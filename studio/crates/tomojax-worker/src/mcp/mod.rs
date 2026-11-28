//! Model Context Protocol over stdio.
mod activity;
mod connection;
mod describe;
mod guidance;
mod images;
mod inspect;
mod jobs;
mod ledger;
mod namespace;
mod parts;
mod schema;
mod summary;
mod tasks;
mod tools;
mod transfers;
mod view;
mod workflow;

use anyhow::{Result, bail};
use rmcp::{
    ErrorData as McpError, RoleServer, ServerHandler, ServiceExt, model::*, service::RequestContext,
};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

type Context = RequestContext<RoleServer>;

fn response<T: DeserializeOwned>(mut value: Value, ctx: &Context) -> Result<T, McpError> {
    if ctx
        .protocol_version()
        .is_some_and(|v| v == ProtocolVersion::V_2026_07_28)
    {
        value["resultType"] = json!("complete");
        value["_meta"] = json!({"io.modelcontextprotocol/serverInfo":identity()});
    }
    serde_json::from_value(value).map_err(|err| McpError::internal_error(err.to_string(), None))
}

fn identity() -> Implementation {
    let version = include_str!("../../../../../pyproject.toml")
        .lines()
        .find_map(|line| {
            line.strip_prefix("version = \"")
                .and_then(|v| v.strip_suffix('"'))
        })
        .expect("TomoJAX package version");
    Implementation::new("tomojax", version)
}

#[derive(Clone)]
struct Server {
    tasks: tasks::Tasks,
    parts: std::sync::Arc<std::sync::Mutex<parts::Parts>>,
    connections: std::sync::Arc<connection::Connections>,
}

impl Server {
    async fn execute(&self, name: &str, arguments: Value) -> Result<Value, McpError> {
        let reply = match tools::call(&self.connections, name, arguments).await {
            Ok(reply) => reply,
            Err(error) => {
                if let Some(lost) = error.downcast_ref::<namespace::Lost>() {
                    return Err(McpError::internal_error(lost.to_string(), None));
                }
                let message = error.to_string();
                let wire = error
                    .downcast_ref::<connection::WorkerError>()
                    .map(|e| &e.0);
                let kind = wire
                    .and_then(|e| e["kind"].as_str())
                    .unwrap_or("invalid_parameter");
                let invalid = error.downcast_ref::<schema::ArgumentError>();
                let checkpoint = wire.and_then(|w| w["checkpoint"].as_str());
                let next = match kind {
                    _ if invalid.is_some() => invalid.unwrap().next.clone(),
                    _ if checkpoint.is_some() => format!(
                        "Its state is saved in {}: call {name} again with the same arguments, on the same input, to resume (inspect shows what the checkpoint holds).",
                        checkpoint.unwrap()
                    ),
                    "superseded" => "Use the newer job.".into(),
                    "out_of_memory" => {
                        "Try dry_run, slab reconstruction, binning or different devices.".into()
                    }
                    "canceled" => "Resubmit when ready.".into(),
                    "input" if name == "python" => {
                        "Read the error, correct the Python code and retry.".into()
                    }
                    "input" => "Inspect the input path and metadata.".into(),
                    "internal" => "Check status for worker and device information.".into(),
                    _ => format!("Check the {name} tool parameters and retry."),
                };
                let path = invalid
                    .map(|e| e.path.as_str())
                    .or_else(|| wire.and_then(|e| e["path"].as_str()));
                let mut data = json!({"kind":kind,"path":path,"message":message,"next":next});
                if let Some(file) = checkpoint {
                    data["checkpoint"] = json!(file);
                }
                json!({"content":[{"type":"text","text":format!("{kind}: {message}. {next}")}],"structuredContent":data,"isError":true})
            }
        };
        Ok(self.parts.lock().unwrap().bounded(reply))
    }
}

impl ServerHandler for Server {
    fn get_info(&self) -> ServerConfig {
        let capabilities = ServerCapabilities::builder()
            .enable_resources()
            .enable_prompts()
            .enable_tasks()
            .enable_tools()
            .build();
        ServerConfig::new(capabilities).with_server_info(identity()).with_instructions(
            "Use inspect to identify an acquisition and open to load it. Use describe for parameters and units. Check geometry and labelled slices before choosing a reconstruction. Handles last until closed, the executor restarts or the session ends. For work that must survive disconnection, submit a job with saved input paths. Use jobs to recover it. Data paths belong to the connected machine. The laptop paths of upload and download are relative to the client project directory."
        )
    }
    async fn list_tools(
        &self,
        _: Option<PaginatedRequestParams>,
        ctx: Context,
    ) -> Result<ListToolsResult, McpError> {
        response(
            json!({"tools":tools::list(self.connections.name=="local")}),
            &ctx,
        )
    }
    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        ctx: Context,
    ) -> Result<CallToolResponse, McpError> {
        let name = request.name.as_ref();
        let arguments = Value::Object(request.arguments.unwrap_or_default());
        if tasks::negotiated(&ctx)
            && (name == "python" && self.connections.name == "local"
                || workflow::OPERATIONS.contains(&name))
            && arguments["dry_run"] != true
        {
            return self.tasks.start(self.clone(), name.into(), arguments);
        }
        let activity = activity::Activity::default();
        let call = tasks::CANCEL.scope(
            ctx.ct.clone(),
            activity::CURRENT.scope(activity.clone(), self.execute(name, arguments)),
        );
        tokio::pin!(call);
        let token = ctx.meta.get_progress_token().or_else(|| {
            ctx.meta
                .get("progress_token")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
        });
        let reply = if let Some(token) = token {
            tokio::select! {reply=&mut call=>reply,_=activity.notify(&ctx,token)=>call.await}
        } else {
            call.await
        };
        let result: CallToolResult = response(reply?, &ctx)?;
        Ok(result.into())
    }
    async fn get_task(&self, req: GetTaskParams, ctx: Context) -> Result<GetTaskResult, McpError> {
        response(self.tasks.lookup(&req.task_id, false)?, &ctx)
    }
    async fn cancel_task(&self, req: CancelTaskParams, _ctx: Context) -> Result<(), McpError> {
        self.tasks.lookup(&req.task_id, true)?;
        Ok(())
    }
    async fn update_task(&self, req: UpdateTaskParams, _ctx: Context) -> Result<(), McpError> {
        self.tasks.lookup(&req.task_id, false)?;
        Ok(())
    }
    async fn list_prompts(
        &self,
        _: Option<PaginatedRequestParams>,
        ctx: Context,
    ) -> Result<ListPromptsResult, McpError> {
        response(json!({"prompts":guidance::prompts()}), &ctx)
    }
    async fn get_prompt(
        &self,
        request: GetPromptRequestParams,
        ctx: Context,
    ) -> Result<GetPromptResponse, McpError> {
        let result: GetPromptResult = response(guidance::prompt(&request)?, &ctx)?;
        Ok(result.into())
    }
    async fn list_resource_templates(
        &self,
        _: Option<PaginatedRequestParams>,
        ctx: Context,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        response(
            json!({"resourceTemplates":[{"name":"read","uriTemplate":"tomojax://result/{call}/{n}","mimeType":"application/json","description":"Read a page by index. JSON escaping can at most double its ASCII length."},{"name":"describe","uriTemplate":"tomojax://describe/{operation}","mimeType":"application/json","description":"Read the parameters, choices and units for one worker operation."},{"name":"guide","description":"Read a reference of the packaged Agent Skill, the single source of the guides.","uriTemplate":"tomojax://guide/{topic}","mimeType":"text/markdown"}]}),
            &ctx,
        )
    }
    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        ctx: Context,
    ) -> Result<ReadResourceResponse, McpError> {
        let (text, mime) = if let Some(key) = request.uri.strip_prefix("tomojax://result/") {
            let page = self
                .parts
                .lock()
                .unwrap()
                .read(key, std::time::Instant::now())
                .map_err(|_| {
                    McpError::internal_error(
                        format!("Error creating resource from template {}", request.uri),
                        Some(json!({"uri":request.uri})),
                    )
                })?;
            (
                serde_json::to_string_pretty(&page).unwrap(),
                "application/json",
            )
        } else if let Some(operation) = request.uri.strip_prefix("tomojax://describe/") {
            let (data, _) = tools::run(
                &self.connections,
                "describe",
                json!({"operation":operation}),
            )
            .await
            .map_err(|e| McpError::resource_not_found(e.to_string(), None))?;
            (
                serde_json::to_string_pretty(&data).unwrap(),
                "application/json",
            )
        } else {
            let topic = request.uri.strip_prefix("tomojax://guide/").unwrap_or("");
            (
                guidance::guide(topic).ok_or_else(|| {
                    McpError::resource_not_found(
                        "Unknown TomoJAX guide",
                        Some(json!({"uri":request.uri})),
                    )
                })?,
                "text/markdown",
            )
        };
        let result: ReadResourceResult = response(
            json!({"contents":[{"uri":request.uri,"text":text,"mimeType":mime}]}),
            &ctx,
        )?;
        Ok(result.into())
    }
}

#[tokio::main]
pub async fn run(mut args: impl Iterator<Item = String>) -> Result<()> {
    let name = match (args.next(), args.next(), args.next()) {
        (None, None, None) => String::from("local"),
        (Some(flag), Some(name), None) if flag == "--connection" && !name.is_empty() => name,
        _ => bail!("Usage: tomojax-worker mcp [--connection NAME]"),
    };
    let connections = std::sync::Arc::new(connection::Connections::new(name));
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let mut interrupt = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
    let serving = async {
        tasks::Service(Server {
            tasks: Default::default(),
            parts: Default::default(),
            connections: connections.clone(),
        })
        .serve(rmcp::transport::stdio())
        .await?
        .waiting()
        .await?;
        anyhow::Ok(())
    };
    let result = tokio::select! {
        done=serving=>done.map(|()| None),
        _=terminate.recv()=>Ok(Some(signal_hook::consts::SIGTERM)),
        _=interrupt.recv()=>Ok(Some(signal_hook::consts::SIGINT)),
    };
    connections.close("Server shutdown").await;
    if let Some(signal) = result? {
        signal_hook::low_level::emulate_default_handler(signal)?;
    }
    Ok(())
}
