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
mod thumbnail;
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
        Ok(parts::Parts::bound(&self.parts, reply))
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
            r"Use TomoJAX for parallel-beam CT, cone-beam CT and laminography.
Start with inspect to identify an acquisition and open to load it. Use describe for
parameter choices and units. status lists devices, current jobs and live data.
A bare handle supplies the whole object to save/export, or the primary array to an array
field (simulation phantom, reconstruction volume, alignment working image, scan projections).
For a scan operation use a scan handle or .scan on a simulation, alignment or reconstruction.
Select another array with .member, e.g. .poses (alpha, beta, phi in degrees, then x, z, y
in the geometry length unit).
Handles last until closed, worker restart or server exit. Array fields accept handle members,
session arrays, .npy files and file.h5::/dataset in storage order. For acquisitions, open
and select .projections, or open with kind=frames and select .counts. For a first look use
inspect with preview=true, or preview with correction_preview for corrected rows.
View and project.volume also accept saved TomoJAX volumes by path in logical (z, y, x) order.
Workflow source fields accept acquisition paths. Relative data paths use the worker directory.
Use ./@name for a literal path starting with @.
The usual workflow is inspect, open, correct, align the center, reconstruct, view slices
and save. Volumes use (z, y, x), projections (view, v, u). Voxel and detector spacings
share one physical length unit. Check slices before trusting a residual. Use dry_run
before large jobs on tools that support it. See tools/list for installed workflows.
To show the user a structure, draw view marks in array indices and ask; make the agreed
region a feature and outline it with view features. On the local connection, use python for sweeps or custom measures.
Use submit with saved inputs for a job that must survive disconnection. Reconnect and use
jobs to read its receipt and result slices. The laptop paths of upload and download use
the client project directory."
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
            json!({"resourceTemplates":[{"name":"handle","uriTemplate":"tomojax://handle/{id}","mimeType":"application/json","description":"Read a live handle's summary and provenance."},{"name":"read","uriTemplate":"tomojax://result/{call}/{n}","mimeType":"application/json","description":"Read a page by index. JSON escaping can at most double its ASCII length."},{"name":"describe","uriTemplate":"tomojax://describe/{operation}","mimeType":"application/json","description":"Read the parameters, choices and units for one worker operation."},{"name":"guide","description":"Read a reference of the packaged Agent Skill, the single source of the guides.","uriTemplate":"tomojax://guide/{topic}","mimeType":"text/markdown"}]}),
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
        } else if let Some(id) = request.uri.strip_prefix("tomojax://handle/") {
            let missing = |e: anyhow::Error| McpError::resource_not_found(e.to_string(), None);
            let (session, _) = self.connections.open(true).await.map_err(missing)?;
            let ledger = session.space.ledger.lock().unwrap();
            let entry = ledger.record(id).map_err(missing)?;
            let mut data = entry["summary"].clone();
            data["handle"] = json!(id);
            data["manifest"] = entry["manifest"].clone();
            (
                serde_json::to_string_pretty(&data).unwrap(),
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
