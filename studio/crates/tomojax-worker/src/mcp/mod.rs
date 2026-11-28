//! Model Context Protocol over stdio.
mod connection;
mod describe;
mod guidance;
mod images;
mod inspect;
mod jobs;
mod parts;
mod tools;

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

struct Server {
    parts: std::sync::Mutex<parts::Parts>,
    connections: std::sync::Arc<connection::Connections>,
}

impl ServerHandler for Server {
    fn get_info(&self) -> ServerConfig {
        let capabilities = ServerCapabilities::builder()
            .enable_resources()
            .enable_prompts()
            .enable_tools()
            .build();
        ServerConfig::new(capabilities).with_server_info(identity())
    }
    async fn list_tools(
        &self,
        _: Option<PaginatedRequestParams>,
        ctx: Context,
    ) -> Result<ListToolsResult, McpError> {
        response(json!({"tools":tools::list()}), &ctx)
    }
    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        ctx: Context,
    ) -> Result<CallToolResponse, McpError> {
        let name = request.name.as_ref();
        let arguments = Value::Object(request.arguments.unwrap_or_default());
        let reply = match tools::call(&self.connections, name, arguments).await {
            Ok(reply) => reply,
            Err(error) => {
                let message = error.to_string();
                let next = format!("Check the {name} tool parameters and retry.");
                json!({"content":[{"type":"text","text":format!("invalid_parameter: {message}. {next}")}],
                    "structuredContent":{"kind":"invalid_parameter","path":null,"message":message,"next":next},"isError":true})
            }
        };
        let result: CallToolResult = response(self.parts.lock().unwrap().bounded(reply), &ctx)?;
        Ok(result.into())
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
    let serving = Server {
        parts: Default::default(),
        connections: connections.clone(),
    }
    .serve(rmcp::transport::stdio())
    .await?
    .waiting()
    .await;
    connections.close().await;
    serving?;
    Ok(())
}
