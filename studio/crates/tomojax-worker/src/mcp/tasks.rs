//! An MCP task keeps its result in this server and forwards cancellation to its worker.
use super::{Context, Server, identity};
use rmcp::{
    ErrorData as McpError,
    model::{CallToolResponse, CreateTaskResult, ProtocolVersion},
};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

tokio::task_local! { pub static CANCEL: CancellationToken; }
const TTL: Duration = Duration::from_secs(86_400);

struct Entry {
    activity: super::activity::Activity,
    started: std::time::Instant,
    state: Value,
    cancel: CancellationToken,
}
#[derive(Default, Clone)]
pub struct Tasks(Arc<Mutex<HashMap<String, Entry>>>);

pub fn negotiated(ctx: &Context) -> bool {
    ctx.protocol_version() == Some(ProtocolVersion::V_2026_07_28)
        && ctx
            .meta
            .client_capabilities()
            .is_some_and(|c| c.supports_tasks())
}

impl Tasks {
    pub fn start(
        &self,
        server: Server,
        name: String,
        args: Value,
    ) -> Result<CallToolResponse, McpError> {
        let id = Uuid::new_v4().simple().to_string();
        let now = jiff::Timestamp::now().to_string();
        let state = json!({"taskId":id,"status":"working","statusMessage":format!("Starting {name}"),"createdAt":now,"lastUpdatedAt":now,"ttlMs":86_400_000,"pollIntervalMs":2000});
        let mut created = state.clone();
        created["resultType"] = json!("task");
        created["_meta"] = json!({"io.modelcontextprotocol/serverInfo":identity()});
        let task: CreateTaskResult = serde_json::from_value(created)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        let cancel = CancellationToken::new();
        let activity = super::activity::Activity::default();
        self.0.lock().unwrap().insert(
            id.clone(),
            Entry {
                activity: activity.clone(),
                started: std::time::Instant::now(),
                state,
                cancel: cancel.clone(),
            },
        );
        let entries = Arc::downgrade(&self.0);
        let key = id.clone();
        tokio::spawn(async move {
            tokio::time::sleep(TTL).await;
            if let Some(entries) = entries.upgrade()
                && let Some(item) = entries.lock().unwrap().remove(&key)
            {
                item.cancel.cancel();
            }
        });
        tokio::spawn(async move {
            let reply = CANCEL
                .scope(
                    cancel,
                    super::activity::CURRENT.scope(activity, server.execute(&name, args)),
                )
                .await;

            if let Some(task) = server.tasks.0.lock().unwrap().get_mut(&id) {
                let state = &mut task.state;
                state.as_object_mut().unwrap().shift_remove("statusMessage");
                match reply {
                    Err(e) => {
                        state["status"] = json!("failed");
                        state["error"] = json!(e);
                    }
                    Ok(reply)
                        if reply.pointer("/structuredContent/kind") == Some(&json!("canceled")) =>
                    {
                        state["status"] = json!("cancelled");
                    }
                    Ok(mut reply) => {
                        state["status"] = json!("completed");
                        reply["resultType"] = json!("complete");
                        state["result"] = reply;
                    }
                }
                state["lastUpdatedAt"] = json!(jiff::Timestamp::now().to_string());
            }
        });
        Ok(task.into())
    }

    pub fn lookup(&self, id: &str, cancel: bool) -> Result<Value, McpError> {
        let tasks = self.0.lock().unwrap();
        let entry = tasks
            .get(id)
            .ok_or_else(|| McpError::invalid_params("Unknown or expired task ID", None))?;
        if cancel {
            entry.cancel.cancel();
        }
        let mut state = entry.state.clone();
        if state["status"] == "working" {
            entry.activity.update(&mut state, entry.started);
        }
        Ok(state)
    }
}

/// Keep the actionable capability error on the wire. rmcp validates task methods
/// before it calls ServerHandler, so this belongs at the service boundary.
pub struct Service(pub Server);
impl rmcp::Service<rmcp::RoleServer> for Service {
    async fn handle_request(
        &self,
        req: rmcp::model::ClientRequest,
        ctx: Context,
    ) -> Result<rmcp::model::ServerResult, McpError> {
        use rmcp::model::ClientRequest::*;
        let task = matches!(
            &req,
            GetTaskRequest(_) | CancelTaskRequest(_) | UpdateTaskRequest(_)
        );
        rmcp::Service::handle_request(&self.0, req, ctx)
            .await
            .map(|result| match result {
                rmcp::model::ServerResult::TaskAckResult(_) => rmcp::model::CustomResult::new(json!({"resultType":"complete","_meta":{"io.modelcontextprotocol/serverInfo":identity()}})).into(),
                result => result,
            })
            .map_err(|mut e| {
                if task && e.code == rmcp::model::ErrorCode::MISSING_REQUIRED_CLIENT_CAPABILITY {
                    e.message = "Declare the tasks extension on this request".into();
                }
                e
            })
    }
    async fn handle_notification(
        &self,
        req: rmcp::model::ClientNotification,
        ctx: rmcp::service::NotificationContext<rmcp::RoleServer>,
    ) -> Result<(), McpError> {
        rmcp::Service::handle_notification(&self.0, req, ctx).await
    }
    fn get_info(&self) -> rmcp::model::ServerConfig {
        rmcp::ServerHandler::get_info(&self.0)
    }
}
