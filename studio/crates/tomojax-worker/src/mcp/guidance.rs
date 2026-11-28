//! The packaged agent guides also supply MCP resources and prompt bodies.
use super::{GetPromptRequestParams, McpError, Value, json};

const WORKFLOWS: &[&str] = &[
    "reconstruct-scan",
    "lab-ct-start-to-finish",
    "find-the-center",
    "check-a-reconstruction",
];
macro_rules! references {
    ($($topic:literal),+ $(,)?) => {
        const GUIDES: &[(&str, &str)] = &[$(($topic, include_str!(concat!(
            "../../../../../src/tomojax/skill/tomojax/references/", $topic, ".md"
        )))),+];
    };
}
references!(
    "reconstruct-scan",
    "lab-ct-start-to-finish",
    "find-the-center",
    "check-a-reconstruction",
    "python",
    "solvers",
    "geometry",
    "calibration",
    "pointing-and-features",
    "tuning-without-truth",
    "thin-shell",
    "large-scans",
    "reporting-bugs"
);

fn reference(topic: &str) -> Option<&'static str> {
    GUIDES
        .iter()
        .find(|(name, _)| *name == topic)
        .map(|(_, text)| text.trim())
}
pub fn guide(topic: &str) -> Option<String> {
    if topic == "python" {
        return Some(
            ["python", "solvers", "geometry", "calibration"]
                .map(|name| reference(name).unwrap())
                .join("\n\n"),
        );
    }
    reference(topic).map(str::to_string)
}
pub fn prompts() -> Vec<Value> {
    WORKFLOWS.iter().map(|topic| {
        let description = reference(topic).unwrap().lines().next().unwrap().trim_start_matches("# ");
        json!({"name":topic,"description":description,"arguments":[{"name":"source","required":false}]})
    }).collect()
}
pub fn prompt(request: &GetPromptRequestParams) -> Result<Value, McpError> {
    if !WORKFLOWS.contains(&request.name.as_str()) {
        return Err(McpError::invalid_params("Unknown TomoJAX workflow", None));
    }
    let mut text = reference(&request.name).unwrap().to_owned();
    if let Some(source) = request
        .arguments
        .as_ref()
        .and_then(|args| args.get("source"))
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty())
    {
        text.push_str("\n\nInput: ");
        text.push_str(source);
    }
    let description = text.lines().next().unwrap().trim_start_matches("# ");
    Ok(
        json!({"description":description,"messages":[{"role":"user","content":{"type":"text","text":text}}]}),
    )
}
