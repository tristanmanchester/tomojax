//! Python request IDs are permanent within a session. Each inline script therefore
//! borrows its own IDs, including for control and cleanup. The log uses the script's IDs.
use crate::executor::Executor;
use serde_json::Value;
use std::{collections::HashMap, io, time::Duration};

pub(crate) struct InlineIds<'a> {
    executor: Box<dyn Executor + 'a>,
    prefix: &'a str,
    to_executor: HashMap<String, String>,
    to_script: HashMap<String, String>,
}

impl<'a> InlineIds<'a> {
    pub(crate) fn new(executor: Box<dyn Executor + 'a>, prefix: &'a str) -> Self {
        Self {
            executor,
            prefix,
            to_executor: HashMap::new(),
            to_script: HashMap::new(),
        }
    }
}

fn translate(value: &mut Value, ids: &HashMap<String, String>) {
    if let Some(mapped) = value.as_str().and_then(|id| ids.get(id)) {
        *value = Value::String(mapped.clone());
    }
}

fn handles(value: &mut Value, ids: &HashMap<String, String>) {
    match value {
        Value::Object(fields) => {
            for (key, child) in fields {
                if key == "handle" {
                    translate(child, ids);
                } else {
                    handles(child, ids);
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(|v| handles(v, ids)),
        _ => {}
    }
}

impl Executor for InlineIds<'_> {
    fn send(&mut self, request: &Value) -> io::Result<()> {
        let id = request["id"]
            .as_str()
            .ok_or_else(|| io::Error::other("missing request ID"))?;
        let wire = format!("@{}:{}", self.prefix, self.to_executor.len());
        self.to_executor.insert(id.into(), wire.clone());
        self.to_script.insert(wire.clone(), id.into());
        let mut request = request.clone();
        request["id"] = Value::String(wire);
        let command = &mut request["command"];
        if command["type"] == "control" {
            translate(&mut command["target"], &self.to_executor);
        }
        handles(command, &self.to_executor);
        self.executor.send(&request)
    }

    fn receive(&mut self, timeout: Duration) -> io::Result<Option<Value>> {
        let Some(mut row) = self.executor.receive(timeout)? else {
            return Ok(None);
        };
        if let Some(request) = row.get_mut("request") {
            translate(request, &self.to_script);
            handles(&mut row, &self.to_script);
        }
        Ok(Some(row))
    }
}
