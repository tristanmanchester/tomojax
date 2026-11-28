//! Transfer a file between the worker and the MCP client's project directory.
use super::{
    connection::{Inventory, Session},
    jobs::path,
};
use crate::transfer::{Download, Upload};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

fn laptop(name: &Path) -> Result<PathBuf> {
    let base = std::env::var_os("CLAUDE_PROJECT_DIR")
        .map(PathBuf::from)
        .unwrap_or(std::env::current_dir()?);
    Ok(base.join(name))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Get {
    path: String,
    to: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Put {
    from: PathBuf,
    path: String,
}

pub async fn download(session: &Session, inventory: &Inventory, args: Value) -> Result<Value> {
    let Get { path: remote, to } = serde_json::from_value(args)?;
    let src = path(inventory, &remote)?;
    let dest = laptop(&to)?;
    let reply = session
        .request(json!({"type":"download","path":src}))
        .await?;
    let result = reply.result()?;
    let bytes = result["bytes"].as_u64().context("Missing download size")?;
    let digest = result["sha256"]
        .as_str()
        .context("Missing download digest")?
        .to_owned();
    let report = json!({"path":src,"to":dest,"bytes":bytes,"sha256":digest});
    tokio::task::spawn_blocking(move || {
        let mut stage = Upload::start(dest, bytes, digest)?;
        stage.write(&reply.bytes)?;
        stage.publish()
    })
    .await??;
    Ok(report)
}

pub async fn upload(session: &Session, inventory: &Inventory, args: Value) -> Result<Value> {
    let Put { from, path: remote } = serde_json::from_value(args)?;
    let src = laptop(&from)?;
    let dest = path(inventory, &remote)?;
    let file = src.clone();
    let (data, digest) = tokio::task::spawn_blocking(move || -> Result<_> {
        let mut source = Download::open(&file)?;
        let bytes = source.read(source.size.try_into()?)?;
        Ok((bytes, source.digest()))
    })
    .await??;
    let bytes = data.len();
    let cmd =
        json!({"type":"upload","path":dest,"bytes":bytes,"sha256":digest,"notify_ready":true});
    let ack = session.upload(cmd, data).await?;
    ensure!(
        ack.result()?["type"] == "ack",
        "Worker did not acknowledge the upload"
    );
    Ok(json!({"from":src,"path":dest,"bytes":bytes,"sha256":digest}))
}
