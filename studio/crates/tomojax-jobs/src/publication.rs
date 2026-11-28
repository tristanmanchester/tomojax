//! A receipt and all of its storage are published in one rename.
use crate::directory::{create_record, optional};
use serde_json::{Value, json};
use std::io::Write;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs, io,
    path::{Path, PathBuf},
};

fn visit(value: &Value, tokens: &mut BTreeSet<String>) {
    match value {
        Value::Object(map) if value["type"] == "npy" => {
            if let Some(token) = map.get("artifact").and_then(Value::as_str) {
                tokens.insert(token.into());
            }
        }
        Value::Object(map) => map.values().for_each(|v| visit(v, tokens)),
        Value::Array(items) => items.iter().for_each(|v| visit(v, tokens)),
        _ => {}
    }
}

pub fn tokens(value: &Value) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    visit(value, &mut found);
    found
}

pub struct Publication<'a> {
    stage: &'a Path,
    published: &'a Path,
    copies: BTreeMap<PathBuf, PathBuf>,
}

impl<'a> Publication<'a> {
    pub fn new(stage: &'a Path, published: &'a Path) -> Self {
        Self {
            stage,
            published,
            copies: BTreeMap::new(),
        }
    }

    /// Preserve descriptors including coordinate arrays. Keys in info.arrays are JSON
    /// pointers and must never become file names.
    pub fn arrays(&mut self, value: &mut Value) -> io::Result<()> {
        match value {
            Value::Object(obj)
                if matches!(
                    obj.get("type").and_then(Value::as_str),
                    Some("npy" | "hdf5")
                ) =>
            {
                let src = PathBuf::from(
                    obj["path"]
                        .as_str()
                        .ok_or_else(|| io::Error::other("storage missing path"))?,
                );
                let path = if let Ok(rel) = src.strip_prefix(self.stage) {
                    self.published.join(rel)
                } else if let Some(path) = self.copies.get(&src) {
                    path.clone()
                } else {
                    let ext = if obj["type"] == "npy" { "npy" } else { "h5" };
                    let name = format!("array-{}.{ext}", self.copies.len());
                    fs::copy(&src, self.stage.join(&name))?;
                    let dest = self.published.join(&name);
                    self.copies.insert(src, dest.clone());
                    dest
                };
                obj.insert("path".into(), json!(path));
                obj.remove("artifact");
            }
            Value::Object(obj) if obj.get("type").and_then(Value::as_str) == Some("member") => {
                return Err(io::Error::other(
                    "an executor member cannot be published as file storage",
                ));
            }
            Value::Object(obj) => {
                for child in obj.values_mut() {
                    self.arrays(child)?;
                }
            }
            Value::Array(items) => {
                for child in items {
                    self.arrays(child)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

pub fn receipt(
    dir: &Path,
    command: &Value,
    output: Value,
    stored: &dyn Fn(&Path) -> io::Result<Value>,
) -> io::Result<Value> {
    let stage = dir.join("work/result");
    let dest = dir.join("result");
    let mut files = Publication::new(&stage, &dest);
    let product = &output["value"];
    let mut info = output["info"].clone();
    files.arrays(&mut info["arrays"])?;
    let kind = product["type"]
        .as_str()
        .ok_or_else(|| io::Error::other("executor output missing product"))?;
    let mut result = json!({"product":kind,"info":info,"manifest":output["manifest"]});
    if stage.join("result.nxs").try_exists()? {
        let mut arrays = stored(&stage.join("result.nxs"))?;
        files.arrays(&mut arrays)?;
        result["type"] = json!("dataset");
        result["path"] = json!(dest.join("result.nxs"));
        result["arrays"] = arrays;
        if command.pointer("/operation/type") == Some(&json!("reconstruct_slabs")) {
            result["product"] = json!("reconstruction");
        }
    } else if kind == "frames" && command["type"] == "open" {
        result["type"] = json!("source");
        result["open_parameters"] = command.clone();
    } else if kind == "files" {
        result["type"] = json!("files");
        result["paths"] = product["paths"].clone();
    } else if kind == "inspection" || kind.ends_with("_plan") || kind == "validated" {
        result["type"] = json!("report");
        result["report"] = product.clone();
    } else {
        let mut arrays = product.clone();
        let fields = arrays
            .as_object_mut()
            .ok_or_else(|| io::Error::other("array product must be an object"))?;
        fields.remove("type");
        fields.remove("handle");
        files.arrays(&mut arrays)?;
        result["type"] = json!("arrays");
        result["arrays"] = arrays;
    }
    Ok(result)
}

pub fn publish(dir: &Path, receipt: &Value) -> io::Result<()> {
    let stage = dir.join("work/result");
    create_record(&stage.join("receipt.json"), receipt)?;
    rustix::fs::renameat_with(
        rustix::fs::CWD,
        stage,
        rustix::fs::CWD,
        dir.join("result"),
        rustix::fs::RenameFlags::NOREPLACE,
    )?;
    Ok(())
}

pub fn preview(dir: &Path, attempt: u64, n: u64, mut row: Value) -> io::Result<()> {
    let root = dir.join("preview");
    fs::create_dir_all(&root)?;
    let generation = format!("{attempt}-{n}");
    let dest = root.join(&generation);
    let stage = tempfile::Builder::new()
        .prefix(".preview-")
        .tempdir_in(&root)?;
    Publication::new(stage.path(), &dest).arrays(&mut row["record"]["arrays"])?;
    row["generation"] = json!(generation);
    create_record(&stage.path().join("preview.json"), &row)?;
    fs::rename(stage.path(), &dest)?;
    // Retain the current and immediately preceding generation, across attempts.
    let previous = optional(fs::read_to_string(root.join("current")))?;
    let mut current = tempfile::NamedTempFile::new_in(&root)?;
    current.write_all(generation.as_bytes())?;
    current.persist(root.join("current"))?;
    for entry in fs::read_dir(&root)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name != generation
            && previous.as_deref() != Some(&name)
            && name
                .split_once('-')
                .is_some_and(|(a, b)| a.parse::<u64>().is_ok() && b.parse::<u64>().is_ok())
        {
            fs::remove_dir_all(entry.path())?;
        }
    }
    Ok(())
}

pub fn current_preview(dir: &Path) -> io::Result<Option<Value>> {
    let root = dir.join("preview");
    let Some(generation) = optional(fs::read_to_string(root.join("current")))? else {
        return Ok(None);
    };
    optional(fs::read(root.join(generation).join("preview.json")))?
        .map(|data| Ok(serde_json::from_slice(&data)?))
        .transpose()
}
