use crate::directory::optional;
use serde_json::Value;
use std::{
    fs::{File, OpenOptions},
    io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write},
    path::Path,
};

pub struct Log(File);

impl Log {
    pub fn open(dir: &Path) -> io::Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(dir.join("replies.jsonl"))?;
        let length = file.metadata()?.len();
        let mut end = length;
        let mut buf = [0; 4096];
        while end > 0 {
            let count = end.min(buf.len() as u64) as usize;
            file.seek(SeekFrom::Start(end - count as u64))?;
            file.read_exact(&mut buf[..count])?;
            if let Some(at) = buf[..count].iter().rposition(|b| *b == b'\n') {
                end -= (count - 1 - at) as u64;
                break;
            }
            end -= count as u64;
        }
        if end != length {
            file.set_len(end)?;
        }
        Ok(Self(file))
    }
    pub fn append(&mut self, record: &Value) -> io::Result<()> {
        let mut line = serde_json::to_vec(record)?;
        line.push(b'\n');
        self.0.write_all(&line)
    }
}

pub fn page(dir: &Path, mut offset: u64, limit: u64) -> io::Result<(Vec<Value>, u64)> {
    let Some(mut file) = optional(File::open(dir.join("replies.jsonl")))? else {
        return Ok((vec![], offset));
    };
    file.seek(SeekFrom::Start(offset))?;
    let mut reader = BufReader::new(file);
    let mut records: Vec<Value> = Vec::new();
    let mut used = 0;
    let mut line = Vec::new();
    loop {
        let bytes = reader.read_until(b'\n', &mut line)? as u64;
        if !line.ends_with(b"\n") || (used > 0 && used + bytes > limit) {
            break;
        }
        let row: Value = serde_json::from_slice(&line)?;
        if let Some(last) = records.last()
            && row.pointer("/record/event") == Some(&Value::String("advanced".into()))
            && last.pointer("/record/event") == row.pointer("/record/event")
            && last["request"] == row["request"]
            && last["record"]["task"] == row["record"]["task"]
        {
            records.pop();
        }
        records.push(row);
        offset += bytes;
        used += bytes;
        if used >= limit {
            break;
        }
        line.clear();
    }
    Ok((records, offset))
}
