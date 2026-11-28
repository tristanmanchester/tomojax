//! An SSH session runs the same executable as its client, addressed by its digest.
use super::{
    Failure, Profile,
    command::{Route, run_probe},
};
use flate2::{Compression, write::GzEncoder};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Seek, copy};
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

/// Return the executable on the last hop, publishing it there on a cache miss.
pub fn worker_path(profile: &Profile, binary: &Path) -> Result<String, Failure> {
    let Some(route) = Route::resolve(profile)? else {
        return Ok(binary.canonicalize()?.to_string_lossy().into());
    };
    let run = |command: String, input: Stdio, timeout: Duration| {
        let depth = profile.hops().len();
        let output = run_probe(
            &mut route.command(depth, command, true),
            timeout,
            false,
            input,
        )
        .map_err(|e| route.classify(e, depth))?;
        if output.code == Some(255) {
            return Err(route.classify(Failure::ssh(output.err.trim()), depth));
        }
        if !output.success {
            return Err(Failure::blocked(output.err));
        }
        Ok(output.out)
    };
    let mut file = File::open(binary)?;
    let mut hash = Sha256::new();
    copy(&mut file, &mut hash)?;
    let hash = format!("{:x}", hash.finalize());
    let path = format!(".cache/tomojax/worker-{hash}");
    let existing = run(
        format!("cd \"$HOME\" && if test -x '{path}'; then echo present; fi"),
        Stdio::null(),
        Duration::from_secs(20),
    )?;
    if existing.trim() == "present" {
        return Ok(path);
    }
    file.rewind()?;
    let mut gzip = GzEncoder::new(tempfile::tempfile()?, Compression::fast());
    copy(&mut file, &mut gzip)?;
    let mut compressed = gzip.finish()?;
    compressed.rewind()?;
    let script = format!(
        r#"set -eu
cd "$HOME"
umask 077
mkdir -p .cache/tomojax
part=$(mktemp .cache/tomojax/.worker-XXXXXXXXXX)
trap 'rm -f -- "$part"' EXIT
gzip -dc > "$part"
printf '%s  %s\n' '{hash}' "$part" | sha256sum --check --status
chmod 700 "$part"
ln -T -- "$part" '{path}' || {{ test -x '{path}' && printf '%s  %s\n' '{hash}' '{path}' | sha256sum --check --status; }}
"#
    );
    run(script, compressed.into(), Duration::from_secs(300))?;
    Ok(path)
}
