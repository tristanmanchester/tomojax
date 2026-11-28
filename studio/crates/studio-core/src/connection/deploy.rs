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

/// Install on the login host, where uv can reach the package index. The
/// interpreter path printed here must also be visible to the compute nodes.
pub(super) fn install(route: Option<&Route<'_>>) -> Result<String, Failure> {
    let script = format!(
        r#"set -eu
version='{version}'
base="$HOME/.local/share/tomojax/envs"
mkdir -p -- "$base"
envdir=$(mktemp -d "$base/$version-XXXXXXXXXX")
trap 'rm -rf -- "$envdir"' EXIT
uv venv --python '>=3.12,<3.15' "$envdir" >&2
uv pip install --python "$envdir/bin/python" "tomojax[cuda12]==$version" >&2
actual=$("$envdir/bin/python" -m tomojax.cli.main --version)
if [ "$actual" != "tomojax $version" ]; then
    echo "Expected tomojax $version, got $actual" >&2
    exit 1
fi
ln -s -- "$envdir" "$envdir/.publish"
# A signal after rename must never remove the published environment.
trap - EXIT
mv -Tf -- "$envdir/.publish" "$base/$version"
printf '%s\n' "$base/$version/bin/python"
"#,
        version = crate::TOMOJAX_VERSION
    );
    let mut command = if let Some(route) = route {
        route.command(1, script, true)
    } else {
        let mut command = std::process::Command::new("sh");
        command.args(["-c", &script]);
        command
    };
    let result = run_probe(
        &mut command,
        Duration::from_secs(1800),
        false,
        Stdio::null(),
    )?;
    if result.success {
        Ok(result.out)
    } else if result.code == Some(255)
        && let Some(route) = route
    {
        Err(route.classify(Failure::ssh(&result.err), 1))
    } else {
        Err(Failure::blocked(result.err))
    }
}
