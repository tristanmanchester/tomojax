#!/usr/bin/env bash
set -euo pipefail

extra=auto
dry_run=false
for flag in "$@"; do
    case "$flag" in
        --cpu|--cuda)
            if [[ "$extra" != auto ]]; then
                printf 'Choose one of --cpu or --cuda.\n' >&2
                exit 2
            fi
            extra=cpu
            if [[ "$flag" == --cuda ]]; then
                extra=cuda12
            fi
            ;;
        --dry-run) dry_run=true ;;
        *)
            printf 'Unknown option: %s\nUsage: %s [--cpu | --cuda] [--dry-run]\n' "$flag" "$0" >&2
            exit 2
            ;;
    esac
done

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
tmp_cache="$HOME/.cache/tomojax-tmp"
export TMPDIR="${TMPDIR:-$tmp_cache}"
cache_dirs=("$tmp_cache")
platform=$(uname -s)
if [[ "$platform" == Linux ]]; then
    cache_dirs+=("$HOME/.cache/tomojax-studio-target")
fi

if [[ "$extra" == auto ]]; then
    extra=cpu
    if [[ "$platform" == Linux ]] && nvidia-smi >/dev/null 2>&1; then
        extra=cuda12
    fi
fi
sync_command=(uv sync --locked --extra "$extra" --dev)

if [[ "$dry_run" == true ]]; then
    printf 'Worktree: %s\n' "$PWD"
    printf 'mkdir -p -- %q\n' "${cache_dirs[@]}"
    printf 'export TMPDIR=%q\n' "$TMPDIR"
    printf '%s\n' "${sync_command[*]}"
else
    mkdir -p -- "${cache_dirs[@]}"
    exec "${sync_command[@]}"
fi
