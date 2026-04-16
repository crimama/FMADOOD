#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SSH_PASSWORD:-}" ]]; then
  echo "SSH_PASSWORD is not set" >&2
  exit 1
fi

printf '%s\n' "${SSH_PASSWORD}"
