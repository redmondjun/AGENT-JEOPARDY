#!/usr/bin/env bash
# Grep-based secret scan over a staged zip directory. Only ever prints
# FILENAMES, never the matched text, so a real secret can't end up echoed
# into a CI log by the tool meant to catch it.
set -euo pipefail

STAGE_DIR="${1:?usage: _scan_secrets.sh <staged-dir>}"
FAIL=0

PATTERNS=(
  'AKIA[0-9A-Z]{16}:AWS access key id'
  '-----BEGIN [A-Z ]*PRIVATE KEY-----:private key block'
  'xox[baprs]-[0-9A-Za-z-]+:Slack token'
  'sk-[A-Za-z0-9]{20,}:generic sk- secret'
  'ghp_[A-Za-z0-9]{30,}:GitHub personal access token'
  'team_[A-Za-z0-9]{10,}:a jeopardy TEAM_API_KEY-shaped string'
)

for entry in "${PATTERNS[@]}"; do
  regex="${entry%%:*}"
  label="${entry#*:}"
  matches="$(grep -RIlE "$regex" "$STAGE_DIR" 2>/dev/null || true)"
  if [[ -n "$matches" ]]; then
    echo "SECRET SCAN: possible $label found in:" >&2
    echo "$matches" | sed "s|$STAGE_DIR/||" >&2
    FAIL=1
  fi
done

# The live team key, if this shell has one exported, must never appear
# hardcoded in a staged file (the runner injects it as an env var; nothing
# in the zip should need to know its own value literally).
if [[ -n "${TEAM_API_KEY:-}" ]]; then
  matches="$(grep -RIlF "$TEAM_API_KEY" "$STAGE_DIR" 2>/dev/null || true)"
  if [[ -n "$matches" ]]; then
    echo "SECRET SCAN: current TEAM_API_KEY value is hardcoded in:" >&2
    echo "$matches" | sed "s|$STAGE_DIR/||" >&2
    FAIL=1
  fi
fi

if [[ -e "$STAGE_DIR/.env" ]]; then
  echo "SECRET SCAN: .env is staged for packaging — should never happen" >&2
  FAIL=1
fi

if [[ "$FAIL" -ne 0 ]]; then
  echo "Secret scan failed — aborting build." >&2
  exit 1
fi

echo "Secret scan: clean"
