#!/usr/bin/env bash
# Build agent.zip from an EXPLICIT allowlist — never `zip -r .`, which would
# happily sweep in .git, .env, tests, __pycache__, and downloaded task data.
#
# Usage: scripts/build_agent.sh [output_zip_path]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_ZIP="${1:-$ROOT_DIR/agent.zip}"
STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT

# Required at the zip root — the build fails loudly if either is missing,
# rather than shipping a zip that dies with ModuleNotFoundError on the host.
REQUIRED_ROOT_FILES=(main.py jeopardy.py requirements.txt)

# Present once Nandh's contracts.py lands; optional until then so this
# script keeps working through the whole integration timeline.
OPTIONAL_ROOT_FILES=(contracts.py)

# Team-owned packages. Included only if they exist yet (Gate 1+), so this
# script is useful from day one instead of blocking on every workstream.
PACKAGE_DIRS=(orchestrator solver tools)

echo "==> Staging allowlisted files into $STAGE_DIR"

for f in "${REQUIRED_ROOT_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: required file missing from repo root: $f" >&2
    exit 1
  fi
  cp "$f" "$STAGE_DIR/"
  echo "  + $f"
done

for f in "${OPTIONAL_ROOT_FILES[@]}"; do
  if [[ -f "$f" ]]; then
    cp "$f" "$STAGE_DIR/"
    echo "  + $f"
  else
    echo "  . skipping (not present yet): $f"
  fi
done

for d in "${PACKAGE_DIRS[@]}"; do
  if [[ ! -d "$d" ]]; then
    echo "  . skipping (not present yet): $d/"
    continue
  fi
  echo "  + $d/ (filtered)"
  while IFS= read -r -d '' rel; do
    mkdir -p "$STAGE_DIR/$d/$(dirname "$rel")"
    cp "$d/$rel" "$STAGE_DIR/$d/$rel"
  done < <(cd "$d" && find . -type f \
      ! -path '*/__pycache__/*' \
      ! -name '*.pyc' \
      ! -path '*/.pytest_cache/*' \
      ! -path '*/tests/*' \
      ! -path '*/fixtures/*' \
      ! -name '*.log' \
      ! -name '.DS_Store' \
      -print0)
done

echo "==> Scanning staged files for credentials"
"$ROOT_DIR/scripts/_scan_secrets.sh" "$STAGE_DIR"

# Reproducibility: `cp` stamps copy-time mtimes and can pick up local umask
# permissions, so two builds of the identical commit would otherwise produce
# byte-different zips. Pin every staged file to the commit's own timestamp
# and a fixed mode, and feed zip a SORTED file list (filesystem readdir
# order isn't guaranteed stable across runs/machines either).
COMMIT_EPOCH="$(git -C "$ROOT_DIR" log -1 --format=%ct 2>/dev/null || echo 0)"
find "$STAGE_DIR" -type f -exec chmod 644 {} +
python3 - "$STAGE_DIR" "$COMMIT_EPOCH" <<'PY'
import os, sys
stage, epoch = sys.argv[1], int(sys.argv[2])
for root, _dirs, files in os.walk(stage):
    for name in files:
        path = os.path.join(root, name)
        os.utime(path, (epoch, epoch))
PY

echo "==> Writing $OUT_ZIP from an explicit, sorted file list"
rm -f "$OUT_ZIP"
( cd "$STAGE_DIR" && find . -type f | sed 's|^\./||' | LC_ALL=C sort | zip -X -q "$OUT_ZIP" -@ )

COMMIT_SHA="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo "unknown (not a git checkout)")"
if command -v shasum >/dev/null 2>&1; then
  CHECKSUM="$(shasum -a 256 "$OUT_ZIP" | awk '{print $1}')"
else
  CHECKSUM="$(sha256sum "$OUT_ZIP" | awk '{print $1}')"
fi

echo "==> Build complete"
echo "commit_sha=$COMMIT_SHA"
echo "zip_sha256=$CHECKSUM"
echo "zip_path=$OUT_ZIP"

echo "==> Verifying $OUT_ZIP"
"$ROOT_DIR/scripts/verify_zip.sh" "$OUT_ZIP"
