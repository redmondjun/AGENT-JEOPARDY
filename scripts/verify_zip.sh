#!/usr/bin/env bash
# Verify a built agent.zip is safe and correct to submit: main.py at root,
# under the size limits, free of excluded paths, and actually importable in
# a CLEAN directory (never trusting the repo's own environment/state).
#
# Usage: scripts/verify_zip.sh [path-to-agent.zip]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZIP_PATH="${1:-$ROOT_DIR/agent.zip}"

# README: "20 MB compressed, 200 MB uncompressed."
MAX_COMPRESSED_BYTES=$((20 * 1024 * 1024))
MAX_UNCOMPRESSED_BYTES=$((200 * 1024 * 1024))

# The hosted image's preinstalled baseline (README "Set up your Python
# environment") — used only to make the clean-directory import check
# realistic without requiring network access to actually pip-install them
# when `uv` is available.
HOSTED_BASELINE="anthropic,httpx,requests,beautifulsoup4,lxml,numpy,pandas"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "ERROR: zip not found: $ZIP_PATH" >&2
  exit 1
fi

echo "==> Verifying $ZIP_PATH"

# 1. main.py must be at the zip root, not nested in a subfolder.
if ! unzip -l "$ZIP_PATH" | awk '{print $NF}' | grep -qx 'main.py'; then
  echo "ERROR: main.py is not at the zip root" >&2
  exit 1
fi
echo "  [ok] main.py is at the zip root"

# 2. Compressed size.
if command -v stat >/dev/null 2>&1 && stat -f%z "$ZIP_PATH" >/dev/null 2>&1; then
  COMPRESSED_BYTES=$(stat -f%z "$ZIP_PATH")            # BSD/macOS stat
else
  COMPRESSED_BYTES=$(stat -c%s "$ZIP_PATH")             # GNU stat
fi
if (( COMPRESSED_BYTES > MAX_COMPRESSED_BYTES )); then
  echo "ERROR: compressed size ${COMPRESSED_BYTES}B exceeds ${MAX_COMPRESSED_BYTES}B limit" >&2
  exit 1
fi
echo "  [ok] compressed size: ${COMPRESSED_BYTES}B (limit ${MAX_COMPRESSED_BYTES}B)"

# 3. Uncompressed size, from zip's own totals line.
UNCOMPRESSED_BYTES=$(unzip -l "$ZIP_PATH" | awk 'END{print $1}')
if (( UNCOMPRESSED_BYTES > MAX_UNCOMPRESSED_BYTES )); then
  echo "ERROR: uncompressed size ${UNCOMPRESSED_BYTES}B exceeds ${MAX_UNCOMPRESSED_BYTES}B limit" >&2
  exit 1
fi
echo "  [ok] uncompressed size: ${UNCOMPRESSED_BYTES}B (limit ${MAX_UNCOMPRESSED_BYTES}B)"

# 4. No excluded paths made it in.
BAD_ENTRIES=$(unzip -l "$ZIP_PATH" | awk '{print $NF}' \
  | grep -E '(^|/)(\.env|\.git|__pycache__|\.pytest_cache|tests|fixtures)(/|$)|\.log$' || true)
if [[ -n "$BAD_ENTRIES" ]]; then
  echo "ERROR: zip contains excluded paths:" >&2
  echo "$BAD_ENTRIES" >&2
  exit 1
fi
echo "  [ok] no .env/.git/caches/tests/logs present"

# 5. Credential scan on the actual zip contents (belt-and-suspenders on top
#    of build_agent.sh's pre-zip scan of the staging dir).
SCAN_DIR="$(mktemp -d)"
unzip -q "$ZIP_PATH" -d "$SCAN_DIR"
"$ROOT_DIR/scripts/_scan_secrets.sh" "$SCAN_DIR"

# 6. Compile every module in a clean directory — catches syntax errors with
#    zero dependencies required.
echo "==> Compiling every .py in a clean extracted copy"
python3 -m compileall -q "$SCAN_DIR"
echo "  [ok] compiles"

# 7. Actually import main.py in a clean directory, with the hosted image's
#    baseline packages available, so a forgotten import dies here instead of
#    on the host. Prefers `uv` (README's own documented pattern); falls back
#    to a plain import that only catches errors in the TEAM's own modules if
#    uv isn't available.
echo "==> Importing main.py in a clean extracted copy"
IMPORT_CHECK='
import importlib, sys
sys.path.insert(0, ".")
importlib.import_module("main")
print("main.py imported OK")
'
if command -v uv >/dev/null 2>&1; then
  ( cd "$SCAN_DIR" && \
    JEOPARDY_BASE_URL="https://dummy.invalid" TEAM_API_KEY="dummy_verification_key" \
    uv run -p 3.12 --with "$HOSTED_BASELINE" python -c "$IMPORT_CHECK" )
else
  echo "  [warn] uv not found — falling back to the current interpreter." >&2
  echo "  [warn] third-party ModuleNotFoundError here may be a LOCAL gap, not a real zip defect." >&2
  ( cd "$SCAN_DIR" && \
    JEOPARDY_BASE_URL="https://dummy.invalid" TEAM_API_KEY="dummy_verification_key" \
    python3 -c "$IMPORT_CHECK" )
fi
echo "  [ok] main.py imports cleanly"
rm -rf "$SCAN_DIR"

COMMIT_SHA="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo "unknown (not a git checkout)")"
if command -v shasum >/dev/null 2>&1; then
  CHECKSUM="$(shasum -a 256 "$ZIP_PATH" | awk '{print $1}')"
else
  CHECKSUM="$(sha256sum "$ZIP_PATH" | awk '{print $1}')"
fi

echo "==> Verification passed"
echo "commit_sha=$COMMIT_SHA"
echo "zip_sha256=$CHECKSUM"
