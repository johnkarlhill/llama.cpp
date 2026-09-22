#!/usr/bin/env bash
# session-end.sh — session-end command support script.
# Takes the PROJECT DIRECTORY as $1 (the model passes its own cwd, which is
# the session's project dir — interpolation runs from the server cwd which is
# NOT project-specific, so we must NOT rely on pwd here).
# 1. Finds the current session for THAT project (main session, no parentID)
# 2. Exports the FULL session via `opencode export` (info + messages)
# 3. Prints a CONTEXT BLOCK for the model to use when writing the handoff
set -u

TIMESTAMP=$(date +%Y%m%d%H%M%S)

PROJECT_DIR="${1:-$(pwd)}"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "META_ERROR: project directory does not exist: $PROJECT_DIR"
    echo "TIMESTAMP=$TIMESTAMP"
    exit 0
fi

# Find the current session ID via the API:
# main session (no parentID) in PROJECT_DIR, most recently updated.
SESSION_ID=$(PROJ_DIR="$PROJECT_DIR" curl -s http://127.0.0.1:4096/api/session 2>/dev/null | python3 -c "
import json, os, sys
try:
    with open('/dev/stdin') as f:
        data = json.load(f)
except Exception:
    print(''); sys.exit(0)
sessions = data.get('data', data) if isinstance(data, dict) else data
if not isinstance(sessions, list) or not sessions:
    print(''); sys.exit(0)
cwd = os.environ.get('PROJ_DIR', '')
def loc_dir(s):
    loc = s.get('location') or {}
    return loc.get('directory') or s.get('directory') or ''
main = [s for s in sessions if not s.get('parentID') and loc_dir(s) == cwd]
if not main:
    main = [s for s in sessions if not s.get('parentID')]
main.sort(key=lambda s: s.get('time', {}).get('updated', 0), reverse=True)
print(main[0].get('id', '') if main else '')
" ) || true

if [ -z "$SESSION_ID" ]; then
    echo "META_ERROR: could not determine current session ID for project $PROJECT_DIR"
    echo "TIMESTAMP=$TIMESTAMP"
    exit 0
fi

mkdir -p "$PROJECT_DIR/sessions"
EXPORT_FILE="$PROJECT_DIR/sessions/session_export_${TIMESTAMP}.json"

# Full export via opencode CLI (info + messages, the REAL backup)
/home/johnk/.opencode/bin/opencode export "$SESSION_ID" > "$EXPORT_FILE" 2>/tmp/oc_export_err.txt
if [ ! -s "$EXPORT_FILE" ]; then
    echo "EXPORT_ERROR: opencode export produced empty output"
    echo "  stderr: $(head -c 200 /tmp/oc_export_err.txt)"
    echo "TIMESTAMP=$TIMESTAMP"
    echo "SESSION_ID=$SESSION_ID"
    echo "PROJECT_DIR=$PROJECT_DIR"
    exit 0
fi

# Print metadata block for the handoff
python3 - "$SESSION_ID" "$TIMESTAMP" "$EXPORT_FILE" "$PROJECT_DIR" <<'PYEOF'
import json, sys, os
sid, ts, ef, pdir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(ef) as f:
    data = json.load(f)
info = data.get('info', {})
tokens = info.get('tokens', {})
cache = tokens.get('cache', {})
print(f'TITLE={info.get("title", "unknown")}')
print(f'ID={sid}')
model = info.get('model') or {}
print(f'MODEL={model.get("id", "unknown") if isinstance(model, dict) else "unknown"}')
print(f'AGENT={info.get("agent", "unknown")}')
print(f'IN={tokens.get("input", 0)}')
print(f'OUT={tokens.get("output", 0)}')
print(f'REASONING={tokens.get("reasoning", 0)}')
print(f'CACHE={cache.get("read", 0)}')
print(f'COST={info.get("cost", 0)}')
print(f'DIR={pdir}')
msgs = data.get('messages', [])
print(f'MESSAGES={len(msgs)}')
print(f'TIMESTAMP={ts}')
print(f'EXPORT_FILE={ef}')
print(f'EXPORT_BYTES={os.path.getsize(ef)}')
PYEOF
echo "--- GIT STATUS (from project dir) ---"
(cd "$PROJECT_DIR" && git status --short 2>/dev/null)
echo "--- GIT DIFFSTAT ---"
(cd "$PROJECT_DIR" && git diff --stat 2>/dev/null)
echo "--- SESSIONS DIR (newest first) ---"
ls -lt "$PROJECT_DIR/sessions/" 2>/dev/null | head -6

