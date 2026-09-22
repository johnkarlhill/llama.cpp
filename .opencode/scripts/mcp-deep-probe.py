#!/usr/bin/env python3
"""Deep-probe one MCP server: dump the FULL tools/list response and stderr."""
import json
import subprocess
import sys

name = sys.argv[1] if len(sys.argv) > 1 else "searxng"

CMDS = {
    "searxng": ["/home/johnk/.nvm/versions/node/v22.23.1/bin/mcp-searxng"],
    "puppeteer": ["/home/johnk/.local/bin/playwright-mcp-wrapper",
                  "--browser", "chrome",
                  "--executable-path",
                  "/home/johnk/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome"],
    "memory-keeper": ["/home/johnk/.nvm/versions/node/v22.23.1/bin/node",
                      "/home/johnk/.config/opencode/mcp-memory-keeper-wrapper.js"],
}

cmd = CMDS.get(name, CMDS["searxng"])
print(f"Probing {name}: {cmd}")
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True, bufsize=1)

# initialize
proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                             "params": {"protocolVersion": "2024-11-05",
                                        "capabilities": {}, "clientInfo": {"name": "probe", "version": "1.0"}}}) + "\n")
proc.stdin.flush()

# Read lines until we get the initialize RESULT (id=1); skip notifications
init = None
for _ in range(20):
    line = proc.stdout.readline()
    if not line:
        break
    print(f"INIT LINE: {line[:300]}")
    try:
        obj = json.loads(line)
    except Exception:
        continue
    if obj.get("id") == 1:
        init = obj
        break

if init is None:
    stderr = proc.stderr.read()[:300]
    print(f"NO INIT RESULT (stderr: {stderr})")
else:
    print(f"INIT RESULT serverInfo: {init.get('result', {}).get('serverInfo')}")
    print(f"INIT RESULT capabilities: {init.get('result', {}).get('capabilities')}")

proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
proc.stdin.flush()

proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}) + "\n")
proc.stdin.flush()

# Read lines until we get the tools/list RESULT (id=2); skip notifications
for _ in range(20):
    line = proc.stdout.readline()
    if not line:
        break
    print(f"TOOLS LINE: {line[:1500]}")
    try:
        obj = json.loads(line)
    except Exception:
        continue
    if obj.get("id") == 2:
        break

# Wait a moment for stderr
import time
time.sleep(1)
stderr = proc.stderr.read()[:500]
if stderr:
    print(f"STDERR: {stderr}")
try:
    proc.kill()
except Exception:
    pass
