#!/usr/bin/env python3
"""Probe MCP servers for tool availability.
Local servers: spawn command, speak JSON-RPC over stdio.
Remote servers: POST tools/list over HTTP.
Prints per-server tool count or error.
"""
import json
import subprocess
import sys
import urllib.request

TIMEOUT = 15  # seconds per server

SERVERS = [
    # (name, type, config)
    ("searxng", "local", ["/home/johnk/.local/bin/mcp-searxng-wrapper"]),
    ("puppeteer", "local", ["/home/johnk/.local/bin/playwright-mcp-wrapper",
                            "--browser", "chrome",
                            "--executable-path",
                            "/home/johnk/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome"]),
    ("memory-keeper", "local", ["/home/johnk/.nvm/versions/node/v22.23.1/bin/node",
                                "/home/johnk/.config/opencode/mcp-memory-keeper-wrapper.js"]),
    ("context7", "remote", "https://mcp.context7.com/mcp"),
    ("gh_grep", "remote", "https://mcp.grep.app"),
]


def probe_local(name, cmd):
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
    except FileNotFoundError as e:
        return f"FAIL: {e}"
    try:
        # MCP handshake: initialize, then tools/list
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05",
                           "capabilities": {}, "clientInfo": {"name": "probe", "version": "1.0"}}}
        proc.stdin.write(json.dumps(init) + "\n")
        proc.stdin.flush()
        # Loop past startup notifications until we get the id=1 result
        init_result = None
        for _ in range(20):
            line = proc.stdout.readline()
            if not line:
                err = proc.stderr.read()[:200]
                return f"FAIL: no response (stderr: {err})"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("id") == 1:
                init_result = obj
                break
        if init_result is None:
            return f"FAIL: no initialize result"
        # notifications/initialized
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        proc.stdin.flush()
        # tools/list — loop past notifications until id=2 result
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}) + "\n")
        proc.stdin.flush()
        tools = []
        for _ in range(20):
            line = proc.stdout.readline()
            if not line:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("id") == 2:
                if "error" in obj:
                    return f"FAIL: {obj['error']}"
                tools = obj.get("result", {}).get("tools", [])
                break
        if not tools:
            return f"FAIL: server connected but exposed 0 tools"
        names = [t.get("name", "?") for t in tools]
        return f"OK: {len(tools)} tools — {', '.join(names[:6])}"
    except Exception as e:
        return f"FAIL: {type(e).__name__}: {e}"
    finally:
        try:
            proc.kill()
        except Exception:
            pass


def probe_remote(name, url):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    })
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read().decode()
        # SSE format: "data: {...}" lines
        tools = []
        for line in raw.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if "result" in obj and "tools" in obj["result"]:
                    tools = obj["result"]["tools"]
        if not tools:
            # try plain JSON
            try:
                obj = json.loads(raw)
                tools = obj.get("result", {}).get("tools", [])
            except json.JSONDecodeError:
                pass
        if not tools:
            return f"FAIL: no tools in response (first 120 chars: {raw[:120]!r})"
        names = [t.get("name", "?") for t in tools]
        return f"OK: {len(tools)} tools — {', '.join(names[:6])}"
    except Exception as e:
        return f"FAIL: {type(e).__name__}: {e}"


def main():
    print("MCP TOOL AVAILABILITY PROBE")
    print("=" * 60)
    ok = 0
    total = 0
    for name, kind, cfg in SERVERS:
        total += 1
        if kind == "local":
            result = probe_local(name, cfg)
        else:
            result = probe_remote(name, cfg)
        status = result.startswith("OK")
        if status:
            ok += 1
        print(f"[{'OK ' if status else 'BAD'}] {name}: {result}")
    print("=" * 60)
    print(f"RESULT: {ok}/{total} servers have tools available")
    if ok < total:
        print("WARNING: at least one MCP server is connected but NOT exposing tools")


if __name__ == "__main__":
    main()
