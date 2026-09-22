---
description: Start session: read handoff and resume work.
---

You are starting a new session. Read the most recent handoff and resume work immediately.

STEP 1 — Determine YOUR project directory from this session's context (not `pwd` — your terminal cwd may be the server home, not the project). You know which project you are working in from this conversation: the files you are in, the repository, the session title. Write it down as PROJECT_DIR (e.g. /home/johnk/projects/<name>).

Confirm it exists:
```
ls -d <PROJECT_DIR>
```

STEP 2 — Find the most recent handoff in YOUR project (not any other project). Run:
```
ls -t <PROJECT_DIR>/sessions/SESSION_HANDOFF_*.md 2>/dev/null | head -1 || echo "NO_HANDOFF_FOUND"
```
If the output is "NO_HANDOFF_FOUND", tell me "No handoff files found in this project. What are we working on?" and stop.

STEP 3 — Read the handoff file found in Step 2. Extract these sections:
- Current State — what's working, build status
- Pending Work — exact TODOs with file paths and line numbers
- Next Steps — the immediate next action (do this FIRST)
- Obstacles — what went wrong before, avoid repeating
- File Map — which files matter and their state

STEP 4 — Verify the codebase matches the handoff. Run:
```
git -C <PROJECT_DIR> status --short && git -C <PROJECT_DIR> diff --stat
```
If files are different than the handoff describes, note the discrepancy. CODE is authoritative over the handoff.

STEP 5 — Check MCP tool availability (informational — does NOT block). Run:
```
python3 /home/johnk/.opencode/scripts/mcp-probe.py
```
Read the probe output and note any problems, but DO NOT stop or wait — continue to STEP 6 and do the work. At the END of your work, if any server showed `[BAD]` or `0 tools`, ask the user: "<name> MCP server is down or showing 0 tools. Do you want me to investigate?" If all servers are `[OK]`, say nothing about MCP.

STEP 6 — Begin working. Start with the FIRST item in "Next Steps" from the handoff. Do NOT:
- Re-read files already documented in the handoff
- Re-explore things already decided
- Ask me what to do (the handoff says what to do)
- Start with "Let me understand the codebase..."

Just do the work. The handoff is truth.
