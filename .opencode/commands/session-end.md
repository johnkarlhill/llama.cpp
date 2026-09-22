---
description: End session: export JSON, write handoff, skip compact.
---

You are ending this session. Execute the steps IN ORDER. Do not skip steps. Do not ask for confirmation.

STEP 1 — Determine YOUR project directory from this session's context (not `pwd` — your terminal cwd may be the server home, not the project). You know which project you are working in from this conversation: the files you have edited, the repository you are in, the session title. Write it down as PROJECT_DIR (e.g. /home/johnk/projects/<name>).

Confirm it exists:
```
ls -d <PROJECT_DIR>
```

STEP 2 — Create the sessions directory if needed:
```
mkdir -p <PROJECT_DIR>/sessions
```

STEP 3 — Write the session handoff file FIRST: <PROJECT_DIR>/sessions/SESSION_HANDOFF_<timestamp>.md
Use a timestamp in the same format the project has used before (YYYYMMDDHHMMSS, e.g. 20260815153926). If unsure, run `date +%Y%m%d%H%M%S`.

The file MUST have ALL of these sections filled with REAL content from our conversation — not placeholders:

## Current State
What is working right now. What files were changed and their current state.
Include exact test results, build outputs, numerical findings.
Be specific: file paths, line numbers, function names, error messages.

## Key Decisions & Rationale
Every significant decision made and WHY. Include alternatives rejected.
Root cause explanations for any bugs fixed.

## Pending Work
Exact TODOs with file paths and line numbers.
Format: [ ] Task description — file_path:line_number
Include what was tried and failed, to avoid dead ends.

## Next Steps
Ordered list. Be specific: "edit file X at line Y to do Z because of finding W."
What blocks everything else goes first.

## Obstacles & Gotchas
Bugs hit, workarounds in place, non-obvious behaviors.
Commands that failed and why. Environment quirks.

## Relevant File Map
Key files touched with one-line description and state.
Format: path/to/file — description (state: modified/stable/broken)

## Session Stats
You may include token/cost stats if you can read them, otherwise note "see export JSON".

STEP 4 — NOW run the export script, passing YOUR project directory explicitly (this is critical — the script cannot guess it):
```
bash /home/johnk/.opencode/scripts/session-end.sh <PROJECT_DIR>
```
Read ALL of its output. It will print TIMESTAMP, EXPORT_FILE, metadata (TITLE, ID, MODEL, AGENT, IN, OUT, CACHE, COST), and git status. If it prints META_ERROR or EXPORT_ERROR, report the error to the user and stop.

STEP 5 — Match the export to the handoff. If the script's TIMESTAMP differs from your handoff's timestamp, note both:
- Handoff: <PROJECT_DIR>/sessions/SESSION_HANDOFF_<your timestamp>.md
- Export: <PROJECT_DIR>/sessions/session_export_<script timestamp>.json
Both live in the same directory. That is fine — they are the pair for this session.

STEP 6 — Verify BOTH files exist in <PROJECT_DIR>/sessions/. Run:
```
ls -la <PROJECT_DIR>/sessions/session_export_*.json <PROJECT_DIR>/sessions/SESSION_HANDOFF_*.md | tail -6
```
Check both are listed. If either is missing, report: "Missing <filename> — <what went wrong>".

STEP 7 — Report:
"Session ended cleanly. Handoff: <PROJECT_DIR>/sessions/SESSION_HANDOFF_<timestamp>.md — use /session-start in the new session to resume."

Do NOT run /compact. The handoff replaces compaction entirely.
