# Handoff contract

One format for every role-to-role message — brief *down*, report *up*. Roles + ranks: see `CLAUDE.md` org chart.

**Deliver:** try `SendMessage` to the target agent. On failure, write `handoff-<to>-<topic>-<4-char hex>.md` (e.g. `handoff-em-rf-a3f1.md`; `openssl rand -hex 2` for the suffix; read before write) for the user to carry across devices. Save in `handoffs/` folder.

**Block:**
```
TO: <role>   FROM: <role>   RE: <one line>
CONTEXT: ≤3 bullets — only what the receiver lacks. Reference repo artifacts by path/URL; don't restate them.
ASK: the directive / what to produce.
CONSTRAINTS: device, budget, stop-conditions, deadline.
RETURN: exact artifacts wanted back.
```

**Rules:** never pass raw transcripts/logs up the chain — compact first, lose no key evidence. Don't duplicate repo content (plan, `REPORT.md`, `*.jsonl`) — link it.
