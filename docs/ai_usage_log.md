# Generative AI usage log

Mandatory for the report appendix *"Declaration of Generative AI Tool Usage"*. Add an entry **every time**
AI meaningfully shapes code, analysis or text. Be honest about errors — at least one documented
incorrect/suboptimal AI output (and how you caught it) is **required**.

| # | Date | Member | Tool | Scope | Prompt (short) | What we kept / changed | AI error? How detected & fixed |
|---|---|---|---|---|---|---|---|
| 1 | 2026-11-12 | Alejandro | Claude (Cowork) | Repo scaffolding | "Create an initial repo for the Global Superstore project from the assignment PDF" | Repo structure, `src/superstore` modules, tests, report skeleton. Reviewed by: _TBD_ | Yes — first date parser used `format="mixed"`, which silently swapped day/month on some US-style rows (`11/8/2016` → 11 Aug). Caught by unit test `test_parse_dates_detects_monthfirst`; fixed by inferring a single format per column. |
| 2 | | | | | | | |

## Scope categories
code generation · refactoring · bug fixing · visualisation ideas · theory explanation · writing/editing · literature

## Verification checklist for AI-generated code
- [ ] We understand every line and could re-derive it
- [ ] Unit test or manual check on a known case
- [ ] No data leakage (temporal order respected)
- [ ] Vectorised (no row-wise loops)
