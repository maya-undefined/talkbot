# TalkBot Agent Guidelines

Scope: entire repository.

## Project conventions
- Keep existing endpoints stable: `/ingest_pg` and `/chat_pg` must remain backward compatible.
- Prefer minimal, incremental changes with clear module boundaries.
- Use existing stack first: FastAPI, Pydantic, SQLAlchemy, Postgres/pgvector.
- Add explicit type hints and concise docstrings for new/updated modules.
- Avoid broad refactors and repository-wide reformatting unless explicitly requested.
- If reformatting is needed, do it in a dedicated commit.

## Definition of Done (DoD)
- Feature is implemented behind clear API/model boundaries.
- Existing endpoints still function (no breaking changes to request/response contracts).
- New/changed code includes tests for core behavior and error cases.
- Tests run locally (or command outputs clearly indicate environment limitations).
- Documentation is updated when API surface or workflows change.

## Testing commands
- `python -m pytest -q`
- `python -m pytest -q tests/test_agent_run.py` (when agent runtime endpoint changes)

## Commit style
- Keep commits small and PR-style: one feature at a time.
- Commit message format recommendation: `<area>: <short imperative summary>`.
