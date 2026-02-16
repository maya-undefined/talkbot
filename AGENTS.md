# AGENTS.md

## Scope
These instructions apply to the entire repository.

## Project conventions
- Preserve existing API behavior for `POST /ingest_pg` and `POST /chat_pg` unless a task explicitly asks to change them.
- Prefer incremental, PR-sized changes. Keep diffs small and focused.
- Use existing stack first: FastAPI, Pydantic, SQLAlchemy, pytest.
- Add type hints and concise docstrings to all newly added modules, classes, and functions.
- Avoid broad refactors or whole-repo formatting changes.
- If formatting is needed, keep it limited to touched files.

## Definition of Done
- Code compiles and app imports successfully.
- New/changed behavior is covered by tests.
- Existing endpoints remain mounted.
- Relevant docs are updated when new APIs/contracts are introduced.
- Changes are committed with a clear, focused message.

## Testing commands
- `pytest -q`
- Optional static check (if available in environment): `ruff check .`

## Delivery checklist
- Provide a short architecture sketch for planned work.
- Provide a phased implementation plan.
- Implement one smallest valuable increment first, validate it, and commit.
