# AGENTS.md

# Coding Agent Instructions for `jepa-robotics`

Keep this repository easy to run, test, and resume. Prefer small, maintainable changes that follow the existing package layout and configuration style.

## Required Quality Gate

Before considering any code, docs, config, or experiment-workflow change complete, run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=src/jepa_robotics --cov-report=term-missing --cov-fail-under=90
```

When touching type-sensitive Python code, also run:

```bash
uv run mypy src tests
```

If formatting fails, apply:

```bash
uv run ruff format .
```

Then rerun the quality gate. Do not claim a gate passed unless it actually ran and passed. If a command cannot run in the current environment, state the exact command and reason.

## Environment

Use `uv` for dependency and environment management:

```bash
uv sync
uv add <package>
uv add --dev <package>
uv run <command>
```

Do not add `pip install` instructions except when explaining what `uv` replaces.

## Documentation

Update `README.md` whenever behavior, commands, configs, architecture, outputs, experiment workflow, plotting, metrics, or troubleshooting changes.

Document any new or changed:

- CLI command or flag
- YAML config field
- model, planner, dataset, metric, or artifact format
- experiment output path
- plotting or analysis workflow

Keep README commands current and runnable from the repository root.

## Git and Artifacts

Do not commit local machine artifacts or generated experiment outputs unless explicitly requested. Leave these out of commits by default:

- `.venv/`
- `.coverage*`
- `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`
- `outputs/`
- `logs/`
- checkpoints, temporary MuJoCo files, wandb caches, and ad hoc run scripts

When committing, stage only files relevant to the requested change.

## Implementation Defaults

Keep CLI files thin: parse arguments, load configs, call library code, and return clear status. Keep reusable logic in `src/jepa_robotics/` modules and add focused tests for behavior changes.
