"""Parallel, resumable experiment-suite driver."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jepa_robotics.config.load import dump_config, load_config
from jepa_robotics.config.schema import BenchmarkConfig
from jepa_robotics.utils.provenance import platform_metadata
from jepa_robotics.utils.serialization import write_json

PHASE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "jepa_sac": ("state_jepa",),
}

PHASE_PRESETS: dict[str, tuple[str, ...]] = {
    "stage1_fetch_reach": (
        "phase1_fetch_reach",
        "state_jepa",
        "jepa_sac",
        "jepa_mpc",
    ),
}


@dataclass(frozen=True)
class SuiteChild:
    phase: str
    seed: int
    config_path: Path
    output_dir: Path
    command: list[str]
    required_outputs: list[Path]
    status_path: Path


def run(
    phases: list[str],
    seeds: list[int] | None = None,
    mode: str = "iteration",
    max_parallel: int = 1,
    smoke_test: bool = False,
    force: bool = False,
    output_dir: str | Path | None = None,
    command: list[str] | None = None,
    generate_report: bool = True,
) -> Path:
    started = time.time()
    requested_phases = list(phases)
    phases = _expand_phase_presets(phases)
    resolved_seeds = seeds or [0]
    suite_root = Path(output_dir) if output_dir is not None else Path("outputs") / f"{mode}_suite"
    suite_root.mkdir(parents=True, exist_ok=True)
    status_dir = suite_root / "suite_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    children = [
        _build_child(phase, seed, mode, suite_root, status_dir, smoke_test)
        for phase in phases
        for seed in resolved_seeds
    ]
    metadata: dict[str, Any] = {
        "command": shlex.join(command or sys.argv),
        "mode": mode,
        "requested_phases": requested_phases,
        "phases": phases,
        "seeds": resolved_seeds,
        "max_parallel": max_parallel,
        "smoke_test": smoke_test,
        "generate_report": generate_report,
        "start_time": started,
        **platform_metadata(Path.cwd()),
        "children": [],
    }
    statuses: list[dict[str, Any]] = []
    runnable: list[SuiteChild] = []
    for child in children:
        if not force and _outputs_complete(child.required_outputs):
            status = _child_status(child, "skipped", returncode=0, elapsed_seconds=0.0)
            write_json(child.status_path, status)
            statuses.append(status)
        else:
            runnable.append(child)
    waves = _topological_waves(runnable, phases)
    workers = max(1, max_parallel)
    failed_phase_seeds: set[tuple[str, int]] = set()
    for wave in waves:
        wave_runnable = [
            child for child in wave if not _has_failed_dependency(child, failed_phase_seeds)
        ]
        for child in wave:
            if child in wave_runnable:
                continue
            status = _child_status(
                child,
                "skipped_dependency_failed",
                returncode=0,
                elapsed_seconds=0.0,
                stderr="Upstream phase failed.",
            )
            write_json(child.status_path, status)
            statuses.append(status)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_child, child) for child in wave_runnable]
            for future in as_completed(futures):
                result = future.result()
                statuses.append(result)
                if result["status"] == "failed":
                    failed_phase_seeds.add((result["phase"], result["seed"]))
    metadata["end_time"] = time.time()
    metadata["elapsed_seconds"] = metadata["end_time"] - started
    metadata["children"] = sorted(statuses, key=lambda item: (item["phase"], item["seed"]))
    write_json(suite_root / "suite_metadata.json", metadata)
    failed = [status for status in statuses if status["status"] == "failed"]
    if failed:
        failed_names = ", ".join(f"{item['phase']}/seed_{item['seed']}" for item in failed)
        raise RuntimeError(f"Suite failed for: {failed_names}")
    if generate_report:
        from jepa_robotics.plotting.generate import generate_report_artifacts

        generate_report_artifacts(suite_root)
    return suite_root


def _build_child(
    phase: str,
    seed: int,
    mode: str,
    suite_root: Path,
    status_dir: Path,
    smoke_test: bool,
) -> SuiteChild:
    source_config = _phase_config_path(phase, mode)
    config = load_config(source_config)
    if smoke_test:
        config = config.smoke_copy()
    config_data = config.model_dump(mode="python")
    config_data["experiment"]["name"] = phase
    config_data["experiment"]["output_dir"] = suite_root / phase
    config_data["experiment"]["seeds"] = [seed]
    resolved = BenchmarkConfig.model_validate(config_data)
    child_config = suite_root / "suite_configs" / f"{phase}_seed_{seed}.yaml"
    dump_config(resolved, child_config)
    command, required = _phase_command(phase, seed, child_config, suite_root / phase)
    status_path = status_dir / f"{phase}_seed_{seed}.json"
    return SuiteChild(
        phase=phase,
        seed=seed,
        config_path=child_config,
        output_dir=suite_root / phase,
        command=command,
        required_outputs=required,
        status_path=status_path,
    )


def _phase_config_path(phase: str, mode: str) -> Path:
    candidates = [
        Path("configs") / "experiments" / mode / f"{phase}.yaml",
        Path("configs") / "experiments" / f"{phase}.yaml",
        Path("configs") / "jepa" / f"{phase}.yaml",
    ]
    if phase == "autoencoder":
        candidates.insert(0, Path("configs") / "jepa" / "autoencoder.yaml")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No config found for phase {phase!r} in mode {mode!r}.")


def _expand_phase_presets(phases: list[str]) -> list[str]:
    expanded: list[str] = []
    for phase in phases:
        preset = PHASE_PRESETS.get(phase)
        if preset is None:
            expanded.append(phase)
            continue
        for preset_phase in preset:
            if preset_phase not in expanded:
                expanded.append(preset_phase)
    return expanded


def _phase_command(
    phase: str,
    seed: int,
    config_path: Path,
    output_dir: Path,
) -> tuple[list[str], list[Path]]:
    if phase in {
        "phase0_reacher_debug",
        "phase1_fetch_reach",
        "phase2_fetch_push_dense",
        "phase3_fetch_push_dense",
    }:
        method = "sac"
        return _train_rl_command(config_path, method, seed), [
            output_dir / "rl" / method / f"seed_{seed}" / "model.zip",
            output_dir / "rl" / method / f"seed_{seed}" / "metrics.csv",
        ]
    if phase == "phase3_fetch_push_sparse":
        method = "sac_her"
        return _train_rl_command(config_path, method, seed), [
            output_dir / "rl" / method / f"seed_{seed}" / "model.zip",
            output_dir / "rl" / method / f"seed_{seed}" / "metrics.csv",
        ]
    if phase == "phase3_fetch_push_sparse_tqc":
        method = "tqc_her"
        return _train_rl_command(config_path, method, seed), [
            output_dir / "rl" / method / f"seed_{seed}" / "model.zip",
            output_dir / "rl" / method / f"seed_{seed}" / "metrics.csv",
        ]
    if phase == "state_jepa":
        return [
            sys.executable,
            "-m",
            "jepa_robotics.cli.train_jepa",
            "--config",
            str(config_path),
            "--seed",
            str(seed),
        ], [
            output_dir / "jepa" / "state_jepa" / f"seed_{seed}" / "encoder.pt",
            output_dir / "jepa" / "state_jepa" / f"seed_{seed}" / "train_metrics.csv",
            output_dir / "jepa" / "state_jepa" / f"seed_{seed}" / "val_metrics.csv",
        ]
    if phase == "jepa_mpc":
        return [
            sys.executable,
            "-m",
            "jepa_robotics.cli.evaluate",
            "--config",
            str(config_path),
            "--seed",
            str(seed),
        ], [
            output_dir / "mpc" / "jepa_mpc" / f"seed_{seed}" / "metrics.csv",
            output_dir / "mpc" / "jepa_mpc" / f"seed_{seed}" / "mpc_diagnostics.csv",
        ]
    if phase == "autoencoder":
        return [
            sys.executable,
            "-m",
            "jepa_robotics.cli.train_autoencoder",
            "--config",
            str(config_path),
            "--seed",
            str(seed),
        ], [
            output_dir / "autoencoder" / "state_autoencoder" / f"seed_{seed}" / "encoder.pt",
            output_dir / "autoencoder" / "state_autoencoder" / f"seed_{seed}" / "train_metrics.csv",
        ]
    if phase == "jepa_sac":
        checkpoint = (
            output_dir.parent / "state_jepa" / "jepa" / "state_jepa" / f"seed_{seed}" / "encoder.pt"
        )
        method = "sac_jepa"
        command = _train_rl_command(config_path, method, seed)
        command.extend(
            [
                "--feature-extractor",
                "jepa",
                "--encoder-checkpoint",
                str(checkpoint),
                "--freeze-encoder",
            ]
        )
        return command, [
            output_dir / "rl" / method / f"seed_{seed}" / "model.zip",
            output_dir / "rl" / method / f"seed_{seed}" / "metrics.csv",
        ]
    raise ValueError(f"Unsupported suite phase: {phase}")


def _train_rl_command(config_path: Path, method: str, seed: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "jepa_robotics.cli.train_rl",
        "--config",
        str(config_path),
        "--method",
        method,
        "--seed",
        str(seed),
    ]


def _outputs_complete(paths: list[Path]) -> bool:
    return bool(paths) and all(path.exists() for path in paths)


def _topological_waves(children: list[SuiteChild], phases: list[str]) -> list[list[SuiteChild]]:
    selected = set(phases)
    remaining = list(children)
    completed_phase_seeds: set[tuple[str, int]] = set()
    waves: list[list[SuiteChild]] = []
    while remaining:
        ready: list[SuiteChild] = []
        deferred: list[SuiteChild] = []
        for child in remaining:
            deps = PHASE_DEPENDENCIES.get(child.phase, ())
            active_deps = [dep for dep in deps if dep in selected]
            if all((dep, child.seed) in completed_phase_seeds for dep in active_deps):
                ready.append(child)
            else:
                deferred.append(child)
        if not ready:
            ready = deferred
            deferred = []
        waves.append(ready)
        for child in ready:
            completed_phase_seeds.add((child.phase, child.seed))
        remaining = deferred
    return waves


def _has_failed_dependency(child: SuiteChild, failed: set[tuple[str, int]]) -> bool:
    return any((dep, child.seed) in failed for dep in PHASE_DEPENDENCIES.get(child.phase, ()))


def _run_child(child: SuiteChild) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(child.command, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    completed = result.returncode == 0 and _outputs_complete(child.required_outputs)
    status = _child_status(
        child,
        "completed" if completed else "failed",
        returncode=result.returncode,
        elapsed_seconds=elapsed,
        stdout=result.stdout,
        stderr=result.stderr,
    )
    write_json(child.status_path, status)
    return status


def _child_status(
    child: SuiteChild,
    status: str,
    returncode: int,
    elapsed_seconds: float,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    return {
        "phase": child.phase,
        "seed": child.seed,
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": elapsed_seconds,
        "config_path": str(child.config_path),
        "output_dir": str(child.output_dir),
        "command": child.command,
        "required_outputs": [str(path) for path in child.required_outputs],
        "stdout": stdout,
        "stderr": stderr,
    }


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default="iteration", choices=["iteration", "matched", "confirm", "full"]
    )
    parser.add_argument("--phases", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    print(
        run(
            phases=args.phases,
            seeds=args.seeds,
            mode=args.mode,
            max_parallel=args.max_parallel,
            smoke_test=args.smoke_test,
            force=args.force,
            output_dir=args.output_dir,
            command=sys.argv if argv is None else ["run_suite", *argv],
            generate_report=not args.no_report,
        )
    )


if __name__ == "__main__":
    main()
