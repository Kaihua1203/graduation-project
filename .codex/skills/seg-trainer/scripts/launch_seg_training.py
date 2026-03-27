#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


ENCODER_PATHS = {
    "mocov3-ep80": "outputs/encoders/lits_mocov3_ep80_encoder.pth",
    "simclr-ep80": "outputs/encoders/lits_simclr_ep80_encoder.pth",
    "vicreg-ep80": "outputs/encoders/lits_vicreg_ep80_encoder.pth",
    "vicreg": "outputs/encoders/lits_vicreg_encoder.pth",
}


@dataclass
class Job:
    mode: str
    experiment: str
    encoder: Optional[str]
    config: Optional[str]
    gpu_ids: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate seg configs and launch tmux training jobs.")
    parser.add_argument("--project-dir", default=None, help="Path to 2d-ssl-seg project root.")
    parser.add_argument("--job", action="append", required=True, help="Job spec: key=value pairs separated by commas.")
    parser.add_argument("--allowed-gpus", default=None, help="Optional GPU pool, e.g. 0,1,2,3,4,5.")
    parser.add_argument("--gpus-per-job", type=int, default=2, help="Physical GPUs per training job.")
    parser.add_argument("--startup-wait-seconds", type=float, default=3.0, help="Sleep before verification.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without launching tmux.")
    return parser.parse_args()


def infer_project_dir(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).resolve()
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "2d-ssl-seg"


def parse_job(spec: str) -> Job:
    values: Dict[str, str] = {}
    current_key: Optional[str] = None
    for item in spec.split(","):
        token = item.strip()
        if not token:
            continue
        if "=" not in token:
            if current_key is None:
                raise ValueError(f"Invalid job token: {token!r}")
            values[current_key] = f"{values[current_key]},{token}"
            continue
        key, value = token.split("=", 1)
        current_key = key.strip()
        values[current_key] = value.strip()

    mode = values.get("mode")
    experiment = values.get("experiment")
    if mode not in {"ssl", "random"}:
        raise ValueError(f"Job mode must be 'ssl' or 'random': {spec}")
    if not experiment:
        raise ValueError(f"Job missing experiment name: {spec}")
    encoder = values.get("encoder")
    if mode == "ssl" and not encoder:
        raise ValueError(f"SSL job missing encoder name: {spec}")

    return Job(
        mode=mode,
        experiment=experiment,
        encoder=encoder,
        config=values.get("config"),
        gpu_ids=values.get("gpu_ids"),
    )


def run(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=True)


def normalize_config_path(project_dir: Path, config_value: Optional[str], mode: str) -> Path:
    if config_value:
        candidate = Path(config_value)
        if not candidate.is_absolute():
            candidate = (project_dir / config_value).resolve()
        else:
            candidate = candidate.resolve()
        return candidate

    default_name = "train_ssl_50epochs.yaml" if mode == "ssl" else "train_random_50epochs.yaml"
    return (project_dir / "configs" / "seg" / default_name).resolve()


def resolve_encoder_path(project_dir: Path, encoder_name: str) -> Path:
    if encoder_name in ENCODER_PATHS:
        return (project_dir / ENCODER_PATHS[encoder_name]).resolve()
    candidate = Path(encoder_name)
    if not candidate.is_absolute():
        candidate = (project_dir / encoder_name).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def save_yaml(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def update_config(project_dir: Path, cfg: Dict, job: Job, encoder_path: Optional[Path]) -> Dict:
    output_dir = "outputs/seg_ssl" if job.mode == "ssl" else "outputs/seg_random"
    freeze_epochs = 10 if job.mode == "ssl" else 0

    cfg.setdefault("pretrained_encoder", {})
    cfg["pretrained_encoder"]["enabled"] = job.mode == "ssl"
    cfg["pretrained_encoder"]["path"] = str(encoder_path.relative_to(project_dir)) if encoder_path else ""
    cfg["output_dir"] = output_dir
    cfg.setdefault("train", {})
    cfg["train"]["freeze_encoder_epochs"] = freeze_epochs
    cfg["train"]["gpu_ids"] = [0]
    cfg.setdefault("swanlab", {})
    cfg["swanlab"]["experiment_name"] = job.experiment
    return cfg


def query_gpu_stats() -> List[Dict[str, int]]:
    result = run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    stats = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        index_s, mem_s, util_s = [part.strip() for part in line.split(",")]
        stats.append({"index": int(index_s), "memory_used": int(mem_s), "utilization": int(util_s)})
    return stats


def assign_gpu_ids(jobs: List[Job], allowed_gpus: Optional[str], gpus_per_job: int) -> Dict[str, str]:
    assignments: Dict[str, str] = {}
    explicit_only = all(job.gpu_ids for job in jobs)
    if explicit_only:
        for job in jobs:
            assignments[job.experiment] = job.gpu_ids or ""
        return assignments

    stats = query_gpu_stats()
    if allowed_gpus:
        allowed = {int(token.strip()) for token in allowed_gpus.split(",") if token.strip()}
        stats = [row for row in stats if row["index"] in allowed]
    stats.sort(key=lambda row: (row["memory_used"], row["utilization"], row["index"]))

    used = set()
    for job in jobs:
        if job.gpu_ids:
            assignments[job.experiment] = job.gpu_ids
            used.update(int(token.strip()) for token in job.gpu_ids.split(",") if token.strip())
            continue

        available = [row["index"] for row in stats if row["index"] not in used]
        if len(available) < gpus_per_job:
            raise RuntimeError(f"Not enough free GPUs for job {job.experiment}.")
        chosen = available[:gpus_per_job]
        used.update(chosen)
        assignments[job.experiment] = ",".join(str(idx) for idx in chosen)
    return assignments


def ensure_session_absent(session_name: str) -> None:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        text=True,
        capture_output=True,
    )
    if result.returncode == 0:
        raise RuntimeError(f"tmux session already exists: {session_name}")


def launch_job(project_dir: Path, job: Job, config_path: Path, gpu_ids: str, dry_run: bool) -> Dict[str, str]:
    script_name = "run_seg_train_ssl_tmux.sh" if job.mode == "ssl" else "run_seg_train_random_tmux.sh"
    script_path = project_dir / "scripts" / script_name
    command = [
        "bash",
        str(script_path),
    ]
    env = {
        **dict(os.environ),
        "SESSION_NAME": job.experiment,
        "CONFIG_PATH": str(config_path),
        "GPU_IDS": gpu_ids,
    }

    if dry_run:
        return {
            "session": job.experiment,
            "config": str(config_path),
            "gpu_ids": gpu_ids,
            "status": "dry-run",
            "command": " ".join(shlex.quote(part) for part in command),
        }

    ensure_session_absent(job.experiment)
    result = subprocess.run(command, cwd=project_dir, text=True, capture_output=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"Launch failed: {job.experiment}")

    return {
        "session": job.experiment,
        "config": str(config_path),
        "gpu_ids": gpu_ids,
        "status": "launched",
        "stdout": result.stdout.strip(),
    }


def verify_job(session_name: str) -> Dict[str, str]:
    has = subprocess.run(["tmux", "has-session", "-t", session_name], text=True, capture_output=True)
    if has.returncode != 0:
        return {"status": "failed", "detail": "tmux session exited before verification"}

    pane = run(["tmux", "capture-pane", "-t", session_name, "-p"], check=False)
    tail_lines = "\n".join(pane.stdout.strip().splitlines()[-8:]) if pane.stdout.strip() else ""
    return {"status": "started", "detail": tail_lines}


def main() -> int:
    args = parse_args()
    jobs = [parse_job(spec) for spec in args.job]
    project_dir = infer_project_dir(args.project_dir)
    if not project_dir.exists():
        raise FileNotFoundError(f"Project dir not found: {project_dir}")

    assignments = assign_gpu_ids(jobs, args.allowed_gpus, args.gpus_per_job)
    summaries = []

    for job in jobs:
        base_config = normalize_config_path(project_dir, job.config, job.mode)
        if not base_config.exists():
            raise FileNotFoundError(f"Config not found: {base_config}")

        encoder_path = None
        if job.mode == "ssl":
            encoder_path = resolve_encoder_path(project_dir, job.encoder or "")
            if not encoder_path.exists():
                raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")

        cfg = load_yaml(base_config)
        cfg = update_config(project_dir, cfg, job, encoder_path)
        generated_config = project_dir / "configs" / "seg" / "generated" / f"{job.experiment}.yaml"
        if not args.dry_run:
            save_yaml(generated_config, cfg)

        launch = launch_job(project_dir, job, generated_config, assignments[job.experiment], args.dry_run)
        summaries.append(
            {
                "mode": job.mode,
                "experiment": job.experiment,
                "encoder": job.encoder or "",
                "base_config": str(base_config),
                "generated_config": str(generated_config),
                "gpu_ids": assignments[job.experiment],
                "session": launch["session"],
                "launch_status": launch["status"],
            }
        )

    if not args.dry_run:
        time.sleep(args.startup_wait_seconds)
        verification = {job.experiment: verify_job(job.experiment) for job in jobs}
        gpu_stats = query_gpu_stats()
    else:
        verification = {job.experiment: {"status": "dry-run", "detail": ""} for job in jobs}
        gpu_stats = []

    for item in summaries:
        result = verification[item["experiment"]]
        item["startup_status"] = result["status"]
        item["detail"] = result["detail"]

    for item in summaries:
        print(f"experiment: {item['experiment']}")
        print(f"  mode: {item['mode']}")
        if item["encoder"]:
            print(f"  encoder: {item['encoder']}")
        print(f"  base_config: {item['base_config']}")
        print(f"  generated_config: {item['generated_config']}")
        print(f"  gpu_ids: {item['gpu_ids']}")
        print(f"  session: {item['session']}")
        print(f"  launch_status: {item['launch_status']}")
        print(f"  startup_status: {item['startup_status']}")
        if item["detail"]:
            print("  detail: |")
            for line in item["detail"].splitlines():
                print(f"    {line}")

    if gpu_stats:
        print("gpu_snapshot:")
        for row in gpu_stats:
            print(
                f"  - index={row['index']} memory_used={row['memory_used']}MiB utilization={row['utilization']}%"
            )

    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
