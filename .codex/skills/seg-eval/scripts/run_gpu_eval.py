#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def parse_run_spec(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"Invalid --run value: {spec}. Expected format <ssl|random>:<experiment_name>")
    target, experiment_name = spec.split(":", 1)
    target = target.strip().lower()
    experiment_name = experiment_name.strip()
    if target not in {"ssl", "random"}:
        raise ValueError(f"Invalid target '{target}' in --run value: {spec}")
    if not experiment_name:
        raise ValueError(f"Empty experiment_name in --run value: {spec}")
    return target, experiment_name


def run_eval(project_dir: Path, eval_script: str, target: str, experiment_name: str) -> Dict:
    env = os.environ.copy()
    if target == "ssl":
        env["SEG_SSL_EXPERIMENT_NAME"] = experiment_name
        run_name = "seg-ssl"
    else:
        env["SEG_RANDOM_EXPERIMENT_NAME"] = experiment_name
        run_name = "seg-random"

    cmd = ["bash", eval_script, target]
    proc = subprocess.run(
        cmd,
        cwd=str(project_dir),
        env=env,
        text=True,
        capture_output=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "Evaluation command failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

    log_path = project_dir / "outputs" / "logs" / run_name / "evaluate_history.jsonl"
    if not log_path.is_file():
        raise FileNotFoundError(f"Expected JSONL log not found: {log_path}")

    lines = [line.strip() for line in log_path.read_text().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"No records found in JSONL log: {log_path}")

    last_record = json.loads(lines[-1])
    return {
        "target": target,
        "experiment_name": experiment_name,
        "log_path": str(log_path),
        "result": {
            "timestamp": last_record.get("timestamp"),
            "checkpoint": last_record.get("checkpoint"),
            "dice": last_record.get("dice"),
            "iou": last_record.get("iou"),
            "hd95": last_record.get("hd95"),
            "num_pairs": last_record.get("num_pairs"),
        },
    }


def compare_if_possible(records: List[Dict]) -> Dict:
    by_target = {rec["target"]: rec for rec in records}
    if "ssl" not in by_target or "random" not in by_target:
        return {}

    ssl = by_target["ssl"]["result"]
    random = by_target["random"]["result"]

    def better(high_is_better: bool, a, b) -> str:
        if a is None or b is None:
            return "unknown"
        if a == b:
            return "tie"
        if high_is_better:
            return "ssl" if a > b else "random"
        return "ssl" if a < b else "random"

    return {
        "dice_better": better(True, ssl.get("dice"), random.get("dice")),
        "iou_better": better(True, ssl.get("iou"), random.get("iou")),
        "hd95_better": better(False, ssl.get("hd95"), random.get("hd95")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run segmentation evaluations on GPU and summarize latest JSONL results.")
    parser.add_argument(
        "--project-dir",
        default="/data3/kaihua.wen/code/graduation-project/2d-ssl-seg",
        help="Path to 2d-ssl-seg project root.",
    )
    parser.add_argument(
        "--eval-script",
        default="scripts/run_seg_eval_lits.sh",
        help="Evaluation script path relative to --project-dir, e.g. scripts/run_seg_eval_lits.sh.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Repeatable spec in format <ssl|random>:<experiment_name>.",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    if not project_dir.is_dir():
        raise NotADirectoryError(f"Project directory not found: {project_dir}")

    run_specs = [parse_run_spec(item) for item in args.run]

    records = []
    for target, experiment_name in run_specs:
        records.append(run_eval(project_dir, args.eval_script, target, experiment_name))

    output = {
        "project_dir": str(project_dir),
        "eval_script": args.eval_script,
        "runs": records,
        "comparison": compare_if_possible(records),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
