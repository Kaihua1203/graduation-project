from __future__ import annotations

import os
import sys
from pathlib import Path


def prepare_diffusers_import() -> None:
    try:
        import diffusers  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    env_path = os.environ.get("DIFFUSERS_SRC_PATH")
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            return

    default_candidate = Path(
        "/home/jupyter-wenkaihua/data3_link/kaihua.wen/code/diffusers/src"
    )
    if default_candidate.exists():
        sys.path.insert(0, str(default_candidate))
