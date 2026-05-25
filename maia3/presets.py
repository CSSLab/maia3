import sys
from pathlib import Path

from .uci import main


_LAUNCHER_NAMES = {
    "maia3-3m-ablation",
    "maia3-5m",
    "maia3-23m",
    "maia3-79m",
}


def _is_launcher_path(arg):
    name = Path(arg).name.removesuffix(".exe")
    return name in _LAUNCHER_NAMES


def _drop_model_args(argv):
    cleaned = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--model":
            skip_next = True
            continue
        if arg.startswith("--model="):
            continue
        if _is_launcher_path(arg):
            continue
        cleaned.append(arg)
    return cleaned


def _run_preset(model_name):
    # GUI engine pickers sometimes preserve stale --model arguments. Preset
    # executables should always use their own model while keeping other options.
    main(["--temperature", "0", *_drop_model_args(sys.argv[1:]), "--model", model_name])



def main_3m_ablation():
    _run_preset("maia3-3m-ablation")


def main_5m():
    _run_preset("maia3-5m")


def main_23m():
    _run_preset("maia3-23m")


def main_79m():
    _run_preset("maia3-79m")
