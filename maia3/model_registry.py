from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


CHECKPOINT_EXTENSIONS = (".pt", ".pth", ".ckpt", ".bin")


class ModelResolutionError(RuntimeError):
    """Raised when a model alias or Hugging Face checkpoint cannot be resolved."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    repo_id: str | None = None
    checkpoint_filename: str | None = None
    revision: str | None = None
    aliases: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def url(self) -> str | None:
        if self.repo_id is None:
            return None
        return f"https://huggingface.co/{self.repo_id}"


BASE_SIZE_CONFIG = {
    "history": 8,
    "use_padding": True,
    "include_time_info": False,
    "dim_emb": 128,
    "num_blocks": 8,
    "mlp_ratio": 2.0,
    "dropout": 0.0,
    "use_gab": True,
    "use_relative_bias": False,
    "use_absolute_pe": False,
    "use_rms_norm": True,
    "omit_qkv_biases": True,
    "activation": "gelu",
}


MODEL_SPECS = (
    ModelSpec(
        name="maia3-3m-ablation",
        display_name="Maia3 3M ablation",
        repo_id="UofTCSSLab/Maia3-ablate-3M",
        checkpoint_filename="maia3-3m.pt",
        aliases=("3m", "maia3-3m"),
        config={
            **BASE_SIZE_CONFIG,
            "dim_vit": 192,
            "head_hid_dim": 192,
            "num_heads": 6,
            "gab_gen_size": 64,
            "gab_per_square_dim": 0,
            "gab_intermediate_dim": 64,
        },
    ),
    ModelSpec(
        name="maia3-5m",
        display_name="Maia3 5M",
        repo_id="UofTCSSLab/Maia3-5M",
        checkpoint_filename="maia3-5m.pt",
        aliases=("5m",),
        config={
            **BASE_SIZE_CONFIG,
            "dim_vit": 256,
            "head_hid_dim": 256,
            "num_heads": 8,
            "gab_gen_size": 64,
            "gab_per_square_dim": 0,
            "gab_intermediate_dim": 64,
        },
    ),
    ModelSpec(
        name="maia3-23m",
        display_name="Maia3 23M",
        repo_id="UofTCSSLab/Maia3-23M",
        checkpoint_filename="maia3-23m.pt",
        aliases=("23m",),
        config={
            **BASE_SIZE_CONFIG,
            "dim_vit": 512,
            "head_hid_dim": 512,
            "num_heads": 16,
            "gab_gen_size": 128,
            "gab_per_square_dim": 32,
            "gab_intermediate_dim": 128,
        },
    ),
    ModelSpec(
        name="maia3-79m",
        display_name="Maia3 79M",
        repo_id="UofTCSSLab/Maia3-79M",
        checkpoint_filename="maia3-79m.pt",
        aliases=("79m",),
        config={
            **BASE_SIZE_CONFIG,
            "dim_vit": 1024,
            "head_hid_dim": 1024,
            "num_heads": 32,
            "gab_gen_size": 128,
            "gab_per_square_dim": 32,
            "gab_intermediate_dim": 128,
        },
    ),
)


def _normalize_key(value: str) -> str:
    return value.strip().lower()


def _looks_like_local_checkpoint_path(value: str) -> bool:
    path = Path(value).expanduser()
    return (
        value.startswith(("/", "./", "../", "~"))
        or path.suffix.lower() in CHECKPOINT_EXTENSIONS
    )


def _alias_map() -> dict[str, ModelSpec]:
    aliases: dict[str, ModelSpec] = {}
    for spec in MODEL_SPECS:
        keys = (spec.name, spec.display_name, *(spec.aliases or ()))
        if spec.repo_id is not None:
            keys = (*keys, spec.repo_id, spec.url or "")
        for key in keys:
            if key:
                aliases[_normalize_key(key)] = spec
    return aliases


def parse_huggingface_url(value: str) -> tuple[str, str | None, str | None] | None:
    parsed = urlparse(value)
    if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
        return None

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None

    repo_id = "/".join(parts[:2])
    revision = None
    filename = None

    if len(parts) >= 5 and parts[2] in {"blob", "resolve"}:
        revision = parts[3]
        filename = "/".join(parts[4:])
    elif len(parts) >= 4 and parts[2] == "tree":
        revision = parts[3]

    return repo_id, revision, filename


def resolve_model_spec(model: str) -> ModelSpec:
    """Resolve a built-in alias, Hugging Face repo ID, or Hugging Face URL."""

    value = model.strip()
    if not value:
        raise ModelResolutionError("Model name cannot be empty.")

    aliases = _alias_map()
    alias = aliases.get(_normalize_key(value))
    if alias is not None:
        return alias

    # Some GUI engine pickers pass the selected executable path back as a model
    # argument. Treat paths like /opt/.../maia3-23m as their launcher alias, while
    # still rejecting actual checkpoint paths such as maia3-23m.pt below.
    path_name = Path(value).name
    path_alias = aliases.get(_normalize_key(path_name.removesuffix(".exe")))
    if path_alias is not None:
        return path_alias

    if Path(value).expanduser().exists() or _looks_like_local_checkpoint_path(value):
        raise ModelResolutionError(
            "Local checkpoint files should be passed with `--checkpoint-path`. "
            "Use `--model <alias> --checkpoint-path <file>` to apply a built-in "
            "architecture preset, or pass `--checkpoint-path <file>` with the "
            "matching architecture flags for a custom checkpoint."
        )

    parsed_url = parse_huggingface_url(value)
    if parsed_url is not None:
        repo_id, revision, filename = parsed_url
        spec = aliases.get(_normalize_key(repo_id))
        if spec is None:
            spec = ModelSpec(name=repo_id, display_name=repo_id, repo_id=repo_id)
        return replace(
            spec,
            repo_id=repo_id,
            revision=revision or spec.revision,
            checkpoint_filename=filename or spec.checkpoint_filename,
        )

    if "/" in value and not value.startswith("-"):
        spec = aliases.get(_normalize_key(value))
        if spec is not None:
            return spec
        return ModelSpec(name=value, display_name=value, repo_id=value)

    available = ", ".join(spec.name for spec in MODEL_SPECS)
    raise ModelResolutionError(
        f"Unknown model '{model}'. Use one of: {available}, a Hugging Face repo ID, "
        "or a Hugging Face model URL."
    )


def format_model_list() -> str:
    lines = ["Built-in Maia3 model aliases:"]
    for spec in MODEL_SPECS:
        aliases = ", ".join((spec.name, *spec.aliases))
        lines.append(f"  {spec.display_name:<10} {aliases:<34} {spec.url}")
    lines.append("")
    lines.append("You can also pass a Hugging Face repo ID or URL, for example:")
    lines.append("  maia3-uci --model UofTCSSLab/Maia3-79M")
    lines.append("  maia3-uci --model https://huggingface.co/UofTCSSLab/Maia3-79M")
    return "\n".join(lines)


def apply_model_config(args: Any, spec: ModelSpec) -> None:
    for key, value in spec.config.items():
        setattr(args, key, value)


def resolve_checkpoint_path(
    spec: ModelSpec,
    *,
    checkpoint_filename: str | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    force_download: bool = False,
    token: str | None = None,
) -> str:
    """Return a local checkpoint path for a resolved model spec."""

    local_filename = checkpoint_filename or spec.checkpoint_filename
    if spec.repo_id is None:
        if local_filename is None:
            raise ModelResolutionError("No checkpoint path or Hugging Face repo was provided.")
        local_path = Path(local_filename).expanduser()
        if not local_path.exists():
            raise ModelResolutionError(f"Checkpoint file does not exist: {local_path}")
        return str(local_path)

    hf_revision = revision or spec.revision
    hf_filename = local_filename
    if hf_filename is None:
        hf_filename = _select_checkpoint_filename(
            spec.repo_id,
            revision=hf_revision,
            token=token,
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ModelResolutionError(
            "Downloading from Hugging Face requires huggingface-hub. "
            "Install this repo with `python -m pip install .` or run "
            "`python -m pip install huggingface-hub`."
        ) from exc

    try:
        return hf_hub_download(
            repo_id=spec.repo_id,
            filename=hf_filename,
            revision=hf_revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
        )
    except Exception as exc:
        raise ModelResolutionError(
            f"Could not download '{hf_filename}' from {spec.repo_id}. "
            "If the repository has more than one checkpoint, pass "
            "`--checkpoint-filename <file>`."
        ) from exc


def _select_checkpoint_filename(
    repo_id: str,
    *,
    revision: str | None,
    token: str | None,
) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ModelResolutionError(
            "Auto-discovering Hugging Face checkpoint files requires huggingface-hub."
        ) from exc

    try:
        files = HfApi().list_repo_files(repo_id=repo_id, revision=revision, token=token)
    except Exception as exc:
        raise ModelResolutionError(
            f"Could not list files in Hugging Face repo {repo_id}."
        ) from exc

    candidates = [
        file
        for file in files
        if file.lower().endswith(CHECKPOINT_EXTENSIONS)
    ]
    if not candidates:
        raise ModelResolutionError(
            f"No checkpoint file with extension {CHECKPOINT_EXTENSIONS} found in {repo_id}."
        )

    if len(candidates) == 1:
        return candidates[0]

    preferred = [
        "model.pt",
        "checkpoint.pt",
        "pytorch_model.bin",
        "maia3.pt",
        f"{repo_id.split('/')[-1].lower()}.pt",
    ]
    lower_to_original = {candidate.lower(): candidate for candidate in candidates}
    for name in preferred:
        match = lower_to_original.get(name)
        if match is not None:
            return match

    options = ", ".join(candidates)
    raise ModelResolutionError(
        f"Multiple checkpoint files found in {repo_id}: {options}. "
        "Pass `--checkpoint-filename <file>` to choose one."
    )
