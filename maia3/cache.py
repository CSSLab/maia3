import argparse

from .model_registry import (
    MODEL_SPECS,
    ModelResolutionError,
    resolve_checkpoint_path,
    resolve_model_spec,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Download Maia3 checkpoints into the local Hugging Face cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", action="append", dest="models",
                        help="Model alias, Hugging Face repo ID, or Hugging Face URL. May be repeated")
    parser.add_argument("--all", action="store_true", default=False,
                        help="Cache every built-in Maia3 model")
    parser.add_argument("--checkpoint-filename", "--checkpoint_filename",
                        dest="checkpoint_filename", type=str, default=None,
                        help="Checkpoint filename inside a Hugging Face repo")
    parser.add_argument("--cache-dir", "--cache_dir", dest="cache_dir", type=str, default=None,
                        help="Optional Hugging Face cache directory")
    parser.add_argument("--revision", type=str, default=None,
                        help="Optional Hugging Face revision, branch, or commit")
    parser.add_argument("--force-download", "--force_download", dest="force_download",
                        action="store_true", default=False,
                        help="Force re-downloading the checkpoint")
    parser.add_argument("--hf-token", "--hf_token", dest="hf_token", type=str, default=None,
                        help="Optional Hugging Face token for private model repos")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.all:
        model_names = [spec.name for spec in MODEL_SPECS]
    else:
        model_names = args.models or ["maia3-5m"]

    for model_name in model_names:
        try:
            spec = resolve_model_spec(model_name)
            path = resolve_checkpoint_path(
                spec,
                checkpoint_filename=args.checkpoint_filename,
                cache_dir=args.cache_dir,
                revision=args.revision,
                force_download=args.force_download,
                token=args.hf_token,
            )
        except ModelResolutionError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"{spec.display_name}: {path}")


if __name__ == "__main__":
    main()
