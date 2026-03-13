import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GGUF model from Hugging Face")
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. microsoft/Phi-3-mini-4k-instruct-gguf")
    parser.add_argument("--filename", required=True, help="GGUF filename, e.g. Phi-3-mini-4k-instruct-q4.gguf")
    parser.add_argument("--out", default="models", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_path = hf_hub_download(
        repo_id=args.repo,
        filename=args.filename,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )

    print(f"Downloaded to {file_path}")


if __name__ == "__main__":
    main()
