"""download_dataset.py — fetch the processed RAG artifacts from Hugging Face.

The GitHub repo does NOT ship the large artifacts (chunks/parents/windows
JSONL, embeddings .npy, lexical BM25 index, raw corpus). They live on the
Hugging Face Hub instead:

    https://huggingface.co/datasets/Fayizz/indonesian-legislation-norms

Run this once after cloning:

    python tools/download_dataset.py            # processed artifacts (~1.1 GB)
    python tools/download_dataset.py --corpus   # + raw corpus (~40 MB)
    python tools/download_dataset.py --model qwen3-06b   # one embedding set only

Requires: pip install huggingface_hub (no auth needed — public dataset).
"""

import argparse
import sys
from pathlib import Path

REPO = "Fayizz/indonesian-legislation-norms"
ROOT = Path(__file__).resolve().parents[1]          # repo root
RESULT_DIR = ROOT / "pipeline" / "result"           # artifacts land here (match config.py)
CORPUS_DIR = ROOT / "data" / "corpus"

ARTIFACT_PATTERNS = [
    "chunks.jsonl",
    "parents.jsonl",
    "parents_windows.jsonl",
    "documents.jsonl",
    "validity.jsonl",
    "quarantine.jsonl",
    "manifest.json",
    "embeddings/**",
    "lexical/**",
]
CORPUS_PATTERNS = ["corpus/**"]


def download(patterns: list[str], local_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub belum terpasang. Jalankan: pip install huggingface_hub")
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {REPO} -> {local_dir}")
    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=patterns,
    )
    print("[download] selesai.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        choices=["qwen3-06b", "qwen3-4b"],
        default=None,
        help="Ambil satu set embedding saja (default: keduanya)",
    )
    ap.add_argument(
        "--corpus",
        action="store_true",
        help="Juga unduh korpus mentah Mendeley (units.csv, documents.csv, texts/)",
    )
    ap.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Lewati artefak build (hanya --corpus)",
    )
    args = ap.parse_args()

    patterns = list(ARTIFACT_PATTERNS)
    if args.model:
        patterns = [p for p in patterns if not p.startswith("embeddings/")]
        patterns.append(f"embeddings/{args.model}/**")
    if args.no_artifacts:
        patterns = []

    if patterns:
        download(patterns, RESULT_DIR)
    if args.corpus:
        download(CORPUS_PATTERNS, CORPUS_DIR)

    if args.no_artifacts and not args.corpus:
        ap.error("--no-artifacts tanpa --corpus tidak mendownload apa pun")


if __name__ == "__main__":
    main()
