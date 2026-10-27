"""小规模实验入口：明确训练/验证来源，记录数据哈希和压缩率。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import regex

from .tokenizer import Tokenizer, train_bpe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _measure(tokenizer: Tokenizer, path: Path) -> dict[str, int | float | None]:
    started = time.perf_counter()
    # newline="" 保留 CRLF；固定大小读取避免一整条超长行立即占满内存。
    with path.open(encoding="utf-8", newline="") as handle:
        chunks = iter(lambda: handle.read(64 * 1024), "")
        count = sum(1 for _ in tokenizer.encode_iterable(chunks))
    elapsed = time.perf_counter() - started
    byte_count = path.stat().st_size
    return {
        "utf8_bytes": byte_count,
        "tokens": count,
        "bytes_per_token": byte_count / count if count else None,
        "encode_seconds": elapsed,
        "tokens_per_second": count / elapsed if elapsed else None,
    }


def _train(args: argparse.Namespace) -> None:
    train_path = args.train_file.resolve()
    validation_path = args.validation_file.resolve() if args.validation_file else None
    train_hash = _sha256(train_path)
    validation_hash = _sha256(validation_path) if validation_path else None
    if validation_path and (train_path.samefile(validation_path) or train_hash == validation_hash):
        raise ValueError("训练集和验证集不能是同一文件或内容完全相同的副本")

    specials = args.special_token if args.special_token is not None else ["<|endoftext|>"]
    started = time.perf_counter()
    # 唯一进入训练函数的文件是 train_path；验证文件只在 tokenizer 固定后编码。
    vocab, merges = train_bpe(train_path, args.vocab_size, specials)
    train_seconds = time.perf_counter() - started
    tokenizer = Tokenizer(vocab, merges, specials)
    tokenizer.save(args.output_dir)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "regex_version": regex.__version__,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": len(tokenizer.vocab),
        "merge_count": len(merges),
        "special_tokens": tokenizer.special_tokens,
        "train_seconds": train_seconds,
        "streaming_semantics": "concatenated_text_exact",
        "train": {"path": str(train_path), "sha256": train_hash},
        "validation": None,
    }
    manifest["train"].update(_measure(tokenizer, train_path))
    if validation_path:
        manifest["validation"] = {
            "path": str(validation_path),
            "sha256": validation_hash,
            **_measure(tokenizer, validation_path),
        }
    output = json.dumps(manifest, ensure_ascii=False, indent=2)
    (args.output_dir / "experiment.json").write_text(output + "\n", encoding="utf-8")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="CS336 BPE 学习实验（CPU 即可）")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="只用训练集学词表，随后可在验证集统计压缩率")
    train.add_argument("--train-file", type=Path, required=True)
    train.add_argument("--validation-file", type=Path)
    train.add_argument("--vocab-size", type=int, default=1000)
    train.add_argument("--special-token", action="append", help="可重复；默认 <|endoftext|>")
    train.add_argument("--output-dir", type=Path, required=True)
    encode = commands.add_parser("encode", help="加载固定 tokenizer，展示 token ID 与 bytes")
    encode.add_argument("--tokenizer-dir", type=Path, required=True)
    encode.add_argument("--text", required=True)
    args = parser.parse_args()
    if args.command == "train":
        _train(args)
    else:
        tokenizer = Tokenizer.load(args.tokenizer_dir)
        ids = tokenizer.encode(args.text)
        print("token IDs:", ids)
        print("token bytes:", [tokenizer.vocab[token_id] for token_id in ids])
        print("decode:", repr(tokenizer.decode(ids)))


if __name__ == "__main__":
    main()
