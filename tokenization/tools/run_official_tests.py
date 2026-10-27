"""在临时目录运行官方两个 tokenizer 测试文件，保留原始测试断言。

仅提供本项目的 adapters 和 snapshot fixture，避免引入 A1 其他题目的 torch
依赖及官方 conftest.py 的 Python 3.12 语法。原始官方仓库不会被修改。
官方 fixture 仅用于测试，运行结束后不会保存其训练出的 tokenizer。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

_ADAPTERS = """from cs336_basics import Tokenizer, train_bpe


def get_tokenizer(vocab, merges, special_tokens=None):
    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    return train_bpe(input_path, vocab_size, special_tokens)
"""

_CONFTEST = """import pickle
from pathlib import Path

import pytest


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("测试快照只允许内置数据，不允许加载类或函数")


@pytest.fixture
def snapshot(request):
    class Snapshot:
        def assert_match(self, actual):
            path = Path(__file__).parent / "_snapshots" / (request.node.name + ".pkl")
            with path.open("rb") as handle:
                expected = DataOnlyUnpickler(handle).load()
            assert actual == expected
    return Snapshot()
"""


def _seed_tiktoken_cache(fixtures: Path, cache: Path) -> None:
    """官方 fixture 可重建 GPT-2 原始文件；核对官方 SHA-256 后供 tiktoken 离线使用。"""
    cache.mkdir()
    base = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/"
    files = [
        (
            "encoder.json",
            (fixtures / "gpt2_vocab.json").read_bytes(),
            "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
        ),
        (
            "vocab.bpe",
            b"#version: 0.2\n" + (fixtures / "gpt2_merges.txt").read_bytes(),
            "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
        ),
    ]
    for filename, data, expected_hash in files:
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError(f"官方 GPT-2 fixture 内容不匹配：{filename}")
        cache_key = hashlib.sha1((base + filename).encode()).hexdigest()
        (cache / cache_key).write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-dir", type=Path, required=True, help="官方 A1 仓库根目录")
    args = parser.parse_args()
    source = args.assignment_dir.resolve() / "tests"
    project = Path(__file__).resolve().parents[1]
    with TemporaryDirectory(prefix="cs336-official-tokenizer-") as temporary:
        root = Path(temporary)
        tests = root / "tests"
        tests.mkdir()
        for name in ("test_tokenizer.py", "test_train_bpe.py", "common.py"):
            shutil.copy2(source / name, tests / name)
            print(f"{name} SHA-256: {hashlib.sha256((tests / name).read_bytes()).hexdigest()}")
        shutil.copytree(source / "fixtures", tests / "fixtures")
        (tests / "_snapshots").mkdir()
        shutil.copy2(
            source / "_snapshots" / "test_train_bpe_special_tokens.pkl",
            tests / "_snapshots" / "test_train_bpe_special_tokens.pkl",
        )
        (tests / "__init__.py").write_text("", encoding="utf-8")
        (tests / "adapters.py").write_text(_ADAPTERS, encoding="utf-8")
        (tests / "conftest.py").write_text(_CONFTEST, encoding="utf-8")
        cache = root / "tiktoken-cache"
        _seed_tiktoken_cache(tests / "fixtures", cache)
        environment = os.environ.copy()
        environment["TIKTOKEN_CACHE_DIR"] = str(cache)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(project / "src"), environment.get("PYTHONPATH", "")]
        )
        environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        print("在临时目录执行官方 tokenizer 测试（含原有性能/内存标记）。", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=short", "--durations=3", "tests"],
            cwd=root,
            env=environment,
            check=False,
        )
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
