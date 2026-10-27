"""运行 python examples/trace_bpe.py，看见训练结果和实际编码的每一步。"""

from pathlib import Path
from tempfile import TemporaryDirectory

import regex

from cs336_basics import Tokenizer, train_bpe
from cs336_basics.tokenizer import PAT


def show_encoding(tokenizer: Tokenizer, text: str) -> None:
    """教学用展开打印；最后与正式 encode 的结果核对。"""
    print(f"\n原文：{text!r}")
    print("UTF-8 十进制字节：", list(text.encode("utf-8")))
    ranks = {pair: rank for rank, pair in enumerate(tokenizer.merges)}
    for piece in regex.findall(PAT, text):
        parts = [bytes([value]) for value in piece.encode("utf-8")]
        print(f"  预分词 {piece!r}，初始 token bytes：{parts}")
        while len(parts) > 1:
            candidates = [pair for pair in zip(parts, parts[1:], strict=False) if pair in ranks]
            if not candidates:
                break
            chosen = min(candidates, key=ranks.__getitem__)
            updated = []
            index = 0
            while index < len(parts):
                if index + 1 < len(parts) and (parts[index], parts[index + 1]) == chosen:
                    updated.append(chosen[0] + chosen[1])
                    index += 2
                else:
                    updated.append(parts[index])
                    index += 1
            parts = updated
            print(f"  应用 rank={ranks[chosen]} 的 {chosen!r} → {parts}")
    ids = tokenizer.encode(text)
    print("最终 token bytes：", [tokenizer.vocab[token_id] for token_id in ids])
    print("最终 token IDs：", ids)
    print("decode：", repr(tokenizer.decode(ids)))
    assert tokenizer.decode(ids) == text


def main() -> None:
    with TemporaryDirectory(prefix="bpe-teaching-") as temporary:
        root = Path(temporary)
        train_path = root / "toy_train.txt"
        # 这是刻意设计的玩具训练集，不是验证集，也不是最终实验 tokenizer。
        train_path.write_text("low\nlow\nlow\nlower\n", encoding="utf-8")
        vocab, merges = train_bpe(train_path, vocab_size=258, special_tokens=[])
        tokenizer = Tokenizer(vocab, merges)
        print("训练集：low\\nlow\\nlow\\nlower\\n")
        print("第一轮 (l,o) 和 (o,w) 都出现 4 次，字节字典序较大的 (o,w) 获胜。")
        for rank, (left, right) in enumerate(merges):
            token_id = next(key for key, value in vocab.items() if value == left + right)
            print(f"rank={rank}: {left!r} + {right!r} → {left + right!r}，ID={token_id}")

        show_encoding(tokenizer, "low")
        show_encoding(tokenizer, "lowest")  # 未出现在训练集的词也可以编码。
        show_encoding(tokenizer, "你")  # 尚未学到中文合并，仍由基础 byte 完整表示。
        print("单独 decode 汉字第一个 byte：", repr(tokenizer.decode([228])))
        print("拼齐 3 个 byte 再 decode：", repr(tokenizer.decode([228, 189, 160])))

        tokenizer.save(root / "saved")
        loaded = Tokenizer.load(root / "saved")
        text = "lowest 你好🙂"
        chunks = ["lo", "west ", "你", "好🙂"]
        assert list(loaded.encode_iterable(chunks)) == loaded.encode(text)
        print("\nsave/load 与跨块 streaming 检查通过。")
        print("ID 只是词表编号；另一份 tokenizer 的同一 ID 可能代表不同 bytes。")


if __name__ == "__main__":
    main()
