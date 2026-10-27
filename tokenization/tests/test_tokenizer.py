"""测试算法约束和跨边界行为；这些玩具语料不用于正式实验训练。"""

import json
import random
from pathlib import Path

import pytest
import regex

from cs336_basics import Tokenizer, train_bpe
from cs336_basics.cli import main
from cs336_basics.tokenizer import PAT


def byte_vocab() -> dict[int, bytes]:
    return {index: bytes([index]) for index in range(256)}


@pytest.fixture
def learned(tmp_path: Path) -> Tokenizer:
    train = tmp_path / "unit_train.txt"
    train.write_text(
        ("low lower lowest 你好世界🙂 café we're I'll 123\r\n\t  " * 8)
        + "<x><x>abc<|endoftext|>[.*]",
        encoding="utf-8",
    )
    specials = ["<x>", "<x>abc", "<|endoftext|>", "[.*]"]
    vocab, merges = train_bpe(train, 330, specials)
    return Tokenizer(vocab, merges, specials)


def test_exact_training_tie_break_and_ids(tmp_path: Path) -> None:
    train = tmp_path / "train.txt"
    train.write_text("low\nlow\nlow\nlower\n", encoding="utf-8")
    vocab, merges = train_bpe(train, 258, [])
    assert merges == [(b"o", b"w"), (b"l", b"ow")]
    assert vocab[256] == b"ow"
    assert vocab[257] == b"low"
    tokenizer = Tokenizer(vocab, merges)
    assert tokenizer.encode("low") == [257]
    assert tokenizer.encode("lowest") == [257, 101, 115, 116]


def test_repeated_pairs_count_overlaps_but_merge_nonoverlaps(tmp_path: Path) -> None:
    train = tmp_path / "train.txt"
    train.write_text("aaaa\nbcbcbc", encoding="utf-8")
    vocab, merges = train_bpe(train, 259, [])
    # aa 与 bc 均有 3 次；若 aa 被错误计为 2 次，第二步就会选错。
    assert merges[:2] == [(b"b", b"c"), (b"a", b"a")]
    tokenizer = Tokenizer(vocab, merges)
    assert tokenizer.decode(tokenizer.encode("aaaaa bcbcbc")) == "aaaaa bcbcbc"


def test_training_excludes_specials_and_preserves_boundaries(tmp_path: Path) -> None:
    train = tmp_path / "train.txt"
    train.write_text("ab<x>cd<x>abc", encoding="utf-8")
    vocab, merges = train_bpe(train, 300, ["<x>", "<x>abc"])
    assert merges == [(b"c", b"d"), (b"a", b"b")]
    assert b"abcd" not in vocab.values()
    assert b"<x" not in vocab.values()
    assert vocab[256] == b"<x>"


def test_encoding_uses_merge_rank_not_longest_vocab_match() -> None:
    vocab = {**byte_vocab(), 256: b"bc", 257: b"ab", 258: b"abc"}
    tokenizer = Tokenizer(vocab, [(b"b", b"c"), (b"a", b"b"), (b"ab", b"c")])
    # 虽然词表里有 abc，但先学到的 bc 会先合并，导致 ab+c 无法再应用。
    assert tokenizer.encode("abc") == [97, 256]


def test_pretokenization_boundaries_are_hard() -> None:
    vocab = {**byte_vocab(), 256: b"a!"}
    tokenizer = Tokenizer(vocab, [(b"a", b"!")])
    assert tokenizer.encode("a!") == [97, 33]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "你",
        "你好，世界！",
        "🙂🙃👩🏽‍💻",
        "مرحبا हिन्दी",
        "é e\u0301",
        "\x00\t\r\n  ",
        "we're I'll they'd",
        "未登录词<|endoftext|><x>abc[.*]<x>",
        "𠮷\ufeff\u2028",
    ],
)
def test_unicode_and_special_roundtrip(learned: Tokenizer, text: str) -> None:
    assert learned.decode(learned.encode(text)) == text


def test_decode_joins_bytes_before_utf8() -> None:
    tokenizer = Tokenizer(byte_vocab(), [])
    assert tokenizer.encode("你") == [228, 189, 160]
    assert tokenizer.decode([228, 189, 160]) == "你"
    assert tokenizer.decode([228]) == "�"
    assert tokenizer.decode([255]) == "�"
    assert "".join(tokenizer.decode([value]) for value in [228, 189, 160]) != "你"
    with pytest.raises(ValueError, match="token ID"):
        tokenizer.decode([999])


def test_specials_longest_match_added_without_mutating_input() -> None:
    vocab = byte_vocab()
    tokenizer = Tokenizer(vocab, [], ["<x>", "<x>abc", "[.*]"])
    assert len(vocab) == 256
    assert tokenizer.encode("<x>abc<x>[.*]") == [257, 256, 258]


def test_save_load_preserves_all_bytes_and_configuration(
    learned: Tokenizer, tmp_path: Path
) -> None:
    learned.save(tmp_path)
    loaded = Tokenizer.load(tmp_path)
    assert loaded.vocab == learned.vocab
    assert loaded.merges == learned.merges
    assert loaded.special_tokens == learned.special_tokens
    text = "你好🙂<x>abc\r\nwe're"
    assert loaded.encode(text) == learned.encode(text)
    assert loaded.decode(loaded.encode(text)) == text
    plain = Tokenizer.from_files(tmp_path / "vocab.json", tmp_path / "merges.json", [])
    assert len(plain.encode("<x>abc")) > 1
    document = json.loads((tmp_path / "vocab.json").read_text(encoding="utf-8"))
    document["pretoken_pattern"] = "different"
    (tmp_path / "vocab.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="预分词"):
        Tokenizer.load(tmp_path)


@pytest.mark.parametrize(
    "text",
    [
        "lowest你好🙂",
        "a  b\r\n\r\nz  ",
        "we're I'll they've",
        "<x>abc<x><|endoftext|>[.*]",
        "a<x>abZ<x>ab",
        "\t\n   lowest\n",
        "你<|endoftext|>好",
    ],
)
def test_stream_every_single_split_and_character_chunks(learned: Tokenizer, text: str) -> None:
    expected = learned.encode(text)
    for cut in range(len(text) + 1):
        assert list(learned.encode_iterable([text[:cut], "", text[cut:]])) == expected
    assert list(learned.encode_iterable(iter(text))) == expected


def test_stream_random_boundaries_and_arbitrary_special_prefixes(tmp_path: Path) -> None:
    randomizer = random.Random(336)
    specials = ["a", "abc", "bcda", "'reX", "  z", "你好"]
    train = tmp_path / "train.txt"
    train.write_text(("abcd xyz we're  \n你好🙂123!" * 100), encoding="utf-8")
    vocab, merges = train_bpe(train, 310, specials)
    tokenizer = Tokenizer(vocab, merges, specials)
    for _ in range(150):
        text = "".join(randomizer.choices("abcd xyz're\n\t你好🙂12!", k=100))
        chunks = []
        start = 0
        while start < len(text):
            end = start + randomizer.randint(1, 9)
            chunks.append(text[start:end])
            start = end
        assert list(tokenizer.encode_iterable(chunks)) == tokenizer.encode(text)


def test_stream_is_lazy_and_cache_is_bounded() -> None:
    tokenizer = Tokenizer(byte_vocab(), [], cache_size=3)
    consumed = []

    def chunks():
        for index in range(100):
            consumed.append(index)
            yield f"word{index} next\n"

    ids = tokenizer.encode_iterable(chunks())
    assert consumed == []
    assert next(ids) == ord("w")
    assert len(consumed) < 100
    list(ids)
    assert len(tokenizer._cache) <= 3


def test_empty_training_and_invalid_configuration(tmp_path: Path) -> None:
    train = tmp_path / "empty.txt"
    train.write_text("", encoding="utf-8")
    vocab, merges = train_bpe(train, 300, ["<eos>"])
    assert len(vocab) == 257 and merges == []
    with pytest.raises(ValueError, match="vocab_size"):
        train_bpe(train, 256, ["<eos>"])
    with pytest.raises(ValueError, match="非空"):
        train_bpe(train, 300, [""])
    with pytest.raises(ValueError, match="重复"):
        Tokenizer({0: b"a", 1: b"a"}, [])


def test_crlf_is_not_normalized_during_training(tmp_path: Path) -> None:
    train = tmp_path / "windows.txt"
    train.write_bytes(b"\r\n\r\n\r\n")
    vocab, merges = train_bpe(train, 257, [])
    assert merges == [(b"\r", b"\n")]
    assert Tokenizer(vocab, merges).decode([256]) == "\r\n"


def test_cli_uses_only_train_split_and_rejects_same_content(tmp_path, monkeypatch, capsys) -> None:
    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    output = tmp_path / "output"
    train.write_text("low\nlow\nlower\n", encoding="utf-8")
    validation.write_text("验证集独有词🙂", encoding="utf-8")
    argv = [
        "bpe-study",
        "train",
        "--train-file",
        str(train),
        "--validation-file",
        str(validation),
        "--output-dir",
        str(output),
        "--vocab-size",
        "270",
    ]
    monkeypatch.setattr("sys.argv", argv)
    main()
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["validation"]["tokens"] > 0
    saved = Tokenizer.load(output)
    expected_vocab, expected_merges = train_bpe(train, 270, ["<|endoftext|>"])
    assert saved.vocab == expected_vocab and saved.merges == expected_merges
    validation.write_bytes(train.read_bytes())
    with pytest.raises(ValueError, match="训练集和验证集"):
        main()


def test_regex_keeps_whitespace_and_unicode() -> None:
    text = "你好  world\r\nwe're🙂\t"
    assert "".join(regex.findall(PAT, text)) == text
