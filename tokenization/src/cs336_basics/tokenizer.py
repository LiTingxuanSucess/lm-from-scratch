"""Byte-level BPE：训练学合并规则，编码应用规则，解码拼回字节。

建议阅读顺序：train_bpe → _merge_pair → Tokenizer._encode_piece → encode → decode。
encode_iterable 和持久化是第二遍再读的工程部分。完整推导见项目 README.md。
"""

from __future__ import annotations

import heapq
import json
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import regex

# Python 内置 re 不支持 \p{L}/\p{N}，这里必须使用第三方 regex。
# L 表示 Unicode 字母（包括汉字），N 表示数字；空格也属于文本，不能丢弃。
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PRETOKEN = regex.compile(PAT)
_FORMAT = "cs336-byte-bpe-v1"
Pair = tuple[bytes, bytes]
Word = tuple[bytes, ...]


def _normalize_specials(special_tokens: Iterable[str] | None) -> list[str]:
    """保留注册顺序并去重；空字符串不能作为分隔符。"""
    if special_tokens is None:
        return []
    if isinstance(special_tokens, str):
        raise TypeError("special_tokens 应是字符串列表，不能是一个字符串")
    result: list[str] = []
    for token in special_tokens:
        if not isinstance(token, str) or not token:
            raise ValueError("特殊 token 必须是非空字符串")
        token.encode("utf-8")  # 及早拒绝不能编码为合法 UTF-8 的孤立代理码点。
        if token not in result:
            result.append(token)
    return result


def _special_pattern(special_tokens: list[str]) -> regex.Pattern | None:
    if not special_tokens:
        return None
    # 同一位置优先匹配长 token；escape 防止 |、[、* 等被解释为正则语法。
    ordered = sorted(special_tokens, key=lambda token: (-len(token), token))
    return regex.compile("|".join(regex.escape(token) for token in ordered))


def _segments(text: str, pattern: regex.Pattern | None) -> Iterator[tuple[str, bool]]:
    """返回 (文本片段, 是否为特殊 token)，始终保留特殊 token 两侧的边界。"""
    start = 0
    if pattern is not None:
        for match in pattern.finditer(text):
            if match.start() > start:
                yield text[start : match.start()], False
            yield match.group(), True
            start = match.end()
    if start < len(text):
        yield text[start:], False


def _merge_pair(parts: Word, pair: Pair) -> Word:
    """从左到右、非重叠地替换 pair：aaaa 合并 aa 后得到 [aa, aa]。"""
    merged: list[bytes] = []
    index = 0
    while index < len(parts):
        if index + 1 < len(parts) and (parts[index], parts[index + 1]) == pair:
            merged.append(pair[0] + pair[1])
            index += 2
        else:
            merged.append(parts[index])
            index += 1
    return tuple(merged)


def _pair_counts(parts: Word) -> Counter[Pair]:
    # 统计时允许重叠：aaaa 的 (a, a) 有 3 次，实际替换仍是非重叠的。
    return Counter(zip(parts, parts[1:], strict=False))


@dataclass(frozen=True)
class _Candidate:
    count: int
    pair: Pair

    def __lt__(self, other: _Candidate) -> bool:
        # heapq 默认是最小堆；反转比较，使最高频、字节字典序最大的 pair 先出堆。
        return (self.count, self.pair) > (other.count, other.pair)


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[Pair]]:
    """仅从 input_path 指定的训练集学习 BPE，返回词表和按学习顺序排列的 merges。

    vocab_size 是包含 256 个基础 byte、特殊 token 和合并产物的总大小上限。
    没有可合并 pair 时提前结束，不用无意义的 token 填满词表。
    此学习版整文件读取 UTF-8 语料；不要直接用于几十 GB 的正式语料。
    """
    specials = _normalize_specials(special_tokens)
    vocab = {index: bytes([index]) for index in range(256)}
    known_tokens = set(vocab.values())
    for token in specials:
        token_bytes = token.encode("utf-8")
        if token_bytes not in known_tokens:
            vocab[len(vocab)] = token_bytes
            known_tokens.add(token_bytes)
    if type(vocab_size) is not int or vocab_size < len(vocab):
        raise ValueError(f"vocab_size 至少为 {len(vocab)}（基础 byte + 特殊 token）")

    # 二进制读取再解码，避免 Windows 的 CRLF 被文本模式悄悄改成 LF。
    text = Path(input_path).read_bytes().decode("utf-8")
    frequencies: Counter[Word] = Counter()
    for segment, is_special in _segments(text, _special_pattern(specials)):
        if is_special:
            continue  # 特殊 token 只占词表位置，不参与相邻 pair 计数。
        for match in _PRETOKEN.finditer(segment):
            # 对 bytes 迭代得到整数，因此用 bytes([value]) 变回单 byte token。
            parts = tuple(bytes([value]) for value in match.group().encode("utf-8"))
            frequencies[parts] += 1
    del text

    # 同一预分词只存一次，用频率加权。例如 low 出现 5 次，(l, o) 就贡献 5。
    words = list(frequencies)
    weights = list(frequencies.values())
    del frequencies
    counts: Counter[Pair] = Counter()
    owners: dict[Pair, set[int]] = defaultdict(set)
    for word_id, parts in enumerate(words):
        for pair, occurrences in _pair_counts(parts).items():
            counts[pair] += occurrences * weights[word_id]
            owners[pair].add(word_id)

    heap = [_Candidate(count, pair) for pair, count in counts.items()]
    heapq.heapify(heap)
    merges: list[Pair] = []
    while len(vocab) < vocab_size and heap:
        best = heapq.heappop(heap)
        if counts.get(best.pair, 0) != best.count:
            continue  # 增量更新后，堆里旧的计数可能过期；直接丢弃。

        pair = best.pair
        merges.append(pair)
        token_bytes = pair[0] + pair[1]
        if token_bytes not in known_tokens:
            vocab[len(vocab)] = token_bytes
            known_tokens.add(token_bytes)

        changed: set[Pair] = set()
        # 只更新包含被选 pair 的预分词，避免每轮重扫整个词库。
        for word_id in tuple(owners[pair]):
            before = _pair_counts(words[word_id])
            words[word_id] = _merge_pair(words[word_id], pair)
            after = _pair_counts(words[word_id])
            for affected in before.keys() | after.keys():
                delta = after[affected] - before[affected]
                if delta:
                    counts[affected] += delta * weights[word_id]
                    changed.add(affected)
                if affected not in after:
                    owners[affected].discard(word_id)
                elif affected not in before:
                    owners[affected].add(word_id)

        for affected in changed:
            if counts[affected] > 0:
                heapq.heappush(heap, _Candidate(counts[affected], affected))
            else:
                counts.pop(affected, None)
                owners.pop(affected, None)
        # 防止大语料训练时过期堆条目无限累积；不改变选择规则。
        if len(heap) > max(1024, 4 * len(counts)):
            heap = [_Candidate(count, pair) for pair, count in counts.items()]
            heapq.heapify(heap)

    return vocab, merges


class Tokenizer:
    """由固定词表和固定 merges 编码；encode 永远不会更新训练统计。

    传入的 special_tokens 会被识别为原子 token；未注册的类似字符串按普通文本处理。
    这是教学 API，默认允许所有已注册特殊 token，不提供聊天角色/权限管理。
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
        *,
        cache_size: int = 4096,
    ) -> None:
        if not vocab:
            raise ValueError("词表不能为空")
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size 必须是非负整数")
        self.vocab = dict(vocab)  # 不修改调用者传入的词表。
        for token_id, token_bytes in self.vocab.items():
            if type(token_id) is not int or token_id < 0:
                raise ValueError("token ID 必须是非负整数")
            if not isinstance(token_bytes, bytes) or not token_bytes:
                raise ValueError("词表的值必须是非空 bytes")
        self._token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        if len(self._token_to_id) != len(self.vocab):
            raise ValueError("不同 token ID 不能对应重复的 bytes")

        self.special_tokens = _normalize_specials(special_tokens)
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self._token_to_id:
                token_id = max(self.vocab) + 1  # 也支持 ID 不连续的外部词表。
                self.vocab[token_id] = token_bytes
                self._token_to_id[token_bytes] = token_id
        self._special_ids = {
            token: self._token_to_id[token.encode("utf-8")] for token in self.special_tokens
        }
        self._special_re = _special_pattern(self.special_tokens)
        self._special_prefixes = sorted(
            {token[:end] for token in self.special_tokens for end in range(1, len(token))},
            key=len,
            reverse=True,
        )

        self.merges = [(left, right) for left, right in merges]
        self._merge_ranks: dict[Pair, int] = {}
        for rank, pair in enumerate(self.merges):
            if pair in self._merge_ranks:
                raise ValueError(f"重复的 merge：{pair!r}")
            if any(part not in self._token_to_id for part in (*pair, pair[0] + pair[1])):
                raise ValueError(f"merge 的两侧及合并结果必须存在于词表：{pair!r}")
            self._merge_ranks[pair] = rank
        self._cache_size = cache_size
        self._cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()

    def _encode_piece(self, piece: str) -> tuple[int, ...]:
        """编码一个预分词：先拆 byte，再反复应用当前可用的最早 merge。"""
        cached = self._cache.get(piece)
        if cached is not None:
            self._cache.move_to_end(piece)
            return cached

        parts = tuple(bytes([value]) for value in piece.encode("utf-8"))
        while len(parts) > 1:
            # rank 是训练先后顺序，不是词频，也不是 token ID 或 token 长度。
            best_pair: Pair | None = None
            best_rank = len(self.merges)
            for pair in zip(parts, parts[1:], strict=False):
                rank = self._merge_ranks.get(pair, len(self.merges))
                if rank < best_rank:
                    best_pair, best_rank = pair, rank
            if best_pair is None:
                break
            parts = _merge_pair(parts, best_pair)

        ids = tuple(self._token_to_id[part] for part in parts)
        # 限制条目数和单条长度；长字符串不会挤占大量缓存内存。
        if self._cache_size and len(piece) <= 256:
            self._cache[piece] = ids
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return ids

    def _encode_tokens(self, text: str, stop: int | None = None) -> Iterator[int]:
        # 流式处理时，正则必须仍能看到 stop 后的上下文。
        # 如果先截断 text[:stop]，负向前瞻会把前缀末尾的空白误认成文末空白。
        offset = 0
        for segment, is_special in _segments(text, self._special_re):
            if is_special:
                if stop is not None and offset + len(segment) > stop:
                    return
                yield self._special_ids[segment]
            else:
                for match in _PRETOKEN.finditer(segment):
                    if stop is not None and offset + match.end() > stop:
                        return
                    yield from self._encode_piece(match.group())
            offset += len(segment)

    def encode(self, text: str) -> list[int]:
        """文本 → 特殊 token 分隔 → 预分词 → UTF-8 bytes → BPE → token IDs。"""
        if not isinstance(text, str):
            raise TypeError("encode 接受 str；原始 bytes 请先按 UTF-8 解码")
        return list(self._encode_tokens(text))

    def decode(self, ids: Iterable[int]) -> str:
        """先拼接全部 token 的 bytes，再统一 UTF-8 解码；残缺字节使用 � 替换。"""
        pieces: list[bytes] = []
        for token_id in ids:
            if type(token_id) is not int or token_id not in self.vocab:
                raise ValueError(f"未知或非法的 token ID：{token_id!r}")
            pieces.append(self.vocab[token_id])
        return b"".join(pieces).decode("utf-8", errors="replace")

    def _stream_prefix_end(self, buffer: str) -> int:
        """找出不会因后续输入而改变编码的前缀终点，其后的文本必须暂存。"""
        limit = len(buffer)
        for prefix in self._special_prefixes:
            if buffer.endswith(prefix):
                limit -= len(prefix)
                break  # 最长后缀优先，保留可能被切断的特殊 token。

        start = 0
        if self._special_re is not None:
            for match in self._special_re.finditer(buffer):
                if match.end() > limit:
                    limit = min(limit, match.start())
                    break
                start = match.end()  # 完整、无歧义的特殊 token 提供安全边界。

        # 保留最后两个预分词：例如块尾 "'r"，下一块的 "e" 会组成 "'re"。
        # 空白分支含负向前瞻，块尾的空白归属也可能随下一字符改变。
        last_starts: deque[int] = deque(maxlen=2)
        for match in _PRETOKEN.finditer(buffer, start, limit):
            last_starts.append(match.start())
        return last_starts[0] if last_starts else start

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """惰性编码连续字符串块，结果等于 encode(''.join(iterable))。

        不同块是同一文档的连续部分，不自动添加换行或文档分隔符。
        内存与当前块、未完成预分词和有界缓存相关，不随整文件线性增长。
        极端长的单个预分词必须缓冲；不承诺任意输入下严格固定内存。
        输入是 str；若读取二进制块，先用 UTF-8 incremental decoder 处理字符边界。
        """
        buffer = ""
        for chunk in iterable:
            if not isinstance(chunk, str):
                raise TypeError("encode_iterable 的每一块必须是 str")
            if not chunk:
                continue
            buffer += chunk
            end = self._stream_prefix_end(buffer)
            if end:
                yield from self._encode_tokens(buffer, stop=end)
                buffer = buffer[end:]
        yield from self._encode_tokens(buffer)

    def save(self, directory: str | Path) -> None:
        """保存为可读 JSON；bytes 用十六进制表示，避免错误地单独 UTF-8 解码 token。"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        vocab_document = {
            "format": _FORMAT,
            "pretoken_pattern": PAT,
            "special_tokens": self.special_tokens,
            "vocab": {str(key): value.hex() for key, value in sorted(self.vocab.items())},
        }
        merges_document = {
            "format": _FORMAT,
            "merges": [[left.hex(), right.hex()] for left, right in self.merges],
        }
        for filename, document in (
            ("vocab.json", vocab_document),
            ("merges.json", merges_document),
        ):
            with (directory / filename).open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2)
                handle.write("\n")

    @classmethod
    def load(cls, directory: str | Path) -> Tokenizer:
        """加载 save 生成的目录，恢复词表、merge 顺序和特殊 token 配置。"""
        directory = Path(directory)
        return cls.from_files(directory / "vocab.json", directory / "merges.json")

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """A1 风格的双文件入口，读取本项目的 JSON 格式。

        None 恢复保存的特殊 token；显式传 [] 表示不启用特殊 token 匹配。
        GPT-2 原始 vocab.json/merges.txt 使用另一种字节映射，不能直接传到这里。
        """
        vocab_document = json.loads(Path(vocab_filepath).read_text(encoding="utf-8"))
        merges_document = json.loads(Path(merges_filepath).read_text(encoding="utf-8"))
        if any(document.get("format") != _FORMAT for document in (vocab_document, merges_document)):
            raise ValueError("不支持的 tokenizer 文件格式或版本")
        if vocab_document.get("pretoken_pattern") != PAT:
            raise ValueError("保存文件的预分词规则与当前实现不一致")
        vocab = {int(key): bytes.fromhex(value) for key, value in vocab_document["vocab"].items()}
        merges = [
            (bytes.fromhex(left), bytes.fromhex(right)) for left, right in merges_document["merges"]
        ]
        specials = vocab_document["special_tokens"] if special_tokens is None else special_tokens
        return cls(vocab, merges, specials)
