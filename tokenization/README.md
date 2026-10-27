# 从字符到 token：CS336 A1 BPE 学习实现

**BPE 的 token 是一段 bytes；token ID 是这段 bytes 在词表中的编号。**
训练负责决定哪些 bytes 经常相邻，可以合并成一项；编码负责按训练好的规则合并，并查出编号。
这个过程不需要神经网络，也不会在 `encode()` 时重新学习。

对应笔记：[第 01 讲：Tokenization](../../CS336Spring-2026-Learning-Notes/lectures/01-overview-tokenization.md)。
实现依据：[CS336 官方 A1](https://github.com/stanford-cs336/assignment1-basics)，不是官方参考答案。

## 1. 先把四个容易混淆的东西分开

| 名称 | 例子 | 含义 |
| --- | --- | --- |
| 字符 | `"你"` | Python `str` 中的一个 Unicode 字符 |
| UTF-8 bytes | `b'\xe4\xbd\xa0'` | `"你".encode("utf-8")` 得到 3 个 byte |
| token | `b'\xe4'` 或 `b'\xe4\xbd\xa0'` | 词表中的一个字节片段；长度取决于学到了什么合并 |
| token ID | `228` 或某个新编号 | 这个 token 在当前词表里的索引 |

请注意：`b'\xe4'` 里 `e4` 是十六进制写法，对应十进制 `228`。
`b'\xe4'` 是一个 byte，不是反斜杠、x、e、4 这四个字符。
`"你"` 的 Unicode 码点是 `U+4F60`（十进制 20320），它也不是 UTF-8 byte，更不是天然的 token ID。

本实现初始词表规定 `vocab[i] = bytes([i])`，所以最初单 byte 的编号恰好等于其数值。
例如 `vocab[108] == b'l'`。这是本实现的编号约定；GPT-2 的基础 byte ID 排列不同。

## 2. 手算一次训练：为什么词表里会出现 low

使用以下玩具**训练集**，换行保留：

```text
low
low
low
lower
```

预分词正则把字母串和换行分开。我们关注 `low`（3 次）和 `lower`（1 次）；
单独的换行没有内部相邻 pair，不能参与这一轮合并。

初始分解：

```text
low   → [b'l', b'o', b'w']                  × 3
lower → [b'l', b'o', b'w', b'e', b'r']      × 1
```

第一轮统计 **相邻 token 对出现多少次**：

| pair | 加权次数 |
| --- | ---: |
| `(b'l', b'o')` | 3 + 1 = 4 |
| `(b'o', b'w')` | 3 + 1 = 4 |
| `(b'w', b'e')` | 1 |
| `(b'e', b'r')` | 1 |

前两个并列。A1 规定按 pair 的 byte 字典序选较大的，故先选 `(b'o', b'w')`。
把它合并为 `b'ow'`，加入词表，分配新 ID `256`：

```text
low   → [b'l', b'ow']
lower → [b'l', b'ow', b'e', b'r']
```

第二轮 `(b'l', b'ow')` 出现 4 次，继续合并为 `b'low'`，ID `257`。
当目标词表大小是 `258` 时结束：256 个基础 byte + 2 个合并产物。
这里为了让数字最直观，没有注册特殊 token。

最终得到两份不同的东西：

```python
vocab[256] = b"ow"
vocab[257] = b"low"

merges = [
    (b"o", b"w"),  # rank 0：先学到
    (b"l", b"ow"),  # rank 1：后学到
]
```

`vocab` 告诉你“每个 ID 代表什么 bytes”；`merges` 告诉你“编码时按什么优先级合并”。
新增 token 后旧 token 仍保留，因此 `l`、`o`、`w`、`ow`、`low` 都还在词表中。

## 3. 再看编码：low 怎样真正变成 [257]

训练结束后，`encode("low")` 做的是：

```text
Python 字符串           "low"
预分词                 ["low"]
UTF-8 bytes            [108, 111, 119]
初始 token bytes       [b'l', b'o', b'w']
应用 rank 0            [b'l', b'ow']
应用 rank 1            [b'low']
查词表                 [257]
```

`"lowest"` 没在训练集中出现过，也可以编码：

```text
[b'l', b'o', b'w', b'e', b's', b't']
→ [b'l', b'ow', b'e', b's', b't']
→ [b'low', b'e', b's', b't']
→ [257, 101, 115, 116]
```

后面的 `e/s/t` 没有可用合并，就保持基础 byte token。
模型接收这些整数后，通常再查 embedding 矩阵得到向量；**查 embedding 是后续模型的工作**。

不要把 BPE 编码写成“从词表找最长的开头”。例如先学 `b+c`，后学 `a+b`、`ab+c`，
编码 `abc` 时优先得到 `[a, bc]`；即使词表里有 `abc`，也不应绕过 merge rank 直接选它。

## 4. 中文为什么可能一个字占三个 token

用上面只学过英文合并的 tokenizer 编码 `"你"`：

```text
"你" → UTF-8 [228, 189, 160]
     → [b'\xe4', b'\xbd', b'\xa0']
     → 没有匹配的中文合并规则
     → token IDs [228, 189, 160]
```

如果新的训练语料中“你”很常见，就可能逐步学到：

```text
(b'\xe4', b'\xbd')    → b'\xe4\xbd'
(b'\xe4\xbd', b'\xa0') → b'\xe4\xbd\xa0'
```

此时“你”可以只占一个 token。相邻汉字还可能继续合并；具体结果取决于整个训练集的统计和词表预算。
所以 token 不等于字符，也不等于单词；token 也不一定能够单独解码成完整字符。

解码的正确顺序是：

```python
tokenizer.decode([228, 189, 160])  # 先拼 bytes，再 UTF-8 decode → "你"
tokenizer.decode([228])  # 字节残缺，按 A1 约定 → "�"
```

不能把每个 token 分别 UTF-8 decode 后再拼字符串，否则可能把完整汉字破坏成几个替换字符。

## 5. 文件与阅读顺序

```text
tokenization/
├── src/cs336_basics/
│   ├── tokenizer.py             # BPE 训练、编码、解码、streaming、save/load
│   ├── cli.py                   # 训练/验证实验入口和元数据记录
│   └── __init__.py
├── examples/trace_bpe.py        # 打印上面的每一次合并，建议先运行
├── tests/test_tokenizer.py      # 自建边界测试，不作为正式训练语料
├── tools/run_official_tests.py  # 接入官方测试的独立运行器
├── VERIFICATION.md              # 本次实测结果与复现说明
└── pyproject.toml
```

第一遍按这个顺序读 [tokenizer.py](src/cs336_basics/tokenizer.py)：

1. `_merge_pair`：一次合并到底怎样改 token 列表。
2. `train_bpe`：统计 pair → 选最高频 → 更新词表和 merges。
3. `Tokenizer._encode_piece`：按 rank 合并一个预分词。
4. `Tokenizer.encode` / `decode`：把上述步骤接起来。
5. `save` / `load` / `from_files`：持久化词表与规则。

第二遍再看性能与流式部分。训练用 `pair → owners` 索引只更新受影响的预分词，
用堆选择最高频 pair；遇到过期堆条目就跳过。编码缓存有上限，长片段不缓存。

## 6. 安装与运行

只需要 CPU，Python 3.10+。从本目录运行：

当前工作区已经准备好 `.venv`，可以直接运行：

```bash
cd '/mnt/c/Users/Administrator/Desktop/科研/学习/lm-from-scratch/tokenization'
.venv/bin/python examples/trace_bpe.py
```

如果是在新机器上克隆项目，按以下步骤安装：

```bash
cd '/mnt/c/Users/Administrator/Desktop/科研/学习/lm-from-scratch/tokenization'
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python examples/trace_bpe.py
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

若 Debian/Ubuntu 提示缺少 `ensurepip`，先安装与 Python 版本对应的 `python3-venv` 系统包，
或者用 `uv venv .venv` 建环境。Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活。
本次 WSL 环境缺少 `ensurepip`，已通过 `venv --without-pip` 和宿主 pip 的 `--python` 参数
完成隔离安装，因此本工作区可以直接使用现成的 `.venv/bin/python`。

最小 API 示例（先准备自己的 `data/train.txt`）：

```python
from cs336_basics import Tokenizer, train_bpe

specials = ["<|endoftext|>"]
vocab, merges = train_bpe("data/train.txt", vocab_size=1000, special_tokens=specials)
tokenizer = Tokenizer(vocab, merges, specials)
ids = tokenizer.encode("你好，BPE！<|endoftext|>")
assert tokenizer.decode(ids) == "你好，BPE！<|endoftext|>"

tokenizer.save("artifacts/my_tokenizer")
restored = Tokenizer.load("artifacts/my_tokenizer")
assert restored.encode("你好") == tokenizer.encode("你好")
```

`vocab_size` 包含基础 byte 和特殊 token；语料太小、所有 pair 用尽时会提前结束。
词表与 merges 保存为两个带格式版本的 JSON 文件，其中 bytes 用十六进制无损保存。
`from_files(vocab_filepath, merges_filepath, special_tokens=None)` 可加载同样的格式；
它不直接读取 GPT-2 原始文件。官方测试先把 GPT-2 字节映射还原为 `dict[int, bytes]`，
再调用构造器，因此仍可检验与 tiktoken 的编码兼容性。

## 7. 特殊 token 和 streaming 的约定

注册的 `<|endoftext|>` 整体编码成一个 ID，训练时不进入普通 pair 统计。
它两侧的文本不能跨界合并。注册项有重叠时，在同一位置优先匹配较长字符串。
`[.*]` 这样的注册项也按字面值匹配。此学习 API 会识别所有已注册项，是否赋予它们控制含义由上层应用决定。

`encode_iterable` 接收的是**同一段文本的连续字符串块**：

```python
chunks = ["你", "好<|endo", "ftext|>lo", "west"]
assert list(tokenizer.encode_iterable(chunks)) == tokenizer.encode("".join(chunks))

with open("data/validation.txt", encoding="utf-8", newline="") as handle:
    for token_id in tokenizer.encode_iterable(handle):
        pass  # 实际应用中可写入 token 文件，避免把全部 ID 收集在内存中。
```

实现保留末尾可能变化的预分词，以及未完成的特殊 token；编码已确定的前缀时，
仍让正则看到后续上下文，防止空白前瞻规则改变结果。
它不会把每块当成独立文档，也不会自动添加分隔符。
如果要拼接独立文档，应自己在文档之间加入已注册的结束标记。

这里的输入块是 `str`。如果直接读取二进制块，UTF-8 字符本身可能被切开，
需要先用 `codecs.getincrementaldecoder("utf-8")()` 增量解码，再把字符串块交进来。
单个预分词特别长时，必须缓冲该片段；因此不承诺任意输入下固定字节数的内存上限。

## 8. 正式实验必须先划分训练/验证集

先按文档划分并去重，再只用训练集学 tokenizer。验证集只用于固定 tokenizer 的编码和评估。
同一文档不要一部分分进训练、一部分分进验证；测试集保留到最终评价。

```bash
python -m cs336_basics.cli train \
  --train-file data/train.txt \
  --validation-file data/validation.txt \
  --vocab-size 1000 \
  --output-dir artifacts/experiment_01

python -m cs336_basics.cli encode \
  --tokenizer-dir artifacts/experiment_01 \
  --text '你好，BPE！'
```

脚本只把 `--train-file` 传给 `train_bpe`；拒绝相同文件或内容完全一致的训练/验证副本。
这不能自动检测部分重叠或近似重复，仍需在准备数据时完成去重和来源审查。
`experiment.json` 记录两个文件的 SHA-256、实际词表大小、合并次数、训练时间、编码耗时、
UTF-8 bytes、token 数和 bytes/token。空文件的 bytes/token 记为 `null`。

官方测试为了检查训练算法，会在官方 fixture 上临时训练，这是单元测试的用途。
**不能把这些测试产物，或曾使用验证集训练的调试 tokenizer，作为最终实验 tokenizer。**
本目录未预置正式实验语料或正式 tokenizer；例子只生成临时玩具产物。

训练实现一次读取整个训练文件，编码极长预分词时也使用直观的反复扫描算法。
当前定位是能通过 A1 tokenizer 测试的学习实现；多 GB 训练应进一步增加按特殊边界分块的预分词统计，
并在真实语料上测量时间和内存，不能由小语料测试直接推断其大规模性能。
