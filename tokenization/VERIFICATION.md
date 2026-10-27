# 验证记录

验证日期：2026-10-26。环境：Linux / WSL，Python 3.10.12，regex 2026.9.29，
pytest 9.1.1，tiktoken 0.14.0。仅运行 CPU tokenizer 检查，没有训练语言模型或正式实验 tokenizer。

## 本地测试

`python -m pytest -q`：**31 passed**。

覆盖手算 merges、频次并列、重复 pair 的重叠计数、按 merge rank 编码、预分词边界、
中文/emoji/多语言、组合字符、CRLF、非法 UTF-8 片段解码、特殊 token 的最长匹配、
保存与重载、所有单次切分点/逐字符输入/随机切块、惰性读取、有界缓存，以及 CLI 训练/验证集分离。

`python examples/trace_bpe.py` 已运行，实际输出 `low → [257]`、
`lowest → [257, 101, 115, 116]`、`你 → [228, 189, 160]`，并完成 round trip。

## 官方测试

来源：[stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics/tree/a158843b20107949f1a8d7df1b05cd33b9166712)。
下载归档中记录的 commit：`a158843b20107949f1a8d7df1b05cd33b9166712`。

通过 `tools/run_official_tests.py`，在临时目录执行未修改的 `test_tokenizer.py`、`test_train_bpe.py`：

```text
27 passed, 1 xfailed in 11.70s
```

27 项通过包含官方参考 merges/词表比较、特殊 token 快照、1.5 秒训练速度门槛、
GPT-2/tiktoken token ID 对齐、多语言 round trip 和 streaming。
同机单独测量官方 `corpus.en`、500 词表、一个特殊 token 的训练约为 0.25 秒；耗时依硬件变化。

`xfailed` 是官方 `test_encode_memory_usage` 自带的标记：一次性 `encode` 预计超过它给定的内存预算。
官方 streaming 测试通过不等于证明所有输入都满足严格的 1 MB 内存上限；本实现的实际内存约定见 README。

运行器只替换两个接入函数和 snapshot fixture，不改测试文件和断言。
这样可在 Python 3.10 下单独运行 tokenizer 测试，避免加载官方其他题目的 torch 依赖和 Python 3.12 泛型语法。
快照使用限制对象加载的反序列化器，并对完整结果作相等比较。
tiktoken 读取经其官方 SHA-256 校验的 GPT-2 缓存文件，无需下载或替换 tiktoken 的编码实现。

| 文件 | SHA-256 |
| --- | --- |
| `test_tokenizer.py` | `a935fd4b3da985d957321bfb1794db3e840473da3b759193b6b5409db34b9551` |
| `test_train_bpe.py` | `7272025c12bfd07b2a6fd0b8f6b851ab7b9f2fb7d3d2f0b9ab85fffa4a7d4cb3` |
| `common.py` | `f9f4e859eea004adcbf9ba80e6f114730c5208e7079354b4710a909a1dbd3db1` |

复现（在 tokenization 目录、已激活的虚拟环境中）：

```bash
python -m pip install -e '.[dev,official]'
python tools/run_official_tests.py --assignment-dir /path/to/assignment1-basics
```

官方仓库保留在项目外；运行器不修改它，也不保留 fixture 上训练出来的 tokenizer。
如直接在完整官方环境运行，可将 `tests/adapters.py` 的两个函数体分别接到
`Tokenizer(vocab, merges, special_tokens)` 和 `train_bpe(input_path, vocab_size, special_tokens)`。

## 静态检查

`python -m ruff check .` 和 `python -m ruff format --check .` 均通过。
本地依赖最初安装在 `/tmp/cs336-tokenization-deps` 进行隔离验证，随后安装到本项目 `.venv`，
未修改用户全局 Python 包。常规安装和复现命令见 README。
