# 00-template

复制这个目录当新篇的起点。它不属于任何一篇文章，只用来固定目录形态。

一个篇目目录必须有：

1. **baseline 与 optimized 并列**，两者跑同一个问题、同一份输入
2. **正确性校验先于性能**：optimized 的输出必须先对齐 baseline 再计时
3. **cold 与 hot 两种口径都输出**，CSV 打到 stdout，schema 见 `bench/include/bench/csv.hpp`
4. **一个 `README.md`**，写清这篇要证明什么、预期在哪台机器上收益反号

```bash
cmake --build build --target kernel-00-template -j
python3 tools/run.py --kernel 00-template --machine h200
```

本目录的 triad 例子里，`optimized` 用 128 位访存 + 按 SM 数量定网格。它在
`h20`（ridge point ≈37 FLOP/B）和 `h200`（≈206）上的相对收益不同——这正是整个系列
要反复利用的那条差异。
