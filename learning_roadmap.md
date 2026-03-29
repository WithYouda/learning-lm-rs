# AI 推理系统学习路线图（基于 2026-03-29 复测）

> 这版路线图只基于当前仓库的真实 benchmark 结果，不再沿用历史文档里的旧数据。
> 同时，这版不再给泛泛而谈的“都学一点”，而是围绕你这个项目本身的瓶颈、你的目标（考研复试 + 找工作）来排优先级。

---

## 一、先纠正一个关键事实

上一版 `learning_roadmap.md` 里“已达成的成果”写的：

- `prefill tok/s = 32.79`
- `decode tok/s = 14.74`
- `prefill_ratio vs llama.cpp = 81.24%`
- `decode_ratio vs llama.cpp = 105.28%`

这些**不能代表当前项目的真实现状**。

原因有两个：

1. 那些数字来自历史阶段记录，不是 2026-03-29 当前代码的复测结果。
2. 项目里的 `bench_compare_with_llamacpp_metrics*` 之前默认调用的 `llama-cli` 实际启用了 CUDA；也就是说，旧对照里混入了 **GPU offload 的 llama.cpp**，不能拿来判断你的 **CPU 推理** 水平。

因此，这次我先把项目里的 compare benchmark 改成了 **默认 `-ngl 0`，强制 CPU-only**，然后重新跑测试。

---

## 二、当前真实结果（2026-03-29 复测）

### 1. 本项目本地 benchmark

测试函数：`test::bench_prefill_decode_metrics`

默认配置：

- `LMRS_THREADS=5`
- `LMRS_PREFILL_THREADS=5`
- `LMRS_CPU_MASK=0,1,2,3,4`
- `LMRS_ENABLE_AFFINITY=1`
- `LMRS_PREFILL_BACKEND=gemm`
- `LMRS_HOT_MATRIX_CACHE_MB=1`

5 轮结果：

| 指标 | 当前均值 | 标准差 |
|---|---:|---:|
| prefill tok/s | **15.2402** | 0.1612 |
| decode tok/s | **11.1701** | 0.1427 |
| TTFT (s) | **0.0035** | 0.0002 |

### 2. CPU-only 的 llama.cpp 连续对照

测试函数：`test::bench_compare_with_llamacpp_metrics_continuous`

5 轮结果：

| 指标 | 我方 | llama.cpp (CPU-only) | 比值 |
|---|---:|---:|---:|
| prefill tok/s | 15.2929 | 131.5800 | **0.1162** |
| decode tok/s | 11.3033 | 17.7200 | **0.6378** |

### 3. CPU-only 的 llama.cpp 交错对照

测试函数：`test::bench_compare_with_llamacpp_metrics`

5 轮结果：

| 指标 | 我方 | llama.cpp (CPU-only) | 比值 |
|---|---:|---:|---:|
| prefill tok/s | 14.9342 | 127.9600 | **0.1170** |
| decode tok/s | 11.4726 | 17.4200 | **0.6594** |

### 4. 这组数据说明什么

当前项目的真实状态不是“已经全面超过 llama.cpp 的 CPU 推理”，而是：

- **decode 已经做到 CPU llama.cpp 的约 64%~66%**
- **prefill 只有 CPU llama.cpp 的约 11%~12%**

这和上一版路线图里的判断差别非常大。

---

## 三、现在该怎么判断：CPU 推理还要不要继续深入？

### 结论

**要继续，但只能“有限继续”，而不是无限深挖。**

更准确地说：

- **还值得继续做一轮 CPU 收尾冲刺**
- **不值得再把未来几个月都押在 CPU 微优化上**

### 为什么我这样判断

#### 1. 这个项目的 CPU 部分还没有真正讲圆

如果当前结果是 decode 0.95、prefill 0.85，那我会建议你停止。

但现在不是。

你现在的状态是：

- decode 已经能讲出很多东西：SIMD、量化、KV cache、scratch buffer、线程绑核、benchmark 方法论
- prefill 还明显落后：只有 0.116 左右

这说明你的项目已经有了**很强的工程基础**，但还缺一个“真正把体系讲完整”的收口：

> 你必须能回答：为什么 decode 已经接近了，但 prefill 还差这么远？

这个问题本身就很有研究价值，也很有面试价值。

#### 2. 现在继续做 CPU，学到的不是“再抠 5% AVX2”，而是“定位架构级差距”

现在最有价值的不是继续在 decode 上打补丁，也不是再加一堆实验开关。

真正有价值的是：

- 找出 prefill 为什么离 llama.cpp CPU 还有近 9 倍差距
- 证明这个差距到底来自哪里
  - 量化主路径没有完全对齐？
  - prefill 仍然回落到 dense/GEMM 路径太多？
  - 权重布局复用不够？
  - activation panel / packed layout 的收益没有真正吃满？

这比“继续抠一个 decode 内核”更能体现你的系统能力。

#### 3. 但 CPU 深挖必须设止损线

如果你不设止损线，CPU 这条线很容易变成：

- 不断试各种小改动
- benchmark 反复波动
- 文档越来越多，核心产出越来越少

这对复试和求职都不划算。

所以我建议：

> **再给 CPU 2~3 周，只做“prefill 架构差距定位 + 1 个有效优化 + 1 个质量 bug 修复”。**

如果 2~3 周之后：

- `prefill_ratio` 还是从 `0.11` 提不上去
- 或者你仍然说不清主要差距到底在哪里

那就应该停下来，转向 GPU 推理和推理系统工程。

---

## 四、CPU 部分接下来该怎么做，才是值得的

### 值得继续的方向

#### A. 先把 benchmark 口径彻底收干净

这一步不是形式主义，而是必要前提。

你现在至少要保证：

1. 项目内 compare benchmark 默认是 **CPU-only**
2. 文档里明确区分：
	- 本地 benchmark
	- CPU-only 的 llama.cpp compare
	- GPU offload 的 llama.cpp compare（如果你想保留）
3. 之后所有结论都不再混用历史数字

这一步已经做了一半，但还没完全做完。你还需要把这个口径写回 benchmark 说明和项目文档。

#### B. 只盯 prefill，不要再把时间花在 decode 小优化上

你的 decode 现在已经在 0.64~0.66。

对复试和求职来说，这已经足够支撑你讲：

- 为什么 quantized dot product 要做 SIMD
- 为什么 hot matrix cache 会从正收益变成负收益
- 为什么 CPU affinity / thread pinning 会影响稳定性

但 prefill 0.11 这个缺口太大，不补的话，项目叙事是不完整的。

#### C. 只做“架构级 prefill 差距”，不要再做外围调度花活

根据你现有的 todo 和历史记录，已经证伪或边际收益极低的方向，不该再碰：

- 外围调度层并行化花活
- 反复改 batch 调度
- 再堆更多实验开关
- 继续把时间花在 decode 路径的小收益点上

真正该做的是下面这三件事。

### 未来 2~3 周的 CPU 收尾计划

#### 第 1 周：把 prefill 主瓶颈定位清楚

目标不是“先写代码”，而是拿证据。

要产出的东西：

1. 一份当前 prefill flamegraph / perf 结果
2. 一份 llama.cpp CPU 路径的对照说明
3. 一张表：
	- 哪些算子在你这里走 dense/GEMM
	- 哪些算子在 llama.cpp 里走 quantized mul_mat / vec_dot
	- 差距最大的位置是什么

这一步重点盯：

- `src/core/operators/quant/generic.rs`
- `src/model/llama.rs`
- `src/core/operators/operator.rs`

#### 第 2 周：只做 1 个真正的 prefill 主路径改动

推荐优先级：

1. **继续对齐 `Q4_K/Q6_K` 的 prefill 量化主路径**
2. **减少 prefill 对 dense fallback / 通用 GEMM 的依赖**
3. **让 packed layout / activation panel 的收益真正长期复用起来**

这周的要求不是“做很多”，而是只做 1 个改动，然后跑：

- `bench_prefill_decode_metrics`
- `bench_compare_with_llamacpp_metrics_continuous`
- `bench_compare_with_llamacpp_metrics`

如果没有稳定收益，就回退。

#### 第 3 周：补 1 个正确性问题，结束 CPU 线

优先修：

- **GGUF 首轮首答错误 / 首轮 prefill-decode 路径问题**

原因很简单：

- 性能可以是“进度中”
- 但正确性 bug 会直接削弱项目说服力

如果你能在 CPU 收尾阶段同时交付：

1. benchmark 口径修正
2. 1 个 prefill 主路径优化
3. 1 个 GGUF 正确性修复

那么这个项目在复试和求职里就已经非常能打了。

### CPU 线的停止条件

满足任意一个就停止：

1. `prefill_ratio` 做到 `>= 0.20`，并且你能清楚解释剩余差距
2. 连续 2 周只有小波动，没有结构性改进
3. 你已经能把“为什么 decode 高、prefill 低”讲清楚

一旦满足，就不要继续沉迷 CPU 内核细节了。

---

## 五、在 CPU 收尾之后，下一条主线该是什么

### 我的建议顺序

不是“CUDA、LoRA、系统都学一点”，而是：

1. **GPU 推理工程**
2. **推理服务与调度系统**
3. **LoRA / QLoRA（作为补课，不是主线）**

也就是说：

> 你的下一条主线不该是“训练”，而该是“把你现在这套 CPU 推理思维迁移到 GPU 和服务系统上”。

---

## 六、为什么 GPU 推理应该排第一，而不是 LoRA

### 1. GPU 推理和你现在的项目是直接连续的

你现在已经自己实现过：

- quantized matmul
- online softmax
- KV cache
- prefill / decode 分离
- benchmark 与 profiling

这些东西天然就能接到 GPU 推理上：

- `fused_decode_attn_online` -> FlashAttention 的数学基础
- `kvcache.rs` -> PagedAttention / block table 的设计入口
- `quant/generic.rs` -> GPU 量化 kernel / W4A16 / W8A8 的理解基础
- `server/http.rs` -> 推理服务、调度和流式输出的工程入口

所以 GPU 推理不是“另起炉灶”，而是**你当前项目的自然下一步**。

### 2. LoRA 对面试重要，但对这个项目不形成主线闭环

LoRA 当然值得学，但它和你这个仓库的直接关系不强。

它更像：

- 你为了补齐面试知识面必须学
- 但不是你现在最该投入最多时间的方向

如果你现在先花很多时间在 LoRA 上，会出现一个问题：

- 你当前项目最强的地方是**推理系统和性能工程**
- LoRA 会把你的注意力拉向另一个赛道
- 结果就是两个方向都碰了，但没有一个方向足够深

所以，LoRA 应该是**第二梯队的补课项**，不是主线。

---

## 七、真正适合你的后续学习路径

### 主线 A：GPU 推理工程（优先级最高）

这条线不要泛泛地“学 CUDA”，而要直接围绕你现在这份仓库做迁移。

#### 阶段 A1：把 CPU 项目里的核心算子迁移到 GPU 思维

先做三件小事：

1. 用 Triton 或 CUDA 重写一个你最熟悉的算子
	- 推荐先选 RMSNorm 或 matmul
2. 把 `fused_decode_attn_online` 对应到 FlashAttention 的 online softmax 思路
3. 把 `kvcache.rs` 的连续缓存管理，改写成“分页 KV cache”的设计草图

产出不是“看懂论文”，而是：

- 你能把 CPU 版和 GPU 版的设计差异讲出来
- 你能说清楚为什么 GPU 上必须做 tile、fusion 和 page 管理

#### 阶段 A2：围绕 vLLM / TensorRT-LLM 做一次有针对性的源码阅读

只盯三件事：

1. PagedAttention
2. continuous batching / request scheduler
3. quantized linear + fused attention kernel

不要一上来泛读整库。

你的目标是回答：

> 如果把你现在这个 Rust CPU 项目继续做下去，哪些部分会在 GPU 服务系统里被彻底重写，哪些部分会保留？

#### 阶段 A3：做一个最小 GPU 推理实验

不是要求你立刻写完整 GPU LLM。

你只需要做下面二选一：

1. 一个最小 Triton FlashAttention demo
2. 一个最小 GPU quantized GEMM demo

有了这一步，你的学习路径就从“会讲概念”变成“我做过 CPU，也做过 GPU 的最小实现”。

### 主线 B：推理服务与调度系统（第二优先级）

这是最适合和你当前仓库衔接的第二条线。

你已经有：

- `server/http.rs`
- SSE 流式输出
- chat 入口

接下来最值得补的是：

1. 请求排队与批处理
2. prefix cache
3. 简单 metrics（吞吐、TTFT、并发数）
4. admission control

这条线的价值非常高，因为它能让你的项目从“一个能跑的推理引擎”变成“一个更接近真实服务的系统”。

这对找工作尤其重要。

### 主线 C：LoRA / QLoRA（第三优先级）

LoRA 不是不重要，而是：

- **对复试重要**：面试老师很可能会问
- **对算法岗也重要**：至少要会讲原理
- **但对这份仓库的直接增益最小**

所以它最适合的做法不是“深挖 1 个月”，而是“高质量补课 1~2 周”。

你至少应该补到能稳定回答：

1. LoRA 为什么有效
2. QLoRA 为什么能省显存
3. LoRA 和全参数微调在工程上怎么权衡
4. 推理时 adapter 是怎么加载或 merge 的

建议只做一个最小实践：

- 用 1B 级模型做一次 QLoRA 小实验
- 能跑通训练、保存 adapter、推理加载 adapter 即可

不要把 LoRA 做成你的主战场。

---

## 八、如果目标是“考研复试”，你该怎么排

如果你更偏向复试，优先级应该是：

1. **把当前项目讲圆**
2. **补 GPU 推理基础概念**
3. **补 LoRA/QLoRA 原理**

复试最怕的不是“你不会所有热点”，而是：

- 你项目很多
- 但一问关键 trade-off 就讲不清楚

所以你要重点准备的是下面这些问题：

1. 为什么你的 decode 能做到 0.64，但 prefill 只有 0.11？
2. 为什么 benchmark 必须区分 CPU-only 和 GPU offload？
3. 为什么 hot matrix cache 在某个阶段有用，后面又变成负收益？
4. 为什么量化激活路线会影响 prefill 架构？
5. LoRA 为什么不需要更新全量参数？
6. FlashAttention 为什么能减少显存访问？

只要这 6 个问题你答得扎实，复试竞争力已经很强了。

---

## 九、如果目标是“找推理/系统方向工作”，你该怎么排

如果你更偏向找工作，优先级应该是：

1. **CPU 项目收尾**
2. **GPU 推理**
3. **服务系统化**
4. **LoRA 补课**

原因很直接：

- 企业更看重你能不能把系统做出来、测明白、讲清楚
- 而不是你是否“听说过很多方向”

对求职最有帮助的组合不是：

- CPU + LoRA + RLHF + DPO 全都沾一点

而是：

- 一个讲得很硬的 CPU 推理项目
- 一个哪怕很小但真实的 GPU 推理实验
- 一个更像服务系统的 demo

这三样组合起来，比“热点名词大全”强得多。

---

## 十、最终建议

### 我对你这条线的最终判断是：

#### 1. CPU 推理部分还值得继续，但只值得再做一轮“收尾冲刺”

不是为了死磕 AVX2，而是为了把这个项目从“做过很多优化”变成“我能解释当前真实差距，并完成一次架构级修正”。

#### 2. 收尾之后，主线必须切到 GPU 推理和服务系统

这是你当前项目最自然、也最有价值的下一步。

#### 3. LoRA 要学，但不要抢主线位置

它应该是补齐面试知识面的配套项，而不是你接下来投入最多时间的方向。

---

## 十一、接下来 6 周的具体排法

### 第 1~2 周

- 修正文档中的 benchmark 口径
- 完成 prefill 主瓶颈定位
- 做 1 个 prefill 主路径优化

### 第 3 周

- 修 GGUF 首轮首答 bug
- 整理一版性能报告和项目讲稿

### 第 4~5 周

- 做 1 个 GPU 最小实验
  - Triton FlashAttention demo，或
  - GPU quantized GEMM demo

### 第 6 周

- 补 LoRA/QLoRA
- 只做到“能讲原理 + 跑通 1 个小实验”即可

如果第 1~2 周 CPU 线进展非常差，就提前切到第 4~5 周的 GPU 线，不要拖。

---

## 十二、这份项目现在最适合你的定位

这不是一个“已经把 CPU 做到极致”的项目。

它现在更适合被定位为：

> **一个已经具备很强工程深度、但仍保留一个关键性能缺口（prefill）的 LLM 推理系统项目。**

这个定位其实是好的。

因为它意味着你既可以讲：

- 已经做成了什么
- 为什么某些优化有效
- 为什么另一些优化失败
- 现在最真实的瓶颈是什么
- 下一步为什么该转向 GPU 和服务系统

这比把项目包装成“我已经全面超过 llama.cpp”要真实得多，也强得多。
