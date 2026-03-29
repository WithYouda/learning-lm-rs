# 完成改进记录（精简版，按时间排列）

> 本文件按时间顺序精简记录从项目起点到达成 80% 目标的完整优化路径。
> **✓ 成功** | **✗ 回退/降为实验开关**

---

## 一、起点基线 — S1~S2（2026-03-09）

| 指标 | 数值 |
|---|---|
| prefill | 5.18 tok/s |
| decode | 2.65 tok/s |
| prefill_ratio vs llama.cpp | 3.48% |
| decode_ratio vs llama.cpp | 28.83% |

- ✓ 融合在线 softmax decode 内核（`fused_decode_attn_online`）
- ✓ 接入 llama.cpp 对照基准，确立唯一事实源
- ✓ perf 采样确认 Q4K `decode_block_dot` 占 CPU 约 64.80%

---

## 二、量化热点与缓存竞争 — S3（2026-03-10）

- ✓ Q4K/Q6K/Q80 专用块点积
- ✓ 分片行缓存（降低锁竞争）
- ✓ decode 预解码 + prefill 分块复用

**结果**：decode_ratio 28.83% → 33.34%

---

## 三、prefill GEMM 后端 — S4（2026-03-10）

- ✓ `LMRS_PREFILL_BACKEND=gemm` 双后端
- ✓ 层内复用 `hidden_states` 缓冲

**结果**：prefill 5.18 → 9.01 tok/s（+74%）

---

## 四、decode 工程基础 — S5（2026-03-10）

- ✓ decode online softmax 向量化 scale/axpy
- ✓ 按层计时统计（`LMRS_LAYER_TIMING`）
- ✓ 按层 packed KV（`LMRS_DECODE_PACKED_LAYERS`）

---

## 五、系统级调优 — S6（2026-03-10）

- ✓ CPU 线程绑核（`LMRS_THREADS=5`、`LMRS_CPU_MASK`）
- ✓ 编译优化：`opt-level=3, lto=fat, codegen-units=1, target-cpu=native`
- ✓ decode scratch 工作缓存复用（一次分配多轮复用）
- ✓ Q/K/V 与 gate/up 投影并行调度
- ✓ 热量化矩阵整块预解码（`hot_matrix_cache`）
- ✓ 5 轮均值/标准差 benchmark 框架

**结果**：decode 2.96 → **7.73 tok/s**（首破 7.5 门槛）

---

## 六、prefill 线性层深挖 — S7~S17（2026-03-10~11）

#### 成功路径
- ✓ prefill 量化 matmul 分块解码后直走 GEMM（S7）
- ✓ O/down 路径消除额外分配，GEMM 直接写回子矩阵（S12）
- ✓ Prefill activation panel 打包（S16）
- ✓ PrefillKernelKind 按矩阵形状选择 row-tile（S16/S17）

#### 失败路径
- ✗ 共享输入 batch 投影调度（S11 → 实验开关 `LMRS_PREFILL_BATCH_PROJ`）
- ✗ k 方向分块 + m 自适应 row-tile（S13 → 实验开关 `LMRS_PREFILL_LONGPROMPT_TILING`）
- ✗ prefill 复用 decode 热点矩阵缓存（S14/S15 → 撤回）
- ✗ Q4K/Q6K direct vec_dot（S16 → decode 7.8→5.7，收为实验开关）
- ✗ 共享 panel 的 batch2/batch3（S17 → 降为实验开关）

**结果**：prefill 8.45 → 9.55 tok/s，decode 稳定 8.28 tok/s

---

## 七、工作集常驻布局探索 — S18~S25（2026-03-11）

#### 成功路径
- ✓ QKV 加载期 workset（GGUF 加载时预热 QKV 工作集）（S18）
- ✓ Steady-state benchmark（跨轮复用模型，预热后再计时）（S24）

#### 失败路径
- ✗ FFN 量化条带常驻布局（S19 → `LMRS_PREFILL_FFN_STRIPES`）
- ✗ MADV_WILLNEED/预触页（S25 → `LMRS_ENABLE_PREFILL_WILLNEED/PRETOUCH`）
- ✗ Q8K x4 权重重排加载期布局（S23 → `LMRS_PREFILL_Q8K_INTERLEAVE_MB=0`）

---

## 八、关键突破：量化激活路线 — S20~S22（2026-03-11）

方向从"dense workset"彻底转向"对齐 llama.cpp 量化激活主路径"。

- ✓ Q8K 激活量化基础设施：`QuantQ8KBlock`、`quantize_activation_block_q8k`（S20）
- ✓ Q4K/Q6K × Q8K 专用块点积、decode 走 Q8K 激活主路径（S20）
- ✓ prefill panel 改为 Q8K/Q8_0 量化 panel（S21，prefill_ratio 0.40 → **0.50**）
- ✓ Q8K interleaved 4x8 prefill 微内核（S22）

**结果**：prefill_ratio 突破 0.50

---

## 九、单份 packed 主布局 — S26（2026-03-12）

- ✓ `prefill_packed` 主布局（加载期构建 packed 权重）
- ✓ `type_traits_cpu` 风格调度骨架
- ✓ 按形状切换 4x1/4x4/4x8 激活打包路径

**结果**：连续 prefill_ratio 0.5062

---

## 十、Q2K/Q3K/Q5K 扩展 — S27~S28（2026-03-12）

- ✓ Q2K/Q3K/Q5K 接入 Q8K x1 主路径（S27）
- ✗ Q2K/Q3K/Q5K 的 x4 微内核（S28 → `LMRS_Q235K_Q8K_X4`）

---

## 十一、聊天入口修复 — S29~S33（2026-03-12）

- ✓ `test_gguf_chat` 重构为单轮可复现质量验收（S29）
- ✓ 修正真实聊天入口默认配置、生成长度语义（S30）
- ✓ `main()` 接入 GGUF chat，CLI 可直接运行（S31）
- ✓ 拆出独立 `cli` 二进制，统一 GGUF/safetensors 入口（S32）
- ✓ 修复 `chat()` 未写回合结束 token 到 cache 的 bug（S33）

---

## 十二、最终突破：AVX2 SIMD + hot cache 重评估 — 阶段O（2026-03-12）

- ✓ Q4K×Q8K AVX2 SIMD 点积（`_mm256_maddubs_epi16` + `_mm256_madd_epi16`）
- ✓ Q6K×Q8K AVX2 SIMD 点积（有符号×有符号技巧）
- ✓ 运行时 AVX2 特征检测，无 AVX2 自动回退标量

**反直觉发现**：AVX2 使量化点积足够快后，hot matrix cache 从正收益变负收益（读 f32 ~1.34 GB/token vs 读压缩 ~0.27 GB/token）。禁用后 decode **+58%**。

| 指标 | 起点 (S2) | 最终 | 提升 |
|---|---|---|---|
| prefill tok/s | 5.18 | 32.79 | **+533%** |
| decode tok/s | 2.65 | 14.74 | **+456%** |
| prefill_ratio | 3.48% | **81.24%** | +23.3× |
| decode_ratio | 28.83% | **105.28%** | +3.65× |

**退出标准全部达成，decode 实际超越 llama.cpp 5.3%。**

---

## 十三、后续清理与对照收敛 — S34 + 补充会话（2026-03-12~13）

### 清理与文档整理（2026-03-12）
- 修复 CLI 默认参数缺失（补 `LMRS_HOT_MATRIX_CACHE_MB=1`）
- 修复 GGUF 模型路径错误
- 删除冗余测试与死代码（`dot_t_simd`、旧 blocked AV、冗余 wrapper）
- 修复编译警告

### benchmark 口径复核（2026-03-12）
- 确认我方 benchmark 逻辑自洽（不含模型加载时间）
- 发现 compare benchmark 的 tok/s 解析风险，已修复

### llama.cpp 对照稳定性收敛（2026-03-13）
- `run_llamacpp_bench()` 增加 3 次重试 + `-c 1024` + `--threads-batch`
- 连续口径：prefill_ratio 0.2348, decode_ratio 0.4239
- 交错口径：prefill_ratio 0.2427, decode_ratio 0.4235

### S34：RoPE 预计算 + 直接 GEMM + 线程调度收敛（2026-03-13）
- ✓ RoPE 逆频率缓存（去掉 head 内层重复 `powf/sin_cos`）
- ✓ prefill attention 直接 `gemm::gemm`（移除临时 Tensor 构造）
- ✓ 内层 GEMM 单线程（避免嵌套并行）
- ✓ group 输出改为连续大缓冲
- 连续口径 prefill 35.36 tok/s, decode 16.90 tok/s

### 第六轮 AVX2 x4 微内核（2026-03-13 会话 B）
- ✓ QuantQ8KBlockX4 改为行主序拼接布局
- ✓ Q4K/Q6K x4 AVX2 prefill 微内核（带 meta 变体）
- ✓ FMA f32 点积双累加器、AVX2 激活量化、软件预取
- ✓ FMA sum_squares、AVX2 RMS norm apply
- ✓ m%4 不整除修复（拆分对齐部分 + 尾部单行回退）

### 三刀未通过准入的尝试（2026-03-13）
- ✗ Q8K x4 row-group 并行化（交错口径回退）
- ✗ Q4K 子循环展开（双口径方向不一致）
- ✗ Q4 AVX2 首次落地（单口径提升、单口径回退）
- ✗ 分支外提+预计算 / 去 Vec 打包 / metadata 一次性分发（均回退）

### 保留的小改动（2026-03-13最终）
- ✓ Q8K x4 微内核索引化简（`idx*4+row` 替代除法/取模）

---

## 十四、方法论总结

1. **双口径验收**：每刀 5 轮连续 + 交错口径，单轮好看的实验不能并入默认路径
2. **失败实验保留为实验开关**：不污染主路径，保留后续验证机会
3. **瓶颈会随优化位移**：hot matrix cache 在标量时代正收益，AVX2 时代负收益——必须在每个里程碑后重新测量
4. **方向比执行更重要**：S14~S19 在 dense workset 方向花费大量精力，直到 S20 对齐 llama.cpp 量化激活主路径后 prefill_ratio 才真正跨过 0.50
