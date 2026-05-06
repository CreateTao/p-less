# P-Less Sampling 测试与验证计划

## 安装部署

```bash
# 克隆仓库
git clone <repo-url> && cd p-less

# 按环境选择安装方式：
# CPU 服务器（仅算法测试 + 小模型验证）
pip install -e ".[cpu]"

# GPU 服务器（全量评测 + TOST 等价检验）
pip install -e ".[gpu]"

# 开发环境（lint + jupyter）
pip install -e ".[dev]"
```

> **说明**：
> - `-e` 表示可编辑模式，修改代码后无需重新安装
> - `pip install -e .` 仅安装基础依赖（torch/numpy/scipy/pyyaml/tqdm），不含模型加载和测试
> - `verification/requirements/` 下的分层文件仍可使用：`pip install -r verification/requirements/cpu.txt`

## 问题一：CPU 上能否验证论文结果？

### 结论：**可以，但有规模限制**

p-less 的采样逻辑极其轻量（仅需 `Σprobs²` + mask + multinomial），瓶颈完全在**模型推理**而非采样。CPU 验证的关键是选择合适的模型规模：

| 模型 | 参数量 | CPU 内存估算 | CPU 生成速度 | 可行性 |
|---|---|---|---|---|
| Qwen3-0.6B | 0.6B | ~1.2 GB | ~15-20 tok/s | 完全可行 |
| Qwen3-1.7B | 1.7B | ~3.4 GB | ~5-8 tok/s | 可行（8GB+ RAM） |
| Qwen3-4B | 4B | ~8 GB | ~2-4 tok/s | 可行（16GB+ RAM，较慢） |
| Qwen3-8B+ | 8B+ | >16 GB | <1 tok/s | 不建议（极慢） |

### CPU 验证策略

**Phase 1 — 算法正确性验证（纯数学，无需模型）**
- 构造人工概率分布，验证 p-less 阈值计算、候选集非空保证、重归一化逻辑
- 构造极端分布（单峰、均匀、退化）验证边界行为
- 验证温度兼容性：在不同温度下计算阈值变化曲线

**Phase 2 — 小模型快速验证（Qwen3-0.6B / 1.7B，CPU）**
- GSM8K 50题子集 + CSQA 50题子集
- 对比 greedy / top-p / p-less / p-less-norm
- 约 2-4 小时可完成一轮完整对比

**Phase 3 — 中模型验证（Qwen3-4B，CPU 或单卡 GPU）**
- 全量 GSM8K（1319题）
- 验证 p-less 与原论文趋势是否一致（p-less > top-p > greedy 在精确性任务上）

### CPU 运行示例

```python
# CPU 模式加载（无需 GPU）
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    device_map="cpu",        # 强制 CPU
    torch_dtype=torch.float32  # CPU 用 float32
)
```

---

## 问题二：扩展测试与验证计划

### 总体目标

1. **复现原论文核心结论**：p-less 在精确性任务上优于 top-p/top-k，p-less-norm 在多样性任务上优于 p-less
2. **验证在新模型上的泛化性**：Qwen3 系列（0.6B → 235B MoE）
3. **验证在新任务域上的泛化性**：Function Call、代码生成、多语言推理
4. **验证极端场景鲁棒性**：长上下文、高温/低温、分布退化

---

### 一、模型矩阵

| 类别 | 模型 | 规模 | 测试环境 | 优先级 |
|---|---|---|---|---|
| 小模型（CPU 可行） | Qwen3-0.6B | 0.6B | CPU | P0 |
| 小模型（CPU 可行） | Qwen3-1.7B | 1.7B | CPU / 单卡 | P0 |
| 中模型 | Qwen3-4B | 4B | 单卡 GPU | P0 |
| 中模型 | Qwen3-8B | 8B | 单卡 GPU | P1 |
| 大模型 | Qwen3-14B | 14B | 多卡 GPU | P1 |
| 大模型 | Qwen3-32B | 32B | 多卡 GPU | P2 |
| MoE 模型 | Qwen3-235B-A22B | 235B(激活22B) | 多卡 GPU | P2 |
| 对照模型（原论文） | Llama-3.1-8B | 8B | 单卡 GPU | P1 |
| 对照模型（原论文） | Mistral-7B-v0.3 | 7B | 单卡 GPU | P1 |

---

### 二、数据集矩阵

#### A. 原论文数据集（复现对照）

| 数据集 | 任务类型 | 评测指标 | 样本量 | 优先级 |
|---|---|---|---|---|
| GSM8K | 数学推理 | Exact Match | 1319 题 | P0 |
| GPQA (Diamond) | 专家级推理 | Accuracy | 198 题 | P0 |
| QASC | 科学推理 | Accuracy | ~800 题 | P0 |
| CSQA | 常识推理 | Accuracy | ~1200 题 | P0 |
| Writing Prompts | 创意写作 | MAUVE / Distinct | ~400 题 | P1 |

#### B. 新增 — Function Call 数据集

| 数据集 | 任务类型 | 评测指标 | 说明 | 优先级 |
|---|---|---|---|---|
| BFCL v3 (Berkeley) | 单轮函数调用 | Accuracy | 结构化 JSON 输出，验证 p-less 对格式约束的兼容性 | P0 |
| BFCL v3 Multi-turn | 多轮函数调用 | Accuracy | 长上下文 + 状态追踪 | P1 |
| ToolBench | 工具使用链 | Pass Rate | 多步工具调用组合 | P2 |
| API-Bank | API 调用 | Accuracy | 层级化 API 场景 (L1-L3) | P2 |

> **核心假设验证**：Function Call 需要**精确的结构化输出**（JSON 格式），p-less 的保守截断是否在此场景下优于 top-p/top-k？这是论文未覆盖但极具实用价值的测试点。

#### C. 新增 — 代码编程数据集

| 数据集 | 任务类型 | 评测指标 | 说明 | 优先级 |
|---|---|---|---|---|
| HumanEval | Python 函数生成 | Pass@1 | 164 题，代码精确性测试 | P0 |
| HumanEval+ | Python 函数生成（增强测试） | Pass@1 | 164 题 + 809 个额外测试用例 | P1 |
| MBPP | Python 函数生成 | Pass@1 | ~500 题，基础编程 | P0 |
| LiveCodeBench | 多语言竞赛编程 | Pass@1 / Pass@5 | 持续更新的代码评测 | P1 |
| CRUXEval | 代码推理 | Accuracy | 代码输入/输出预测 | P2 |
| CodeContests | 竞赛编程 | Pass@k | 高难度算法题 | P2 |

> **核心假设验证**：代码生成需要**语法正确 + 逻辑精确**，p-less 是否能减少语法错误率（相比 top-p/top-k 的过度随机性）？

#### D. 新增 — 多语言与长上下文

| 数据集 | 任务类型 | 评测指标 | 说明 | 优先级 |
|---|---|---|---|---|
| MMLU | 多领域知识 | Accuracy | 57 子领域，覆盖广 | P0 |
| MMLU-Pro | 高难度知识 | Accuracy | 更难的 MMLU 版本 | P1 |
| LongBench | 长上下文理解 | 多指标 | 6 子任务，测试长文本下的采样稳定性 | P2 |
| MGSM | 多语言数学 | Exact Match | 10 种语言的 GSM8K | P2 |

---

### 三、采样方法对照矩阵

| 方法 | 超参数 | 说明 | 是否需调参 |
|---|---|---|---|
| Greedy | — | 确定性基线 | 否 |
| Top-p (nucleus) | p=0.9, 0.95 | 原论文对照 | 是（需选 p） |
| Top-k | k=10, 40, 100 | 原论文对照 | 是（需选 k） |
| Min-p | min_p=0.05, 0.1 | 相对阈值方法 | 是 |
| Epsilon-sampling | ε=0.001, 0.01 | 绝对阈值方法 | 是 |
| Eta-sampling | η=0.3, 0.5 | 熵自适应方法 | 是 |
| Mirostat v2 | target_entropy=3.0 | 熵追踪方法 | 是 |
| **p-less** | — | **零超参数（本方法）** | **否** |
| **p-less-norm** | — | **零超参数（本方法放松版）** | **否** |

---

### 四、温度兼容性测试

验证 p-less 论文核心主张：阈值随温度自适应变化。

| 温度 | 测试内容 | 预期 |
|---|---|---|
| T=0.01 | 近 greedy 行为 | p-less 阈值→max(probs)，趋近 greedy |
| T=0.3 | 低随机性 | p-less 保持精确性 |
| T=0.7 | 常用温度 | 标准对比点 |
| T=1.0 | 默认温度 | 标准对比点 |
| T=1.5 | 高随机性 | p-less 阈值下降，允许更多 token |
| T=3.0 | 极高温度 | p-less 仍有有效截断，top-p/top-k 失控 |

> 对照：top-p(0.9) 在 T→0 时所有 token 都满足 p≥0.9×max，等于无截断；T→∞ 时几乎无 token 满足，等于 greedy。p-less 无此退化问题。

---

### 五、详细测试步骤

#### Step 1：算法正确性单元测试（1-2 小时，纯代码）

```python
# 测试项清单
1. p_less_decode: 均匀分布 → 验证阈值 = 1/V，所有 token 保留
2. p_less_decode: 单峰分布 → 验证仅保留高概率 token
3. p_less_decode: 退化分布（一个 token p=1）→ 验证阈值为 1，返回该 token
4. p_less_norm_decode: 均匀分布 → 验证 p_norm = 0，保留所有 token
5. p_less_norm_decode: 单峰分布 → 验证 p_norm < p_less，保留更多 token
6. 温度兼容性: 对 logits/T 做 softmax，验证阈值随 T 平滑变化
7. 批量测试: batch_size > 1 时各 batch 独立正确
8. 数值稳定性: 极大词表 (V=128K) 下的精度
```

#### Step 2：CPU 小模型验证（Qwen3-0.6B，4-6 小时）

```bash
# GSM8K 50题快速验证
python verification/scripts/run_cpu_benchmark.py \
  --config verification/configs/experiments/cpu_step2_gsm8k.yaml

# 如需 CSQA 验证，需创建单独的 YAML 配置文件
# （CPU benchmark 每次运行仅支持一个数据集）
```

> **说明**：CPU benchmark 采用 YAML 配置驱动，所有参数（模型、数据集、方法、样本数）均在配置文件中定义。
> 如需对其他数据集（如 CSQA）运行，需参照 `cpu_step2_gsm8k.yaml` 创建新配置。

验证目标：
- p-less 在 GSM8K 上 Accuracy ≥ greedy（与原论文趋势一致）
- p-less-norm 在 CSQA 上 Diversity ≥ p-less（与原论文趋势一致）
- 无空候选集崩溃（100% 样本正常生成）

#### Step 3：CPU 中模型验证（Qwen3-1.7B，8-12 小时）

```bash
python verification/scripts/run_cpu_benchmark.py \
  --config verification/configs/experiments/cpu_step3_full.yaml
```

新增验证：
- HumanEval Pass@1 对比（代码精确性）
- 引入 min-p 对照（同为相对阈值方法，但需调参）

#### Step 4：GPU 超参搜索（20%子集，找 baseline 最优参数）

先在 20% 数据子集上对每种 baseline 方法（top-p、top-k、min-p、epsilon、eta）进行网格搜索，
找到各自的最优超参数，供 Step 5 全量评测使用。

```bash
python verification/scripts/run_gpu_search.py \
  --model Qwen/Qwen3-8B \
  --datasets gsm8k humaneval bfcl gpqa \
  --methods top_p top_k min_p epsilon eta \
  --subset-fraction 0.2 \
  --results-dir verification/outputs/results/gpu_search
```

> **说明**：搜索结果保存为 `grid_search_*.json`，后续 Step 5 会自动加载最优参数。
> 可选参数：`--methods`（默认搜索 top_p/top_k/min_p/epsilon/eta）、`--subset-fraction`（默认 0.2）。

#### Step 5：GPU 全量评测 + TOST 等价检验（核心实验）

使用 Step 4 搜索到的最优 baseline 参数，在全量数据集上运行评测，
并对 p-less vs 各 baseline 做 TOST 等价检验（δ=2%）。

```bash
# P0 数据集
python verification/scripts/run_gpu_evaluation.py \
  --model Qwen/Qwen3-8B \
  --datasets gsm8k gpqa humaneval mbpp bfcl csqa \
  --methods top_p top_k min_p p_less p_less_norm \
  --temperatures 0.3 0.7 1.0 \
  --search-dir verification/outputs/results/gpu_search \
  --results-dir verification/outputs/results/gpu_eval

# 如需对其他模型运行
python verification/scripts/run_gpu_evaluation.py \
  --model meta-llama/Llama-3.1-8B \
  --datasets gsm8k gpqa humaneval mbpp bfcl csqa
```

> **说明**：
> - `--search-dir` 指向 Step 4 的搜索结果，脚本会自动加载各 baseline 的最优参数
> - `--temperatures` 同时覆盖了温度兼容性测试（见下方说明）
> - p-less 在所有指定温度下评测；baseline 仅在其最优温度下评测

**温度兼容性测试**（原 Step 5 专项实验，现已合并）：

```bash
# 扩展温度范围进行温度兼容性验证
python verification/scripts/run_gpu_evaluation.py \
  --model Qwen/Qwen3-8B \
  --datasets gsm8k \
  --methods top_p top_k min_p p_less p_less_norm \
  --temperatures 0.01 0.1 0.3 0.7 1.0 1.5 2.0 3.0 \
  --search-dir verification/outputs/results/gpu_search \
  --results-dir verification/outputs/results/gpu_temperature
```

输出：
- 各方法 Accuracy-vs-Temperature 数据（通过 `run_analysis.py` 生成曲线图）
- 各方法 候选集平均大小-vs-Temperature 数据
- p-less 阈值分布统计（均值、方差、min、max）

#### Step 6：统计分析与论文级报告

```bash
python verification/scripts/run_analysis.py \
  --results-dir verification/outputs/results/gpu_eval \
  --search-dir verification/outputs/results/gpu_search \
  --output-dir verification/outputs/report
```

分析内容：
1. Accuracy 对比表（Group A: 框架默认配置，Group B: 推荐配置）
2. Bootstrap 95% CI：各方法 Accuracy 置信区间
3. 配对 t 检验 / Wilcoxon 检验：p-less vs 各 baseline
4. Win/Tie/Lose 计数表：p-less 在多少配置下 Win
5. TOST 等价检验汇总：p-less 与 baseline 是否等价（δ=2%）
6. 温度鲁棒性曲线图

---

### 六、预期结论与风险分析

#### 预期正向结论

| 场景 | 预期 | 论文依据 |
|---|---|---|
| 数学推理 (GSM8K/GPQA) | p-less ≥ top-p/top-k | 原论文已验证 |
| Function Call (BFCL) | p-less ≥ top-p/top-k | **新假设**：精确结构化输出受益于保守截断 |
| 代码生成 (HumanEval) | p-less ≥ top-p/top-k | **新假设**：语法精确性受益于保守截断 |
| 创意写作 | p-less-norm ≥ p-less | 原论文已验证 |
| 高温场景 | p-less 稳定，top-p/top-k 退化 | 原论文理论推导 |

#### 潜在风险与反向结论

| 风险场景 | 可能结果 | 应对措施 |
|---|---|---|
| 代码多样性不足 | p-less Pass@1 高但 Pass@10 低 | p-less-norm 可作为中间方案 |
| Function Call 格式过于保守 | 某些 API 参数值被截断 | 分析阈值 vs 参数 token 分布 |
| MoE 模型分布异常 | 专家路由导致多峰分布 | p-less 保留多峰 token，可能反而更合适 |
| 小模型能力不足 | 所有方法 Accuracy 都很低 | 差异可能不显著，需增大模型 |

---

### 七、输出交付物

1. `results/` 目录：所有原始评测 JSON 结果
2. `figures/` 目录：
   - Accuracy 对比柱状图（每数据集 × 每模型）
   - Temperature-vs-Accuracy 曲线图
   - 候选集大小分布图
   - Win/Tie/Lose 热力图
3. `tables/` 目录：
   - 完整结果表格（Markdown + LaTeX）
   - 统计显著性标记表
4. `CODE_PLAN.md`：本文件（计划文档）
5. 补充论文/技术报告：扩展实验的完整描述

---

### 八、时间线估算

| 阶段 | 时间 | 环境 | 产出 |
|---|---|---|---|
| Step 1: 算法正确性 | 1-2 小时 | 纯代码 | 单元测试通过报告 |
| Step 2: CPU 小模型 | 4-6 小时 | CPU | GSM8K/CSQA 初步对比 |
| Step 3: CPU 中模型 | 8-12 小时 | CPU | HumanEval 初步结果 |
| Step 4: GPU 超参搜索 | 4-6 小时 | GPU | 各 baseline 最优参数 |
| Step 5: GPU 全量评测 + 温度 | 2-3 天 | GPU | 核心实验结果 + 温度曲线 |
| Step 6: 统计分析 | 4-6 小时 | 代码 | 论文级报告 + 图表 |
| **总计** | **~4-5 天** | | |

---

## 验证执行进度

> **更新时间：2026-05-06** — 每次执行验证后更新此表，便于中断后恢复。

### Step 1：算法正确性单元测试 ✅ 完成

- 状态：已完成并通过
- 执行时间：2026-05-05
- 结果：5个测试文件全部通过，float32精度容差已修正
- Commit：`55c1982 Fix Step 1 test tolerances for float32 precision and GBK encoding`

### Step 2：CPU 小模型验证（Qwen3-0.6B, GSM8K 50题）✅ 完成

| 方法 | Accuracy | 95% CI | Avg Candidate Set | 状态 |
|---|---|---|---|---|
| greedy | 0.20 | [0.10, 0.32] | 1.0 | ✅ |
| temperature_1.0 (vllm_default) | 0.22 | [0.12, 0.34] | 151936.0 | ✅ |
| top_k_50 (hf_default) | 0.22 | [0.12, 0.34] | 50.0 | ✅ |
| top_p_0.9 | 0.16 | [0.08, 0.28] | 2.3 | ✅ |
| p_less_t0.7 | **0.24** | [0.14, 0.36] | 1.1 | ✅ |
| p_less_t1.0 | **0.24** | [0.14, 0.36] | 1.1 | ✅ |

- 执行时间：2026-05-06
- 核心发现：p-less 在 GSM8K 上 Accuracy=0.24 ≥ greedy(0.20)，与论文趋势一致
- p-less 平均候选集大小仅 1.1，说明 0.6B 模型分布高度集中，截断非常保守
- top_p_0.9 在此小模型上表现最差(0.16)，可能因为截断过于激进
- 所有方法均达到 256 token 上限（max_tokens 限制），模型倾向于生成较长的推理链
- 修复：`run_cpu_benchmark.py` 断点续跑 bug（已有结果应加载而非跳过导致除零错误）

### Step 3：CPU 中模型验证（Qwen3-1.7B, GSM8K 100题）🔄 执行中（中途终止，待迁移 GPU 继续）

- 目标：GSM8K 100题子集（全量1319题需~370h，暂用100题验证趋势）
- 对比方法：greedy / vllm_default / hf_default / top_p(0.9, 0.95) / top_k(40) / min_p(0.05) / p_less / p_less_norm
- max_tokens: 512（相比 Step 2 的 256 翻倍，允许更完整推理链）
- 配置：`verification/configs/experiments/cpu_step3_full.yaml`
- 模型来源：ModelScope（自动下载）
- 预计时间：~28小时（CPU）
- GPU 可大幅加速：单卡 T4/A10 约 1-2 小时，A100 约 30-40 分钟

**执行进度（2026-05-06）：**

| 方法 | 完成题数 / 100 | 状态 |
|---|---|---|
| greedy | 16 / 100 | 🔄 中途中断 |
| vllm_default | 0 / 100 | ⏳ 未开始 |
| hf_default | 0 / 100 | ⏳ 未开始 |
| top_p_0.9 | 0 / 100 | ⏳ 未开始 |
| top_p_0.95 | 0 / 100 | ⏳ 未开始 |
| top_k_40 | 0 / 100 | ⏳ 未开始 |
| min_p_0.05 | 0 / 100 | ⏳ 未开始 |
| p_less_t0.7 | 0 / 100 | ⏳ 未开始 |
| p_less_t1.0 | 0 / 100 | ⏳ 未开始 |
| p_less_norm_t0.7 | 0 / 100 | ⏳ 未开始 |
| p_less_norm_t1.0 | 0 / 100 | ⏳ 未开始 |

**已有结果：**
- `verification/outputs/results/cpu/gsm8k/Qwen3-1.7B/greedy/t1.0/` — 16 个 JSON 文件（q_0000 ~ q_0015）
- 脚本支持断点续跑，已有结果会被加载跳过

**GPU 迁移注意事项：**
- 当前结果存储路径为 `verification/outputs/results/cpu/gsm8k/Qwen3-1.7B/`，GPU 上运行时需确认路径一致或做适配
- 模型在 CPU 上通过 ModelScope 下载，GPU 服务器需重新下载或从 HuggingFace 加载
- 建议在 GPU 服务器上直接运行同一命令，脚本的断点续跑功能会跳过已完成的 greedy 前 16 题
- 如需从 greedy 第 17 题继续，确保 `verification/outputs/results/cpu/gsm8k/Qwen3-1.7B/` 目录已完整迁移到 GPU 服务器

### Step 3.5：CPU MoE 模型验证（Qwen3-30B-A3B）⏳ 待执行

> **说明**：MoE 模型验证的特殊性在于专家路由机制可能导致多峰分布，这对 p-less 采样行为有独特影响。

**模型信息：**
- HuggingFace ID: `Qwen/Qwen3-30B-A3B`
- 总参数: ~30B，激活参数: ~3B（32 experts, 4 active per token）
- 架构：Mixture-of-Experts，`Qwen3MoeForCausalLM`

**CPU 可行性分析：**
| 量化方式 | 模型大小 | 内存需求 | 推理速度 | 可行性 |
|---|---|---|---|---|
| FP32 | ~120 GB | ~128 GB | ~3-5 tok/s | ❌ 不可行 |
| FP16/BF16 | ~60 GB | ~64 GB | ~5-8 tok/s | ❌ 内存不足 |
| INT4 (bitsandbytes NF4) | ~18 GB | ~24 GB | ~1-3 tok/s | ⚠️ 需 GPU+CPU offload，较慢 |
| GGUF Q4_K_M (llama.cpp) | ~17 GB | ~20 GB | ~8-12 tok/s | ✅ 推荐，但需改用 llama.cpp |

**验证目标：**
1. p-less 在 MoE 模型上的阈值分布是否有别于 dense 模型（多峰分布假设）
2. p-less 候选集大小是否大于 dense 模型同参数量级
3. p-less 在 MoE 模型上 Accuracy ≥ greedy（与 dense 模型趋势一致）
4. MoE 专家路由对采样分布的影响分析

**待解决的技术问题：**
- 当前验证框架基于 transformers + AutoModelForCausalLM，bitsandbytes 4-bit 量化对 MoE 模型兼容性不稳定
- 推荐方案：使用 GGUF 量化 + llama.cpp 推理，但需改造 GenerationEngine
- 备选方案：使用 bitsandbytes NF4 量化 + device_map="auto" CPU offload（需 GPU，推理慢）

**执行策略：**
- 优先使用 GPU 服务器（如有）以 bitsandbytes 4-bit 加载
- 如纯 CPU 环境，需等待 GGUF/llama.cpp 适配或足够内存的机器
- 可先在 Step 5 (GPU 全量评测) 中一并测试

### Step 4：GPU 超参搜索 ⏳ 未开始

### Step 5：GPU 全量评测 + TOST等价检验 ⏳ 未开始

### Step 6：统计分析与论文级报告 ⏳ 未开始

---

## 代码框架实现状态

### 已完成（63 文件）

| 模块 | 文件数 | 状态 |
|---|---|---|
| `verification/__init__.py` + 子包 `__init__.py` | 9 | ✅ |
| `samplers/` — 采样策略（p-less包装器 + 9种baseline） | 12 | ✅ |
| `generation/` — 通用生成引擎 + 结果数据类 | 3 | ✅ |
| `datasets/` — 9个数据集Handler（GSM8K/CSQA/QASC/HumanEval/BFCL/IFEval/GPQA/WritingPrompts/MBPP） | 11 | ✅ |
| `config/` — YAML加载 + 数据类校验 | 3 | ✅ |
| `stats/` — 统计分析（paired t-test, Wilcoxon, Bootstrap CI, TOST, Win/Tie/Lose, 报告生成） | 7 | ✅ |
| `storage/` — JSON结果存储 + schema | 3 | ✅ |
| `tests/` — 算法性质pytest测试（5个测试文件 + conftest） | 6 | ✅ |
| `scripts/` — 运行入口（math_properties, cpu_benchmark, gpu_search, gpu_evaluation, analysis） | 6 | ✅ |
| `configs/` — YAML配置（models/methods/datasets/experiments） | 10 | ✅ |
| `requirements/` — 分层依赖（base/cpu/gpu/dev） | 4 | ✅ |
| `outputs/.gitignore` | 1 | ✅ |

### 运行命令速查

```bash
# Step 1: 算法正确性测试
python verification/scripts/run_math_properties.py

# Step 2: CPU 小模型验证（GSM8K 50题）
python verification/scripts/run_cpu_benchmark.py \
  --config verification/configs/experiments/cpu_step2_gsm8k.yaml

# Step 3: CPU 中模型验证（全量）
python verification/scripts/run_cpu_benchmark.py \
  --config verification/configs/experiments/cpu_step3_full.yaml

# Step 4: GPU 超参搜索（20%子集找baseline最优参数）
python verification/scripts/run_gpu_search.py \
  --model Qwen/Qwen3-8B \
  --datasets gsm8k humaneval bfcl gpqa

# Step 5: GPU 全量评测 + TOST等价检验
python verification/scripts/run_gpu_evaluation.py \
  --model Qwen/Qwen3-8B \
  --datasets gsm8k gpqa humaneval mbpp bfcl csqa \
  --search-dir verification/outputs/results/gpu_search

# Step 5 扩展: 温度兼容性验证
python verification/scripts/run_gpu_evaluation.py \
  --model Qwen/Qwen3-8B \
  --datasets gsm8k \
  --temperatures 0.01 0.1 0.3 0.7 1.0 1.5 2.0 3.0 \
  --search-dir verification/outputs/results/gpu_search

# Step 6: 统计分析 + 报告生成
python verification/scripts/run_analysis.py \
  --results-dir verification/outputs/results/gpu_eval \
  --search-dir verification/outputs/results/gpu_search
```