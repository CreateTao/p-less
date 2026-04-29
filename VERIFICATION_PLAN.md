# P-Less Sampling 验证与扩展测试计划（修订版）

## 核心验证目标

| 环境 | 验证目标 | 核心论点 |
|---|---|---|
| **CPU** | **p-less 提高了准确度** | 无需调参的 p-less 在各任务上 Accuracy ≥ 各 baseline 的默认配置 |
| **GPU** | **p-less 性能不差于最优超参配置** | p-less 无调参 ≥ 各 baseline 经调参后的最优表现 |

---

## 一、CPU 测试：验证 p-less 提高准确度

### 1.1 验证逻辑

CPU 测试要证明的核心命题：

```
p-less（零超参数）在各精确性任务上的准确度
≥ top-p/top-k/min-p 等方法的常用默认配置
```

即：别人还要调参才能得到的结果，p-less 不调参就能达到或超过。

### 1.2 模型选择

| 模型 | 规模 | 内存 | 作用 | CPU 单 token 耗时 |
|---|---|---|---|---|
| **Qwen3-0.6B** | 0.6B | ~1.2 GB | 快速迭代、算法验证 | ~0.1-0.3s |
| **Qwen3-1.7B** | 1.7B | ~3.4 GB | 主力 CPU 测试模型 | ~0.3-1s |
| **Qwen3-4B** | 4B | ~8 GB | 辅助验证（需量化） | ~1-3s |

> 小模型在精确性任务上 baseline 较弱，p-less 的提升更容易显现且统计显著。

### 1.3 数据集与任务设计

CPU 测试只选**有明确 Accuracy 评判标准**的数据集，排除开放式生成任务。

| 数据集 | 任务类型 | 样本量 | 评测指标 | 选择理由 |
|---|---|---|---|---|
| **GSM8K** | 数学推理 | 全集 1319 题 | Exact Match Accuracy | 原论文核心，答案唯一可判 |
| **CSQA** | 常识推理 | 全集 1,020 题 | Accuracy（5选1） | 原论文核心，选择题易评判 |
| **QASC** | 科学推理 | 全集 800 题 | Accuracy（8选1） | 原论文核心，选择题易评判 |
| **HumanEval** | 代码生成 | 全集 164 题 | Pass@1 | 精确性任务新扩展，执行判定 |
| **BFCL v3 (Simple)** | 函数调用 | ~200 题子集 | Accuracy + Format Rate | 精确性任务新扩展，格式可判 |
| **IFEval** | 指令格式遵循 | ~500 题 | Strict Accuracy | 格式约束精确评判 |

### 1.4 采样方法对照与温度配置

CPU 测试对比 p-less 与两类 baseline 配置：

**A 组 — 推理框架默认配置**（最真实的生产场景，大多数人根本不调参）

| 方法 | 配置 | 说明 |
|---|---|---|
| **Greedy** | — | 最高确定性基线 |
| **vLLM/SGLang 默认** | top_p=1.0, top_k=-1, T=1.0 | 实际等于无截断纯温度采样，最常见部署配置 |
| **HuggingFace 默认** | top_p=1.0, top_k=50, T=1.0 | HF generate() 默认，轻微截断 |
| **temperature 纯采样** | T=0.3, 0.7, 1.0 | 无截断、仅温度缩放 |

> 核心观察：框架默认配置下 top_p=1 / top_k=-1 等于完全没有截断，p-less 在此场景下有天然优势——它提供了截断而无需额外调参。

**B 组 — 常用推荐配置**（有人特意调了参数的场景）

| 方法 | 配置 | 说明 |
|---|---|---|
| **top-p** | p=0.9, T=0.7 | 最常见的推荐配置 |
| **top-p** | p=0.95, T=0.7 | 宽松配置 |
| **top-k** | k=40, T=0.7 | 常用配置 |
| **top-k** | k=10, T=0.7 | 精确配置 |
| **min-p** | min_p=0.05, T=0.7 | 常用配置 |
| **p-less** | —（无超参数） | 本方法 |
| **p-less-norm** | —（无超参数） | 本方法放松版 |

**温度变量对 p-less 的意义**

p-less 的阈值 `p = Σ(probs²)` 从 softmax 后的概率分布计算，温度只影响 logits→probs 的映射，阈值自然随温度变化。而 top-p/top-k 的超参数是固定的，在不同温度下语义不同：

| 温度 | top-p/top-k 行为 | p-less 行为 |
|---|---|---|
| T→0 | 分布趋于尖峰，top-p(0.9) 几乎无截断效果 | 阈值趋于最大 token 概率，自动趋近 greedy |
| T=1.0 | 标准行为 | 标准行为 |
| T→∞ | 分布趋于均匀，top-p(0.9) 截断大量 token | 阈值趋于 1/V，自动放宽 |

> 因此 p-less 无需为不同温度重新调参——一个策略覆盖所有温度。CPU 测试中 B 组方法固定用 T=0.7，而 p-less 在 T=0.3/0.7/1.0/1.5 下都用同一策略测试，展示温度兼容性。

### 1.5 测试执行步骤

#### Step 1：算法数学性质验证（纯代码，不依赖 LLM，~1 小时）

**怎么测：** 编写 Python 测试脚本，构造人工概率分布，直接调用 `p_less_decode` / `p_less_norm_decode`，检查输出是否符合理论预期。

| 测试项 | 具体做法 | 通过标准 |
|---|---|---|
| 候选集非空 | 构造 10,000 个随机分布（均匀/尖峰/多峰/退化），运行 `p_less_decode`，检查 `mask.sum() > 0` | 每次候选集 ≥ 1，通过率 100% |
| 候选集非空（norm） | 同上，运行 `p_less_norm_decode` | 均匀分布时所有 token 保留；尖峰时 ≥ 1 个保留 |
| 阈值边界 | 对每个分布，计算 `p = Σprobs²`，检查 `p >= 1/V` 且 `p <= max(probs)` | 100% 在 `[1/V, max(probs)]` 内 |
| 重归一化 | 对 p-less 输出的采样概率检查 `sum == 1.0`（允许 float 误差 1e-6） | 100% 通过 |
| 温度自适应 | 对同一组 logits，施加 T ∈ [0.01, 0.1, 0.3, 0.7, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]，计算各温度下的 `p` 值，绘制曲线 | 曲线平滑单调，无突变；T→0 时 p→max(probs)；T→∞ 时 p→1/V |
| 空集率对比 | 对 top-p(0.9) / top-k(10) / ε-sampling(1e-4) / η-sampling(0.3)，在同分布上统计空候选集出现率 | p-less = 0%；对比方法在极端分布下 > 0% |

**结论判定：** 6 项全部通过 → 算法数学性质验证完成；任何一项不通过 → 算法实现有 bug，需修复后重测。

#### Step 2：GSM8K 精确性对比（Qwen3-0.6B + Qwen3-1.7B，~2-4 天）

**怎么测：** 对每个 (模型, 采样方法, 温度) 配置，在 GSM8K 全集上逐题生成答案，提取最终数值，与标准答案做 Exact Match。

**具体流程：**
```
1. 加载模型（device_map="auto"，CPU 模式）
2. 对每题：格式化 prompt → 自回归生成 → 提取答案数值 → 判定正误
3. 记录每题结果到 JSON：
   {
     "model": "Qwen3-0.6B",
     "method": "p_less",
     "temperature": 0.7,
     "question_id": 0,
     "correct": true/false,
     "generated_tokens": 120,
     "candidate_set_avg_size": 15.3
   }
4. 汇总计算 Accuracy = correct_count / total_count
```

**配置矩阵：**

| 模型 | A组配置 | B组配置 | 每配置题数 |
|---|---|---|---|
| Qwen3-0.6B | greedy, vLLM默认(T=1), HF默认(T=1), T=0.3/0.7/1.0 | top-p(0.9), top-p(0.95), top-k(40), top-k(10), min-p(0.05), p-less, p-less-norm | 1319 |
| Qwen3-1.7B | 同上 | 同上 | 1319 |

> p-less/p-less-norm 在 A 组温度 0.3/0.7/1.0/1.5 下都测，展示一个策略覆盖多个温度；B 组方法固定 T=0.7。

**结论判定：**

| 对比组 | 判定标准 |
|---|---|
| A 组（框架默认） | p-less Accuracy > 框架默认 Accuracy ≥ 3% → **显著提升**；差值 < 3% → 需进一步分析 |
| B 组（推荐配置） | p-less vs 每个 baseline 做配对 t 检验：p<0.05 且 p-less 更高 → **Win**；p≥0.05 → **Tie**；p<0.05 且 p-less 更低 → **Lose** |

#### Step 3：选择题任务对比（CSQA + QASC，~1-2 天）

**怎么测：** 与 Step 2 流程相同，区别在于答案提取方式——选择题直接匹配选项字母（A/B/C/D/E 或 A-H）。

**结论判定：** 同 Step 2，按 A/B 组分别统计 Win/Tie/Lose。

#### Step 4：代码与结构化输出（HumanEval + BFCL Simple，~2-3 天）

**怎么测：**

- **HumanEval：** 对每题生成函数代码，用 `exec()` 运行单元测试，判定 pass/fail。Pass@1 = 单次生成通过率；需额外跑 10 次采样计算 Pass@10。
- **BFCL：** 对每题生成函数调用 JSON，先检查格式合规（JSON 可解析 + 字段完整），再检查调用正确性（参数名/值与预期匹配）。

**结论判定：**

| 指标 | p-less 显著优于框架默认的阈值 | p-less 不劣于推荐配置的阈值 |
|---|---|---|
| Pass@1 | p-less > 框架默认 ≥ 5% | p-less ≥ top-p(0.9) - 2%（非劣性 δ=2%） |
| Format Rate | p-less > 框架默认 ≥ 10% | p-less ≥ top-p(0.9) - 3%（JSON 格式容忍度稍宽） |

> 代码任务阈值更宽松，因为 Pass@1 基线较低（小模型通常 10-30%），5% 的绝对差距已经很显著。

#### Step 5：IFEval 指令遵循（~1 天）

**怎么测：** 对 IFEval 中每条指令生成回复，用 IFEval 官方评测脚本检查格式约束是否满足（如"回复不超过 3 段"、"必须包含关键词 X"等）。

**结论判定：**
- p-less Strict Accuracy ≥ Greedy - 5% → 不因采样策略破坏指令遵循能力
- p-less Strict Accuracy > 框架默认 ≥ 5% → 截断策略在指令遵循上有优势

### 1.6 统计判定方法

每个 Step 的每组对比都需要统计判定。以下说明具体何时用哪种方法、怎么操作、怎么判结论。

#### 1.6.1 配对 t 检验（用于 Step 2-5 的逐题对比）

**何时用：** p-less 与某个 baseline 在**同一组题目**上的逐题正确/错误对比。

**怎么操作：**
```python
# 对每题记录 p-less 是否正确（1/0）和 baseline 是否正确（1/0）
p_less_results = [1, 0, 1, 1, 0, ...]  # 1319 个值
baseline_results = [1, 1, 0, 1, 0, ...]  # 1319 个值
diff = p_less_results - baseline_results  # 每题差值

# 配对 t 检验
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(p_less_results, baseline_results)
```

**判定：**

| p_value | 含义 | 结论 |
|---|---|---|
| < 0.05 且 diff 均值 > 0 | p-less 显著更高 | **Win** |
| ≥ 0.05 | 无显著差异 | **Tie**（p-less 不调参也能匹配 baseline，对 p-less 有利） |
| < 0.05 且 diff 均值 < 0 | p-less 显著更低 | **Lose**（需分析是哪个数据集/哪种 baseline 导致） |

#### 1.6.2 Bootstrap 95% 置信区间（用于所有 Step 的 Accuracy 区间估计）

**何时用：** 需要报告 Accuracy 的置信区间，而非仅报告点估计。

**怎么操作：**
```python
import numpy as np
n_bootstrap = 1000
accuracies = []
for _ in range(n_bootstrap):
    sample = np.random.choice(results, size=len(results), replace=True)
    accuracies.append(sample.mean())
ci_low, ci_high = np.percentile(accuracies, [2.5, 97.5])
```

**判定：** 若 p-less 的 CI 下界 ≥ baseline 的 CI 上界 → **显著优于**；CI 有重叠 → **需看 t 检验结果**。

#### 1.6.3 Wilcoxon 秩和检验（配对 t 检验的非参数替代）

**何时用：** 当题数较少（< 30）或数据明显非正态分布时（如 HumanEval 仅 164 题）。

**怎么操作：**
```python
from scipy.stats import wilcoxon  # 配对版本
stat, p_value = wilcoxon(p_less_results - baseline_results)
```

**判定：** 同 1.6.1 的 p_value 判定逻辑。

#### 1.6.4 Win/Tie/Lose 汇总（最终结论统计）

**怎么操作：** 统计所有 (数据集 × 模型 × baseline方法 × 温度) 组合中 p-less 的 Win/Tie/Lose 计数。

**判定标准：**

| 结论类型 | 定义 | 论文中如何表述 |
|---|---|---|
| **Win** | p<0.05，p-less Accuracy 显著高于 baseline | "p-less 在 X/N 配置中显著优于 [方法名]" |
| **Tie** | p≥0.05，无显著差异 | "p-less 在 Y/N 配置中与 [方法名] 无显著差异，但 p-less 不需调参" |
| **Lose** | p<0.05，p-less Accuracy 显著低于 baseline | "p-less 在 Z/N 配置中低于 [方法名]，集中在 [具体场景]" |

**CPU 测试总体结论判定：**

| 情况 | 结论 |
|---|---|
| A 组中 p-less Win ≥ 80% | **强结论：p-less 在框架默认场景下显著提升准确度** |
| B 组中 p-less Win+Tie ≥ 80% | **强结论：p-less 不调参即达到或超过推荐配置** |
| B 组中 p-less Lose > 20% | **弱结论：需分析 Lose 集中在哪些场景，并讨论 p-less 的局限性** |

### 1.7 结论输出格式

测试完成后，按以下格式整理结果，每张表可直接用于论文。

**表 1：A 组 — p-less vs 推理框架默认配置**

```
| Dataset  | Model     | Greedy  | vLLM默认(T=1) | HF默认(T=1) | p-less(T=0.7) | p-less(T=1) | p-less 提升 |
|----------|-----------|---------|---------------|-------------|----------------|-------------|-------------|
| GSM8K    | Qwen3-0.6B|  XX.X%  |    XX.X%      |   XX.X%     |    XX.X%       |   XX.X%     | +Δ% vs vLLM |
| GSM8K    | Qwen3-1.7B|  XX.X%  |    XX.X%      |   XX.X%     |    XX.X%       |   XX.X%     | +Δ% vs vLLM |
| CSQA     | ...       |         |               |             |                |             |             |
| QASC     | ...       |         |               |             |                |             |             |
| HumanEval| ...       |         |               |             |                |             |             |
| BFCL     | ...       |         |               |             |                |             |             |
| IFEval   | ...       |         |               |             |                |             |             |
```

**表 2：B 组 — p-less vs 推荐配置（Win/Tie/Lose）**

```
| Dataset  | Model     | top-p(0.9) | top-p(0.95) | top-k(40) | top-k(10) | min-p(0.05) | W/T/L 总计 |
|----------|-----------|------------|-------------|-----------|-----------|-------------|------------|
| GSM8K    | Qwen3-0.6B| W/T/L      | W/T/L       | W/T/L     | W/T/L     | W/T/L       | X/Y/Z      |
| GSM8K    | Qwen3-1.7B| W/T/L      | W/T/L       | W/T/L     | W/T/L     | W/T/L       | X/Y/Z      |
| ...      | ...       |            |             |           |           |             |            |

汇总：Win=X, Tie=Y, Lose=Z, 总=N, Win+Tie占比=XX%
```

**表 3：温度兼容性（GSM8K 上各方法的 Accuracy vs Temperature）**

```
| Method        | T=0.3  | T=0.7  | T=1.0  | T=1.5  | Accuracy方差 |
|---------------|--------|--------|--------|--------|-------------|
| top-p(0.9)    | XX.X%  | XX.X%  | XX.X%  | XX.X%  | σ²          |
| top-k(40)     | XX.X%  | XX.X%  | XX.X%  | XX.X%  | σ²          |
| min-p(0.05)   | XX.X%  | XX.X%  | XX.X%  | XX.X%  | σ²          |
| p-less        | XX.X%  | XX.X%  | XX.X%  | XX.X%  | σ²(预期最小)|
```

> p-less Accuracy 方差最小 → 温度变化对其影响最小 → 无需为不同温度调参。

---

## 二、GPU 测试：验证 p-less 性能不劣于最优超参

### 2.1 验证逻辑

GPU 测试要证明的核心命题：

```
p-less（零超参数）在各任务上的表现
≥ 各 baseline 经网格搜索调参后的最优表现
```

即：即使别人花大量精力调出了最优参数，p-less 不调参也能达到同等水平。这是一个更严格的标准——**non-degradation**（不劣于）。

### 2.2 模型选择

| 模型 | 规模 | GPU 需求 | 作用 |
|---|---|---|---|
| **Qwen3-8B** | 8B | 1x 24GB | 主力测试，对标原论文 Llama3-8B |
| **Qwen3-14B** | 14B | 1x 24GB (量化) 或 2x | 中大模型验证 |
| **Llama3-8B-Instruct** | 8B | 1x 24GB | 原论文模型复现 |
| **Mistral-7B-v0.3** | 7B | 1x 24GB | 原论文模型复现 |

### 2.3 数据集矩阵

GPU 测试覆盖**原论文全部 + 新领域核心**数据集，同时增加生成类任务：

| 类别 | 数据集 | 评测指标 | 是否原论文 |
|---|---|---|---|
| 数学推理 | **GSM8K** | Exact Match | 是 |
| 专家推理 | **GPQA** | Accuracy | 是 |
| 科学推理 | **QASC** | Accuracy | 是 |
| 常识推理 | **CSQA** | Accuracy | 是 |
| 创意写作 | **Writing Prompts** | Distinct-1/2, MAUVE | 是 |
| 代码生成 | **HumanEval** | Pass@1, Pass@10 | **新增** |
| 代码生成 | **MBPP** (子集 200 题) | Pass@1 | **新增** |
| 函数调用 | **BFCL v3** (Simple+Multiple) | Accuracy, Format Rate | **新增** |
| 指令遵循 | **IFEval** | Strict/Prompt Accuracy | **新增** |
| 多轮对话 | **MT-Bench** | GPT-4 Judge Score | **新增** |

### 2.4 Baseline 最优超参搜索

GPU 测试的核心难点：需要先找到各 baseline 的**最优超参**，才能与 p-less 对比。

#### 搜索策略

| 方法 | 搜索空间 | 搜索方式 |
|---|---|---|
| top-p | p ∈ {0.8, 0.85, 0.9, 0.95, 0.99} × T ∈ {0.3, 0.5, 0.7, 1.0} | 网格搜索 |
| top-k | k ∈ {5, 10, 20, 40, 100} × T ∈ {0.3, 0.5, 0.7, 1.0} | 网格搜索 |
| min-p | min_p ∈ {0.01, 0.03, 0.05, 0.1, 0.2} × T ∈ {0.3, 0.5, 0.7, 1.0} | 网格搜索 |
| ε-sampling | ε ∈ {1e-2, 1e-3, 1e-4, 1e-5} × T ∈ {0.3, 0.7, 1.0} | 网格搜索 |
| η-sampling | η ∈ {0.1, 0.3, 0.5, 0.7, 0.9} × T ∈ {0.3, 0.7, 1.0} | 网格搜索 |

> 每个方法 × 每个数据集独立搜索最优配置，确保 baseline 有"主场优势"。p-less 对所有数据集使用同一策略（无超参数），体现通用性优势。

#### 搜索流程

```
1. 对每个 (方法, 数据集) 组合，在数据集子集（20%）上做网格搜索
2. 选取最优配置
3. 在全集上用最优配置评测
4. 与 p-less（全集，无搜索）对比
```

### 2.5 Non-Degradation 统计框架

GPU 测试的论点是"不劣于"，因此统计框架需要围绕**等价性检验**而非 superiority 检验。

| 检验方法 | 用途 | 说明 |
|---|---|---|
| **等价性检验 (TOST)** | 证明 p-less 与最优 baseline 等价 | 设定 δ=2% 容忍区间，若 p-less Accuracy 在 [baseline-δ, baseline+δ] 内则等价 |
| **非劣性检验** | 证明 p-less 不劣于最优 baseline | 单侧检验：p-less ≥ baseline - δ，δ=2% |
| **配对 t 检验** | 检测是否有显著差异 | 若 p>0.05 则无显著差异 → Tie（对 non-degradation 论点有利） |

**判定标准：**

| 情况 | 结论 |
|---|---|
| p-less ≥ baseline最优 - δ 且 TOST 通过 | **Non-degradation 成立** |
| p-less 显著 > baseline最优 (p<0.05) | **p-less 优于**（更强的结论） |
| p-less < baseline最优 - δ | **退化**（需分析原因和场景） |

### 2.6 具体测试流程

#### Step 1：Baseline 超参搜索（各数据集 20% 子集，~3-5 天）

```
对每个 (模型, 方法, 数据集) 组合，在子集上搜索最优超参
模型：Qwen3-8B, Llama3-8B
方法：top-p, top-k, min-p, ε-sampling, η-sampling
数据集：GSM8K, GPQA, QASC, CSQA, HumanEval, BFCL, IFEval
```

| 组合总数 | 搜索配置数/组合 | 子集评测次数 |
|---|---|---|
| 2 模型 × 5 方法 × 7 数据集 = 70 | ~16-20 | ~1,120-1,400 |

> 可并行化：不同 (模型, 数据集) 组合可在不同 GPU 上并行搜索。

#### Step 2：全集评测 — Baseline 最优配置（~5-7 天）

```
用 Step 1 找到的最优配置，在全集上正式评测每个 baseline
同时评测 p-less / p-less-norm（全集，无需搜索阶段）
```

| 数据集 | 评测配置数 | 每配置样本量 |
|---|---|---|
| GSM8K | ~12（5 baseline 最优 + greedy + p-less + p-less-norm + p-less×2温度） | 1319 |
| GPQA | ~12 | ~400 |
| QASC | ~12 | 800 |
| CSQA | ~12 | 1020 |
| HumanEval | ~12 + 多次采样(Pass@10需10次) | 164×10 |
| BFCL | ~12 | ~400 |
| IFEval | ~12 | ~500 |

#### Step 3：Writing Prompts 多样性评测（~2 天）

```
生成类任务：对比 p-less / p-less-norm 与 baseline 最优配置的多样性
重点：p-less-norm 在多样性上是否能达到 top-p(最优) 的水平
```

| 对比 | 指标 | 预期 |
|---|---|---|
| p-less vs top-p(最优) | MAUVE | p-less 可能略低（更保守），但不应大幅劣于 |
| p-less-norm vs top-p(最优) | Distinct-1/2 | p-less-norm 应与 top-p(最优) 相当或更高 |
| p-less-norm vs top-p(最优) | MAUVE | 等价或更好 |

#### Step 4：MT-Bench 多轮对话（~2-3 天）

```
需要 GPT-4 或强模型作为 judge，评估对话质量
对比 p-less vs top-p(最优) 在 80 道多轮对话题上的表现
```

#### Step 5：效率对比（贯穿所有步骤，无需额外时间）

在 Step 2-4 的评测中同步记录效率数据：

| 指标 | 记录方式 | 对比目标 |
|---|---|---|
| 平均生成 token 数 | 每个 sample 记录 | p-less 是否更短（论文声称优势） |
| 采样函数耗时 | `time.perf_counter()` 计时 | p-less 采样速度 vs 其他方法 |
| 候选集平均大小 | 每 step 记录保留 token 数 | p-less 候选集大小分布 |
| 总生成耗时 | 端到端计时 | p-less 总时间效率 |

### 2.7 原论文复现对照

GPU 测试必须包含原论文模型的数据集复现，确保扩展测试的基准可信：

| 复现项 | 原论文结果 | 本测试验证 |
|---|---|---|
| Llama-2-7B × GSM8K | 原论文 Table 中的 Accuracy | 复现到 ±1% |
| Mistral-7B × QASC | 原论文 Table 中的 Accuracy | 复现到 ±1% |
| Llama3-8B × CSQA | 原论文 Table 中的 Accuracy | 复现到 ±1% |
| 各方法 × Writing Prompts | 原论文 Diversity 指标 | 复现趋势一致 |

> 若复现结果与原论文偏差 >2%，需排查原因（模型版本、tokenizer 差异等）。

---

## 三、完整测试时间线与资源

### 3.1 时间线

| 阶段 | 环境 | 时间 | 内容 |
|---|---|---|---|
| **CPU-1** 算法性质 | CPU | 1 小时 | Step 1：数学性质验证 |
| **CPU-2** 精确性对比 | CPU | 3-5 天 | Step 2-5：GSM8K/CSQA/QASC/HumanEval/BFCL/IFEval |
| **GPU-1** 超参搜索 | GPU | 3-5 天 | Step 1：各 baseline 最优超参网格搜索 |
| **GPU-2** 全集评测 | GPU | 5-7 天 | Step 2：全集 Accuracy + 效率对比 |
| **GPU-3** 生成+对话 | GPU | 4-5 天 | Step 3-4：Writing Prompts + MT-Bench |
| **GPU-4** 原论文复现 | GPU | 2-3 天 | 原论文模型 × 原论文数据集复现 |
| **总结** | — | 1-2 天 | 统计分析、可视化、撰写结论 |

**总计：约 20-28 天**

### 3.2 GPU 资源需求

| 阶段 | GPU 配置 | 并行策略 |
|---|---|---|
| GPU-1 超参搜索 | 1x RTX 4090 (24GB) | 不同数据集串行，同一数据集不同方法可并行 |
| GPU-2 全集评测 | 1x RTX 4090 | 同上 |
| GPU-3 生成评测 | 1x RTX 4090 | Writing + MT-Bench 串行 |
| GPU-4 复现 | 1x RTX 4090 | 3 个原论文模型串行 |

> 如有多卡，GPU-1 到 GPU-4 可并行化，总时间压缩至 ~10-15 天。

### 3.3 存储需求

| 内容 | 估算 |
|---|---|
| 模型权重 | ~80 GB（4 个 7-8B 模型 + 1 个 14B） |
| 数据集 | ~5 GB |
| 评测结果（JSON/CSV） | ~500 MB |
| 生成文本缓存 | ~2 GB |
| **总计** | ~90 GB |

---

## 四、结果呈现框架

### 4.1 CPU 测试结果表（准确度提升）

```
Table: p-less vs Baseline Default Configs on Small Models (CPU)

| Dataset  | Model     | Greedy | top-p(0.9) | top-k(40) | min-p(0.05) | p-less | p-less-norm | p-less Win/Tie/Lose |
|----------|-----------|--------|------------|-----------|-------------|--------|-------------|---------------------|
| GSM8K    | Qwen3-0.6B|  XX.X% |   XX.X%    |   XX.X%   |   XX.X%     | XX.X%  |   XX.X%     | W/T/L vs each       |
| GSM8K    | Qwen3-1.7B|  XX.X% |   XX.X%    |   XX.X%   |   XX.X%     | XX.X%  |   XX.X%     | W/T/L vs each       |
| CSQA     | ...       |        |            |           |             |        |             |                     |
| HumanEval| ...       |        |            |           |             |        |             |                     |
| BFCL     | ...       |        |            |           |             |        |             |                     |

Aggregate Win/Tie/Lose: X / Y / Z out of N comparisons
```

### 4.2 GPU 测试结果表（non-degradation）

```
Table: p-less vs Optimally-Tuned Baselines on 8B Models (GPU)

| Dataset  | Model     | top-p(best) | top-k(best) | min-p(best) | ε(best) | η(best) | p-less | p-less ≥ best-δ? |
|----------|-----------|-------------|-------------|-------------|---------|---------|--------|-------------------|
| GSM8K    | Qwen3-8B  |   XX.X%     |   XX.X%     |   XX.X%     | XX.X%   | XX.X%   | XX.X%  | Yes/No per method |
| GPQA     | ...       |             |             |             |         |         |        |                   |
| HumanEval| ...       |             |             |             |         |         |        |                   |
| BFCL     | ...       |             |             |             |         |         |        |                   |

Non-degradation rate: p-less ≥ best-δ in M/N cases (δ=2%)
```

### 4.3 效率对比图

```
Figure: Generation Efficiency Comparison

(a) Average token count per sample (bar chart)
(b) Sampling time per token (bar chart)
(c) Candidate set size distribution (box plot)
```

### 4.4 温度鲁棒性图

```
Figure: Accuracy vs Temperature across Methods

X-axis: Temperature (0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)
Y-axis: Accuracy
Lines: p-less, top-p(0.9), top-k(40), min-p(0.05)

Expected: p-less curve flat/stable, baselines dip at extreme temperatures
```

### 4.5 关键发现汇总

```
1. CPU: p-less 在 N 个配置中 Win X / Tie Y / Lose Z
   → 不调参即达到或超过默认配置 baseline

2. GPU: p-less 在 M/N 个 non-degradation 检验中通过
   → 不调参不劣于各 baseline 的最优调参配置

3. Function Call / Code: p-less 在结构化输出任务上 Format Rate 优势显著
   → 无超参数方法在格式敏感场景的特殊价值

4. 温度兼容性: p-less Accuracy 在温度变化下方差最小
   → 无需为不同温度重新调参

5. 效率: p-less 平均 token 数最少，采样速度最快
   → 与原论文效率优势结论一致
```