# P-Less Sampling 测试与验证计划

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
# 运行命令
python run_benchmark.py \
  --model Qwen/Qwen3-0.6B \
  --device cpu \
  --datasets gsm8k csqa \
  --methods greedy top_p_0.9 top_k_40 p_less p_less_norm \
  --num_samples 50 \
  --seed 42 \
  --output results/cpu_step2/
```

验证目标：
- p-less 在 GSM8K 上 Accuracy ≥ greedy（与原论文趋势一致）
- p-less-norm 在 CSQA 上 Diversity ≥ p-less（与原论文趋势一致）
- 无空候选集崩溃（100% 样本正常生成）

#### Step 3：CPU 中模型验证（Qwen3-1.7B，8-12 小时）

```bash
python run_benchmark.py \
  --model Qwen/Qwen3-1.7B \
  --device cpu \
  --datasets gsm8k qasc humaneval \
  --methods greedy top_p_0.9 top_k_40 min_p_0.05 p_less p_less_norm \
  --num_samples full \
  --seed 42 \
  --output results/cpu_step3/
```

新增验证：
- HumanEval Pass@1 对比（代码精确性）
- 引入 min-p 对照（同为相对阈值方法，但需调参）

#### Step 4：GPU 全量评测（核心实验，2-3 天）

按优先级逐数据集运行：

```bash
# P0 数据集
for MODEL in Qwen3-8B Llama-3.1-8B Mistral-7B-v0.3; do
  for DATASET in gsm8k gpqa humaneval mbpp bfcl_v3 mmlu; do
    python run_benchmark.py \
      --model $MODEL --device cuda \
      --datasets $DATASET \
      --methods greedy top_p_0.9 top_p_0.95 top_k_40 min_p_0.05 epsilon_0.001 p_less p_less_norm \
      --temperatures 0.3 0.7 1.0 \
      --seeds 42 123 456 \
      --num_samples full
  done
done

# P1 数据集
for DATASET in csqa qasc humaneval_plus livecodebench writing_prompts bfcl_v3_multi mmlu_pro; do
  ...
done
```

#### Step 5：温度兼容性专项实验

```bash
python run_temperature_study.py \
  --model Qwen3-8B \
  --dataset gsm8k \
  --methods top_p_0.9 top_k_40 p_less p_less_norm \
  --temperatures 0.01 0.1 0.3 0.7 1.0 1.5 2.0 3.0 \
  --output results/temperature_study/
```

输出：
- 各方法 Accuracy-vs-Temperature 曲线图
- 各方法 候选集平均大小-vs-Temperature 曲线图
- p-less 阈值分布统计（均值、方差、min、max）

#### Step 6：统计分析与论文级报告

```python
# 对每组 (model, dataset) 做：
1. 配对 t 检验 / McNemar 检验：p-less vs 各 baseline
2. Bootstrap 95% CI：各方法 Accuracy 置信区间
3. Win/Tie/Lose 计数表：p-less 在多少配置下 Win
4. 效果量 (Cohen's d)：量化 p-less 优势大小
5. 效率分析：平均 token 数 / 生成速度
```

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
| Step 4: GPU 全量评测 | 2-3 天 | GPU | 核心实验完整结果 |
| Step 5: 温度专项 | 4-6 小时 | GPU | 温度兼容性曲线 |
| Step 6: 统计分析 | 4-6 小时 | 代码 | 论文级报告 + 图表 |
| **总计** | **~4-5 天** | | |

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

# Step 6: GPU 超参搜索（20%子集找baseline最优参数）
python verification/scripts/run_gpu_search.py \
  --model Qwen/Qwen3-8B \
  --datasets gsm8k humaneval bfcl gpqa

# Step 7: GPU 全量评测 + TOST等价检验
python verification/scripts/run_gpu_evaluation.py \
  --model Qwen/Qwen3-8B \
  --datasets gsm8k gpqa humaneval mbpp bfcl csqa

# 统计分析 + 报告生成
python verification/scripts/run_analysis.py \
  --results-dir verification/outputs/results/gpu_eval \
  --search-dir verification/outputs/results/gpu_search
```