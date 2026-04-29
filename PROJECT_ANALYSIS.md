# P-Less Sampling 项目分析

## 项目概述

**p-less** 是一个基于信息论的无超参数 LLM 解码采样方法的参考实现，由 Runyan Tan、Shuang Wu、Phillip Howard 提出。

**对应论文：** *p*-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding (arXiv:2509.23234, 2025)

## 解决的问题

在 LLM 自回归解码过程中，需要从 token 概率分布中进行截断采样以生成文本。现有方法（top-p、top-k、epsilon-sampling、min-p、eta-sampling、mirostat）均依赖需要手动调优的超参数，且可能产生退化边缘情况（如候选集为空）。p-less sampling 提出了一种**基于信息论、无需超参数**的替代方案，在每个解码步骤中从完整概率分布动态计算截断阈值。

## 核心算法

项目包含两个采样函数（`p_less_samplers.py`）：

### 1. p-less sampling (`p_less_decode`)

**阈值计算：** 使用概率分布的平方 2-范数（即 Rényi 熵 of order 2 / 逆 Simpson 指数）：

```
p = Σ(probs_i²)
```

- `p` 表示在两次独立采样中抽到相同 token 的概率
- 任何概率 ≥ `p` 的 token 被视为有足够质量而保留
- 低于 `p` 的 token 被屏蔽（置为 0），剩余 token 重归一化后采样

**关键性质：** `p` 始终 ≥ `1/V`（V 为词表大小）且 ≤ 最大 token 概率，因此至少有一个 token 满足条件，**保证候选集永远非空**，无需任何边缘情况处理。

### 2. p-less-norm sampling (`p_less_norm_decode`)

**阈值计算：** 对 p-less 进行归一化放松，适用于偏好多样性而非一致性的任务：

```
p_norm = (V × Σ(probs_i²) - 1) / (V - 1)
```

- 当分布完全均匀时，`p_norm = 0`，所有 token 均被接受（最大多样性）
- 当分布高度集中时，`p_norm` 接近峰值概率（与 p-less 行为趋同）
- 对于中等集中程度的分布，`p_norm` 比 p-less 更宽松，允许更多 token，产生更多样化的输出

## 四大优势

| 优势 | 说明 |
|---|---|
| **动态、分布感知阈值** | 从整个 token 分布计算，而非固定超参数或依赖单个 token |
| **有界且有效** | 始终保证候选集非空，无需边缘情况回退逻辑 |
| **兼容温度缩放** | 阈值随温度自动调整，不像 top-p/top-k 等在极端温度下超参数失去意义 |
| **高效** | 更快的 token 采样速度和更短的生成长度，且不牺牲任务性能 |

## 项目结构

| 文件 | 说明 |
|---|---|
| `p_less_samplers.py` | 核心实现——两个采样函数 |
| `p_less_examples.ipynb` | Jupyter 示例笔记本，演示与 HuggingFace 模型的集成 |
| `README.md` | 项目文档与引用信息 |
| `LICENSE` | MIT 许可证（Copyright 2025 Runyan Tan） |

## 使用方式

作为标准解码策略的**直接替换**：

```python
import torch
from p_less_samplers import p_less_decode, p_less_norm_decode

# 在每个自回归步骤中
probs = torch.softmax(logits[-1], dim=-1)
next_token = p_less_decode(probs)       # 精确、一致的输出
next_token = p_less_norm_decode(probs)   # 更多样化的输出
```

## 依赖

- `torch` (测试版本: 2.6.0)
- `transformers` (测试版本: 4.55.2)
- Python 3.10.12

安装：`pip install torch transformers`

## 论文信息

如果需要将论文 PDF 添加到项目中，对应的论文为：

> Runyan Tan, Shuang Wu, Phillip Howard. "*p*-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding." arXiv:2509.23234, 2025.

论文链接：https://arxiv.org/abs/2509.23234