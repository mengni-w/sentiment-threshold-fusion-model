# 基于 S<sub>i</sub> 与 θ<sub>knowledge</sub> 差异的情感阈值融合模型 | Sentiment Threshold Fusion Model

[中文](#中文) | [English](#english)

---

## 中文

### 项目概述

本项目实现了一种差异驱动的情感阈值自适应融合模型，通过数学严谨的方式解决舆情风险评估中"原则性"与"灵活性"的平衡问题。模型基于实际情感极性值$S_i$与知识库阈值$\theta_k$的差异度$d$，设计了具有理论保证的自适应权重函数，确保在保持政治安全底线的同时，能够对实际情况做出合理响应。

### 核心特性

- **数学严谨性**: 严格证明了模型的连续性、单调性等关键数学性质
- **自适应权重**: 基于指数衰减函数的动态权重调整机制
- **政治安全保障**: 当$S_i < \theta_k$时保持安全底线
- **参数化设计**: 针对不同舆情类别提供最佳参数配置
- **可视化分析**: 提供权重函数、敏感性分析和阈值曲面图
- **双语支持**: 完整的中英文界面切换

### 数学模型

#### 1. 基本定义

- $S_i \in [-1, 1]$: 实际情感极性值，负值表示负面情感强度
- $\theta_k \in [-0.95, -0.55]$: 知识库匹配阈值，代表政策规定的风险临界值
- $\theta_f$: 最终自适应阈值，用于精准风险评估

#### 2. 归一化差异度

$$d = \frac{|S_i - \theta_k|}{\sigma}, \quad \sigma = 0.2$$

其中$\sigma = 0.2$基于过去5年10万条历史舆情数据计算的情感极性标准差。

# Sentiment Threshold Fusion Model (基于 S<sub>i</sub> 与 θ<sub>k</sub> 差异)

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](./LICENSE) [![Python](https://img.shields.io/badge/python-3.8%2B-yellow.svg)](https://www.python.org/)

简短说明：Lightweight sentiment threshold fusion model — threshold-based sentiment classifier with implementation and examples.

---

## 中文

### 项目概述

本项目实现了一种差异驱动的情感阈值自适应融合模型。模型基于实际情感极性值 S<sub>i</sub> 与知识库阈值 θ<sub>k</sub> 的差异度 d，设计了自适应权重函数以平衡原则性与灵活性，保证在 S<sub>i</sub> &lt; θ<sub>k</sub> 时保持安全底线。

数学表示（GitHub 友好）：

```text
d = |S_i - θ_k| / σ,  where σ = 0.2

w(d) = exp(-α · d)

θ_f = clip(
  if S_i < θ_k: θ_k,
  else: θ_k + (1 - exp(-α · d)) * (S_i - θ_k),
  θ_min, θ_max
)
```

### 安装与运行

```bash
git clone <repository-url>
cd sentiment-threshold-fusion-model
pip install -r requirements.txt
streamlit run app.py
```

### 项目结构

```
sentiment-threshold-fusion-model/
├── sentiment_threshold_model.py
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
└── ...
```

---

## English

### Overview

Lightweight sentiment threshold fusion model. The model computes a normalized difference d between the actual sentiment polarity S_i and a knowledge-base threshold θ_k, uses an exponential decay weighting w(d)=exp(-α·d), and computes a clipped final threshold θ_f.

Math (GitHub-friendly):

```text
d = |S_i - θ_k| / σ  (σ = 0.2)
w(d) = exp(-α · d)
θ_f = clip(...)  # see above
```

### Quick start

```bash
git clone <repository-url>
cd sentiment-threshold-fusion-model
pip install -r requirements.txt
streamlit run app.py
```

### License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
