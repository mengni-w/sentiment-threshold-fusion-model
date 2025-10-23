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

#### 3. 自适应权重函数

$$w(d) = e^{-\alpha \cdot d}$$

权重函数具有以下性质：
- 边界行为: $w(0) = 1$, $\lim_{d \to \infty} w(d) = 0$
- 单调性: 严格单调递减
- 凸性: 权重衰减先快后慢

#### 4. 最终阈值计算

$$\theta_f = \text{clip}\left(\begin{cases}
\theta_k, & \text{if } S_i < \theta_k \\
\theta_k + (1 - e^{-\alpha \cdot d}) \cdot (S_i - \theta_k), & \text{if } S_i \geq \theta_k
\end{cases}, \theta_{\min}, \theta_{\max}\right)$$

其中$\theta_{\min} = -0.95$，$\theta_{\max} = -0.55$。

### 舆情类别参数配置

| 舆情类别 | 最佳$\alpha$值 | 特点 |
|---------|---------------|------|
| 意识形态安全与政治风险类 | 0.8 | 政治安全优先，衰减慢 |
| 干部选拔任用与监督管理类 | 1.2 | 平衡原则与实际 |
| 党员教育管理监督类 | 1.3 | 适度信任实际情感 |
| 人才工作与国际人才安全类 | 1.4 | 更关注实际情况 |
| 组织制度与政策执行类 | 1.5 | 高度适应实际情况 |
| 组织系统自身建设与舆情应对类 | 1.3 | 平衡知识库与实际 |
| 区域与特殊领域党建类 | 0.9 | 严格保持知识库主导 |

### 安装与运行

#### 环境要求

- Python 3.8+
- 依赖包见 `requirements.txt`

#### 安装步骤

```bash
# 克隆项目
git clone <repository-url>
cd yuqing

# 安装依赖
pip install -r requirements.txt

# 运行Web应用
streamlit run app.py

# 或直接运行Python脚本
python sentiment_threshold_model.py
```

### 使用方法

#### 1. Web界面使用

1. 运行 `streamlit run app.py`
2. 在浏览器中打开显示的URL
3. 在左侧边栏调整参数
4. 点击"计算阈值"按钮查看结果
5. 使用不同标签页进行批量分析和可视化

#### 2. Python API使用

```python
from sentiment_threshold_model import SentimentThresholdModel

# 创建模型实例
model = SentimentThresholdModel()

# 单次计算
result = model.calculate_adaptive_threshold(
    si=-0.7,
    theta_k=-0.8,
    alpha=1.2,
    category="组织制度与政策执行类"
)

print(f"最终阈值: {result['theta_f_final']:.3f}")
print(f"风险判定: {result['is_high_risk']}")

# 批量计算
data = [
    {'si': -0.6, 'theta_k': -0.7, 'alpha': 1.0},
    {'si': -0.8, 'theta_k': -0.75, 'alpha': 1.2}
]
batch_results = model.batch_calculate(data)
```

### 项目结构

```
yuqing/
├── sentiment_threshold_model.py  # 核心模型实现
├── app.py                       # Streamlit Web应用
├── requirements.txt             # 依赖包列表
├── README.md                    # 项目说明文档
├── main-2.tex                   # LaTeX数学模型文档
└── 舆情.pdf                     # 模型说明PDF
```

### 理论基础

本模型基于以下数学理论：

1. **连续性保证**: 模型在$S_i = \theta_k$处连续，避免阈值跳跃
2. **单调性保证**: 输出随输入单调不减，符合直觉
3. **边界约束**: 确保输出在合理范围内
4. **参数敏感性**: 提供参数变化对结果影响的量化分析

### 贡献指南

欢迎提交Issue和Pull Request来改进项目。

### 许可证

本项目采用MIT许可证。

---

## English

### Project Overview

This project implements a difference-driven adaptive sentiment threshold fusion model that mathematically balances "principle" and "flexibility" in public opinion risk assessment. The model is based on the difference $d$ between actual sentiment polarity $S_i$ and knowledge base threshold $\theta_k$, designing an adaptive weight function with theoretical guarantees to ensure reasonable responses to actual situations while maintaining political security baselines.

### Key Features

- **Mathematical Rigor**: Strictly proves key mathematical properties like continuity and monotonicity
- **Adaptive Weighting**: Dynamic weight adjustment mechanism based on exponential decay function
- **Political Security Assurance**: Maintains safety baseline when $S_i < \theta_k$
- **Parameterized Design**: Provides optimal parameter configurations for different category types
- **Visualization Analysis**: Offers weight function, sensitivity analysis, and threshold surface plots
- **Bilingual Support**: Complete Chinese-English interface switching

### Mathematical Model

#### 1. Basic Definitions

- $S_i \in [-1, 1]$: Actual sentiment polarity value, negative values indicate negative sentiment intensity
- $\theta_k \in [-0.95, -0.55]$: Knowledge base threshold representing policy-defined risk critical value
- $\theta_f$: Final adaptive threshold for precise risk assessment

#### 2. Normalized Difference

$$d = \frac{|S_i - \theta_k|}{\sigma}, \quad \sigma = 0.2$$

Where $\sigma = 0.2$ is the sentiment polarity standard deviation calculated from 100,000 historical public opinion data over the past 5 years.

#### 3. Adaptive Weight Function

$$w(d) = e^{-\alpha \cdot d}$$

The weight function has the following properties:
- Boundary behavior: $w(0) = 1$, $\lim_{d \to \infty} w(d) = 0$
- Monotonicity: Strictly monotonically decreasing
- Convexity: Weight decay is fast initially then slow

#### 4. Final Threshold Calculation

$$\theta_f = \text{clip}\left(\begin{cases}
\theta_k, & \text{if } S_i < \theta_k \\
\theta_k + (1 - e^{-\alpha \cdot d}) \cdot (S_i - \theta_k), & \text{if } S_i \geq \theta_k
\end{cases}, \theta_{\min}, \theta_{\max}\right)$$

Where $\theta_{\min} = -0.95$ and $\theta_{\max} = -0.55$.

### Category Parameter Configuration

| Category Type | Optimal $\alpha$ Value | Characteristics |
|---------------|----------------------|-----------------|
| Ideological Security & Political Risk | 0.8 | Political security priority, slow decay |
| Cadre Selection & Supervision | 1.2 | Balance principle and reality |
| Party Member Education & Management | 1.3 | Moderate trust in actual sentiment |
| Talent Work & International Security | 1.4 | More focus on actual situations |
| Organizational System & Policy Implementation | 1.5 | Highly adaptive to actual situations |
| System Construction & Public Opinion Response | 1.3 | Balance knowledge base and reality |
| Regional & Special Field Party Building | 0.9 | Strictly maintain knowledge base dominance |

### Installation and Running

#### Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

#### Installation Steps

```bash
# Clone the project
git clone <repository-url>
cd yuqing

# Install dependencies
pip install -r requirements.txt

# Run web application
streamlit run app.py

# Or run Python script directly
python sentiment_threshold_model.py
```

### Usage

#### 1. Web Interface Usage

1. Run `streamlit run app.py`
2. Open the displayed URL in browser
3. Adjust parameters in the left sidebar
4. Click "Calculate Threshold" button to view results
5. Use different tabs for batch analysis and visualization

#### 2. Python API Usage

```python
from sentiment_threshold_model import SentimentThresholdModel

# Create model instance
model = SentimentThresholdModel()

# Single calculation
result = model.calculate_adaptive_threshold(
    si=-0.7,
    theta_k=-0.8,
    alpha=1.2,
    category="organizational_system"
)

print(f"Final threshold: {result['theta_f_final']:.3f}")
print(f"High risk: {result['is_high_risk']}")

# Batch calculation
data = [
    {'si': -0.6, 'theta_k': -0.7, 'alpha': 1.0},
    {'si': -0.8, 'theta_k': -0.75, 'alpha': 1.2}
]
batch_results = model.batch_calculate(data)
```

### Project Structure

```
yuqing/
├── sentiment_threshold_model.py  # Core model implementation
├── app.py                       # Streamlit web application
├── requirements.txt             # Dependency list
├── README.md                    # Project documentation
├── main-2.tex                   # LaTeX mathematical model document
└── 舆情.pdf                     # Model description PDF
```

### Theoretical Foundation

This model is based on the following mathematical theories:

1. **Continuity Guarantee**: Model is continuous at $S_i = \theta_k$, avoiding threshold jumps
2. **Monotonicity Guarantee**: Output is monotonically non-decreasing with input, conforming to intuition
3. **Boundary Constraints**: Ensures output within reasonable range
4. **Parameter Sensitivity**: Provides quantitative analysis of parameter changes' impact on results

### Contributing

Welcome to submit Issues and Pull Requests to improve the project.

### License

This project is licensed under the MIT License.
