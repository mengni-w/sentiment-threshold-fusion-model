# 使用说明 | Usage Instructions

## 快速开始 | Quick Start

### 1. 安装依赖 | Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. 运行方式 | Running Options

#### 方式一：使用启动脚本 | Option 1: Use Launcher Script
```bash
python run.py
```
然后选择：
- 1: Web界面 (推荐) | Web Interface (Recommended)
- 2: 命令行演示 | Command Line Demo

#### 方式二：直接启动Web界面 | Option 2: Direct Web Interface
```bash
streamlit run app.py
```

#### 方式三：命令行测试 | Option 3: Command Line Test
```bash
python test_model.py
```

## 功能特性 | Features

### Web界面功能 | Web Interface Features

1. **参数调整** | Parameter Adjustment
   - 实际情感极性值 Si (-1 到 1)
   - 知识库阈值 θk (-0.95 到 -0.55)
   - 差异敏感系数 α (0.1 到 3.0)
   - 舆情类别选择
   - 匹配置信度 (0 到 1)

2. **计算结果** | Calculation Results
   - 归一化差异度
   - 权重值
   - 最终阈值
   - 风险判定
   - 详细结果表格

3. **可视化分析** | Visualization Analysis
   - 权重函数图
   - 参数敏感性分析
   - 阈值曲面图

4. **批量分析** | Batch Analysis
   - CSV文件上传
   - 示例数据测试
   - 结果可视化
   - 结果下载

5. **语言切换** | Language Switching
   - 完整的中英文界面
   - 实时语言切换

### 舆情类别参数 | Category Parameters

| 类别 | α值 | 特点 |
|------|-----|------|
| 意识形态安全与政治风险类 | 0.8 | 政治安全优先 |
| 干部选拔任用与监督管理类 | 1.2 | 平衡原则与实际 |
| 党员教育管理监督类 | 1.3 | 适度信任实际情感 |
| 人才工作与国际人才安全类 | 1.4 | 更关注实际情况 |
| 组织制度与政策执行类 | 1.5 | 高度适应实际情况 |
| 组织系统自身建设与舆情应对类 | 1.3 | 平衡知识库与实际 |
| 区域与特殊领域党建类 | 0.9 | 严格保持知识库主导 |

## API使用 | API Usage

```python
from sentiment_threshold_model import SentimentThresholdModel

# 创建模型实例
model = SentimentThresholdModel()

# 单次计算
result = model.calculate_adaptive_threshold(
    si=-0.7,           # 实际情感极性值
    theta_k=-0.8,      # 知识库阈值
    alpha=1.2,         # 差异敏感系数
    category="组织制度与政策执行类",  # 舆情类别
    confidence=1.0     # 匹配置信度
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

## 数学模型 | Mathematical Model

### 核心公式 | Core Formulas

1. **归一化差异度** | Normalized Difference
   ```
   d = |Si - θk| / σ  (σ = 0.2)
   ```

2. **自适应权重函数** | Adaptive Weight Function
   ```
   w(d) = e^(-α·d)
   ```

3. **最终阈值计算** | Final Threshold Calculation
   ```
   θf = θk                                    if Si < θk
   θf = θk + (1 - e^(-α·d)) × (Si - θk)      if Si ≥ θk
   ```

### 模型特性 | Model Properties

- **连续性**: 在 Si = θk 处连续
- **单调性**: 输出随输入单调不减
- **边界约束**: θf ∈ [-0.95, -0.55]
- **政治安全**: 当 Si < θk 时保持 θf = θk

## 故障排除 | Troubleshooting

### 常见问题 | Common Issues

1. **依赖包安装失败** | Dependency Installation Failed
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Web界面无法启动** | Web Interface Won't Start
   ```bash
   pip install streamlit --upgrade
   streamlit run app.py
   ```

3. **中文显示问题** | Chinese Display Issues
   - Web界面会自动处理中文显示
   - 命令行可能需要设置终端编码为UTF-8

4. **图形显示问题** | Graphics Display Issues
   - 使用Web界面获得最佳可视化效果
   - 命令行版本已简化图形功能

## 技术支持 | Technical Support

如有问题，请检查：
1. Python版本 >= 3.8
2. 所有依赖包已正确安装
3. 网络连接正常（用于Streamlit）

For issues, please check:
1. Python version >= 3.8
2. All dependencies properly installed
3. Network connection (for Streamlit)
