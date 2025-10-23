#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情感阈值融合模型Web应用
Sentiment Threshold Fusion Model Web Application

基于Si与θ_knowledge差异的情感阈值融合模型
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from sentiment_threshold_model import SentimentThresholdModel

# 页面配置
st.set_page_config(
    page_title="情感阈值融合模型 | Sentiment Threshold Fusion Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 语言配置
LANGUAGES = {
    'zh': {
        'title': '基于$S_i$与$\\theta_{knowledge}$差异的情感阈值融合模型',
        'subtitle': '精准风险评估方案',
        'language_selector': '选择语言',
        'model_params': '模型参数',
        'si_input': '实际情感极性值 $S_i$',
        'theta_k_input': '知识库阈值 $\\theta_k$',
        'alpha_input': '差异敏感系数 $\\alpha$',
        'category_input': '舆情类别',
        'confidence_input': '匹配置信度',
        'calculate_btn': '计算阈值',
        'results_title': '计算结果',
        'visualization_title': '可视化分析',
        'batch_analysis': '批量分析',
        'sensitivity_analysis': '敏感性分析',
        'model_description': '模型说明',
        'about_model': '关于模型',
        'normalized_diff': '归一化差异度',
        'weight_value': '权重值',
        'final_threshold': '最终阈值',
        'risk_assessment': '风险判定',
        'risk_level': '风险等级',
        'high_risk': '高风险',
        'safe': '安全',
        'categories': {
            '意识形态安全与政治风险类': '意识形态安全与政治风险类',
            '干部选拔任用与监督管理类': '干部选拔任用与监督管理类',
            '党员教育管理监督类': '党员教育管理监督类',
            '人才工作与国际人才安全类': '人才工作与国际人才安全类',
            '组织制度与政策执行类': '组织制度与政策执行类',
            '组织系统自身建设与舆情应对类': '组织系统自身建设与舆情应对类',
            '区域与特殊领域党建类': '区域与特殊领域党建类'
        }
    },
    'en': {
        'title': 'Sentiment Threshold Fusion Model Based on Difference between $S_i$ and $\\theta_{knowledge}$',
        'subtitle': 'Precise Risk Assessment Solution',
        'language_selector': 'Select Language',
        'model_params': 'Model Parameters',
        'si_input': 'Actual Sentiment Polarity $S_i$',
        'theta_k_input': 'Knowledge Base Threshold $\\theta_k$',
        'alpha_input': 'Difference Sensitivity Coefficient $\\alpha$',
        'category_input': 'Category Type',
        'confidence_input': 'Matching Confidence',
        'calculate_btn': 'Calculate Threshold',
        'results_title': 'Calculation Results',
        'visualization_title': 'Visualization Analysis',
        'batch_analysis': 'Batch Analysis',
        'sensitivity_analysis': 'Sensitivity Analysis',
        'model_description': 'Model Description',
        'about_model': 'About the Model',
        'normalized_diff': 'Normalized Difference',
        'weight_value': 'Weight Value',
        'final_threshold': 'Final Threshold',
        'risk_assessment': 'Risk Assessment',
        'risk_level': 'Risk Level',
        'high_risk': 'High Risk',
        'safe': 'Safe',
        'categories': {
            'ideological_security': 'Ideological Security & Political Risk',
            'cadre_selection': 'Cadre Selection & Supervision',
            'party_member_education': 'Party Member Education & Management',
            'talent_work': 'Talent Work & International Security',
            'organizational_system': 'Organizational System & Policy Implementation',
            'system_construction': 'System Construction & Public Opinion Response',
            'regional_party_building': 'Regional & Special Field Party Building'
        }
    }
}

def init_session_state():
    """初始化会话状态"""
    if 'language' not in st.session_state:
        st.session_state.language = 'zh'
    if 'model' not in st.session_state:
        st.session_state.model = SentimentThresholdModel()

def get_text(key):
    """获取当前语言的文本"""
    return LANGUAGES[st.session_state.language].get(key, key)

def create_header():
    """创建页面头部"""
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.title(get_text('title'))
        st.markdown(f"**{get_text('subtitle')}**")
    
    with col2:
        # 语言选择器
        language_options = {'中文': 'zh', 'English': 'en'}
        selected_lang = st.selectbox(
            get_text('language_selector'),
            options=list(language_options.keys()),
            index=0 if st.session_state.language == 'zh' else 1
        )
        st.session_state.language = language_options[selected_lang]

def create_sidebar():
    """创建侧边栏"""
    st.sidebar.header(get_text('model_params'))
    
    # 输入参数
    si = st.sidebar.slider(
        get_text('si_input'),
        min_value=-1.0,
        max_value=1.0,
        value=-0.7,
        step=0.01,
        help="实际情感极性值，负值表示负面情感强度" if st.session_state.language == 'zh' 
             else "Actual sentiment polarity value, negative values indicate negative sentiment intensity"
    )
    
    theta_k = st.sidebar.slider(
        get_text('theta_k_input'),
        min_value=-0.95,
        max_value=-0.55,
        value=-0.8,
        step=0.01,
        help="知识库匹配阈值，代表政策规定的风险临界值" if st.session_state.language == 'zh'
             else "Knowledge base threshold representing policy-defined risk critical value"
    )
    
    # 类别选择
    categories = get_text('categories')
    category_key = st.sidebar.selectbox(
        get_text('category_input'),
        options=list(categories.keys())
    )
    
    # 根据类别自动设置alpha值
    if st.session_state.language == 'zh':
        alpha_default = st.session_state.model.category_alpha.get(category_key, 1.2)
    else:
        alpha_default = st.session_state.model.category_alpha.get(category_key, 1.2)
    
    alpha = st.sidebar.slider(
        get_text('alpha_input'),
        min_value=0.1,
        max_value=3.0,
        value=alpha_default,
        step=0.1,
        help="控制权重衰减速率的参数" if st.session_state.language == 'zh'
             else "Parameter controlling weight decay rate"
    )
    
    confidence = st.sidebar.slider(
        get_text('confidence_input'),
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.01,
        help="知识库匹配质量的置信度" if st.session_state.language == 'zh'
             else "Confidence in knowledge base matching quality"
    )
    
    return si, theta_k, alpha, category_key, confidence

def display_results(result):
    """显示计算结果"""
    st.subheader(get_text('results_title'))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            get_text('normalized_diff'),
            f"{result['normalized_difference']:.3f}"
        )
    
    with col2:
        st.metric(
            get_text('weight_value'),
            f"{result['weight']:.3f}"
        )
    
    with col3:
        st.metric(
            get_text('final_threshold'),
            f"{result['theta_f_final']:.3f}"
        )
    
    with col4:
        risk_status = get_text('high_risk') if result['is_high_risk'] else get_text('safe')
        st.metric(
            get_text('risk_assessment'),
            risk_status
        )
    
    # 详细结果表格
    if st.session_state.language == 'zh':
        results_df = pd.DataFrame({
            '参数': ['实际情感极性值 Si', '知识库阈值 θk', '差异敏感系数 α', '归一化差异度 d', 
                   '权重 w(d)', '初始阈值 θf_initial', '边界约束后 θf_clipped', '最终阈值 θf_final',
                   '匹配置信度', '风险判定', '风险等级'],
            '数值': [
                f"{result['si']:.3f}",
                f"{result['theta_k']:.3f}",
                f"{result['alpha']:.3f}",
                f"{result['normalized_difference']:.3f}",
                f"{result['weight']:.3f}",
                f"{result['theta_f_initial']:.3f}",
                f"{result['theta_f_clipped']:.3f}",
                f"{result['theta_f_final']:.3f}",
                f"{result['confidence']:.3f}",
                "是" if result['is_high_risk'] else "否",
                result['risk_level']
            ]
        })
    else:
        results_df = pd.DataFrame({
            'Parameter': ['Actual Sentiment Si', 'Knowledge Threshold θk', 'Sensitivity Coefficient α', 
                         'Normalized Difference d', 'Weight w(d)', 'Initial Threshold θf_initial', 
                         'Clipped Threshold θf_clipped', 'Final Threshold θf_final',
                         'Matching Confidence', 'High Risk', 'Risk Level'],
            'Value': [
                f"{result['si']:.3f}",
                f"{result['theta_k']:.3f}",
                f"{result['alpha']:.3f}",
                f"{result['normalized_difference']:.3f}",
                f"{result['weight']:.3f}",
                f"{result['theta_f_initial']:.3f}",
                f"{result['theta_f_clipped']:.3f}",
                f"{result['theta_f_final']:.3f}",
                f"{result['confidence']:.3f}",
                "Yes" if result['is_high_risk'] else "No",
                result['risk_level']
            ]
        })
    
    st.dataframe(results_df, use_container_width=True)

def create_visualizations(si, theta_k, alpha):
    """创建可视化图表"""
    st.subheader(get_text('visualization_title'))
    
    tab1, tab2, tab3 = st.tabs([
        "权重函数 | Weight Function",
        "敏感性分析 | Sensitivity Analysis", 
        "阈值曲面 | Threshold Surface"
    ])
    
    with tab1:
        # 权重函数图
        d_values = np.linspace(0, 4, 100)
        alpha_values = [0.8, 1.2, 1.5, alpha]
        
        fig = go.Figure()
        
        for a in alpha_values:
            weights = [st.session_state.model.weight_function(d, a) for d in d_values]
            label = f'α = {a}' + (' (当前 | Current)' if a == alpha else '')
            fig.add_trace(go.Scatter(
                x=d_values,
                y=weights,
                mode='lines',
                name=label,
                line=dict(width=3 if a == alpha else 2)
            ))
        
        fig.update_layout(
            title="自适应权重函数 | Adaptive Weight Function",
            xaxis_title="归一化差异度 d | Normalized Difference d",
            yaxis_title="权重 w(d) | Weight w(d)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # 敏感性分析
        sensitivity_results = st.session_state.model.sensitivity_analysis(si, theta_k)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("最终阈值变化 | Final Threshold Change", "权重变化 | Weight Change")
        )
        
        fig.add_trace(
            go.Scatter(
                x=sensitivity_results['alpha'],
                y=sensitivity_results['theta_f'],
                mode='lines+markers',
                name='θf',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=sensitivity_results['alpha'],
                y=sensitivity_results['weight'],
                mode='lines+markers',
                name='w(d)',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        # 添加当前α值的垂直线
        fig.add_vline(x=alpha, line_dash="dash", line_color="green", 
                     annotation_text=f"当前α | Current α = {alpha}")
        
        fig.update_layout(
            title="参数敏感性分析 | Parameter Sensitivity Analysis",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # 阈值曲面图（简化版）
        si_range = np.linspace(-1, 0.5, 20)
        theta_k_range = np.linspace(-0.95, -0.55, 20)
        
        Si_mesh, Theta_k_mesh = np.meshgrid(si_range, theta_k_range)
        Theta_f_mesh = np.zeros_like(Si_mesh)
        
        for i in range(len(theta_k_range)):
            for j in range(len(si_range)):
                try:
                    result = st.session_state.model.calculate_adaptive_threshold(
                        Si_mesh[i,j], Theta_k_mesh[i,j], alpha
                    )
                    Theta_f_mesh[i,j] = result['theta_f_final']
                except:
                    Theta_f_mesh[i,j] = np.nan
        
        fig = go.Figure(data=[go.Surface(
            x=Si_mesh,
            y=Theta_k_mesh,
            z=Theta_f_mesh,
            colorscale='viridis'
        )])
        
        fig.update_layout(
            title=f"自适应阈值曲面 (α = {alpha}) | Adaptive Threshold Surface (α = {alpha})",
            scene=dict(
                xaxis_title="实际情感极性值 Si | Actual Sentiment Si",
                yaxis_title="知识库阈值 θk | Knowledge Threshold θk",
                zaxis_title="最终阈值 θf | Final Threshold θf"
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_batch_analysis():
    """创建批量分析功能"""
    st.subheader(get_text('batch_analysis'))
    
    # 示例数据
    if st.session_state.language == 'zh':
        sample_data = [
            {'si': -0.6, 'theta_k': -0.7, 'alpha': 1.0, 'category': '意识形态安全与政治风险类'},
            {'si': -0.8, 'theta_k': -0.75, 'alpha': 1.2, 'category': '干部选拔任用与监督管理类'},
            {'si': -0.5, 'theta_k': -0.8, 'alpha': 1.5, 'category': '组织制度与政策执行类'},
            {'si': -0.9, 'theta_k': -0.7, 'alpha': 0.8, 'category': '区域与特殊领域党建类'}
        ]
    else:
        sample_data = [
            {'si': -0.6, 'theta_k': -0.7, 'alpha': 1.0, 'category': 'ideological_security'},
            {'si': -0.8, 'theta_k': -0.75, 'alpha': 1.2, 'category': 'cadre_selection'},
            {'si': -0.5, 'theta_k': -0.8, 'alpha': 1.5, 'category': 'organizational_system'},
            {'si': -0.9, 'theta_k': -0.7, 'alpha': 0.8, 'category': 'regional_party_building'}
        ]
    
    # 文件上传或使用示例数据
    uploaded_file = st.file_uploader(
        "上传CSV文件 | Upload CSV File" if st.session_state.language == 'zh' else "Upload CSV File",
        type=['csv']
    )
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file).to_dict('records')
        except Exception as e:
            st.error(f"文件读取错误 | File reading error: {e}")
            data = sample_data
    else:
        if st.button("使用示例数据 | Use Sample Data"):
            data = sample_data
        else:
            return
    
    # 批量计算
    try:
        batch_results = st.session_state.model.batch_calculate(data)
        
        # 显示结果
        st.dataframe(batch_results, use_container_width=True)
        
        # 结果可视化
        fig = px.scatter(
            batch_results,
            x='si',
            y='theta_f_final',
            color='is_high_risk',
            size='normalized_difference',
            hover_data=['category', 'risk_level'],
            title="批量分析结果 | Batch Analysis Results"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 下载结果
        csv = batch_results.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="下载结果 | Download Results",
            data=csv,
            file_name="batch_analysis_results.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"批量计算错误 | Batch calculation error: {e}")

def create_model_description():
    """创建模型说明"""
    st.subheader(get_text('about_model'))
    
    if st.session_state.language == 'zh':
        st.markdown("""
        ### 模型概述
        
        本模型提出一种差异驱动的情感阈值自适应融合模型，通过数学严谨的方式解决舆情风险评估中"原则性"与"灵活性"的平衡问题。
        
        ### 核心思想
        
        - **差异驱动**: 基于实际情感极性值$S_i$与知识库阈值$\\theta_k$的差异度$d$
        - **自适应权重**: 使用指数衰减函数$w(d) = e^{-\\alpha \\cdot d}$动态调整权重
        - **政治安全底线**: 当$S_i < \\theta_k$时，保持$\\theta_f = \\theta_k$，确保政治安全
        
        ### 数学公式
        
        1. **归一化差异度**: $d = \\frac{|S_i - \\theta_k|}{\\sigma}$，其中$\\sigma = 0.2$
        
        2. **自适应权重函数**: $w(d) = e^{-\\alpha \\cdot d}$
        
        3. **最终阈值计算**: 
        $$\\theta_f = \\begin{cases}
        \\theta_k, & \\text{if } S_i < \\theta_k \\\\
        \\theta_k + (1 - e^{-\\alpha \\cdot d}) \\cdot (S_i - \\theta_k), & \\text{if } S_i \\geq \\theta_k
        \\end{cases}$$
        
        ### 模型特性
        
        - **连续性**: 在$S_i = \\theta_k$处连续
        - **单调性**: 输出随输入单调不减
        - **边界约束**: $\\theta_f \\in [\\theta_{\\min}, \\theta_{\\max}]$
        """)
    else:
        st.markdown("""
        ### Model Overview
        
        This model proposes a difference-driven adaptive sentiment threshold fusion model that mathematically balances "principle" and "flexibility" in public opinion risk assessment.
        
        ### Core Ideas
        
        - **Difference-Driven**: Based on the difference $d$ between actual sentiment polarity $S_i$ and knowledge base threshold $\\theta_k$
        - **Adaptive Weighting**: Uses exponential decay function $w(d) = e^{-\\alpha \\cdot d}$ for dynamic weight adjustment
        - **Political Security Baseline**: When $S_i < \\theta_k$, maintain $\\theta_f = \\theta_k$ to ensure political security
        
        ### Mathematical Formulas
        
        1. **Normalized Difference**: $d = \\frac{|S_i - \\theta_k|}{\\sigma}$, where $\\sigma = 0.2$
        
        2. **Adaptive Weight Function**: $w(d) = e^{-\\alpha \\cdot d}$
        
        3. **Final Threshold Calculation**: 
        $$\\theta_f = \\begin{cases}
        \\theta_k, & \\text{if } S_i < \\theta_k \\\\
        \\theta_k + (1 - e^{-\\alpha \\cdot d}) \\cdot (S_i - \\theta_k), & \\text{if } S_i \\geq \\theta_k
        \\end{cases}$$
        
        ### Model Properties
        
        - **Continuity**: Continuous at $S_i = \\theta_k$
        - **Monotonicity**: Output is monotonically non-decreasing with input
        - **Boundary Constraints**: $\\theta_f \\in [\\theta_{\\min}, \\theta_{\\max}]$
        """)

def main():
    """主函数"""
    init_session_state()
    create_header()
    
    # 侧边栏参数输入
    si, theta_k, alpha, category_key, confidence = create_sidebar()
    
    # 计算按钮
    if st.sidebar.button(get_text('calculate_btn'), type="primary"):
        try:
            result = st.session_state.model.calculate_adaptive_threshold(
                si, theta_k, alpha, category_key, confidence
            )
            
            # 显示结果
            display_results(result)
            
            # 可视化
            create_visualizations(si, theta_k, alpha)
            
        except Exception as e:
            st.error(f"计算错误 | Calculation Error: {e}")
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs([
        get_text('batch_analysis'),
        get_text('model_description'),
        "使用说明 | Instructions"
    ])
    
    with tab1:
        create_batch_analysis()
    
    with tab2:
        create_model_description()
    
    with tab3:
        if st.session_state.language == 'zh':
            st.markdown("""
            ### 使用说明
            
            1. **参数设置**: 在左侧边栏调整模型参数
            2. **单次计算**: 点击"计算阈值"按钮进行单次计算
            3. **批量分析**: 在"批量分析"标签页上传CSV文件或使用示例数据
            4. **可视化**: 查看权重函数、敏感性分析和阈值曲面图
            5. **语言切换**: 右上角可切换中英文界面
            
            ### CSV文件格式
            
            批量分析需要包含以下列的CSV文件：
            - `si`: 实际情感极性值
            - `theta_k`: 知识库阈值  
            - `alpha`: 差异敏感系数
            - `category`: 舆情类别（可选）
            """)
        else:
            st.markdown("""
            ### Instructions
            
            1. **Parameter Setting**: Adjust model parameters in the left sidebar
            2. **Single Calculation**: Click "Calculate Threshold" button for single calculation
            3. **Batch Analysis**: Upload CSV file or use sample data in "Batch Analysis" tab
            4. **Visualization**: View weight function, sensitivity analysis, and threshold surface plots
            5. **Language Switch**: Switch between Chinese and English interface in the top right corner
            
            ### CSV File Format
            
            Batch analysis requires a CSV file with the following columns:
            - `si`: Actual sentiment polarity value
            - `theta_k`: Knowledge base threshold
            - `alpha`: Difference sensitivity coefficient
            - `category`: Category type (optional)
            """)

if __name__ == "__main__":
    main()
