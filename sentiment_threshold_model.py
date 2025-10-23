#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Si与θ_knowledge差异的情感阈值融合模型
Sentiment Threshold Fusion Model Based on Difference between Si and θ_knowledge

作者: 基于数学模型实现
Author: Based on Mathematical Model Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 如果字体设置失败，使用默认字体
    pass

class SentimentThresholdModel:
    """
    情感阈值融合模型
    Sentiment Threshold Fusion Model
    """
    
    def __init__(self, sigma: float = 0.2, theta_min: float = -0.95, theta_max: float = -0.55):
        """
        初始化模型参数
        Initialize model parameters
        
        Args:
            sigma: 标准差，用于归一化差异度 / Standard deviation for normalized difference
            theta_min: 最小阈值边界 / Minimum threshold boundary  
            theta_max: 最大阈值边界 / Maximum threshold boundary
        """
        self.sigma = sigma
        self.theta_min = theta_min
        self.theta_max = theta_max
        
        # 不同舆情类别的最佳α值 / Optimal α values for different categories
        self.category_alpha = {
            "意识形态安全与政治风险类": 0.8,
            "干部选拔任用与监督管理类": 1.2, 
            "党员教育管理监督类": 1.3,
            "人才工作与国际人才安全类": 1.4,
            "组织制度与政策执行类": 1.5,
            "组织系统自身建设与舆情应对类": 1.3,
            "区域与特殊领域党建类": 0.9,
            # English categories
            "ideological_security": 0.8,
            "cadre_selection": 1.2,
            "party_member_education": 1.3, 
            "talent_work": 1.4,
            "organizational_system": 1.5,
            "system_construction": 1.3,
            "regional_party_building": 0.9
        }
    
    def calculate_normalized_difference(self, si: float, theta_k: float) -> float:
        """
        计算归一化差异度
        Calculate normalized difference
        
        Args:
            si: 实际情感极性值 / Actual sentiment polarity value
            theta_k: 知识库阈值 / Knowledge base threshold
            
        Returns:
            归一化差异度 / Normalized difference
        """
        return abs(si - theta_k) / self.sigma
    
    def weight_function(self, d: float, alpha: float) -> float:
        """
        自适应权重函数
        Adaptive weight function
        
        Args:
            d: 归一化差异度 / Normalized difference
            alpha: 差异敏感系数 / Difference sensitivity coefficient
            
        Returns:
            权重值 / Weight value
        """
        return np.exp(-alpha * d)
    
    def calculate_adaptive_threshold(self, si: float, theta_k: float, alpha: float, 
                                   category: str = None, confidence: float = 1.0) -> Dict:
        """
        计算自适应阈值
        Calculate adaptive threshold
        
        Args:
            si: 实际情感极性值 / Actual sentiment polarity value
            theta_k: 知识库阈值 / Knowledge base threshold
            alpha: 差异敏感系数 / Difference sensitivity coefficient
            category: 舆情类别 / Category type
            confidence: 匹配置信度 / Matching confidence
            
        Returns:
            包含计算结果的字典 / Dictionary containing calculation results
        """
        # 输入验证 / Input validation
        if not (-1 <= si <= 1):
            raise ValueError("Si must be in range [-1, 1]")
        if not (-0.95 <= theta_k <= -0.55):
            raise ValueError("θk must be in range [-0.95, -0.55]")
        if alpha <= 0:
            raise ValueError("α must be positive")
        if not (0 <= confidence <= 1):
            raise ValueError("Confidence must be in range [0, 1]")
        
        # 计算归一化差异度 / Calculate normalized difference
        d = self.calculate_normalized_difference(si, theta_k)
        
        # 计算权重 / Calculate weight
        w = self.weight_function(d, alpha)
        
        # 计算初始阈值 / Calculate initial threshold
        if si < theta_k:
            # 当Si更负面时，保持政治安全底线 / Maintain political security baseline when Si is more negative
            theta_f_initial = theta_k
        else:
            # 使用融合公式 / Use fusion formula
            theta_f_initial = theta_k + (1 - np.exp(-alpha * d)) * (si - theta_k)
        
        # 应用边界约束 / Apply boundary constraints
        theta_f = np.clip(theta_f_initial, self.theta_min, self.theta_max)
        
        # 应用置信度调整 / Apply confidence adjustment
        theta_f_final = confidence * theta_f + (1 - confidence) * theta_k
        
        # 风险判定 / Risk assessment
        is_high_risk = si < theta_f_final
        
        return {
            'si': si,
            'theta_k': theta_k,
            'alpha': alpha,
            'normalized_difference': d,
            'weight': w,
            'theta_f_initial': theta_f_initial,
            'theta_f_clipped': theta_f,
            'theta_f_final': theta_f_final,
            'confidence': confidence,
            'is_high_risk': is_high_risk,
            'category': category,
            'risk_level': self._assess_risk_level(si, theta_f_final)
        }
    
    def _assess_risk_level(self, si: float, theta_f: float) -> str:
        """
        评估风险等级
        Assess risk level
        """
        if si < theta_f:
            diff = theta_f - si
            if diff > 0.3:
                return "极高风险 / Extremely High Risk"
            elif diff > 0.2:
                return "高风险 / High Risk" 
            elif diff > 0.1:
                return "中等风险 / Medium Risk"
            else:
                return "低风险 / Low Risk"
        else:
            return "安全 / Safe"
    
    def batch_calculate(self, data: List[Dict]) -> pd.DataFrame:
        """
        批量计算
        Batch calculation
        
        Args:
            data: 包含si, theta_k, alpha等参数的字典列表
                  List of dictionaries containing si, theta_k, alpha parameters
                  
        Returns:
            计算结果的DataFrame / DataFrame with calculation results
        """
        results = []
        for item in data:
            try:
                result = self.calculate_adaptive_threshold(**item)
                results.append(result)
            except Exception as e:
                print(f"Error processing item {item}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def sensitivity_analysis(self, si: float, theta_k: float, alpha_range: Tuple[float, float] = (0.5, 2.0), 
                           num_points: int = 50) -> pd.DataFrame:
        """
        参数敏感性分析
        Parameter sensitivity analysis
        """
        alphas = np.linspace(alpha_range[0], alpha_range[1], num_points)
        results = []
        
        for alpha in alphas:
            result = self.calculate_adaptive_threshold(si, theta_k, alpha)
            results.append({
                'alpha': alpha,
                'theta_f': result['theta_f_final'],
                'weight': result['weight'],
                'is_high_risk': result['is_high_risk']
            })
        
        return pd.DataFrame(results)
    
    def plot_weight_function(self, alpha_values: List[float] = [0.8, 1.2, 1.5], 
                           d_range: Tuple[float, float] = (0, 4), language: str = 'zh'):
        """
        绘制权重函数图
        Plot weight function
        """
        d_values = np.linspace(d_range[0], d_range[1], 100)
        
        plt.figure(figsize=(10, 6))
        for alpha in alpha_values:
            weights = [self.weight_function(d, alpha) for d in d_values]
            plt.plot(d_values, weights, label=f'α = {alpha}', linewidth=2)
        
        if language == 'zh':
            plt.xlabel('归一化差异度 d')
            plt.ylabel('权重 w(d)')
            plt.title('自适应权重函数')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.xlabel('Normalized Difference d')
            plt.ylabel('Weight w(d)')
            plt.title('Adaptive Weight Function')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return plt
    
    def plot_threshold_surface(self, si_range: Tuple[float, float] = (-1, 0.5),
                             theta_k_range: Tuple[float, float] = (-0.95, -0.55),
                             alpha: float = 1.2, language: str = 'zh'):
        """
        绘制阈值曲面图
        Plot threshold surface
        """
        si_values = np.linspace(si_range[0], si_range[1], 50)
        theta_k_values = np.linspace(theta_k_range[0], theta_k_range[1], 50)
        
        Si, Theta_k = np.meshgrid(si_values, theta_k_values)
        Theta_f = np.zeros_like(Si)
        
        for i in range(len(theta_k_values)):
            for j in range(len(si_values)):
                try:
                    result = self.calculate_adaptive_threshold(Si[i,j], Theta_k[i,j], alpha)
                    Theta_f[i,j] = result['theta_f_final']
                except:
                    Theta_f[i,j] = np.nan
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(Si, Theta_k, Theta_f, cmap='viridis', alpha=0.8)
        
        if language == 'zh':
            ax.set_xlabel('实际情感极性值 Si')
            ax.set_ylabel('知识库阈值 θk')
            ax.set_zlabel('最终阈值 θf')
            ax.set_title(f'自适应阈值曲面 (α = {alpha})')
        else:
            ax.set_xlabel('Actual Sentiment Si')
            ax.set_ylabel('Knowledge Threshold θk')
            ax.set_zlabel('Final Threshold θf')
            ax.set_title(f'Adaptive Threshold Surface (α = {alpha})')
        
        plt.colorbar(surf)
        plt.tight_layout()
        return plt
    
    def export_results(self, results: pd.DataFrame, filename: str):
        """
        导出结果
        Export results
        """
        results.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"Results exported to {filename}")

def demo_usage():
    """
    演示用法
    Demo usage
    """
    print("=" * 60)
    print("情感阈值融合模型演示 / Sentiment Threshold Fusion Model Demo")
    print("=" * 60)
    
    # 创建模型实例 / Create model instance
    model = SentimentThresholdModel()
    
    # 单个计算示例 / Single calculation example
    print("\n1. 单个计算示例 / Single Calculation Example:")
    print("-" * 40)
    
    si = -0.7
    theta_k = -0.8
    alpha = 1.2
    category = "组织制度与政策执行类"
    
    result = model.calculate_adaptive_threshold(si, theta_k, alpha, category)
    
    print(f"输入参数 / Input Parameters:")
    print(f"  实际情感极性值 Si: {si}")
    print(f"  知识库阈值 θk: {theta_k}")
    print(f"  差异敏感系数 α: {alpha}")
    print(f"  舆情类别: {category}")
    
    print(f"\n计算结果 / Calculation Results:")
    print(f"  归一化差异度 d: {result['normalized_difference']:.3f}")
    print(f"  权重 w(d): {result['weight']:.3f}")
    print(f"  最终阈值 θf: {result['theta_f_final']:.3f}")
    print(f"  风险判定: {result['is_high_risk']}")
    print(f"  风险等级: {result['risk_level']}")
    
    # 批量计算示例 / Batch calculation example
    print("\n2. 批量计算示例 / Batch Calculation Example:")
    print("-" * 40)
    
    test_data = [
        {'si': -0.6, 'theta_k': -0.7, 'alpha': 1.0, 'category': '意识形态安全与政治风险类'},
        {'si': -0.8, 'theta_k': -0.75, 'alpha': 1.2, 'category': '干部选拔任用与监督管理类'},
        {'si': -0.5, 'theta_k': -0.8, 'alpha': 1.5, 'category': '组织制度与政策执行类'},
        {'si': -0.9, 'theta_k': -0.7, 'alpha': 0.8, 'category': '区域与特殊领域党建类'}
    ]
    
    batch_results = model.batch_calculate(test_data)
    print(batch_results[['si', 'theta_k', 'alpha', 'theta_f_final', 'is_high_risk', 'risk_level']].to_string(index=False))
    
    # 敏感性分析示例 / Sensitivity analysis example
    print("\n3. 敏感性分析示例 / Sensitivity Analysis Example:")
    print("-" * 40)
    
    sensitivity_results = model.sensitivity_analysis(-0.7, -0.8)
    print(f"α值变化对最终阈值的影响:")
    print(f"α范围: {sensitivity_results['alpha'].min():.2f} - {sensitivity_results['alpha'].max():.2f}")
    print(f"θf范围: {sensitivity_results['theta_f'].min():.3f} - {sensitivity_results['theta_f'].max():.3f}")
    
    return model, result, batch_results

if __name__ == "__main__":
    model, result, batch_results = demo_usage()
    
    # 可选：生成可视化图表 / Optional: Generate visualization charts
    print("\n4. 可视化功能 / Visualization Features:")
    print("-" * 40)
    print("可视化功能已集成到Web界面中，请运行以下命令启动Web应用:")
    print("Visualization features are integrated into the web interface. Run the following command to start the web app:")
    print("streamlit run app.py")
    print("\n或使用启动脚本: python run.py")
    print("Or use the launcher script: python run.py")
    
    print("\n" + "=" * 60)
    print("演示完成 / Demo Completed")
    print("=" * 60)
