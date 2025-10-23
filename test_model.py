#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试脚本 | Simple Test Script
"""

import numpy as np
import pandas as pd

class SentimentThresholdModel:
    """简化版情感阈值融合模型"""
    
    def __init__(self, sigma: float = 0.2, theta_min: float = -0.95, theta_max: float = -0.55):
        self.sigma = sigma
        self.theta_min = theta_min
        self.theta_max = theta_max
    
    def calculate_normalized_difference(self, si: float, theta_k: float) -> float:
        """计算归一化差异度"""
        return abs(si - theta_k) / self.sigma
    
    def weight_function(self, d: float, alpha: float) -> float:
        """自适应权重函数"""
        return np.exp(-alpha * d)
    
    def calculate_adaptive_threshold(self, si: float, theta_k: float, alpha: float) -> dict:
        """计算自适应阈值"""
        # 计算归一化差异度
        d = self.calculate_normalized_difference(si, theta_k)
        
        # 计算权重
        w = self.weight_function(d, alpha)
        
        # 计算初始阈值
        if si < theta_k:
            theta_f_initial = theta_k
        else:
            theta_f_initial = theta_k + (1 - np.exp(-alpha * d)) * (si - theta_k)
        
        # 应用边界约束
        theta_f = np.clip(theta_f_initial, self.theta_min, self.theta_max)
        
        # 风险判定
        is_high_risk = si < theta_f
        
        return {
            'si': si,
            'theta_k': theta_k,
            'alpha': alpha,
            'normalized_difference': d,
            'weight': w,
            'theta_f_final': theta_f,
            'is_high_risk': is_high_risk
        }

def test_model():
    """测试模型"""
    print("=" * 60)
    print("情感阈值融合模型测试 | Sentiment Threshold Fusion Model Test")
    print("=" * 60)
    
    # 创建模型实例
    model = SentimentThresholdModel()
    
    # 测试数据
    test_cases = [
        {'si': -0.7, 'theta_k': -0.8, 'alpha': 1.2},
        {'si': -0.6, 'theta_k': -0.7, 'alpha': 1.0},
        {'si': -0.9, 'theta_k': -0.75, 'alpha': 0.8},
        {'si': -0.5, 'theta_k': -0.8, 'alpha': 1.5}
    ]
    
    print("\n测试结果 | Test Results:")
    print("-" * 60)
    
    for i, case in enumerate(test_cases, 1):
        result = model.calculate_adaptive_threshold(**case)
        
        print(f"\n测试案例 {i} | Test Case {i}:")
        print(f"  输入 | Input: Si={result['si']:.3f}, θk={result['theta_k']:.3f}, α={result['alpha']:.1f}")
        print(f"  差异度 | Difference: d={result['normalized_difference']:.3f}")
        print(f"  权重 | Weight: w={result['weight']:.3f}")
        print(f"  最终阈值 | Final Threshold: θf={result['theta_f_final']:.3f}")
        print(f"  风险判定 | Risk Assessment: {'高风险 | High Risk' if result['is_high_risk'] else '安全 | Safe'}")
    
    print("\n" + "=" * 60)
    print("测试完成 | Test Completed")
    print("=" * 60)
    
    return model

if __name__ == "__main__":
    test_model()
