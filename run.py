#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动脚本 | Launch Script
情感阈值融合模型 | Sentiment Threshold Fusion Model
"""

import subprocess
import sys
import os

def check_dependencies():
    """检查依赖包是否安装"""
    try:
        import streamlit
        import pandas
        import numpy
        import matplotlib
        import plotly
        print("✅ 所有依赖包已安装 | All dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖包 | Missing dependency: {e}")
        print("请运行: pip install -r requirements.txt")
        print("Please run: pip install -r requirements.txt")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("情感阈值融合模型启动器 | Sentiment Threshold Fusion Model Launcher")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    print("\n选择运行模式 | Choose running mode:")
    print("1. Web界面 | Web Interface (Streamlit)")
    print("2. 命令行演示 | Command Line Demo")
    print("3. 退出 | Exit")
    
    while True:
        choice = input("\n请输入选择 (1-3) | Please enter choice (1-3): ").strip()
        
        if choice == '1':
            print("\n🚀 启动Web界面... | Starting Web Interface...")
            try:
                subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
            except KeyboardInterrupt:
                print("\n👋 Web界面已关闭 | Web interface closed")
            except Exception as e:
                print(f"\n❌ 启动失败 | Launch failed: {e}")
            break
            
        elif choice == '2':
            print("\n🔬 运行命令行演示... | Running command line demo...")
            try:
                subprocess.run([sys.executable, "sentiment_threshold_model.py"], check=True)
            except Exception as e:
                print(f"\n❌ 运行失败 | Run failed: {e}")
            break
            
        elif choice == '3':
            print("\n👋 再见! | Goodbye!")
            break
            
        else:
            print("❌ 无效选择，请重新输入 | Invalid choice, please try again")

if __name__ == "__main__":
    main()
