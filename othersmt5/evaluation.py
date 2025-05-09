import matplotlib
matplotlib.use('Agg')  # 设置为非交互模式

# 设置中文字体支持
from matplotlib import font_manager
import platform

# 根据操作系统选择合适的中文字体
system = platform.system()
if system == 'Windows':
    # Windows系统使用微软雅黑
    font_path = 'C:/Windows/Fonts/msyh.ttc'  # 微软雅黑字体路径
elif system == 'Darwin':
    # macOS系统使用苹方字体
    font_path = '/System/Library/Fonts/PingFang.ttc'
else:
    # Linux系统尝试使用文泉驿字体
    font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'

# 尝试设置中文字体
try:
    font_prop = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = 'sans-serif'
    if system == 'Windows':
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    elif system == 'Darwin':
        matplotlib.rcParams['font.sans-serif'] = ['PingFang SC']
    else:
        matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
except Exception as e:
    print(f"设置中文字体时出错: {e}")
    # 如果指定字体设置失败，尝试使用matplotlib内置的DejaVu Sans字体
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

import matplotlib.pyplot as plt
import numpy as np
import json
import re
from rouge import Rouge
import os
import jieba

def clean_sentence_spaces(text):
    # 按标点符号分句并去除每句开头的空格
    sentences = re.split('([。！？!?])', text)
    
    # 重组句子并确保每句开头没有空格
    result = ""
    for i in range(0, len(sentences), 2):
        if i < len(sentences):
            # 当前部分是句子内容
            current = sentences[i].strip()
            if current:  # 只有当句子不为空时才添加
                result += current
                
                # 添加标点符号（如果存在）
                if i+1 < len(sentences):
                    result += sentences[i+1]
    
    return result

def read_summary_data():
    """从文件读取原文、参考摘要和各种生成的摘要"""
    data = {
        "article": "",
        "reference": "",
        "mt5": "",
        "textrank": ""
    }
    
    # 读取mt5摘要
    if os.path.exists("c:\\Users\\jackx\\Desktop\\transformers\\othersmt5\\mt5_summary.txt"):
        with open("c:\\Users\\jackx\\Desktop\\transformers\\othersmt5\\mt5_summary.txt", "r", encoding="utf-8") as f:
            content = f.read()
        
        article_match = re.search(r'ARTICLE:(.*?)(?=REFERENCE:|$)', content, re.DOTALL)
        reference_match = re.search(r'REFERENCE:(.*?)(?=MT5:|$)', content, re.DOTALL)
        mt5_match = re.search(r'MT5:(.*?)$', content, re.DOTALL)
        
        if article_match:
            data["article"] = clean_sentence_spaces(article_match.group(1))
        if reference_match:
            data["reference"] = clean_sentence_spaces(reference_match.group(1))
        if mt5_match:
            data["mt5"] = clean_sentence_spaces(mt5_match.group(1))
    
    # 读取TextRank摘要
    if os.path.exists("c:\\Users\\jackx\\Desktop\\transformers\\othersmt5\\mt5TextRank_results.txt"):
        with open("c:\\Users\\jackx\\Desktop\\transformers\\othersmt5\\mt5TextRank_results.txt", "r", encoding="utf-8") as f:
            content = f.read()
        
        textrank_match = re.search(r'TextRank摘要: (.*?)(?:\n\n|$)', content, re.DOTALL)
        if textrank_match:
            data["textrank"] = clean_sentence_spaces(textrank_match.group(1))
    
    return data

def evaluate_rouge(hypothesis, reference):
    """计算ROUGE分数"""
    # 对中文文本进行分词
    hypothesis = ' '.join(jieba.cut(hypothesis))
    reference = ' '.join(jieba.cut(reference))
    
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)[0]
    
    return {
        'rouge1': scores['rouge-1']['f'],
        'rouge2': scores['rouge-2']['f'],
        'rougeL': scores['rouge-l']['f']
    }

def generate_evaluation_chart(scores):
    """生成评估结果的柱状图"""
    methods = list(scores.keys())
    rouge1_scores = [scores[method]['rouge1'] for method in methods]
    rouge2_scores = [scores[method]['rouge2'] for method in methods]
    rougeL_scores = [scores[method]['rougeL'] for method in methods]
    
    x = np.arange(len(methods))  # 方法标签的位置
    width = 0.25  # 柱状图的宽度
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    # 使用更高对比度的颜色
    rects1 = ax.bar(x - width, rouge1_scores, width, label='ROUGE-1', color='#1f77b4')
    rects2 = ax.bar(x, rouge2_scores, width, label='ROUGE-2', color='#ff7f0e')
    rects3 = ax.bar(x + width, rougeL_scores, width, label='ROUGE-L', color='#2ca02c')
    
    # 设置中文标题和标签，确保使用中文字体
    ax.set_title('摘要方法评估结果对比', fontsize=16)
    ax.set_xlabel('摘要方法', fontsize=14)
    ax.set_ylabel('ROUGE分数', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12)
    
    # 在柱状图上显示具体数值
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10)
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    # 调整布局，确保所有元素都可见
    fig.tight_layout()
    
    # 保存图表，使用更高的DPI以获得更清晰的文本
    plt.savefig('c:\\Users\\jackx\\Desktop\\transformers\\othersmt5\\eval_chart.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # 读取摘要数据
    data = read_summary_data()
    
    if not data["reference"]:
        print("错误: 未找到参考摘要。请先运行mt5Summary.py生成数据。")
        return
    
    # 存储评估结果的字典
    evaluation_results = {}
    
    # 评估每种摘要方法
    if data["mt5"]:
        evaluation_results["mT5"] = evaluate_rouge(data["mt5"], data["reference"])
        print(f"mT5摘要评估完成: ROUGE-1={evaluation_results['mT5']['rouge1']:.4f}, "
              f"ROUGE-2={evaluation_results['mT5']['rouge2']:.4f}, "
              f"ROUGE-L={evaluation_results['mT5']['rougeL']:.4f}")
    
    if data["textrank"]:
        evaluation_results["TextRank"] = evaluate_rouge(data["textrank"], data["reference"])
        print(f"TextRank摘要评估完成: ROUGE-1={evaluation_results['TextRank']['rouge1']:.4f}, "
              f"ROUGE-2={evaluation_results['TextRank']['rouge2']:.4f}, "
              f"ROUGE-L={evaluation_results['TextRank']['rougeL']:.4f}")
    
    # 如果有评估结果，生成图表
    if evaluation_results:
        # 生成评估图表
        generate_evaluation_chart(evaluation_results)
        print("评估图表已生成: eval_chart.png")
        
        # 将评估结果保存为JSON
        with open("c:\\Users\\jackx\\Desktop\\transformers\\othersmt5\\eval_results.json", "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
        print("评估结果已保存: eval_results.json")
    else:
        print("错误: 没有可用的摘要来评估。")

if __name__ == "__main__":
    main()