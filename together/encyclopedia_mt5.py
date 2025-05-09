# -*- coding:utf-8 -*-
import re
import random
import argparse
import os
import pandas as pd
from openai import OpenAI

# 处理文本中每个句子前的空格
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

# 使用大模型API生成摘要
def generate_summary_with_api(text):
    try:
        client = OpenAI(api_key="sk-a17fda1fa3bf42e186d7e3868c88f3a8", base_url="https://api.deepseek.com")
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的中文文本摘要助手，针对百科知识内容生成简明摘要。生成的摘要应当清晰地概括主要概念和关键信息。"},
                {"role": "user", "content": f"请帮我对下面这段百科知识文本生成简明摘要：{text}"},
            ],
            stream=False
        )
        
        # 获取生成的摘要
        summary = response.choices[0].message.content.strip()
        return clean_sentence_spaces(summary)
    except Exception as e:
        print(f"调用API生成摘要时出错: {e}")
        # 如果API调用失败，返回一个简单的摘要（取文本的前部分）
        summary = text[:min(150, len(text))]
        if "。" in summary:
            summary = summary[:summary.rindex("。") + 1]
        return clean_sentence_spaces(summary)

def main(web_mode=False):
    # 决定是随机选择样本还是使用预先选择的样本
    if web_mode:
        # 从文件读取当前选择的文章
        try:
            current_file = os.path.join(os.path.dirname(__file__), "current_encyclopedia.txt")
            with open(current_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            article_match = re.search(r'ARTICLE:(.*?)(?=REFERENCE:|$)', content, re.DOTALL)
            reference_match = re.search(r'REFERENCE:(.*?)(?=INDEX:|$)', content, re.DOTALL)
            
            article_text = clean_sentence_spaces(article_match.group(1)) if article_match else ""
            reference_summary = clean_sentence_spaces(reference_match.group(1)) if reference_match else ""
            
            if not article_text:
                print("错误: 未能从文件中读取到文章，请先点击'随机选择文章'按钮")
                return
                
        except Exception as e:
            print(f"读取预选文章时出错: {e}")
            return
    else:
        # 加载百科数据集
        dataset_path = os.path.join(os.path.dirname(__file__), "..", "datasets/baike_wiki/train-00000-of-00001.parquet")
        
        # 读取parquet文件
        df = pd.read_parquet(dataset_path)
            
        print(f"百科数据集总条目数: {len(df)}")

        # 随机选择一个测试样本
        example_index = random.randint(0, len(df) - 1)
        print(f"随机选择的样本索引: {example_index}")
        
        sample = df.iloc[example_index]
        article_text = clean_sentence_spaces(sample["document"])  # 使用document键访问文章内容
        
        # 使用第一段作为参考摘要
        first_para_end = article_text.find("。") + 1
        reference_summary = article_text[:first_para_end] if first_para_end > 0 else article_text[:100]
    
    if not web_mode:
        print("原文：")
        print(article_text[:200] + "..." if len(article_text) > 200 else article_text)
        print("\n参考摘要：")
        print(reference_summary)

    # ------ 使用API生成摘要 ------
    print("\n正在使用大模型API生成摘要...")
    mt5_summary = generate_summary_with_api(article_text)

    if not web_mode:
        print("\nAPI生成的摘要：")
        print(mt5_summary)

    # 将结果保存到文件，供评估程序使用
    output_file = os.path.join(os.path.dirname(__file__), "encyclopedia_mt5_summary.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"ARTICLE:{clean_sentence_spaces(article_text)}\n")
        f.write(f"REFERENCE:{clean_sentence_spaces(reference_summary)}\n")
        f.write(f"MT5:{clean_sentence_spaces(mt5_summary)}\n")
    
    if not web_mode:
        print("\n数据已保存到encyclopedia_mt5_summary.txt文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成百科知识摘要')
    parser.add_argument('--web_mode', action='store_true', help='是否以Web模式运行')
    args = parser.parse_args()
    
    main(web_mode=args.web_mode)