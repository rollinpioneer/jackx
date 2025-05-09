# -*- coding:utf-8 -*-
import re
import numpy as np
import nltk
import argparse
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba

# 下载NLTK必要资源
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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

# ------ 基线方法: TextRank摘要法 ------
def textrank_summarize(text, sentences_count=3):
    # 先清理输入文本中每个句子前的空格
    text = clean_sentence_spaces(text)
    
    try:
        # 中文分句处理
        sentences = re.split('([。！？])', text)
        sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2] + [''])]
        # 清理句子，去除空白句子和句子开头结尾的空格
        sentences = [sent.strip() for sent in sentences if len(sent.strip()) > 0]
        
        # 如果句子数少于请求摘要句子数，直接返回原文
        if len(sentences) <= sentences_count:
            return text
        
        # 创建句子的TF-IDF矩阵
        sentence_tokens = [' '.join(jieba.cut(sentence)) for sentence in sentences]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentence_tokens)
        
        # 计算句子间相似度
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # 计算TextRank分数
        scores = np.zeros(len(sentences))
        damping = 0.85  # 阻尼系数
        iterations = 30
        
        # 初始化得分
        for i in range(len(sentences)):
            scores[i] = 1.0 / len(sentences)
        
        # 进行TextRank迭代
        for _ in range(iterations):
            new_scores = np.zeros(len(sentences))
            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    if i != j and similarity_matrix[i, j] > 0:
                        new_scores[i] += (similarity_matrix[i, j] / np.sum(similarity_matrix[j])) * scores[j]
                new_scores[i] = (1 - damping) + damping * new_scores[i]
            scores = new_scores
        
        # 选择得分最高的句子
        ranked_indices = np.argsort(scores)[::-1]
        top_indices = sorted(ranked_indices[:sentences_count])
        
        # 生成摘要 - 使用正确的拼接方式防止空格问题
        summary = "。".join([sentences[i] for i in top_indices]) + "。"
        # 再次确保摘要中没有句子前的空格
        return clean_sentence_spaces(summary)
        
    except Exception as e:
        print(f"TextRank处理错误: {e}")
        # 最后的备选：返回文章的前几句，确保没有多余空格
        sentences = [s.strip() for s in text.split("。") if s.strip()][:sentences_count]
        return clean_sentence_spaces("。".join(sentences) + "。" if len(sentences) > 0 else text[:200])

def read_summarization_data():
    """从文件读取原文和Transformer摘要"""
    try:
        file_path = os.path.join(os.path.dirname(__file__), "media_mt5_summary.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 解析数据
        article_match = re.search(r'ARTICLE:(.*?)(?=MT5:|$)', content, re.DOTALL)
        mt5_match = re.search(r'MT5:(.*?)$', content, re.DOTALL)
        
        # 确保去除文本中每个句子开头的空格
        article = clean_sentence_spaces(article_match.group(1)) if article_match else ""
        mt5 = clean_sentence_spaces(mt5_match.group(1)) if mt5_match else ""
        
        return article, mt5
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return "", ""

def main(web_mode=False):
    # 读取原文和Transformer摘要
    article_text, mt5_summary = read_summarization_data()
    
    if not article_text:
        print("错误: 无法读取数据。请先运行media_mt5.py生成数据。")
        return
    
    if not web_mode:
        print("\n正在使用TextRank生成摘要...")
    
    # 生成基线摘要前清理原文中每个句子的空格
    article_text = clean_sentence_spaces(article_text)
    
    # 为社交媒体文本确定合适的摘要句子数 - 短文本摘要可以更简短
    sentences = re.split('([。！？])', article_text)
    sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2] + [''])]
    sentences = [sent.strip() for sent in sentences if len(sent.strip()) > 0]
    
    # 根据社交媒体文本长度确定摘要句子数 - 使用较小的比例
    sentences_count = max(1, min(3, int(len(sentences) * 0.15)))  # 至少1句，最多3句，或者原文15%的句子数
    
    textrank_summary = textrank_summarize(article_text, sentences_count=sentences_count)
    
    # 确保所有摘要的每个句子前没有空格
    summaries = {
        'Transformer': clean_sentence_spaces(mt5_summary),
        'TextRank': clean_sentence_spaces(textrank_summary)
    }
    
    if not web_mode:
        # 打印信息
        print("\n原文摘录:")
        print(article_text[:200] + "..." if len(article_text) > 200 else article_text)
        
        print("\n摘要结果:")
        for method, summary in summaries.items():
            print(f"\n{method}摘要:")
            print(summary)
        
        # 添加"是否继续迭代？"的提示
        while True:
            continue_choice = input("\n是否继续迭代？(y/n): ").strip().lower()
            if continue_choice == 'y':
                # 保存当前结果
                with open(os.path.join(os.path.dirname(__file__), "current_news.txt"), "w", encoding="utf-8") as f:
                    f.write(article_text)
                print("\n已保存当前文章，请运行media_mt5.py继续迭代！")
                break
            elif continue_choice == 'n':
                print("\n已完成摘要生成。")
                break
            else:
                print("无效输入，请输入'y'或'n'。")
    
    # 修改这里的输出格式，使其与app.py匹配的正则表达式一致
    output_file = os.path.join(os.path.dirname(__file__), "media_textrank_results.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"原文: {clean_sentence_spaces(article_text)}\n\n")
        f.write(f"Transformer摘要: {summaries['Transformer']}\n\n")
        f.write(f"TextRank: {summaries['TextRank']}\n\n")  # 改为 TextRank: 而不是 TextRank摘要:
    
    if not web_mode and continue_choice != 'y':
        print("\n摘要结果已保存到media_textrank_results.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成社交媒体TextRank摘要')
    parser.add_argument('--web_mode', action='store_true', help='是否以Web模式运行')
    args = parser.parse_args()
    
    main(web_mode=args.web_mode)