# -*- coding: utf-8 -*-
"""
评估抽取-生成式摘要在NLPCC数据集上的表现
使用TextRank进行抽取，mT5进行生成
"""

import json
import torch
import numpy as np
import re
import jieba
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from together.evaluate_summary import evaluate_summary

# mT5模型路径
MODEL_PATH = r"C:\Users\jackx\.cache\huggingface\hub\models--csebuetnlp--mT5_multilingual_XLSum\snapshots\2437a524effdbadc327ced84595508f1e32025b3"

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

# TextRank抽取式摘要
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
            return "。".join(sentences) + "。"
        
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
        
        # 返回选出的句子组成的文本
        selected_text = "。".join([sentences[i] for i in top_indices]) + "。"
        return selected_text
        
    except Exception as e:
        # 发生错误时返回文章的前几句
        sentences = [s.strip() for s in text.split("。") if s.strip()][:sentences_count]
        return "。".join(sentences) + "。" if sentences else text[:200]

# 使用mT5模型生成摘要
def mt5_summarize(text, model, tokenizer, device, max_length=150):
    try:
        inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(device)
        
        # 生成摘要
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                num_beams=4,
                min_length=30,
                max_length=max_length,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        # 解码生成的摘要
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return clean_sentence_spaces(summary)
    except Exception as e:
        return ""

# 抽取-生成联合摘要
def extractive_abstractive_summarize(text, model, tokenizer, device, sentences_count=5):
    # 1. 抽取阶段: 使用TextRank抽取关键句子
    extracted_text = textrank_summarize(text, sentences_count=sentences_count)
    
    # 2. 生成阶段: 使用mT5基于抽取内容生成摘要
    abstractive_summary = mt5_summarize(extracted_text, model, tokenizer, device)
    
    return abstractive_summary

# 加载NLPCC数据集前50条
def load_nlpcc_data(file_path, num_samples=50):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data.append(json.loads(line))
    return data

def main():
    print("正在评估抽取-生成式摘要方法...")
    
    # 加载模型和分词器
    print("正在加载mT5模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"模型已加载到 {device}")
    
    # 加载NLPCC数据集
    print("正在加载NLPCC数据集...")
    dataset_path = "datasets/nlpcc_data.jsonl"
    nlpcc_data = load_nlpcc_data(dataset_path)
    
    # 初始化结果存储
    results = []
    
    # 处理每个样本
    print(f"正在评估 {len(nlpcc_data)} 个样本...")
    for i, item in enumerate(tqdm(nlpcc_data)):
        # 提取文本和参考摘要
        text = item["text"]
        reference = item["summary"]
        
        # 生成摘要 (使用5句句子进行抽取)
        generated_summary = extractive_abstractive_summarize(
            text, model, tokenizer, device, sentences_count=5
        )
        
        # 使用ROUGE评估
        evaluation = evaluate_summary(generated_summary, reference)
        
        # 存储结果
        result = {
            "sample_id": i,
            "rouge_1_f": evaluation["scores"]["rouge-1"]["f"],
            "rouge_2_f": evaluation["scores"]["rouge-2"]["f"],
            "rouge_l_f": evaluation["scores"]["rouge-l"]["f"]
        }
        results.append(result)
    
    # 计算平均分数
    avg_rouge1 = sum(r["rouge_1_f"] for r in results) / len(results)
    avg_rouge2 = sum(r["rouge_2_f"] for r in results) / len(results)
    avg_rougel = sum(r["rouge_l_f"] for r in results) / len(results)
    
    # 创建最终结果字典，ROUGE值乘以100
    final_results = {
        "rouge-1": {
            "f": float(avg_rouge1 * 100)
        },
        "rouge-2": {
            "f": float(avg_rouge2 * 100)
        },
        "rouge-l": {
            "f": float(avg_rougel * 100)
        },
        "num_samples": len(results)
    }
    
    # 打印平均分数
    print("\n抽取-生成式摘要方法评估结果:")
    print(f"ROUGE-1 F1: {avg_rouge1*100:.2f}")
    print(f"ROUGE-2 F1: {avg_rouge2*100:.2f}")
    print(f"ROUGE-L F1: {avg_rougel*100:.2f}")
    
    # 将最终平均结果保存为JSON
    with open("extractive_abstractive_rouge_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    print("ROUGE结果已保存至 extractive_abstractive_rouge_results.json")

if __name__ == "__main__":
    main()