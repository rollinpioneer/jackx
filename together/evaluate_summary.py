#!/usr/bin/env python
# -*- coding:utf-8 -*-
import re
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 处理文本中每个句子前的空格
def clean_sentence_spaces(text):
    if not text:
        return ""
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

def segment_text(text):
    """使用jieba分词对文本进行分词"""
    return " ".join(jieba.cut(text))

def evaluate_summary(summary, reference):
    """评估摘要质量，计算ROUGE和BLEU分数"""
    if not summary:
        return {
            "status": "error",
            "message": "摘要内容为空",
            "scores": {}
        }
    
    if not reference or not reference.strip():
        return {
            "status": "error",
            "message": "参考摘要为空，无法进行评估",
            "scores": {}
        }
    
    try:
        # 分词
        summary_seg = segment_text(summary)
        reference_seg = segment_text(reference)
        
        # ROUGE评分
        rouge = Rouge()
        scores = rouge.get_scores(summary_seg, reference_seg)[0]
        
        # BLEU评分
        smoothie = SmoothingFunction().method1
        summary_tokens = list(jieba.cut(summary))
        reference_tokens = list(jieba.cut(reference))
        bleu_score = sentence_bleu([reference_tokens], summary_tokens, smoothing_function=smoothie)
        
        # 计算关键词覆盖率
        unique_ref_tokens = set([token for token in reference_tokens if len(token) > 1])
        unique_sum_tokens = set([token for token in summary_tokens if len(token) > 1])
        
        # 关键词覆盖率
        keyword_coverage = len(unique_sum_tokens.intersection(unique_ref_tokens)) / len(unique_ref_tokens) if unique_ref_tokens else 0
        
        # 长度比例评分
        length_ratio = min(len(summary) / len(reference), 1.0) if reference else 0
        
        # 简洁性评分（句子平均长度的反向值）
        summary_sentences = re.split('[。！？!?]', summary)
        avg_sent_len = np.mean([len(s.strip()) for s in summary_sentences if s.strip()]) if summary_sentences else 0
        conciseness = 1.0 if avg_sent_len == 0 else min(20.0 / avg_sent_len, 1.0)
        
        # 综合评分
        comprehensive_score = (
            scores['rouge-1']['f'] * 0.3 + 
            scores['rouge-2']['f'] * 0.2 + 
            scores['rouge-l']['f'] * 0.2 + 
            bleu_score * 0.1 + 
            keyword_coverage * 0.1 + 
            length_ratio * 0.05 + 
            conciseness * 0.05
        )
        
        # 保存评分结果
        evaluation_results = {
            "rouge-1": {"f": float(scores['rouge-1']['f']), "p": float(scores['rouge-1']['p']), "r": float(scores['rouge-1']['r'])},
            "rouge-2": {"f": float(scores['rouge-2']['f']), "p": float(scores['rouge-2']['p']), "r": float(scores['rouge-2']['r'])},
            "rouge-l": {"f": float(scores['rouge-l']['f']), "p": float(scores['rouge-l']['p']), "r": float(scores['rouge-l']['r'])},
            "bleu": float(bleu_score),
            "keyword_coverage": float(keyword_coverage),
            "length_ratio": float(length_ratio),
            "conciseness": float(conciseness),
            "comprehensive_score": float(comprehensive_score)
        }
        
        # 生成评估图表
        generate_evaluation_chart(evaluation_results)
        
        return {
            "status": "success",
            "message": "评估完成",
            "scores": evaluation_results
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {
            "status": "error",
            "message": f"评估过程中出错: {str(e)}\n{error_details}",
            "scores": {}
        }

def evaluate_summary_without_reference(summary, original_text):
    """无参考摘要时评估摘要质量，基于原文进行评估"""
    if not summary:
        return {
            "status": "error", 
            "message": "摘要内容为空",
            "scores": {}
        }
    
    if not original_text:
        return {
            "status": "error",
            "message": "原文内容为空，无法进行评估",
            "scores": {}
        }
    
    try:
        # 分词
        summary_tokens = list(jieba.cut(summary))
        original_tokens = list(jieba.cut(original_text))
        
        # 计算TF-IDF权重
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([segment_text(original_text)])
            feature_names = vectorizer.get_feature_names_out()
            
            # 提取原文中的关键词（TF-IDF值最高的词）
            tfidf_scores = zip(feature_names, tfidf_matrix.toarray()[0])
            sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
            keywords = [word for word, score in sorted_scores[:min(20, len(sorted_scores))]]
            
        except:
            # 如果TF-IDF处理失败，则使用词频
            word_counts = {}
            for token in original_tokens:
                if len(token) > 1:  # 只考虑长度大于1的词
                    word_counts[token] = word_counts.get(token, 0) + 1
            
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            keywords = [word for word, count in sorted_words[:min(20, len(sorted_words))]]
        
        # 计算关键词覆盖率
        unique_orig_keywords = set(keywords)
        unique_sum_tokens = set([token for token in summary_tokens if len(token) > 1])
        keyword_coverage = len(unique_sum_tokens.intersection(unique_orig_keywords)) / len(unique_orig_keywords) if unique_orig_keywords else 0
        
        # 计算信息密度 - 摘要中非停用词的比例
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        summary_content_words = [w for w in summary_tokens if w not in stopwords and len(w) > 1]
        information_density = len(summary_content_words) / len(summary_tokens) if summary_tokens else 0
        
        # 摘要长度比例 - 理想的摘要长度约为原文的10-20%
        ideal_ratio = 0.15  # 理想的摘要/原文长度比
        actual_ratio = len(summary) / len(original_text) if original_text else 0
        length_ratio = 1.0 - min(abs(actual_ratio - ideal_ratio) / ideal_ratio, 1.0)
        
        # 简洁性评分（句子平均长度的合理性）
        summary_sentences = re.split('[。！？!?]', summary)
        avg_sent_len = np.mean([len(s.strip()) for s in summary_sentences if s.strip()]) if summary_sentences else 0
        # 理想的平均句子长度为15-25个字
        if avg_sent_len == 0:
            conciseness = 0.5
        elif avg_sent_len < 10:  # 太短
            conciseness = 0.7 * (avg_sent_len / 10)
        elif 10 <= avg_sent_len <= 30:  # 适中
            conciseness = 1.0 - abs(20 - avg_sent_len) / 20
        else:  # 太长
            conciseness = 0.7 * (60 / avg_sent_len) if avg_sent_len <= 60 else 0.7 * (60 / avg_sent_len)
        conciseness = max(0.1, min(conciseness, 1.0))
        
        # 计算句子的连贯性 - 使用余弦相似度
        summary_sentences = [s for s in re.split('[。！？!?]', summary) if s.strip()]
        if len(summary_sentences) > 1:
            sentence_vectors = []
            for sent in summary_sentences:
                # 为每个句子创建一个词袋向量
                sent_tokens = list(jieba.cut(sent))
                sent_vector = {}
                for token in sent_tokens:
                    if len(token) > 1:
                        sent_vector[token] = sent_vector.get(token, 0) + 1
                sentence_vectors.append(sent_vector)
            
            # 计算相邻句子的连贯性分数
            coherence_scores = []
            for i in range(len(sentence_vectors) - 1):
                # 计算当前句子与下一句子的相似度
                v1, v2 = sentence_vectors[i], sentence_vectors[i + 1]
                # 找出两个句子共有的词
                common_words = set(v1.keys()) & set(v2.keys())
                if not common_words:  # 没有共同词
                    coherence_scores.append(0.1)  # 设置一个低但非零的基线分数
                    continue
                
                # 计算点积
                dot_product = sum(v1[w] * v2[w] for w in common_words)
                # 计算模长
                mag_v1 = np.sqrt(sum(v1[w]**2 for w in v1))
                mag_v2 = np.sqrt(sum(v2[w]**2 for w in v2))
                
                # 计算相似度
                similarity = dot_product / (mag_v1 * mag_v2) if mag_v1 * mag_v2 > 0 else 0
                coherence_scores.append(similarity)
            
            # 取平均值作为总体连贯性分数
            coherence = np.mean(coherence_scores) if coherence_scores else 0.5
        else:
            # 只有一个句子，连贯性设为中等值
            coherence = 0.5
        
        # 计算摘要的多样性 - 基于词汇丰富度
        if summary_tokens:
            unique_tokens = set([t for t in summary_tokens if len(t) > 1])
            lexical_diversity = min(len(unique_tokens) / len(summary_tokens), 1.0)
        else:
            lexical_diversity = 0
        
        # 综合评分
        comprehensive_score = (
            keyword_coverage * 0.35 +  # 关键词覆盖率权重最高
            information_density * 0.20 +
            length_ratio * 0.15 +
            conciseness * 0.10 +
            coherence * 0.10 +
            lexical_diversity * 0.10
        )
        
        # 保存评分结果 - 使用统一的字段名便于前端显示
        evaluation_results = {
            "keyword_coverage": float(keyword_coverage),
            "information_density": float(information_density),  
            "coherence": float(coherence),
            "lexical_diversity": float(lexical_diversity),
            "length_ratio": float(length_ratio),
            "conciseness": float(conciseness),
            "comprehensive_score": float(comprehensive_score),
            # 保留兼容字段，但不在雷达图中显示
            "rouge-1": {"f": float(keyword_coverage), "p": float(keyword_coverage), "r": float(keyword_coverage)},
            "rouge-2": {"f": float(information_density), "p": float(information_density), "r": float(information_density)},
            "rouge-l": {"f": float(coherence), "p": float(coherence), "r": float(coherence)},
            "bleu": float(lexical_diversity)
        }
        
        # 生成评估图表 - 使用自定义指标显示
        generate_evaluation_chart_no_reference(evaluation_results)
        
        return {
            "status": "success",
            "message": "无参考摘要评估完成",
            "scores": evaluation_results
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {
            "status": "error",
            "message": f"评估过程中出错: {str(e)}\n{error_details}",
            "scores": {}
        }

def generate_evaluation_chart(scores):
    """生成评估分数的雷达图"""
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 支持中文
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 提取要在雷达图中显示的指标 - 只显示三个指标
    metrics = [
        '关键词覆盖率', '长度比例', '简洁性'
    ]
    
    # 对应的值
    values = [
        scores['keyword_coverage'],
        scores['length_ratio'],
        scores['conciseness']
    ]
    
    # 创建雷达图
    plt.figure(figsize=(10, 8))
    
    # 创建极坐标子图
    ax = plt.subplot(111, polar=True)
    
    # 设置角度（每个指标的位置）
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    values.append(values[0])  # 闭合雷达图
    angles.append(angles[0])  # 闭合雷达图
    metrics.append(metrics[0])  # 闭合标签
    
    # 绘制雷达图
    ax.plot(angles, values, 'o-', linewidth=2, label='评分')
    ax.fill(angles, values, alpha=0.25)
    
    # 设置标签
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics[:-1])
    
    # 设置网格的一圈圈同心圆
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    
    # 添加标题
    plt.title('摘要质量评估', size=20, y=1.1, fontproperties='SimHei')
    
    # 设置y轴最大值
    ax.set_ylim(0, 1)
    
    # 保存图表
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'summary_eval_chart.png'), dpi=100)
    plt.close()
    
    # 在JSON文件中保存原始分数，前端可以根据需要决定显示方式
    with open(os.path.join(os.path.dirname(__file__), 'summary_eval_results.json'), 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

def generate_evaluation_chart_no_reference(scores):
    """生成无参考摘要评估分数的雷达图"""
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 支持中文
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 提取要在雷达图中显示的指标 - 显示所有自评估指标
    metrics = [
        '关键词覆盖率', '信息密度', '连贯性', 
        '词汇丰富度', '长度比例', '简洁性'
    ]
    values = [
        scores['keyword_coverage'],
        scores['information_density'],
        scores['coherence'],
        scores['lexical_diversity'],
        scores['length_ratio'], 
        scores['conciseness']
    ]
    
    # 创建雷达图
    plt.figure(figsize=(10, 8))
    
    # 创建极坐标子图
    ax = plt.subplot(111, polar=True)
    
    # 设置角度（每个指标的位置）
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    values.append(values[0])  # 闭合雷达图
    angles.append(angles[0])  # 闭合雷达图
    metrics.append(metrics[0])  # 闭合标签
    
    # 绘制雷达图
    ax.plot(angles, values, 'o-', linewidth=2, label='评分')
    ax.fill(angles, values, alpha=0.25)
    
    # 设置标签
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics[:-1])
    
    # 设置网格的一圈圈同心圆
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    
    # 添加标题
    plt.title('摘要质量自评估', size=20, y=1.1, fontproperties='SimHei')
    
    # 设置y轴最大值
    ax.set_ylim(0, 1)
    
    # 保存图表
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'summary_eval_chart.png'), dpi=100)
    plt.close()
    
    # 保存评估结果到JSON文件
    with open(os.path.join(os.path.dirname(__file__), 'summary_eval_results.json'), 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

def evaluate_from_file(file_path, summary_type='mt5'):
    """从文件中读取文本和摘要，进行评估"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取原文、参考摘要和生成摘要
        article_match = re.search(r'ARTICLE:(.*?)(?=REFERENCE:|$)', content, re.DOTALL)
        reference_match = re.search(r'REFERENCE:(.*?)(?=MT5:|TextRank:|Transformer:|$)', content, re.DOTALL)
        
        article_text = clean_sentence_spaces(article_match.group(1)) if article_match else ""
        reference_summary = clean_sentence_spaces(reference_match.group(1)) if reference_match else ""
        
        # 根据摘要类型提取相应的摘要内容
        if summary_type.lower() == 'mt5':
            # 匹配 MT5: 或 Transformer: 开头的摘要
            summary_match = re.search(r'(?:MT5|Transformer):(.*?)(?=TextRank:|$)', content, re.DOTALL)
        else:  # TextRank
            # 匹配 TextRank: 开头的摘要，并允许多种可能的格式
            summary_match = re.search(r'TextRank:(.*?)(?=MT5:|Transformer:|$|\Z)', content, re.DOTALL)
            
            # 如果没有匹配到，再尝试其他可能的格式
            if not summary_match:
                summary_match = re.search(r'TextRank摘要:(.*?)(?=MT5:|Transformer:|$|\Z)', content, re.DOTALL)
        
        generated_summary = clean_sentence_spaces(summary_match.group(1)) if summary_match else ""
        
        if not generated_summary:
            return {
                "status": "error",
                "message": f"未找到{summary_type}摘要",
                "scores": {}
            }
        
        # 如果有参考摘要就使用标准评估，否则使用无参考评估
        if reference_summary and reference_summary.strip():
            return evaluate_summary(generated_summary, reference_summary)
        else:
            # 使用无参考摘要评估方法
            return evaluate_summary_without_reference(generated_summary, article_text)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {
            "status": "error",
            "message": f"从文件读取摘要时出错: {str(e)}\n{error_details}",
            "scores": {}
        }

def main():
    parser = argparse.ArgumentParser(description='评估摘要质量')
    parser.add_argument('--file', type=str, help='摘要文件路径')
    parser.add_argument('--type', type=str, default='mt5', choices=['mt5', 'textrank'], help='摘要类型')
    args = parser.parse_args()
    
    if args.file:
        results = evaluate_from_file(args.file, args.type)
        print(json.dumps(results, ensure_ascii=False, indent=4))
    else:
        print("请提供摘要文件路径")

if __name__ == "__main__":
    main()