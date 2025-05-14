# -*- coding: utf-8 -*-
import json
import torch
import time
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from together.evaluate_summary import evaluate_summary
from openai import OpenAI

# 模型路径
model_path = r"C:\Users\jackx\.cache\huggingface\hub\models--csebuetnlp--mT5_multilingual_XLSum\snapshots\2437a524effdbadc327ced84595508f1e32025b3"

# DeepSeek API配置
deepseek_api_key = "sk-a17fda1fa3bf42e186d7e3868c88f3a8"
deepseek_base_url = "https://api.deepseek.com"

def load_model_and_tokenizer():
    """加载mT5模型和分词器"""
    print("正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("正在加载模型...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # 如果可用，将模型移至GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"模型已加载到 {device}")
    
    return model, tokenizer, device

def generate_summary(text, model, tokenizer, device, max_length=150):
    """使用mT5模型生成摘要"""
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
    return summary

def split_text_by_deepseek(text, num_segments=3):
    """
    使用DeepSeek模型将文本分割成指定数量的片段
    """
    print("使用DeepSeek模型进行文本分段...")
    
    # 初始化OpenAI客户端（使用DeepSeek API）
    client = OpenAI(api_key=deepseek_api_key, base_url=deepseek_base_url)
    
    # 构造分段请求
    prompt = f"""请将以下文本分成{num_segments}段，尽量保持每段的内容完整和连贯。
    不要添加任何评论、分析或额外内容，只需将原文分成{num_segments}段即可。
    使用"[段落X]"（X为段落编号）标记每段的开始。
    
    原文：
    {text}
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的中文文本处理助手，擅长进行文本分段。"},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        
        # 获取分段后的文本
        segmented_text = response.choices[0].message.content
        
        # 按标记分割文本
        segments = []
        for i in range(1, num_segments + 1):
            pattern = fr"\[段落{i}\](.*?)(?=\[段落{i+1}\]|$)"
            match = re.search(pattern, segmented_text, re.DOTALL)
            if match:
                segment = match.group(1).strip()
                segments.append(segment)
        
        # 检查是否获取到足够的段落
        if len(segments) < num_segments:
            # 如果DeepSeek返回的段落数不够，使用备用方法分割
            return split_text_by_sentence(text, num_segments)
        
        return segments
    except Exception as e:
        print(f"DeepSeek API调用失败：{e}")
        # 调用失败时使用备用方法
        return split_text_by_sentence(text, num_segments)

def split_text_by_sentence(text, num_segments=3):
    """
    备用方法：将长文本按句子分割成指定数量的片段
    用于DeepSeek API不可用时
    """
    print("使用句子分割方法进行文本分段...")
    
    # 首先按句子分割
    # 中文句子通常以句号、问号、感叹号结尾
    sentences = re.split(r'([。！？])', text)
    # 将分割符放回句子中
    sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2] + [''])]
    sentences = [s for s in sentences if s.strip()]  # 移除空句子
    
    # 如果句子数量太少，尝试按字符长度分割
    if len(sentences) < num_segments:
        # 计算每个片段应包含的平均字符数
        chars_per_segment = len(text) // num_segments
        segments = []
        for i in range(num_segments):
            start = i * chars_per_segment
            end = start + chars_per_segment if i < num_segments - 1 else len(text)
            segment = text[start:end]
            if segment.strip():  # 确保片段不为空
                segments.append(segment)
    else:
        # 如果句子数足够，则按句子分配到不同片段
        # 计算每个片段应包含的句子数
        sentences_per_segment = len(sentences) // num_segments
        extra_sentences = len(sentences) % num_segments
        
        segments = []
        start_idx = 0
        
        for i in range(num_segments):
            # 为前extra_sentences个片段多分配一个句子
            num_sentences = sentences_per_segment + (1 if i < extra_sentences else 0)
            end_idx = start_idx + num_sentences
            
            segment = ''.join(sentences[start_idx:end_idx])
            if segment.strip():  # 确保片段不为空
                segments.append(segment)
            start_idx = end_idx
    
    return segments

def hierarchical_summarize(text, model, tokenizer, device, num_segments=3):
    """
    分层摘要方法：
    1. 使用DeepSeek将长文本分割成指定数量的片段
    2. 为每个片段生成摘要（使用mT5）
    3. 将这些摘要合并，生成最终摘要（使用mT5）
    """
    # 使用DeepSeek分割文本
    segments = split_text_by_deepseek(text, num_segments)
    
    # 为每个非空片段生成摘要（使用mT5）
    intermediate_summaries = []
    for segment in segments:
        if segment.strip():  # 确保片段不为空
            summary = generate_summary(segment, model, tokenizer, device, max_length=100)
            intermediate_summaries.append(summary)
    
    # 将中间摘要合并起来
    combined_intermediate = '\n'.join(intermediate_summaries)
    
    # 对合并后的中间摘要再次生成最终摘要（使用mT5）
    final_summary = generate_summary(combined_intermediate, model, tokenizer, device, max_length=150)
    
    return final_summary, intermediate_summaries

def load_nlpcc_data(file_path, num_samples):
    """从NLPCC数据集加载前n个样本"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data.append(json.loads(line))
    return data

def main():
    # 加载模型和分词器
    model, tokenizer, device = load_model_and_tokenizer()
    
    # 加载NLPCC数据集
    print("正在加载NLPCC数据集...")
    dataset_path = "datasets/nlpcc_data.jsonl"
    nlpcc_data = load_nlpcc_data(dataset_path, num_samples=50)
    
    # 初始化结果存储
    results = []
    
    # 开始计时
    start_time = time.time()
    
    # 处理每个样本
    print(f"正在评估 {len(nlpcc_data)} 个样本...")
    for i, item in enumerate(tqdm(nlpcc_data)):
        # 提取文本和参考摘要
        text = item["text"]
        reference = item["summary"]
        
        # 生成层次化摘要，固定为3段
        final_summary, _ = hierarchical_summarize(text, model, tokenizer, device, num_segments=3)
        
        # 使用ROUGE评估
        evaluation = evaluate_summary(final_summary, reference)
        
        # 存储结果
        result = {
            "sample_id": i,
            "rouge_1_f": evaluation["scores"]["rouge-1"]["f"],
            "rouge_2_f": evaluation["scores"]["rouge-2"]["f"],
            "rouge_l_f": evaluation["scores"]["rouge-l"]["f"]
        }
        results.append(result)
    
    # 计算总耗时
    total_time = time.time() - start_time
    print(f"\n总耗时: {total_time:.2f} 秒")
    
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
        "num_samples": len(results),
        "total_time": total_time
    }
    
    # 打印平均分数
    print("\n平均评估分数:")
    print(f"ROUGE-1 F1: {avg_rouge1*100:.2f}")
    print(f"ROUGE-2 F1: {avg_rouge2*100:.2f}")
    print(f"ROUGE-L F1: {avg_rougel*100:.2f}")
    
    # 将最终平均结果保存为JSON
    with open("ceng_mt5_nlpcc_rouge_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    print("ROUGE结果已保存至 ceng_mt5_nlpcc_rouge_results.json")

if __name__ == "__main__":
    main()