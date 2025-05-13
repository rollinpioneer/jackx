# -*- coding: utf-8 -*-
import json
import torch
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from together.evaluate_summary import evaluate_summary

# 模型路径
model_path = r"C:\Users\jackx\.cache\huggingface\hub\models--csebuetnlp--mT5_multilingual_XLSum\snapshots\2437a524effdbadc327ced84595508f1e32025b3"

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

def generate_summary_batch(texts, model, tokenizer, device, max_length=150, batch_size=8):
    """使用批处理和动态填充生成摘要"""
    all_summaries = []
    
    # 按批次处理文本
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # 使用动态填充，仅在批次内填充到最长序列长度
        inputs = tokenizer(
            batch_texts,
            max_length=512,
            truncation=True,
            padding='longest',  # 动态填充到批次内最长序列
            return_tensors="pt"
        ).to(device)
        
        # 生成摘要
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_beams=4,
                min_length=30,
                max_length=max_length,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        # 解码生成的摘要
        batch_summaries = [tokenizer.decode(ids, skip_special_tokens=True) for ids in summary_ids]
        all_summaries.extend(batch_summaries)
    
    return all_summaries

def generate_summary_batch_uniform_length(texts, model, tokenizer, device, max_length=150, batch_size=8):
    """使用均匀长度批处理和动态填充生成摘要"""
    # 计算每个文本的标记长度
    text_lengths = []
    for text in texts:
        tokens = tokenizer.encode(text, truncation=True, max_length=512)
        text_lengths.append(len(tokens))
    
    # 将文本按长度排序
    texts_with_lengths = list(zip(texts, text_lengths, range(len(texts))))
    texts_with_lengths.sort(key=lambda x: x[1])
    
    # 保持原始索引的映射
    sorted_texts = [t[0] for t in texts_with_lengths]
    original_indices = [t[2] for t in texts_with_lengths]
    
    # 按批次处理排序后的文本
    all_summaries = [""] * len(texts)  # 预分配空间
    
    for i in range(0, len(sorted_texts), batch_size):
        batch_texts = sorted_texts[i:i+batch_size]
        batch_indices = original_indices[i:i+batch_size]
        
        # 使用动态填充
        inputs = tokenizer(
            batch_texts,
            max_length=512,
            truncation=True,
            padding='longest',  # 动态填充到批次内最长序列
            return_tensors="pt"
        ).to(device)
        
        # 生成摘要
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_beams=4,
                min_length=30,
                max_length=max_length,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        # 解码生成的摘要并放回原始位置
        batch_summaries = [tokenizer.decode(ids, skip_special_tokens=True) for ids in summary_ids]
        for j, summary in enumerate(batch_summaries):
            all_summaries[batch_indices[j]] = summary
    
    return all_summaries

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
    
    # 提取文本和参考摘要
    texts = [item["text"] for item in nlpcc_data]
    references = [item["summary"] for item in nlpcc_data]
    
    # 初始化结果存储
    results_batch = []
    results_uniform = []
    
    # 1. 测试批处理方法（动态填充）
    print("\n测试批处理方法（动态填充）...")
    start_time_batch = time.time()
    
    # 批量生成摘要
    batch_size = 8
    generated_summaries = generate_summary_batch(texts, model, tokenizer, device, batch_size=batch_size)
    
    # 评估每个摘要
    for i, (generated_summary, reference) in enumerate(zip(generated_summaries, references)):
        evaluation = evaluate_summary(generated_summary, reference)
        
        result = {
            "sample_id": i,
            "rouge_1_f": evaluation["scores"]["rouge-1"]["f"],
            "rouge_2_f": evaluation["scores"]["rouge-2"]["f"],
            "rouge_l_f": evaluation["scores"]["rouge-l"]["f"]
        }
        results_batch.append(result)
    
    batch_time = time.time() - start_time_batch
    print(f"批处理方法用时: {batch_time:.2f} 秒")
    
    # 2. 测试均匀长度批处理方法
    print("\n测试均匀长度批处理方法...")
    start_time_uniform = time.time()
    
    # 批量生成摘要（使用均匀长度批处理）
    generated_summaries = generate_summary_batch_uniform_length(texts, model, tokenizer, device, batch_size=batch_size)
    
    # 评估每个摘要
    for i, (generated_summary, reference) in enumerate(zip(generated_summaries, references)):
        evaluation = evaluate_summary(generated_summary, reference)
        
        result = {
            "sample_id": i,
            "rouge_1_f": evaluation["scores"]["rouge-1"]["f"],
            "rouge_2_f": evaluation["scores"]["rouge-2"]["f"],
            "rouge_l_f": evaluation["scores"]["rouge-l"]["f"]
        }
        results_uniform.append(result)
    
    uniform_time = time.time() - start_time_uniform
    print(f"均匀长度批处理方法用时: {uniform_time:.2f} 秒")
    print(f"相比普通批处理速度提升: {batch_time / uniform_time:.2f}x")
    
    # 计算各方法的平均ROUGE分数
    methods = [
        ("批处理方法", results_batch),
        ("均匀长度批处理", results_uniform)
    ]
    
    final_results = {
        "timing": {
            "batch_time": batch_time,
            "uniform_time": uniform_time,
            "uniform_speedup": batch_time / uniform_time
        },
        "rouge_scores": {}
    }
    
    for method_name, results in methods:
        avg_rouge1 = sum(r["rouge_1_f"] for r in results) / len(results)
        avg_rouge2 = sum(r["rouge_2_f"] for r in results) / len(results)
        avg_rougel = sum(r["rouge_l_f"] for r in results) / len(results)
        
        print(f"\n{method_name}平均评估分数:")
        print(f"ROUGE-1 F1: {avg_rouge1*100:.2f}")
        print(f"ROUGE-2 F1: {avg_rouge2*100:.2f}")
        print(f"ROUGE-L F1: {avg_rougel*100:.2f}")
        
        final_results["rouge_scores"][method_name] = {
            "rouge-1": {"f": float(avg_rouge1 * 100)},
            "rouge-2": {"f": float(avg_rouge2 * 100)},
            "rouge-l": {"f": float(avg_rougel * 100)}
        }
    
    # 将最终结果保存为JSON
    with open("p_mt5_nlpcc_rouge_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    print("\n结果已保存至 p_mt5_nlpcc_rouge_results.json")

if __name__ == "__main__":
    main()