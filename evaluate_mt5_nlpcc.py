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
        
        # 生成摘要
        generated_summary = generate_summary(text, model, tokenizer, device)
        
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
    with open("mt5_nlpcc_rouge_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    print("ROUGE结果已保存至 mt5_nlpcc_rouge_results.json")

if __name__ == "__main__":
    main()