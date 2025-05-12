# -*- coding: utf-8 -*-
import json
import time
from tqdm import tqdm
from openai import OpenAI
from together.evaluate_summary import evaluate_summary

def setup_client():
    """初始化DeepSeek API客户端"""
    client = OpenAI(api_key="sk-a17fda1fa3bf42e186d7e3868c88f3a8", base_url="https://api.deepseek.com")
    return client

def generate_summary(text, client, max_retries=3, retry_delay=2):
    """使用DeepSeek API生成摘要"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的中文文本摘要助手，只输出简明摘要。"},
                    {"role": "user", "content": f"请帮我对下面这段文本生成简明摘要：{text}"},
                ],
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"请求出错 (尝试 {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print("达到最大重试次数，返回空摘要")
                return ""

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
    # 初始化DeepSeek API客户端
    print("正在初始化DeepSeek API客户端...")
    client = setup_client()
    
    # 加载NLPCC数据集
    print("正在加载NLPCC数据集...")
    dataset_path = "datasets/nlpcc_data.jsonl"
    nlpcc_data = load_nlpcc_data(dataset_path, num_samples=50)  # 与mT5评估相同样本数
    
    # 初始化结果存储
    results = []
    
    # 处理每个样本
    print(f"正在评估 {len(nlpcc_data)} 个样本...")
    for i, item in enumerate(tqdm(nlpcc_data)):
        # 提取文本和参考摘要
        text = item["text"]
        reference = item["summary"]
        
        # 生成摘要
        generated_summary = generate_summary(text, client)
        
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
        
        # 每10个样本打印一次进度以及当前摘要示例
        if i % 10 == 0:
            print(f"\n示例 {i}:")
            print(f"生成的摘要: {generated_summary[:100]}...")
            print(f"参考摘要: {reference[:100]}...")
        
        # 请求之间添加短暂延迟，避免API请求过于频繁
        time.sleep(0.5)
    
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
    print("\n平均评估分数:")
    print(f"ROUGE-1 F1: {avg_rouge1*100:.2f}")
    print(f"ROUGE-2 F1: {avg_rouge2*100:.2f}")
    print(f"ROUGE-L F1: {avg_rougel*100:.2f}")
    
    # 将最终平均结果保存为JSON
    with open("dmx_nlpcc_rouge_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    print("ROUGE结果已保存至 dmx_nlpcc_rouge_results.json")

if __name__ == "__main__":
    main()