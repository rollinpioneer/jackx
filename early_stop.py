# -*- coding: utf-8 -*-
import json
import torch
import time
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, StoppingCriteriaList, StoppingCriteria
from tqdm import tqdm
from together.evaluate_summary import evaluate_summary

# 模型路径
model_path = r"C:\Users\jackx\.cache\huggingface\hub\models--csebuetnlp--mT5_multilingual_XLSum\snapshots\2437a524effdbadc327ced84595508f1e32025b3"

class KeywordsStoppingCriteria(StoppingCriteria):
    """
    自定义停止条件：当生成特定关键词时提前停止
    """
    def __init__(self, tokenizer, stop_tokens, device):
        self.tokenizer = tokenizer
        self.stop_tokens_ids = [tokenizer.encode(token, add_special_tokens=False) for token in stop_tokens]
        self.device = device
        
    def __call__(self, input_ids, scores, **kwargs):
        for stop_ids in self.stop_tokens_ids:
            if input_ids[0][-len(stop_ids):].tolist() == stop_ids:
                return True
        return False

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

def generate_summary_baseline(text, model, tokenizer, device, max_length=150):
    """early_stopping=True"""
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(device)
    
    # 开始计时
    start_time = time.time()
    
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
    
    # 结束计时
    generation_time = time.time() - start_time
    
    # 解码生成的摘要
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary, generation_time

def generate_summary_no_early_stop(text, model, tokenizer, device, max_length=150):
    """不使用早停策略生成摘要"""
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(device)
    
    # 开始计时
    start_time = time.time()
    
    # 生成摘要（early_stopping设为False）
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            min_length=30,
            max_length=max_length,
            no_repeat_ngram_size=2,
            early_stopping=False
        )
    
    # 结束计时
    generation_time = time.time() - start_time
    
    # 解码生成的摘要
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary, generation_time

def generate_summary_keyword_stop(text, model, tokenizer, device, max_length=150):
    """使用自定义关键词停止条件生成摘要"""
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(device)
    
    # 定义可能表示摘要结束的停止词
    stop_words = ["。", "！", "？", ".", "!", "?"]
    stopping_criteria = KeywordsStoppingCriteria(tokenizer, stop_words, device)
    
    # 开始计时
    start_time = time.time()
    
    # 生成摘要（使用自定义停止条件）
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            min_length=30,
            max_length=max_length,
            no_repeat_ngram_size=2,
            stopping_criteria=StoppingCriteriaList([stopping_criteria])
        )
    
    # 结束计时
    generation_time = time.time() - start_time
    
    # 解码生成的摘要
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary, generation_time

def load_nlpcc_data(file_path, num_samples):
    """从NLPCC数据集加载前n个样本"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data.append(json.loads(line))
    return data

def run_experiment(strategy, model, tokenizer, device, samples):
    """运行指定策略的实验"""
    results = []
    
    for i, item in enumerate(tqdm(samples, desc=f"生成摘要（{strategy}）")):
        text = item["text"]
        reference = item["summary"]
        
        # 选择摘要生成策略
        if strategy == "baseline":
            summary, gen_time = generate_summary_baseline(text, model, tokenizer, device)
        elif strategy == "no_early_stop":
            summary, gen_time = generate_summary_no_early_stop(text, model, tokenizer, device)
        elif strategy == "keyword_stop":
            summary, gen_time = generate_summary_keyword_stop(text, model, tokenizer, device)
            
        # 评估摘要质量
        evaluation = evaluate_summary(summary, reference)
        
        # 存储结果
        result = {
            "sample_id": i,
            "summary": summary,
            "generation_time": gen_time,
            "rouge_1_f": evaluation["scores"]["rouge-1"]["f"],
            "rouge_2_f": evaluation["scores"]["rouge-2"]["f"],
            "rouge_l_f": evaluation["scores"]["rouge-l"]["f"]
        }
        results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="评估早停策略对摘要生成的影响")
    parser.add_argument("--samples", type=int, default=50, help="要处理的样本数量")
    args = parser.parse_args()
    
    # 加载模型和分词器
    model, tokenizer, device = load_model_and_tokenizer()
    
    # 加载NLPCC数据集
    print("正在加载NLPCC数据集...")
    dataset_path = "datasets/nlpcc_data.jsonl"
    nlpcc_data = load_nlpcc_data(dataset_path, num_samples=args.samples)
    
    # 运行不同的实验策略
    strategies = ["baseline", "no_early_stop", "keyword_stop"]
    all_results = {}
    
    for strategy in strategies:
        print(f"\n开始实验：{strategy}")
        start_time = time.time()
        
        results = run_experiment(strategy, model, tokenizer, device, nlpcc_data)
        
        # 计算平均时间和分数
        avg_time = sum(r["generation_time"] for r in results) / len(results)
        avg_rouge1 = sum(r["rouge_1_f"] for r in results) / len(results)
        avg_rouge2 = sum(r["rouge_2_f"] for r in results) / len(results)
        avg_rougel = sum(r["rouge_l_f"] for r in results) / len(results)
        
        # 存储结果摘要
        all_results[strategy] = {
            "avg_generation_time": avg_time,
            "rouge-1": {"f": float(avg_rouge1 * 100)},
            "rouge-2": {"f": float(avg_rouge2 * 100)},
            "rouge-l": {"f": float(avg_rougel * 100)},
            "total_time": time.time() - start_time,
            "num_samples": len(results)
        }
        
        # 打印平均分数
        print(f"\n{strategy} 策略评估结果:")
        print(f"平均生成时间: {avg_time:.4f} 秒/样本")
        print(f"ROUGE-1 F1: {avg_rouge1*100:.2f}")
        print(f"ROUGE-2 F1: {avg_rouge2*100:.2f}")
        print(f"ROUGE-L F1: {avg_rougel*100:.2f}")
    
    # 打印对比结果
    print("\n\n早停策略对比结果:")
    print("-" * 60)
    print(f"{'策略':<15} {'平均生成时间(秒)':<20} {'ROUGE-L F1':<12}")
    print("-" * 60)
    
    for strategy, result in all_results.items():
        print(f"{strategy:<15} {result['avg_generation_time']:<20.4f} {result['rouge-l']['f']:<12.2f}")
    
    # 计算加速比
    baseline_time = all_results["baseline"]["avg_generation_time"]
    for strategy in strategies[1:]:
        speedup = baseline_time / all_results[strategy]["avg_generation_time"]
        print(f"\n{strategy} 相比 baseline 的加速比: {speedup:.2f}x")
    
    # 将详细结果保存为JSON
    with open("early_stop_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    print("\n实验结果已保存至 early_stop_experiment_results.json")

if __name__ == "__main__":
    main()