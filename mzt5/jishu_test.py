import os
import random
import torch
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

# 处理空白的函数
WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

# 定义模型路径（使用相对路径）
model_path = "../results/baike_wiki-creative/final_model"

# 确保使用绝对路径
model_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
model_abs_path = model_abs_path.replace("\\", "/")  # 将反斜杠替换为正斜杠

# 检查模型路径是否存在
if not os.path.exists(model_abs_path):
    raise FileNotFoundError(f"模型路径不存在: {model_abs_path}")

# 定义数据集路径（使用相对路径）
dataset_path = "../datasets/baike_wiki"

# 确保使用绝对路径
dataset_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), dataset_path))
dataset_abs_path = dataset_abs_path.replace("\\", "/")  # 将反斜杠替换为正斜杠

# 检查数据集路径是否存在
if not os.path.exists(dataset_abs_path):
    raise FileNotFoundError(f"数据集路径不存在: {dataset_abs_path}")

# 加载模型和分词器
print("正在加载模型和分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_abs_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_abs_path)
model.eval()  # 设置为评估模式

# 读取数据集
print("正在加载数据集...")
dataset = load_dataset("parquet", data_files={
    "train": os.path.join(dataset_abs_path, "train-00000-of-00001.parquet")
})["train"]

# 提取document和summary字段
def extract_document_summary(example):
    article = example['article']
    if isinstance(article, dict):
        document = article.get('document', [])
        summary = article.get('summary', [])
        
        if isinstance(document, list):
            document = ' '.join(document)
        if isinstance(summary, list):
            summary = ' '.join(summary)
            
        return {
            'document': document,
            'summary': summary
        }
    else:
        return {
            'document': '',
            'summary': ''
        }

# 应用转换函数
processed_dataset = dataset.map(extract_document_summary)

# 为了测试，与jishu.py中相同的方式分割数据集
print("分割数据集为训练集和测试集...")
train_test_split = processed_dataset.train_test_split(test_size=0.1, seed=42)
test_dataset = train_test_split["test"]

print(f"测试集大小: {len(test_dataset)} 样本")

# 随机选择一条测试数据
print("随机选择测试数据...")
idx = random.randint(0, len(test_dataset) - 1)
test_sample = test_dataset[idx]

article = test_sample["document"]
reference_summary = test_sample.get("summary", "无参考摘要")

# 确保参考摘要不为空
reference_summary = reference_summary if reference_summary.strip() else "无参考摘要"

print("\n原始文章:")
print(article[:500] + "..." if len(article) > 500 else article)
print("\n参考摘要:")
print(reference_summary)

# 预处理文本
cleaned_article = WHITESPACE_HANDLER(article)

# 对文章进行分词和编码
inputs = tokenizer(
    cleaned_article,
    return_tensors="pt",
    max_length=512,
    truncation=True
)

# 使用模型生成摘要
with torch.no_grad():
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

# 解码生成的摘要
generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

print("\n生成的摘要:")
print(generated_summary)

# 打印统计信息
print("\n统计信息:")
print(f"原文长度: {len(article)} 字符")
print(f"生成摘要长度: {len(generated_summary)} 字符")
print(f"参考摘要长度: {len(reference_summary)} 字符")
print(f"生成摘要与原文比例: {len(generated_summary)/len(article)*100:.2f}%" if len(article) > 0 else "原文为空，无法计算比例")