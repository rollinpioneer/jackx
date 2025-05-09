import os
import json
import random
import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 处理空白的函数
WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

# 定义模型路径（使用相对路径）
model_path = "../results/media-creative/final_model"

# 确保使用绝对路径
model_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
model_abs_path = model_abs_path.replace("\\", "/")  # 将反斜杠替换为正斜杠

# 检查模型路径是否存在
if not os.path.exists(model_abs_path):
    raise FileNotFoundError(f"模型路径不存在: {model_abs_path}")

# 定义数据集路径（使用相对路径）
dataset_path = "../datasets/media_LCSTS/test_public.jsonl"

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

# 读取测试数据集
print("正在加载测试数据集...")
with open(dataset_abs_path, 'r', encoding='utf-8') as f:
    test_data = [json.loads(line) for line in f]

# 随机选择一条测试数据
print("随机选择测试数据...")
test_sample = random.choice(test_data)

# 修改数据集字段名以适应新的格式
article = test_sample["text"]
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
print(f"生成摘要与原文比例: {len(generated_summary)/len(article)*100:.2f}%")