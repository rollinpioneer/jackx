from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from datasets import load_dataset
import os

# 定义模型路径（使用相对路径）
model_path = "../models/BERT"

# 确保使用绝对路径
model_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
model_abs_path = model_abs_path.replace("\\", "/")  # 将反斜杠替换为正斜杠

# 检查模型路径是否存在
if not os.path.exists(model_abs_path):
    raise FileNotFoundError(f"模型路径不存在: {model_abs_path}")

# 确保路径兼容性
try:
    # 使用Auto类从本地加载模型和分词器
    auto_model = AutoModel.from_pretrained(model_abs_path)
    auto_tokenizer = AutoTokenizer.from_pretrained(model_abs_path)
except Exception as e:
    raise RuntimeError(f"加载模型或分词器失败: {e}")

print(f"模型和分词器已成功加载，路径: {model_abs_path}")

# 定义数据集路径（使用相对路径）
dataset_path = "../datasets/media_LCSTS"

# 确保使用绝对路径
dataset_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), dataset_path))
dataset_abs_path = dataset_abs_path.replace("\\", "/")  # 将反斜杠替换为正斜杠

# 检查数据集路径是否存在
if not os.path.exists(dataset_abs_path):
    raise FileNotFoundError(f"数据集路径不存在: {dataset_abs_path}")

# 加载数据集
try:
    dataset = load_dataset("json", data_files={
        "train": os.path.join(dataset_abs_path, "train.jsonl"),
        "validation": os.path.join(dataset_abs_path, "valid.jsonl"),
        "test": os.path.join(dataset_abs_path, "test_public.jsonl")
    })
except Exception as e:
    raise RuntimeError(f"加载数据集失败: {e}")

print(f"数据集已成功加载，路径: {dataset_abs_path}")



