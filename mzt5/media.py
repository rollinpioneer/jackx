from transformers import AutoModelForSeq2SeqLM, AutoModel, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_dataset
import os
import torch
import numpy as np
from rouge import Rouge
import nltk
from nltk.tokenize import word_tokenize

# 确保所需的 nltk 资源已下载
try:
    nltk.data.find('tokenizers/punkt')  # 检查 punkt 是否已下载
except LookupError:
    print("未找到 punkt 资源，正在下载...")
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')  # 检查 punkt_tab 是否已下载
except LookupError:
    print("未找到 punkt_tab 资源，正在下载...")
    nltk.download('punkt_tab')

# 定义模型路径（使用绝对路径）
model_path = "../models/mengzit5"

# 确保使用绝对路径
model_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
model_abs_path = model_abs_path.replace("\\", "/")  # 将反斜杠替换为正斜杠

# 检查模型路径是否存在
if not os.path.exists(model_abs_path):
    raise FileNotFoundError(f"模型路径不存在: {model_abs_path}")

# 修正模型加载：使用AutoModelForSeq2SeqLM而不是AutoModel
try:
    auto_tokenizer = AutoTokenizer.from_pretrained(model_abs_path)
    auto_model = AutoModelForSeq2SeqLM.from_pretrained(model_abs_path)
except Exception as e:
    raise RuntimeError(f"加载模型或分词器失败: {e}")

print(f"模型和分词器已成功加载，路径: {model_abs_path}")

# 定义数据集路径（使用绝对路径）
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

# 数据预处理函数
def preprocess_function(examples):
    # 数据集中的字段是text(文章内容)和summary(摘要)
    inputs = examples["text"]
    targets = examples["summary"]
    
    # 为模型准备输入
    model_inputs = auto_tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    # 为模型准备标签
    labels = auto_tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# 应用预处理
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names  # 移除原始列
)

print("数据预处理完成")

# 定义评估指标
def compute_metrics(eval_pred):
    rouge = Rouge()
    predictions, labels = eval_pred
    
    # 替换 -100 为 pad_token_id
    predictions = np.where(predictions != -100, predictions, auto_tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, auto_tokenizer.pad_token_id)
    
    # 解码生成的文本和参考文本
    decoded_preds = auto_tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = auto_tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 将生成的文本和参考文本转换为可计算ROUGE的格式
    decoded_preds = [" ".join(word_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = [" ".join(word_tokenize(label.strip())) for label in decoded_labels]
    
    # 计算ROUGE分数
    rouge_scores = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    
    result = {
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-2": rouge_scores["rouge-2"]["f"],
        "rouge-l": rouge_scores["rouge-l"]["f"],
    }
    
    return result

# 定义训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/media-creative")),
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),  # 如果有GPU则使用半精度训练
    logging_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/media-creative/logs")),
    logging_steps=100,
    save_strategy="epoch",
)

# 创建数据整理器
data_collator = DataCollatorForSeq2Seq(
    auto_tokenizer,
    model=auto_model,
    padding=True,
    return_tensors="pt"
)

# 初始化训练器
trainer = Seq2SeqTrainer(
    model=auto_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=auto_tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 开始训练
print("开始训练模型...")
trainer.train()

# 保存最终模型
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/media-creative/final_model"))
trainer.save_model(output_dir)
auto_tokenizer.save_pretrained(output_dir)
print(f"模型已保存到 {output_dir}")

# 在测试集上评估模型
print("在测试集上评估模型...")
results = trainer.evaluate(tokenized_datasets["test"])
print(f"测试结果: {results}")
