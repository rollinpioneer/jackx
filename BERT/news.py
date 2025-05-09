from transformers import BertForTokenClassification, Trainer, TrainingArguments, AutoModel, AutoTokenizer
from datasets import load_dataset
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re
import torch
import nltk
from transformers import DataCollatorForTokenClassification

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

# 定义数据集路径（使用相对路径）
dataset_path = "../datasets/news_chinese_simplified_XLSum_v2.0"

# 确保使用绝对路径
dataset_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), dataset_path))
dataset_abs_path = dataset_abs_path.replace("\\", "/")  # 将反斜杠替换为正斜杠

# 检查数据集路径是否存在
if not os.path.exists(dataset_abs_path):
    raise FileNotFoundError(f"数据集路径不存在: {dataset_abs_path}")

# 加载数据集
try:
    dataset = load_dataset("json", data_files={
        "train": os.path.join(dataset_abs_path, "chinese_simplified_train.jsonl"),
        "validation": os.path.join(dataset_abs_path, "chinese_simplified_val.jsonl"),
        "test": os.path.join(dataset_abs_path, "chinese_simplified_test.jsonl")
    })
except Exception as e:
    raise RuntimeError(f"加载数据集失败: {e}")

# 下载必要的nltk资源
try:
    nltk.download('punkt')
except:
    print("无法下载nltk资源，但这不影响程序运行")

# 设置训练参数
model_name = "news-extractive-summarizer"
output_dir = os.path.join(os.path.dirname(__file__), f"../results/{model_name}")
log_dir = os.path.join(output_dir, "logs")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# 确定文章和标签列的名称
article_column = "text"
summary_column = "summary"

# 定义预处理函数 - 完全重写
def preprocess_function(examples):
    articles = examples[article_column]
    summaries = examples[summary_column]
    
    # 清理文本
    cleaned_articles = [WHITESPACE_HANDLER(text) for text in articles]
    cleaned_summaries = [WHITESPACE_HANDLER(text) for text in summaries]
    
    tokenized_inputs = auto_tokenizer(
        cleaned_articles,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # 创建序列标签 (BIO标记)
    labels = []
    for article, summary in zip(cleaned_articles, cleaned_summaries):
        # 分词
        article_tokens = auto_tokenizer.tokenize(article)
        summary_tokens = auto_tokenizer.tokenize(summary)
        
        # 获取文章的token IDs
        article_ids = auto_tokenizer.convert_tokens_to_ids(article_tokens)
        
        # 创建标签，初始化为0（不在摘要中）
        label = [0] * len(article_ids)
        
        # 标记摘要中的token (简化版本 - 根据tokens是否在摘要中)
        for i, token_id in enumerate(article_ids):
            if i < len(article_tokens) and article_tokens[i] in summary_tokens:
                label[i] = 1  # 标记为摘要的一部分
                
        # 填充到最大长度
        if len(label) < 512:
            label = label + [0] * (512 - len(label))
        else:
            label = label[:512]
            
        labels.append(label)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 处理空白的函数
WHITESPACE_HANDLER = lambda k: re.sub('\\s+', ' ', re.sub('\n+', ' ', k.strip()))

# 处理和标记化数据集
print("正在处理数据集...")
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# 加载标记分类模型
print("加载模型...")
model = BertForTokenClassification.from_pretrained(
    model_abs_path,
    num_labels=2  # 二分类：0=不在摘要中，1=在摘要中
)

# 使用标准的DataCollator
data_collator = DataCollatorForTokenClassification(
    tokenizer=auto_tokenizer
)

# 定义计算指标的函数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # 展平预测和标签，去除填充标记
    true_predictions = [
        [p for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for l in label if l != -100]
        for label in labels
    ]
    
    # 展平以便计算整体指标
    flattened_predictions = np.concatenate(true_predictions)
    flattened_labels = np.concatenate(true_labels)
    
    # 计算指标
    accuracy = accuracy_score(flattened_labels, flattened_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        flattened_labels, flattened_predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 配置训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,   
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir=log_dir,
    logging_strategy="steps",
    logging_steps=100,
    report_to=["tensorboard"],
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    tokenizer=auto_tokenizer,
    data_collator=data_collator
)

# 开始训练
print(f"开始训练抽取式摘要模型，TensorBoard日志将保存到 {log_dir}")
trainer.train()

# 在测试集上评估模型
print("在测试集上评估模型...")
results = trainer.evaluate(tokenized_datasets["test"])
print(f"测试结果: {results}")

# 保存模型
trainer.save_model(os.path.join(output_dir, "final_model"))
print(f"模型已保存到 {os.path.join(output_dir, 'final_model')}")

# 提示用户如何查看TensorBoard
print(f"\n训练完成！通过以下命令查看TensorBoard可视化结果:")
print(f"tensorboard --logdir={log_dir}")




