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

# 定义模型路径（使用相对路径）
model_path = "../models/mengzit5"

# 确保使用绝对路径
model_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
model_abs_path = model_abs_path.replace("\\", "/")  # 将反斜杠替换为正斜杠

# 检查模型路径是否存在
if not os.path.exists(model_abs_path):
    raise FileNotFoundError(f"模型路径不存在: {model_abs_path}")

# 修正模型加载：使用AutoModelForSeq2SeqLM而不是AutoModel
try:
    # 使用Auto类从本地加载模型和分词器
    auto_tokenizer = AutoTokenizer.from_pretrained(model_abs_path)
    auto_model = AutoModelForSeq2SeqLM.from_pretrained(model_abs_path)
except Exception as e:
    raise RuntimeError(f"加载模型或分词器失败: {e}")

print(f"模型和分词器已成功加载，路径: {model_abs_path}")

# 定义数据集路径（使用相对路径）
dataset_path = "../datasets/baike_wiki"

# 确保使用绝对路径
dataset_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), dataset_path))
dataset_abs_path = dataset_abs_path.replace("\\", "/")  # 将反斜杠替换为正斜杠

# 检查数据集路径是否存在
if not os.path.exists(dataset_abs_path):
    raise FileNotFoundError(f"数据集路径不存在: {dataset_abs_path}")

# 加载数据集
try:
    dataset = load_dataset("parquet", data_files={
        "train": os.path.join(dataset_abs_path, "train-00000-of-00001.parquet")
    })
except Exception as e:
    raise RuntimeError(f"加载数据集失败: {e}")

print(f"数据集已成功加载，路径: {dataset_abs_path}")

# 查看数据集样例结构
print("数据集样例结构:")
print(dataset["train"][0])

# 定义数据转换函数，从嵌套的article字典中提取document和summary
def extract_document_summary(example):
    article = example['article']
    # 确保article是字典类型
    if isinstance(article, dict):
        document = article.get('document', [])
        summary = article.get('summary', [])
        
        # 连接列表中的字符串（如果是列表形式）
        if isinstance(document, list):
            document = ' '.join(document)
        if isinstance(summary, list):
            summary = ' '.join(summary)
            
        return {
            'document': document,
            'summary': summary
        }
    else:
        # 如果article不是字典，返回空字符串
        return {
            'document': '',
            'summary': ''
        }

# 应用转换函数
processed_dataset = dataset.map(extract_document_summary)

# 为了训练，需要将数据集分割为训练集和验证集
train_test_split = processed_dataset["train"].train_test_split(test_size=0.1)
dataset = {
    "train": train_test_split["train"],
    "validation": train_test_split["test"]
}

# 数据预处理函数
def preprocess_function(examples):
    inputs = examples["document"]
    targets = examples["summary"]
    
    # 为模型准备输入
    model_inputs = auto_tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    # 为模型准备标签
    labels = auto_tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# 应用预处理
tokenized_datasets = {
    split: dataset[split].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset[split].column_names  # 移除原始列
    )
    for split in dataset.keys()
}

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
    output_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/baike_wiki-creative")),
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),  # 如果有GPU则使用半精度训练
    logging_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/baike_wiki-creative/logs")),
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True
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
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/baike_wiki-creative/final_model"))
trainer.save_model(output_dir)
auto_tokenizer.save_pretrained(output_dir)
print(f"模型已保存到 {output_dir}")

# 在验证集上评估模型
print("在验证集上评估模型...")
results = trainer.evaluate(tokenized_datasets["validation"])
print(f"验证结果: {results}")