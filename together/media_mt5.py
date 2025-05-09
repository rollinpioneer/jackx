# -*- coding:utf-8 -*-
import re
import random
import argparse
import os
import json
from openai import OpenAI

# 处理文本中每个句子前的空格
def clean_sentence_spaces(text):
    # 按标点符号分句并去除每句开头的空格
    sentences = re.split('([。！？!?])', text)
    
    # 重组句子并确保每句开头没有空格
    result = ""
    for i in range(0, len(sentences), 2):
        if i < len(sentences):
            # 当前部分是句子内容
            current = sentences[i].strip()
            if current:  # 只有当句子不为空时才添加
                result += current
                
                # 添加标点符号（如果存在）
                if i+1 < len(sentences):
                    result += sentences[i+1]
    
    return result

# 使用大模型API生成摘要
def generate_summary_with_api(text):
    try:
        client = OpenAI(api_key="sk-a17fda1fa3bf42e186d7e3868c88f3a8", base_url="https://api.deepseek.com")
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的中文文本摘要助手，针对社交媒体内容生成简明摘要。请输出不超过两句话的简短摘要。"},
                {"role": "user", "content": f"请帮我对下面这段社交媒体文本生成简明摘要：{text}"},
            ],
            stream=False
        )
        
        # 获取生成的摘要
        summary = response.choices[0].message.content.strip()
        return clean_sentence_spaces(summary)
    except Exception as e:
        print(f"调用API生成摘要时出错: {e}")
        # 如果API调用失败，返回一个简单的摘要（取文本的前部分）
        summary = text[:min(100, len(text))]
        if "。" in summary:
            summary = summary[:summary.rindex("。") + 1]
        return clean_sentence_spaces(summary)

def main(web_mode=False):
    # 决定是随机选择样本还是使用预先选择的样本
    if web_mode:
        # 从文件读取当前选择的文章
        try:
            current_file = os.path.join(os.path.dirname(__file__), "current_media.txt")
            with open(current_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            article_match = re.search(r'ARTICLE:(.*?)(?=INDEX:|$)', content, re.DOTALL)
            
            article_text = clean_sentence_spaces(article_match.group(1)) if article_match else ""
            
            if not article_text:
                print("错误: 未能从文件中读取到文章，请先点击'随机选择文章'按钮")
                return
                
        except Exception as e:
            print(f"读取预选文章时出错: {e}")
            return
    else:
        # 加载媒体数据集
        dataset_path = os.path.join(os.path.dirname(__file__), "..", "datasets/media_LCSTS/test_public.jsonl")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f]
            
        print(f"测试集总条目数: {len(test_data)}")

        # 随机选择一个测试样本
        example_index = random.randint(0, len(test_data) - 1)
        print(f"随机选择的样本索引: {example_index}")
        article_text = clean_sentence_spaces(test_data[example_index]["text"])  # 使用text键访问文章内容
    
    if not web_mode:
        print("原文：")
        print(article_text[:200] + "..." if len(article_text) > 200 else article_text)

#   # ------ Transformer模型摘要生成 ------
#     # 加载模型和分词器 - 使用fine-tuned的模型
#     model_path = os.path.join(os.path.dirname(__file__), "..", "results/media-creative/final_model")
    
#     # 如果模型路径不存在，使用默认的mT5模型
#     if not os.path.exists(model_path):
#         model_path = r"C:\Users\jackx\.cache\huggingface\hub\models--csebuetnlp--mT5_multilingual_XLSum\snapshots\2437a524effdbadc327ced84595508f1e32025b3"
    
#     # 加载模型
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

#     # 检查GPU可用性并将模型加载到GPU
#     import torch
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
    
#     # 生成摘要
#     input_ids = tokenizer(
#         [WHITESPACE_HANDLER(article_text)],
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True,
#         max_length=512
#     )["input_ids"].to(device)

#     output_ids = model.generate(
#         input_ids=input_ids,
#         max_length=64,  # 社交媒体文本摘要通常更短
#         no_repeat_ngram_size=2,
#         num_beams=4
#     )[0]

#     mt5_summary = tokenizer.decode(
#         output_ids,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False
#     )
#     # 清理生成摘要中每个句子前的空格
#     mt5_summary = clean_sentence_spaces(mt5_summary)

#     if not web_mode:
#         print("\nTransformer生成的摘要：")
#         print(mt5_summary)
    # ------ 使用API生成摘要 ------
    print("\n正在使用大模型API生成摘要...")
    mt5_summary = generate_summary_with_api(article_text)

    if not web_mode:
        print("\nAPI生成的摘要：")
        print(mt5_summary)

    # 将结果保存到文件，供评估程序使用
    output_file = os.path.join(os.path.dirname(__file__), "media_mt5_summary.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"ARTICLE:{clean_sentence_spaces(article_text)}\n")
        f.write(f"MT5:{clean_sentence_spaces(mt5_summary)}\n")
    
    if not web_mode:
        print("\n数据已保存到media_mt5_summary.txt文件")
        
        # 添加是否继续迭代的提示
        continue_iteration = input("\n是否继续迭代？(y/n): ")
        if continue_iteration.lower() == 'y':
            # 导入media_textrank模块并运行
            try:
                import media_textrank
                media_textrank.main(web_mode=False)
            except ImportError:
                print("无法导入media_textrank模块，请确保它可用")
                print("您可以手动运行: python media_textrank.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成社交媒体摘要')
    parser.add_argument('--web_mode', action='store_true', help='是否以Web模式运行')
    args = parser.parse_args()
    
    main(web_mode=args.web_mode)