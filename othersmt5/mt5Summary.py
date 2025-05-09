# -*- coding:utf-8 -*-
import re
import random
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

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

# 处理空白的函数
WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

def main(web_mode=False):
    # 决定是随机选择样本还是使用预先选择的样本
    if web_mode:
        # 从文件读取当前选择的文章
        try:
            with open("c:\\Users\\jackx\\Desktop\\transformers\\othersmt5\\current_article.txt", "r", encoding="utf-8") as f:
                content = f.read()
            
            article_match = re.search(r'ARTICLE:(.*?)(?=REFERENCE:|$)', content, re.DOTALL)
            reference_match = re.search(r'REFERENCE:(.*?)(?=INDEX:|$)', content, re.DOTALL)
            
            article_text = clean_sentence_spaces(article_match.group(1)) if article_match else ""
            reference_summary = clean_sentence_spaces(reference_match.group(1)) if reference_match else ""
            
            if not article_text:
                print("错误: 未能从文件中读取到文章，请先点击'随机选择文章'按钮")
                return
                
        except Exception as e:
            print(f"读取预选文章时出错: {e}")
            return
    else:
        # 加载XLSum数据集的中文简体版本
        dataset = load_dataset("csebuetnlp/xlsum", "chinese_simplified")
        test_data = dataset["test"]
        print(f"测试集总条目数: {len(test_data)}")

        # 随机选择一个测试样本
        example_index = random.randint(0, len(test_data) - 1)
        print(f"随机选择的样本索引: {example_index}")
        article_text = clean_sentence_spaces(test_data[example_index]["text"])  # 去除每个句子前的空格
        reference_summary = clean_sentence_spaces(test_data[example_index]["summary"])  # 去除每个句子前的空格
    
    if not web_mode:
        print("原文：")
        print(article_text[:200] + "..." if len(article_text) > 200 else article_text)
        print("\n参考摘要：")
        print(reference_summary)

    # ------ mT5模型摘要生成 ------
    # 加载模型和分词器
    model_path = r"C:\Users\jackx\.cache\huggingface\hub\models--csebuetnlp--mT5_multilingual_XLSum\snapshots\2437a524effdbadc327ced84595508f1e32025b3"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    """模型参数量"""
    print(f"模型参数量: {model.num_parameters()}")
    # 生成摘要
    input_ids = tokenizer(
        [WHITESPACE_HANDLER(article_text)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=84,
        no_repeat_ngram_size=2,
        num_beams=4
    )[0]

    mt5_summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    # 清理生成摘要中每个句子前的空格
    mt5_summary = clean_sentence_spaces(mt5_summary)

    if not web_mode:
        print("\nmT5生成的摘要：")
        print(mt5_summary)

    # 将结果保存到文件，供评估程序使用
    with open("c:\\Users\\jackx\\Desktop\\transformers\\othersmt5\\mt5_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"ARTICLE:{clean_sentence_spaces(article_text)}\n")
        f.write(f"REFERENCE:{clean_sentence_spaces(reference_summary)}\n")
        f.write(f"MT5:{clean_sentence_spaces(mt5_summary)}\n")
    
    if not web_mode:
        print("\n数据已保存到mt5_summary.txt文件")
        print("可以运行TextRankSummary.py进行基线方法对比和评估")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成mT5摘要')
    parser.add_argument('--web_mode', action='store_true', help='是否以Web模式运行')
    args = parser.parse_args()
    
    main(web_mode=args.web_mode)