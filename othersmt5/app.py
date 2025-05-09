from flask import Flask, render_template, request, jsonify
import subprocess
import os
import random
from datasets import load_dataset
import re
import sys
import json
import base64

app = Flask(__name__)
app.static_folder = 'static'

# 全局变量存储当前选择的文章和摘要
current_data = {
    "article": "",
    "reference": "",
    "index": -1
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/random_article', methods=['GET'])
def random_article():
    try:
        # 加载XLSum数据集的中文简体版本
        dataset = load_dataset("csebuetnlp/xlsum", "chinese_simplified")
        test_data = dataset["test"]
        
        # 随机选择一个测试样本
        example_index = random.randint(0, len(test_data) - 1)
        article_text = clean_sentence_spaces(test_data[example_index]["text"])
        reference_summary = clean_sentence_spaces(test_data[example_index]["summary"])
        
        # 保存当前选择的文章和参考摘要
        current_data["article"] = article_text
        current_data["reference"] = reference_summary
        current_data["index"] = example_index
        
        # 将数据写入文件供其他脚本使用
        with open("c:\\Users\\jackx\\Desktop\\transformers\\othersmt5\\current_article.txt", "w", encoding="utf-8") as f:
            f.write(f"ARTICLE:{article_text}\n")
            f.write(f"REFERENCE:{reference_summary}\n")
            f.write(f"INDEX:{example_index}\n")
        
        return jsonify({
            "status": "success",
            "article": article_text,
            "reference": reference_summary,
            "index": example_index
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/generate_mt5', methods=['POST'])
def generate_mt5():
    try:
        # 运行mt5Summary.py
        result = subprocess.run(
            [sys.executable, "c:\\Users\\jackx\\Desktop\\transformers\\othersmt5\\mt5Summary.py", "--web_mode"],
            capture_output=True, 
            text=True, 
            encoding='utf-8'
        )
        
        # 读取生成的摘要文件
        with open("c:\\Users\\jackx\\Desktop\\transformers\\othersmt5\\mt5_summary.txt", "r", encoding="utf-8") as f:
            content = f.read()
        
        mt5_match = re.search(r'MT5:(.*?)$', content, re.DOTALL)
        mt5_summary = clean_sentence_spaces(mt5_match.group(1)) if mt5_match else ""
        
        return jsonify({
            "status": "success", 
            "mt5_summary": mt5_summary,
            "log": result.stdout
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/generate_textrank', methods=['POST'])
def generate_textrank():
    try:
        # 运行TextRankSummary.py
        result = subprocess.run(
            [sys.executable, "c:\\Users\\jackx\\Desktop\\transformers\\othersmt5\\TextRankSummary.py", "--web_mode"],
            capture_output=True, 
            text=True, 
            encoding='utf-8'
        )
        
        # 读取生成的摘要文件
        with open("c:\\Users\\jackx\\Desktop\\transformers\\othersmt5\\mt5TextRank_results.txt", "r", encoding="utf-8") as f:
            content = f.read()
        
        textrank_match = re.search(r'TextRank摘要: (.*?)(?:\n\n|$)', content, re.DOTALL)
        textrank_summary = clean_sentence_spaces(textrank_match.group(1)) if textrank_match else ""
        
        return jsonify({
            "status": "success", 
            "textrank_summary": textrank_summary,
            "log": result.stdout
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        # 运行evaluation.py
        result = subprocess.run(
            [sys.executable, "c:\\Users\\jackx\\Desktop\\transformers\\othersmt5\\evaluation.py"],
            capture_output=True, 
            text=True, 
            encoding='utf-8'
        )
        
        # 读取评估结果
        if os.path.exists("c:\\Users\\jackx\\Desktop\\transformers\\othersmt5\\eval_results.json"):
            with open("c:\\Users\\jackx\\Desktop\\transformers\\othersmt5\\eval_results.json", "r", encoding="utf-8") as f:
                eval_results = json.load(f)
            
            # 读取图片并转换为base64
            with open("c:\\Users\\jackx\\Desktop\\transformers\\othersmt5\\eval_chart.png", "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            return jsonify({
                "status": "success", 
                "results": eval_results,
                "image": img_data,
                "log": result.stdout
            })
        else:
            return jsonify({"status": "error", "message": "评估结果文件不存在"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# 清理文本空格的函数
def clean_sentence_spaces(text):
    sentences = re.split('([。！？!?])', text)
    result = ""
    for i in range(0, len(sentences), 2):
        if i < len(sentences):
            current = sentences[i].strip()
            if current:
                result += current
                if i+1 < len(sentences):
                    result += sentences[i+1]
    return result

if __name__ == '__main__':
    app.run(debug=True, port=5000)
