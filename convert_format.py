#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

def convert_json_to_jsonl():
    """
    将nlpcc_data.json的格式转换为与test_public.jsonl相同的格式
    nlpcc_data.json格式：
    {
      "version": "0.0.1",
      "data": [
        {
          "title": "...",
          "content": "..."
        },
        ...
      ]
    }
    
    test_public.jsonl格式：
    {"summary": "", "text": "..."}
    {"summary": "", "text": "..."}
    ...
    """
    input_file = "datasets/nlpcc_data.json"
    output_file = "datasets/nlpcc_data_converted.jsonl"
    
    print(f"开始转换 {input_file} 为 {output_file}")
    
    try:
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 打开输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            # 遍历数据数组
            for item in data['data']:
                # 创建新的对象，将title映射为summary，content映射为text
                new_item = {
                    "summary": item["title"],
                    "text": item["content"]
                }
                # 写入一行
                f.write(json.dumps(new_item, ensure_ascii=False) + '\n')
        
        print(f"转换完成! 共转换 {len(data['data'])} 条记录.")
        print(f"输出文件保存在: {output_file}")
        
    except Exception as e:
        print(f"转换过程中发生错误: {e}")

if __name__ == "__main__":
    convert_json_to_jsonl()