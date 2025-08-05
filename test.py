#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
查询向量数据库并使用AI生成回复的脚本
"""

import sys
import json
import openai
from openai import OpenAI
from component.toEmbeding import TextEmbedder
from component.toDatabase import MemoryDatabase

# 从配置文件中读取API密钥和模型配置
with open("./config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    API_KEY = config.get("apiKey")
    CHAT_MODEL = config.get("chatModel")

# 初始化OpenAI客户端
client = OpenAI(api_key=API_KEY)

def query_database(query_text, n_results=5):
    """
    使用查询文本向量搜索数据库
    
    Args:
        query_text: 查询文本
        n_results: 返回结果数量
        
    Returns:
        数据库查询结果
    """
    # 初始化数据库和文本嵌入器
    db = MemoryDatabase()
    embedder = TextEmbedder()
    
    # 生成查询文本的向量
    query_embedding = embedder.get_embedding(query_text)
    
    # 在所有集合中搜索相似项
    results = {}
    
    # 查询实体
    try:
        entities_result = db.entities_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        results['entities'] = entities_result
    except Exception as e:
        print(f"查询实体时出错: {e}")
        results['entities'] = {}
    
    # 查询关系
    try:
        relations_result = db.relations_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        results['relations'] = relations_result
    except Exception as e:
        print(f"查询关系时出错: {e}")
        results['relations'] = {}
    
    # 查询摘要
    try:
        summaries_result = db.summaries_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        results['summaries'] = summaries_result
    except Exception as e:
        print(f"查询摘要时出错: {e}")
        results['summaries'] = {}
    
    return results

def format_context(results):
    """
    格式化查询结果作为上下文
    
    Args:
        results: 数据库查询结果
        
    Returns:
        格式化后的上下文字符串
    """
    context_parts = []
    
    # 处理实体
    if 'entities' in results and results['entities'].get('documents'):
        context_parts.append("相关实体:")
        for doc_list in results['entities']['documents']:
            for doc in doc_list:
                context_parts.append(f"- {doc}")
    
    # 处理关系
    if 'relations' in results and results['relations'].get('documents'):
        context_parts.append("\n相关关系:")
        for doc_list in results['relations']['documents']:
            for doc in doc_list:
                context_parts.append(f"- {doc}")
    
    # 处理摘要
    if 'summaries' in results and results['summaries'].get('documents'):
        context_parts.append("\n相关摘要:")
        for doc_list in results['summaries']['documents']:
            for doc in doc_list:
                context_parts.append(f"- {doc}")
    
    return '\n'.join(context_parts)

def generate_response(query_text, context):
    """
    使用AI根据查询和上下文生成回复
    
    Args:
        query_text: 用户查询
        context: 数据库查询结果上下文
        
    Returns:
        AI生成的回复
    """
    # 构造提示词
    if context.strip():
        prompt = f"""
        基于以下背景信息回答用户问题：
        
        背景信息：
        {context}
        
        用户问题：
        {query_text}
        
        请根据背景信息回答问题。如果背景信息与问题无关或没有背景信息，请直接回答问题或说明无法根据提供的信息回答问题。
        """
    else:
        prompt = f"请回答以下问题：{query_text}"
    
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "你是一个智能助手，能够根据提供的背景信息回答问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"生成回复时出错: {e}"

def main():
    """
    主函数
    """
    print("向量数据库查询与AI回复系统")
    print("=" * 40)
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        query_text = ' '.join(sys.argv[1:])
        print(f"查询: {query_text}")
    else:
        # 交互式输入
        query_text = input("请输入您的查询: ")
        if not query_text.strip():
            print("查询不能为空")
            return
    
    print("\n正在查询数据库...")
    results = query_database(query_text)
    
    print("正在格式化上下文...")
    context = format_context(results)
    
    print("\n找到以下相关信息:")
    if context.strip():
        print(context)
    else:
        print("未找到相关数据")
    
    print("\n正在生成AI回复...")
    response = generate_response(query_text, context)
    
    print("\nAI回复:")
    print(response)

if __name__ == "__main__":
    main()