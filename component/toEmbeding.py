"""
将提取到的消息向量化
"""
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import openai
from openai import OpenAI
from component.extract import extract_entity_relation

class TextEmbedder:
    def __init__(self, api_key: str = None):
        """
        初始化文本向量化器
        
        Args:
            api_key: OpenAI API密钥
        """
        # 使用OpenAI的embedding模型
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()  # 从环境变量获取API密钥
        self.embedding_model = "text-embedding-ada-002"
    
    def get_embedding(self, text: str) -> List[float]:
        """
        获取单个文本的向量表示
        
        Args:
            text: 输入文本
            
        Returns:
            文本的向量表示
        """
        text = text.replace("\n", " ")  # 清理文本
        response = self.client.embeddings.create(
            input=[text],
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        获取多个文本的向量表示
        
        Args:
            texts: 文本列表
            
        Returns:
            文本的向量表示列表
        """
        # 清理文本
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        
        # 批量获取嵌入
        response = self.client.embeddings.create(
            input=cleaned_texts,
            model=self.embedding_model
        )
        
        # 提取嵌入向量
        embeddings = [item.embedding for item in response.data]
        return embeddings

def process_message_for_database(msg_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理从getMessage获取的消息，生成向量并格式化为适合toDatabase的格式
    
    Args:
        msg_data: 包含msg和uuid的消息数据
        
    Returns:
        格式化后的数据，包含实体、关系、摘要及其向量表示
    """
    from component.extract import extract_entity_relation
    
    msg = msg_data.get('msg', '')
    uuid = msg_data.get('uuid', '')
    
    # 使用extract模块中的函数提取实体、关系和摘要
    extracted_data = extract_entity_relation(msg)
    
    # 处理提取错误的情况
    if "error" in extracted_data:
        # 错误处理：创建一个包含错误信息的文本表示
        error_text = f"提取错误: {extracted_data['error']}"
        if 'raw_output' in extracted_data:
            error_text += f", 原始输出: {extracted_data['raw_output']}"
        
        # 为错误信息创建向量表示
        embedder = TextEmbedder()
        error_embedding = embedder.get_embedding(error_text)
        
        # 准备元数据
        timestamp = datetime.now().isoformat()
        base_metadata = {
            "uuid": uuid,
            "timestamp": timestamp,
            "source_msg": msg,
            "type": "error"
        }
        
        return {
            "entities": [],
            "relations": [],
            "summaries": [error_text],
            "entity_embeddings": [],
            "relation_embeddings": [],
            "summary_embeddings": [error_embedding],
            "entities_metadata": [],
            "relations_metadata": [],
            "summaries_metadata": [base_metadata],
            "uuid": uuid
        }
    
    # 提取实体、关系和摘要
    entities_data = extracted_data.get('entities', [])
    relations_data = extracted_data.get('relations', [])
    summary_text = extracted_data.get('summary', '')
    
    # 将实体转换为文本列表
    entities = []
    for entity in entities_data:
        entities.append(f"{entity['name']} ({entity['type']})")
    
    # 将关系转换为文本列表
    relations = []
    for relation in relations_data:
        relations.append(f"{relation['subject']} {relation['relation']} {relation['object']}")
    
    # 摘要已经是文本形式
    summaries = [summary_text] if summary_text else []
    
    # 创建嵌入器实例
    embedder = TextEmbedder()
    
    # 为实体、关系和摘要生成向量
    entity_embeddings = []
    relation_embeddings = []
    summary_embeddings = []
    
    if entities:
        entity_embeddings = embedder.get_embeddings(entities)
    
    if relations:
        relation_embeddings = embedder.get_embeddings(relations)
    
    if summaries:
        summary_embeddings = embedder.get_embeddings(summaries)
    
    # 准备元数据
    timestamp = datetime.now().isoformat()
    base_metadata = {
        "uuid": uuid,
        "timestamp": timestamp,
        "source_msg": msg
    }
    
    entities_metadata = [{**base_metadata, "type": "entity"} for _ in entities]
    relations_metadata = [{**base_metadata, "type": "relation"} for _ in relations]
    summaries_metadata = [{**base_metadata, "type": "summary"} for _ in summaries]
    
    return {
        "entities": entities,
        "relations": relations,
        "summaries": summaries,
        "entity_embeddings": entity_embeddings,
        "relation_embeddings": relation_embeddings,
        "summary_embeddings": summary_embeddings,
        "entities_metadata": entities_metadata,
        "relations_metadata": relations_metadata,
        "summaries_metadata": summaries_metadata,
        "uuid": uuid
    }

def format_for_database(processed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将处理后的数据格式化为适合数据库存储的格式
    
    Args:
        processed_data: 处理后的数据
        
    Returns:
        格式化后的数据，可直接传递给toDatabase模块
    """
    return {
        "entities": processed_data.get("entities", []),
        "relations": processed_data.get("relations", []),
        "summaries": processed_data.get("summaries", []),
        "entities_metadata": processed_data.get("entities_metadata", None),
        "relations_metadata": processed_data.get("relations_metadata", None),
        "summaries_metadata": processed_data.get("summaries_metadata", None),
        "uuid": processed_data.get("uuid", None)
    }