"""
将提取到的消息向量化
"""
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import openai
from openai import OpenAI
import json
import logging
from component.extract import extract_entity_relation

# 配置日志
logger = logging.getLogger(__name__)

# 从配置文件中读取API密钥和模型配置
with open("./config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    API_KEY = config.get("apiKey")
    EMBEDDING_MODEL = config.get("embeddingModel", "text-embedding-ada-002")

class TextEmbedder:
    def __init__(self, api_key: str = None):
        """
        初始化文本向量化器
        
        Args:
            api_key: OpenAI API密钥
        """
        logger.info("初始化TextEmbedder")
        # 使用OpenAI的embedding模型
        # 优先使用传入的api_key，否则使用配置文件中的api_key
        if api_key:
            self.client = OpenAI(api_key=api_key)
            logger.debug("使用传入的API密钥初始化OpenAI客户端")
        elif API_KEY:
            self.client = OpenAI(api_key=API_KEY)
            logger.debug("使用配置文件中的API密钥初始化OpenAI客户端")
        else:
            logger.error("OpenAI API密钥未设置")
            raise ValueError("OpenAI API密钥未设置。请在config.json中配置apiKey或在初始化时传递api_key参数。")
        
        self.embedding_model = EMBEDDING_MODEL
        logger.info(f"使用的嵌入模型: {self.embedding_model}")
    
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
        embedding = response.data[0].embedding
        logger.info(f"向量维度: {len(embedding)}")
        return embedding
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        获取多个文本的向量表示
        
        Args:
            texts: 文本列表
            
        Returns:
            文本的向量表示列表
        """
        logger.debug(f"为{texts}个文本生成嵌入向量")
        if not texts:
            return []
            
        # 清理文本
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        
        # 批量获取嵌入
        response = self.client.embeddings.create(
            input=cleaned_texts,
            model=self.embedding_model
        )
        
        # 提取嵌入向量
        embeddings = [item.embedding for item in response.data]
        logger.info(f"向量维度: {[len(e) for e in embeddings]}")
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
        error_text = "提取错误"
        if 'raw_output' in extracted_data:
            error_text += ", 包含原始输出"
        
        # 为错误信息创建向量表示
        embedder = TextEmbedder()
        error_embedding = embedder.get_embedding(error_text)
        logger.info(f"向量维度: {len(error_embedding)}")
        
        # 准备元数据
        timestamp = datetime.now().isoformat()
        base_metadata = {
            "uuid": uuid,
            "timestamp": timestamp,
            "source_msg": msg,
            "type": "error"
        }
        
        result = {
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
        logger.info("错误处理完成")
        return result
    
    # 提取实体、关系和摘要
    entities_data = extracted_data.get('entities', [])
    relations_data = extracted_data.get('relations', [])
    summary_text = extracted_data.get('summary', '')
    
    # 提取到实体和关系的数量
    
    # 将实体转换为文本列表，过滤掉空实体
    entities = []
    for entity in entities_data:
        if entity.get('name', '').strip():  # 只保留有名称的实体
            entity_text = f"{entity['name']} ({entity['type']})"
            entities.append(entity_text)
    
    # 将关系转换为文本列表，过滤掉无效关系
    relations = []
    for relation in relations_data:
        subject = relation.get('subject', '').strip()
        relation_type = relation.get('relation', '').strip()
        object_name = relation.get('object', '').strip()
        
        # 如果关系中的任何部分为空，尝试从其他部分推断
        if not subject and object_name and relation_type:
            # 如果主语为空，但从宾语和关系可以推断，使用宾语作为主语
            subject = object_name
        elif not object_name and subject and relation_type:
            # 如果宾语为空，但从主语和关系可以推断，使用主语作为宾语
            object_name = subject
            
        if subject and relation_type and object_name:
            relation_text = f"{subject} {relation_type} {object_name}"
            relations.append(relation_text)
    
    # 摘要已经是文本形式
    summaries = [summary_text] if summary_text else []
    
    # 创建嵌入器实例
    # 创建嵌入器实例
    embedder = TextEmbedder()
    
    # 为实体、关系和摘要生成向量
    entity_embeddings = []
    relation_embeddings = []
    summary_embeddings = []
    
    if entities:
        entity_embeddings = embedder.get_embeddings(entities)
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
    
    result = {
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
    
    return result

def format_for_database(processed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将处理后的数据格式化为适合数据库存储的格式
    
    Args:
        processed_data: 处理后的数据
        
    Returns:
        格式化后的数据，可直接传递给toDatabase模块
    """
    logger.info("开始格式化处理后的数据以准备数据库存储")
    
    result = {
        "entities": processed_data.get("entities", []),
        "relations": processed_data.get("relations", []),
        "summaries": processed_data.get("summaries", []),
        "entities_metadata": processed_data.get("entities_metadata", None),
        "relations_metadata": processed_data.get("relations_metadata", None),
        "summaries_metadata": processed_data.get("summaries_metadata", None),
        "uuid": processed_data.get("uuid", None)
    }
    
    logger.info("数据格式化完成")
    return result
