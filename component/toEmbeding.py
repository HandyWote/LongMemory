"""
Enhanced message vectorization with configuration management
增强的消息向量化处理，包含配置管理
"""
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import openai
from openai import OpenAI
import json
import logging
from component.config_manager import setup_system_config

# Configure logging / 配置日志
logger = logging.getLogger(__name__)

# Initialize configuration / 初始化配置
config_manager = setup_system_config()

class TextEmbedder:
    def __init__(self, api_key: str = None):
        """
        Initialize text vectorizer
        初始化文本向量化器
        
        Args:
            api_key: OpenAI API key / OpenAI API密钥
        """
        logger.info("Initializing TextEmbedder")
        # Use OpenAI's embedding model / 使用OpenAI的嵌入模型
        # Prioritize using the passed api_key, otherwise use the api_key from configuration / 优先使用传入的api_key，否则使用配置中的api_key
        if api_key:
            self.client = OpenAI(api_key=api_key)
            logger.debug("Initializing OpenAI client with passed API key")
        else:
            self.client = OpenAI(api_key=config_manager.get_api_key())
            logger.debug("Initializing OpenAI client with configuration API key")
        
        self.embedding_model = config_manager.get_embedding_model()
        logger.info(f"Using embedding model: {self.embedding_model}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get vector representation of a single text
        获取单个文本的向量表示
        
        Args:
            text: Input text / 输入文本
            
        Returns:
            Vector representation of the text / 文本的向量表示
        """
        text = text.replace("\n", " ")  # Clean text / 清理文本
        response = self.client.embeddings.create(
            input=[text],
            model=self.embedding_model
        )
        embedding = response.data[0].embedding
        logger.info(f"Vector dimension: {len(embedding)}")
        return embedding
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get vector representations of multiple texts
        获取多个文本的向量表示
        
        Args:
            texts: Text list / 文本列表
            
        Returns:
            List of vector representations of texts / 文本的向量表示列表
        """
        logger.debug(f"Generating embeddings for {len(texts)} texts")
        if not texts:
            return []
            
        # Clean texts / 清理文本
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        
        # Batch get embeddings / 批量获取嵌入
        response = self.client.embeddings.create(
            input=cleaned_texts,
            model=self.embedding_model
        )
        
        # Extract embedding vectors / 提取嵌入向量
        embeddings = [item.embedding for item in response.data]
        logger.info(f"Vector dimensions: {[len(e) for e in embeddings]}")
        return embeddings

def process_message_for_database(msg_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process messages obtained from getMessage, generate vectors and format for toDatabase
    处理从getMessage获取的消息，生成向量并格式化为适合toDatabase的格式
    
    Args:
        msg_data: Message data containing msg and uuid / 包含msg和uuid的消息数据
        
    Returns:
        Formatted data containing entities, relationships, summaries and their vector representations / 格式化后的数据，包含实体、关系、摘要及其向量表示
    """
    from component.extract import extract_entity_relation
    
    msg = msg_data.get('msg', '')
    uuid = msg_data.get('uuid', '')
    
    # Use functions from extract module to extract entities, relationships and summaries / 使用extract模块中的函数提取实体、关系和摘要
    extracted_data = extract_entity_relation(msg)
    
    # Handle extraction errors / 处理提取错误的情况
    if "error" in extracted_data:
        # Error handling: Create a text representation containing error information / 错误处理：创建一个包含错误信息的文本表示
        error_text = "Extraction error"
        if 'raw_output' in extracted_data:
            error_text += ", contains raw output"
        
        # Create vector representation for error information / 为错误信息创建向量表示
        embedder = TextEmbedder()
        error_embedding = embedder.get_embedding(error_text)
        logger.info(f"Vector dimension: {len(error_embedding)}")
        
        # Prepare metadata / 准备元数据
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
        logger.info("Error handling completed")
        return result
    
    # Extract entities, relationships and summaries / 提取实体、关系和摘要
    entities_data = extracted_data.get('entities', [])
    relations_data = extracted_data.get('relations', [])
    summary_text = extracted_data.get('summary', '')
    
    # Convert entities to text list, filter out empty entities / 将实体转换为文本列表，过滤掉空实体
    entities = []
    for entity in entities_data:
        if entity.get('name', '').strip():  # Only keep entities with names / 只保留有名称的实体
            entity_text = f"{entity['name']} ({entity['type']})"
            entities.append(entity_text)
    
    # Convert relationships to text list, filter out invalid relationships / 将关系转换为文本列表，过滤掉无效关系
    relations = []
    for relation in relations_data:
        subject = relation.get('subject', '').strip()
        relation_type = relation.get('relation', '').strip()
        object_name = relation.get('object', '').strip()
        
        # If any part of the relationship is empty, try to infer from other parts / 如果关系中的任何部分为空，尝试从其他部分推断
        if not subject and object_name and relation_type:
            # If subject is empty, but object and relation can be inferred, use object as subject / 如果主语为空，但从宾语和关系可以推断，使用宾语作为主语
            subject = object_name
        elif not object_name and subject and relation_type:
            # If object is empty, but subject and relation can be inferred, use subject as object / 如果宾语为空，但从主语和关系可以推断，使用主语作为宾语
            object_name = subject
            
        if subject and relation_type and object_name:
            relation_text = f"{subject} {relation_type} {object_name}"
            relations.append(relation_text)
    
    # Summary is already in text form / 摘要已经是文本形式
    summaries = [summary_text] if summary_text else []
    
    # Create embedder instance / 创建嵌入器实例
    embedder = TextEmbedder()
    
    # Generate vectors for entities, relationships and summaries / 为实体、关系和摘要生成向量
    entity_embeddings = []
    relation_embeddings = []
    summary_embeddings = []
    
    if entities:
        entity_embeddings = embedder.get_embeddings(entities)
        relation_embeddings = embedder.get_embeddings(relations)
    
    if summaries:
        summary_embeddings = embedder.get_embeddings(summaries)
    
    # Prepare metadata / 准备元数据
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
    Format processed data for database storage
    将处理后的数据格式化为适合数据库存储的格式
    
    Args:
        processed_data: Processed data / 处理后的数据
        
    Returns:
        Formatted data that can be directly passed to the toDatabase module / 格式化后的数据，可直接传递给toDatabase模块
    """
    logger.info("Starting to format processed data for database storage")
    
    result = {
        "entities": processed_data.get("entities", []),
        "relations": processed_data.get("relations", []),
        "summaries": processed_data.get("summaries", []),
        "entities_metadata": processed_data.get("entities_metadata", None),
        "relations_metadata": processed_data.get("relations_metadata", None),
        "summaries_metadata": processed_data.get("summaries_metadata", None),
        "uuid": processed_data.get("uuid", None)
    }
    
    logger.info("Data formatting completed")
    return result
