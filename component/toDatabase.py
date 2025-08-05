"""
将向量化的数据格式化存入数据库
"""
import chromadb
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional

# 配置日志
logger = logging.getLogger(__name__)

class MemoryDatabase:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        初始化记忆数据库
        
        Args:
            persist_directory: 数据库存储目录
        """
        logger.info(f"初始化记忆数据库，存储目录: {persist_directory}")
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.entities_collection = self.client.get_or_create_collection("entities")
        self.relations_collection = self.client.get_or_create_collection("relations")
        self.summaries_collection = self.client.get_or_create_collection("summaries")
        logger.info("记忆数据库初始化完成")
    
    def store_entities(self, texts: List[str], metadatas: Optional[List[Dict]] = None, 
                      embeddings: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        """
        存储实体信息
        
        Args:
            texts: 原文列表
            metadatas: 元数据列表
            embeddings: 向量列表
            
        Returns:
            存储结果
        """
        logger.info(f"开始存储实体，实体数量: {len(texts)}")
        logger.debug(f"实体文本: {texts}")
        logger.debug(f"实体元数据: {metadatas}")
        logger.debug(f"实体嵌入向量: {bool(embeddings)}")
        
        # 检查是否为空列表
        if not texts:
            logger.warning("实体文本列表为空，跳过存储")
            return {"ids": []}
            
        ids = [f"entity_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}" for i in range(len(texts))]
        logger.debug(f"生成的实体IDs: {ids}")
        
        self.entities_collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas or [{} for _ in texts],
            embeddings=embeddings
        )
        logger.info("实体存储完成")
        return {"ids": ids}
    
    def store_relations(self, texts: List[str], metadatas: Optional[List[Dict]] = None, 
                       embeddings: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        """
        存储关系信息
        
        Args:
            texts: 原文列表
            metadatas: 元数据列表
            embeddings: 向量列表
            
        Returns:
            存储结果
        """
        logger.info(f"开始存储关系，关系数量: {len(texts)}")
        logger.debug(f"关系文本: {texts}")
        logger.debug(f"关系元数据: {metadatas}")
        logger.debug(f"关系嵌入向量: {bool(embeddings)}")
        
        # 检查是否为空列表
        if not texts:
            logger.warning("关系文本列表为空，跳过存储")
            return {"ids": []}
            
        ids = [f"relation_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}" for i in range(len(texts))]
        logger.debug(f"生成的关系IDs: {ids}")
        
        self.relations_collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas or [{} for _ in texts],
            embeddings=embeddings
        )
        logger.info("关系存储完成")
        return {"ids": ids}
    
    def store_summaries(self, texts: List[str], metadatas: Optional[List[Dict]] = None, 
                       embeddings: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        """
        存储摘要信息
        
        Args:
            texts: 原文列表
            metadatas: 元数据列表
            embeddings: 向量列表
            
        Returns:
            存储结果
        """
        logger.info(f"开始存储摘要，摘要数量: {len(texts)}")
        logger.debug(f"摘要文本: {texts}")
        logger.debug(f"摘要元数据: {metadatas}")
        logger.debug(f"摘要嵌入向量: {bool(embeddings)}")
        
        # 检查是否为空列表
        if not texts:
            logger.warning("摘要文本列表为空，跳过存储")
            return {"ids": []}
            
        ids = [f"summary_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}" for i in range(len(texts))]
        logger.debug(f"生成的摘要IDs: {ids}")
        
        self.summaries_collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas or [{} for _ in texts],
            embeddings=embeddings
        )
        logger.info("摘要存储完成")
        return {"ids": ids}

def store_knowledge_triple(entities: List[str], relations: List[str], summaries: List[str],
                          entities_metadata: Optional[List[Dict]] = None,
                          relations_metadata: Optional[List[Dict]] = None,
                          summaries_metadata: Optional[List[Dict]] = None,
                          uuid: Optional[str] = None) -> Dict[str, Any]:
    """
    存储知识三元组（实体、关系、摘要）
    
    Args:
        entities: 实体列表
        relations: 关系列表
        summaries: 摘要列表
        entities_metadata: 实体元数据
        relations_metadata: 关系元数据
        summaries_metadata: 摘要元数据
        uuid: 用户UUID，用于区分不同用户的数据
        
    Returns:
        包含各部分存储结果的字典
    """
    logger.info("开始存储知识三元组")
    logger.info(f"待存储数据 - 实体: {len(entities)}, 关系: {len(relations)}, 摘要: {len(summaries)}")
    logger.debug(f"实体: {entities}")
    logger.debug(f"关系: {relations}")
    logger.debug(f"摘要: {summaries}")
    logger.debug(f"UUID: {uuid}")
    
    db = MemoryDatabase()
    
    # 如果提供了UUID，则将其添加到所有元数据中
    if uuid:
        logger.info(f"添加UUID到元数据: {uuid}")
        entities_metadata = entities_metadata or [{} for _ in entities]
        relations_metadata = relations_metadata or [{} for _ in relations]
        summaries_metadata = summaries_metadata or [{} for _ in summaries]
        
        for meta in entities_metadata:
            meta["uuid"] = uuid
            
        for meta in relations_metadata:
            meta["uuid"] = uuid
            
        for meta in summaries_metadata:
            meta["uuid"] = uuid
    
    logger.info("开始存储实体")
    entities_result = db.store_entities(entities, entities_metadata)
    logger.info("开始存储关系")
    relations_result = db.store_relations(relations, relations_metadata)
    logger.info("开始存储摘要")
    summaries_result = db.store_summaries(summaries, summaries_metadata)
    
    result = {
        "entities": entities_result,
        "relations": relations_result,
        "summaries": summaries_result
    }
    
    logger.info("知识三元组存储完成")
    logger.debug(f"存储结果: {result}")
    return result