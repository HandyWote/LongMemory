"""
将向量化的数据格式化存入数据库
"""
import chromadb
from datetime import datetime
from typing import Dict, List, Any, Optional

class MemoryDatabase:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        初始化记忆数据库
        
        Args:
            persist_directory: 数据库存储目录
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.entities_collection = self.client.get_or_create_collection("entities")
        self.relations_collection = self.client.get_or_create_collection("relations")
        self.summaries_collection = self.client.get_or_create_collection("summaries")
    
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
        ids = [f"entity_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}" for i in range(len(texts))]
        self.entities_collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas or [{} for _ in texts],
            embeddings=embeddings
        )
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
        ids = [f"relation_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}" for i in range(len(texts))]
        self.relations_collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas or [{} for _ in texts],
            embeddings=embeddings
        )
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
        ids = [f"summary_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}" for i in range(len(texts))]
        self.summaries_collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas or [{} for _ in texts],
            embeddings=embeddings
        )
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
    db = MemoryDatabase()
    
    # 如果提供了UUID，则将其添加到所有元数据中
    if uuid:
        entities_metadata = entities_metadata or [{} for _ in entities]
        relations_metadata = relations_metadata or [{} for _ in relations]
        summaries_metadata = summaries_metadata or [{} for _ in summaries]
        
        for meta in entities_metadata:
            meta["uuid"] = uuid
            
        for meta in relations_metadata:
            meta["uuid"] = uuid
            
        for meta in summaries_metadata:
            meta["uuid"] = uuid
    
    result = {
        "entities": db.store_entities(entities, entities_metadata),
        "relations": db.store_relations(relations, relations_metadata),
        "summaries": db.store_summaries(summaries, summaries_metadata)
    }
    
    return result