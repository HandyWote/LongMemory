"""
Enhanced database storage with configuration management
增强的数据库存储，包含配置管理
"""
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional
from component.config_manager import setup_system_config

# Configure logging / 配置日志
logger = logging.getLogger(__name__)

# Initialize configuration / 初始化配置
config_manager = setup_system_config()

class MemoryDatabase:
    def __init__(self, reset_database: bool = False):
        """
        Initialize memory database
        初始化记忆数据库
        
        Args:
            reset_database: Whether to reset the database (delete all data) / 是否重置数据库（删除所有数据）
        """
        db_config = config_manager.get_database_config()
        persist_directory = db_config.get("persist_directory", "./chroma_db")
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Only delete collections when explicitly requested to reset / 只有在明确要求重置时才删除集合
        if reset_database:
            try:
                self.client.delete_collection("entities")
                self.client.delete_collection("relations")
                self.client.delete_collection("summaries")
                print("Database has been reset")
            except:
                pass
        
        # Get or create collections / 获取或创建集合
        self.entities_collection = self.client.get_or_create_collection(
            name="entities"
        )
        self.relations_collection = self.client.get_or_create_collection(
            name="relations"
        )
        self.summaries_collection = self.client.get_or_create_collection(
            name="summaries"
        )
        
        print(f"Database initialization completed - entities: {self.entities_collection.count()}, relations: {self.relations_collection.count()}, summaries: {self.summaries_collection.count()}")
    
    def store_entities(self, texts: List[str], metadatas: Optional[List[Dict]] = None, 
                      embeddings: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        """
        Store entity information
        存储实体信息
        
        Args:
            texts: Original text list / 原文列表
            metadatas: Metadata list / 元数据列表
            embeddings: Vector list / 向量列表
            
        Returns:
            Storage result / 存储结果
        """
        # Check if it's an empty list / 检查是否为空列表
        if not texts:
            return {"ids": []}
            
        ids = [f"entity_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}" for i in range(len(texts))]
        
        # Record vector dimension information / 记录向量维度信息
        if embeddings:
            logger.info(f"Vector dimensions: {[len(e) for e in embeddings]}")
        
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
        Store relationship information
        存储关系信息
        
        Args:
            texts: Original text list / 原文列表
            metadatas: Metadata list / 元数据列表
            embeddings: Vector list / 向量列表
            
        Returns:
            Storage result / 存储结果
        """
        # Check if it's an empty list / 检查是否为空列表
        if not texts:
            return {"ids": []}
            
        ids = [f"relation_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}" for i in range(len(texts))]
        
        # Record vector dimension information / 记录向量维度信息
        if embeddings:
            logger.info(f"Vector dimensions: {[len(e) for e in embeddings]}")
        
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
        Store summary information
        存储摘要信息
        
        Args:
            texts: Original text list / 原文列表
            metadatas: Metadata list / 元数据列表
            embeddings: Vector list / 向量列表
            
        Returns:
            Storage result / 存储结果
        """
        # Check if it's an empty list / 检查是否为空列表
        if not texts:
            return {"ids": []}
            
        ids = [f"summary_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}" for i in range(len(texts))]
        
        # Record vector dimension information / 记录向量维度信息
        if embeddings:
            logger.info(f"Vector dimensions: {[len(e) for e in embeddings]}")
        
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
    Store knowledge triples (entities, relationships, summaries)
    存储知识三元组（实体、关系、摘要）
    
    Args:
        entities: Entity list / 实体列表
        relations: Relationship list / 关系列表
        summaries: Summary list / 摘要列表
        entities_metadata: Entity metadata / 实体元数据
        relations_metadata: Relationship metadata / 关系元数据
        summaries_metadata: Summary metadata / 摘要元数据
        uuid: User UUID, used to distinguish data from different users / 用户UUID，用于区分不同用户的数据
        
    Returns:
        Dictionary containing storage results for each part / 包含各部分存储结果的字典
    """
    db = MemoryDatabase()
    
    # If UUID is provided, add it to all metadata / 如果提供了UUID，则将其添加到所有元数据中
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
    
    entities_result = db.store_entities(entities, entities_metadata)
    relations_result = db.store_relations(relations, relations_metadata)
    summaries_result = db.store_summaries(summaries, summaries_metadata)
    
    result = {
        "entities": entities_result,
        "relations": relations_result,
        "summaries": summaries_result
    }
    
    return result
