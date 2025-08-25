"""
Production-grade forgetting mechanism based on database scoring
生产级遗忘机制 - 基于数据库评分的智能遗忘系统
"""
import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
from component.config_manager import setup_system_config

logger = logging.getLogger(__name__)

# Initialize configuration / 初始化配置
config_manager = setup_system_config()

class ForgettingStage(Enum):
    """Forgetting stage definitions / 遗忘阶段定义"""
    INTACT = "intact"        # Intact retention / 完整保留
    COMPRESSED = "compressed"  # Compressed / 已压缩
    ARCHIVED = "archived"    # Archived / 已归档
    FORGOTTEN = "forgotten"  # Forgotten / 已遗忘

@dataclass
class MemoryScore:
    """Memory scoring data structure / 记忆评分数据结构"""
    memory_id: str
    base_score: float        # Base importance score (0-1) / 基础重要性评分 (0-1)
    recency_boost: float     # Recency weighting / 时效性加权
    emotional_weight: float  # Emotional weight / 情感权重
    access_frequency: int    # Access frequency / 访问频率
    compression_ratio: float # Compression ratio / 压缩率
    final_score: float       # Final score / 最终评分

class AdvancedForgettingSystem:
    """Advanced forgetting system / 高级遗忘系统"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config_manager.get_database_config().get("database", "memory.db")
        self._init_database()
        
    def _init_database(self):
        """Initialize database schema / 初始化数据库架构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Memory scoring table / 记忆评分表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_scores (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content_hash TEXT,
                base_score REAL DEFAULT 0.5,
                recency_boost REAL DEFAULT 1.0,
                emotional_weight REAL DEFAULT 0.0,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                compression_ratio REAL DEFAULT 1.0,
                forgetting_stage TEXT DEFAULT 'intact',
                compressed_content TEXT,
                semantic_tags TEXT  -- JSON array
            )
        ''')
        
        # Forgetting log table / 遗忘日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS forgetting_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT,
                old_stage TEXT,
                new_stage TEXT,
                reason TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (memory_id) REFERENCES memory_scores(memory_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def calculate_memory_score(self, memory_data: Dict) -> MemoryScore:
        """Calculate comprehensive memory score / 计算记忆综合评分"""
        now = datetime.now()
        created_at = datetime.fromisoformat(memory_data['created_at'])
        age_days = (now - created_at).days
        
        # Time decay factor / 时间衰减系数
        recency_boost = np.exp(-age_days / 30)  # 30-day half-life / 30天半衰期
        
        # Emotional weight calculation / 情感权重计算
        emotional_weight = memory_data.get('emotional_score', 0.0)
        
        # Access frequency weight / 访问频率权重
        access_freq = memory_data.get('access_count', 0)
        frequency_boost = min(access_freq * 0.1, 0.5)
        
        # Base score adjustment / 基础评分调整
        base_score = memory_data.get('importance_score', 0.5)
        
        # Final score calculation / 最终评分计算
        final_score = (
            base_score * 0.4 +
            recency_boost * 0.3 +
            emotional_weight * 0.2 +
            frequency_boost * 0.1
        )
        
        return MemoryScore(
            memory_id=memory_data['memory_id'],
            base_score=base_score,
            recency_boost=recency_boost,
            emotional_weight=emotional_weight,
            access_frequency=access_freq,
            compression_ratio=memory_data.get('compression_ratio', 1.0),
            final_score=final_score
        )
    
    def should_forget(self, memory_id: str, threshold: float = None) -> Tuple[bool, str]:
        """Determine if memory should be forgotten / 判断是否应该遗忘"""
        if threshold is None:
            threshold = config_manager.get_memory_config().get("forgetting_threshold", 0.2)
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM memory_scores WHERE memory_id = ?
        ''', (memory_id,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False, "Memory not found"
            
        # Convert row to dict / 转换行为字典
        columns = [desc[0] for desc in cursor.description]
        memory_data = dict(zip(columns, row))
        
        score = self.calculate_memory_score(memory_data)
        
        # Forgetting decision logic / 遗忘决策逻辑
        if score.final_score < threshold:
            reason = f"Low score: {score.final_score:.2f} < {threshold}"
            should_forget = True
        elif memory_data['forgetting_stage'] == 'intact' and score.compression_ratio > 0.8:
            reason = "Ready for compression"
            should_forget = False
        elif memory_data['forgetting_stage'] == 'compressed' and score.final_score < 0.5:
            reason = "Ready for archival"
            should_forget = False
        else:
            reason = "Memory retained"
            should_forget = False
            
        conn.close()
        return should_forget, reason
    
    def compress_memory(self, memory_id: str, content: str) -> str:
        """Intelligently compress memory content / 智能压缩记忆内容"""
        # Here we can integrate LLM for intelligent compression
        # 这里可以集成LLM进行智能压缩
        
        # Extract key information / 提取关键信息
        sentences = content.split('. ')
        if len(sentences) <= 3:
            return content
            
        # Keep first, last, and longest sentences / 保留首尾句和最长句
        key_sentences = [sentences[0], sentences[-1]]
        longest = max(sentences[1:-1], key=len) if len(sentences) > 2 else ""
        if longest:
            key_sentences.insert(1, longest)
            
        compressed = '. '.join(key_sentences)
        
        # Add compression marker / 添加压缩标记
        return f"[COMPRESSED] {compressed}"
    
    def execute_forgetting(self, user_id: str, dry_run: bool = False) -> Dict[str, List[str]]:
        """Execute forgetting process / 执行遗忘过程"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all user memories / 获取用户所有记忆
        cursor.execute('''
            SELECT memory_id FROM memory_scores 
            WHERE user_id = ? AND forgetting_stage != 'forgotten'
            ORDER BY created_at ASC
        ''', (user_id,))
        
        memories = cursor.fetchall()
        actions = {
            "compressed": [],
            "archived": [],
            "forgotten": [],
            "retained": []
        }
        
        for (memory_id,) in memories:
            should_forget, reason = self.should_forget(memory_id)
            
            cursor.execute('''
                SELECT forgetting_stage, content FROM memory_scores 
                WHERE memory_id = ?
            ''', (memory_id,))
            current_stage, content = cursor.fetchone()
            
            if dry_run:
                # Analysis only / 仅分析不执行
                if should_forget:
                    actions["forgotten"].append(memory_id)
                else:
                    actions["retained"].append(memory_id)
                continue
            
            # Execute forgetting stage transition / 执行遗忘阶段转换
            new_stage = current_stage
            compressed_content = None
            
            if current_stage == "intact" and self.should_compress(memory_id):
                new_stage = "compressed"
                compressed_content = self.compress_memory(memory_id, content)
                actions["compressed"].append(memory_id)
                
            elif current_stage == "compressed" and should_forget:
                new_stage = "archived"
                actions["archived"].append(memory_id)
                
            elif current_stage == "archived" and should_forget:
                new_stage = "forgotten"
                actions["forgotten"].append(memory_id)
            else:
                actions["retained"].append(memory_id)
            
            # Update memory state / 更新记忆状态
            cursor.execute('''
                UPDATE memory_scores 
                SET forgetting_stage = ?, 
                    compressed_content = ?,
                    last_accessed = ?
                WHERE memory_id = ?
            ''', (new_stage, compressed_content, datetime.now(), memory_id))
            
            # Log forgetting action / 记录遗忘日志
            cursor.execute('''
                INSERT INTO forgetting_log (memory_id, old_stage, new_stage, reason)
                VALUES (?, ?, ?, ?)
            ''', (memory_id, current_stage, new_stage, reason))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Forgetting process completed for user {user_id}: {actions}")
        return actions
    
    def should_compress(self, memory_id: str) -> bool:
        """Determine if memory should be compressed / 判断是否应该压缩"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT compression_ratio FROM memory_scores WHERE memory_id = ?
        ''', (memory_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0]:
            return row[0] > 0.7  # Trigger compression when compression ratio is greater than 70% / 压缩率大于70%时触发压缩
        return True
    
    def update_access_count(self, memory_id: str):
        """Update access count / 更新访问计数"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE memory_scores 
            SET access_count = access_count + 1,
                last_accessed = ?
            WHERE memory_id = ?
        ''', (datetime.now(), memory_id))
        
        conn.commit()
        conn.close()

# Global forgetting system instance / 全局遗忘系统实例
forgetting_system = AdvancedForgettingSystem()

# Convenient forgetting interface / 便捷的遗忘接口
def should_forget_memory(memory_id: str, threshold: float = None) -> Tuple[bool, str]:
    """Check if memory should be forgotten / 判断记忆是否应该遗忘"""
    return forgetting_system.should_forget(memory_id, threshold)

def execute_user_forgetting(user_id: str, dry_run: bool = False) -> Dict[str, List[str]]:
    """Execute user-level forgetting / 执行用户级别的遗忘"""
    return forgetting_system.execute_forgetting(user_id, dry_run)