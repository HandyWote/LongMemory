"""
Production-grade user profile generator based on vector database
生产级用户画像生成器 - 基于向量数据库的智能画像系统
"""
import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from openai import OpenAI
from component.config_manager import setup_system_config

logger = logging.getLogger(__name__)

# Initialize configuration / 初始化配置
config_manager = setup_system_config()

@dataclass
class UserProfile:
    """User profile data structure / 用户画像数据结构"""
    user_id: str
    personality_traits: Dict[str, float]
    knowledge_areas: List[str]
    emotional_patterns: Dict[str, any]
    communication_style: str
    relationship_timeline: List[Dict]
    preference_vector: List[float]
    last_updated: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format / 转换为字典格式"""
        return {
            "user_id": self.user_id,
            "personality_traits": self.personality_traits,
            "knowledge_areas": self.knowledge_areas,
            "emotional_patterns": self.emotional_patterns,
            "communication_style": self.communication_style,
            "relationship_timeline": self.relationship_timeline,
            "preference_vector": self.preference_vector,
            "last_updated": self.last_updated.isoformat()
        }

class AdvancedUserProfiler:
    """Advanced user profiling system / 高级用户画像系统"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config_manager.get_database_config().get("database", "memory.db")
        self.client = OpenAI(api_key=config_manager.get_api_key())
        self._init_database()
        
    def _init_database(self):
        """Initialize user profile database / 初始化用户画像数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User profiles table / 用户画像表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                profile_data TEXT NOT NULL,  -- JSON format / JSON格式
                profile_vector TEXT,         -- Vector representation / 向量表示
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        ''')
        
        # User interaction history / 用户交互历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                interaction_type TEXT,
                content_summary TEXT,
                emotional_tone REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def extract_user_traits(self, user_id: str) -> Dict[str, float]:
        """Extract user personality traits / 提取用户性格特征"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all user memory content / 获取用户所有记忆内容
        cursor.execute('''
            SELECT content FROM memory_scores 
            WHERE user_id = ? AND forgetting_stage != 'forgotten'
            ORDER BY created_at DESC
            LIMIT 100
        ''', (user_id,))
        
        memories = cursor.fetchall()
        conn.close()
        
        if not memories:
            return self._get_default_traits()
        
        # Combine all content for analysis / 合并所有内容进行分析
        combined_content = " ".join([m[0] for m in memories])
        
        # Use AI to analyze personality traits / 使用AI分析性格特征
        try:
            response = self.client.chat.completions.create(
                model=config_manager.get_chat_model(),
                messages=[
                    {
                        "role": "system",
                        "content": """Analyze the user's personality traits based on their conversation history.
                        基于对话历史分析用户性格特征，返回JSON格式：
                        {
                            "analytical": 0.0-1.0,
                            "emotional": 0.0-1.0,
                            "creative": 0.0-1.0,
                            "social": 0.0-1.0,
                            "detail_oriented": 0.0-1.0,
                            "open_minded": 0.0-1.0
                        }"""
                    },
                    {
                        "role": "user",
                        "content": combined_content[:4000]  # Limit length / 限制长度
                    }
                ],
                temperature=0.3
            )
            
            traits = json.loads(response.choices[0].message.content)
            return traits
            
        except Exception as e:
            logger.error(f"Error extracting traits for user {user_id}: {e}")
            return self._get_default_traits()
    
    def identify_knowledge_areas(self, user_id: str) -> List[str]:
        """Identify user knowledge areas / 识别用户知识领域"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get entities and relations / 获取实体和关系
        cursor.execute('''
            SELECT entities, relations FROM toDatabase 
            WHERE uuid = ? 
            ORDER BY timestamp DESC
            LIMIT 50
        ''', (user_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        # Extract keywords / 提取关键词
        keywords = []
        for entities, relations in results:
            if entities:
                try:
                    entities_list = json.loads(entities)
                    keywords.extend(entities_list)
                except:
                    keywords.extend(entities.split(','))
        
        # Deduplicate and sort / 去重和排序
        knowledge_areas = list(set([k.strip() for k in keywords if len(k.strip()) > 2]))
        return knowledge_areas[:20]  # Limit number of areas / 限制领域数量
    
    def analyze_emotional_patterns(self, user_id: str) -> Dict[str, any]:
        """Analyze user emotional patterns / 分析用户情感模式"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get last 30 days interactions / 获取最近30天的交互
        thirty_days_ago = datetime.now() - timedelta(days=30)
        
        cursor.execute('''
            SELECT emotional_tone, timestamp FROM user_interactions 
            WHERE user_id = ? AND timestamp > ?
            ORDER BY timestamp ASC
        ''', (user_id, thirty_days_ago.isoformat()))
        
        interactions = cursor.fetchall()
        conn.close()
        
        if not interactions:
            return {
                "average_tone": 0.0,
                "mood_stability": 1.0,
                "emotional_triggers": [],
                "comfort_topics": []
            }
        
        # Calculate emotional statistics / 计算情感统计
        tones = [i[0] for i in interactions if i[0] is not None]
        if not tones:
            tones = [0.0]
        
        avg_tone = np.mean(tones)
        mood_stability = 1.0 - np.std(tones)  # Higher stability, lower variance / 稳定性越高，方差越小
        
        # Identify emotional triggers / 识别情感触发器
        triggers = self._identify_emotional_triggers(user_id)
        
        return {
            "average_tone": float(avg_tone),
            "mood_stability": float(mood_stability),
            "emotional_triggers": triggers,
            "comfort_topics": self._identify_comfort_topics(user_id)
        }
    
    def _identify_emotional_triggers(self, user_id: str) -> List[str]:
        """Identify emotional trigger topics / 识别情感触发话题"""
        # Simplified implementation / 简化实现
        return ["work", "family", "health", "relationships"][:3]
    
    def _identify_comfort_topics(self, user_id: str) -> List[str]:
        """Identify comfort topics / 识别舒适话题"""
        # Simplified implementation / 简化实现
        return ["hobbies", "food", "travel", "learning"][:3]
    
    def determine_communication_style(self, user_id: str) -> str:
        """Determine user communication style / 确定用户沟通风格"""
        traits = self.extract_user_traits(user_id)
        
        # Determine style based on traits / 基于性格特征判断风格
        if traits.get("analytical", 0) > 0.7:
            return "analytical_precise"
        elif traits.get("emotional", 0) > 0.7:
            return "empathetic_supportive"
        elif traits.get("creative", 0) > 0.7:
            return "creative_exploratory"
        else:
            return "balanced_adaptive"
    
    def build_relationship_timeline(self, user_id: str) -> List[Dict]:
        """Build relationship development timeline / 构建关系发展时间线"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get important interaction nodes / 获取重要交互节点
        cursor.execute('''
            SELECT content_summary, emotional_tone, timestamp
            FROM user_interactions 
            WHERE user_id = ?
            ORDER BY timestamp ASC
        ''', (user_id,))
        
        interactions = cursor.fetchall()
        conn.close()
        
        timeline = []
        for content, tone, timestamp in interactions[-10:]:  # Last 10 times / 最近10次
            timeline.append({
                "milestone": content[:50] + "...",
                "emotional_context": tone,
                "date": timestamp,
                "relationship_depth": self._calculate_relationship_depth(tone)
            })
        
        return timeline
    
    def _calculate_relationship_depth(self, emotional_tone: float) -> str:
        """Calculate relationship depth / 计算关系深度"""
        if emotional_tone > 0.7:
            return "deep_trust"
        elif emotional_tone > 0.3:
            return "established"
        elif emotional_tone > -0.3:
            return "developing"
        else:
            return "initial"
    
    def _get_default_traits(self) -> Dict[str, float]:
        """Get default personality traits / 获取默认性格特征"""
        return {
            "analytical": 0.5,
            "emotional": 0.5,
            "creative": 0.5,
            "social": 0.5,
            "detail_oriented": 0.5,
            "open_minded": 0.5
        }
    
    def generate_user_profile(self, user_id: str) -> UserProfile:
        """Generate complete user profile / 生成完整用户画像"""
        traits = self.extract_user_traits(user_id)
        knowledge = self.identify_knowledge_areas(user_id)
        emotional = self.analyze_emotional_patterns(user_id)
        communication = self.determine_communication_style(user_id)
        timeline = self.build_relationship_timeline(user_id)
        
        # Generate preference vector / 生成偏好向量
        preference_vector = [
            traits.get("analytical", 0.5),
            traits.get("emotional", 0.5),
            emotional.get("average_tone", 0.0),
            len(knowledge) / 20.0,  # Normalized / 归一化
            emotional.get("mood_stability", 1.0)
        ]
        
        profile = UserProfile(
            user_id=user_id,
            personality_traits=traits,
            knowledge_areas=knowledge,
            emotional_patterns=emotional,
            communication_style=communication,
            relationship_timeline=timeline,
            preference_vector=preference_vector,
            last_updated=datetime.now()
        )
        
        # Save to database / 保存到数据库
        self._save_profile(profile)
        
        return profile
    
    def _save_profile(self, profile: UserProfile):
        """Save user profile to database / 保存用户画像到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Use UPSERT to update or insert / 使用UPSERT更新或插入
        cursor.execute('''
            INSERT OR REPLACE INTO user_profiles 
            (user_id, profile_data, profile_vector, updated_at)
            VALUES (?, ?, ?, ?)
        ''', (
            profile.user_id,
            json.dumps(profile.to_dict()),
            json.dumps(profile.preference_vector),
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
    
    def get_user_profile_prompt(self, user_id: str) -> str:
        """Get user profile prompt / 获取用户画像提示词"""
        profile = self.generate_user_profile(user_id)
        
        # Generate personalized prompt / 生成个性化提示词
        prompt = f"""
        User Profile Analysis:
        用户画像分析：
        
        Basic characteristics:
        基本特征：
        - Personality: {', '.join([k for k, v in profile.personality_traits.items() if v > 0.7])}
        - 性格：{', '.join([k for k, v in profile.personality_traits.items() if v > 0.7])}
        - Knowledge areas: {', '.join(profile.knowledge_areas[:5])}
        - 知识领域：{', '.join(profile.knowledge_areas[:5])}
        - Communication style: {profile.communication_style}
        - 沟通风格：{profile.communication_style}
        
        Emotional patterns:
        情感模式：
        - Average emotional tone: {profile.emotional_patterns['average_tone']:.2f}
        - 平均情感基调：{profile.emotional_patterns['average_tone']:.2f}
        - Mood stability: {profile.emotional_patterns['mood_stability']:.2f}
        - 情绪稳定性：{profile.emotional_patterns['mood_stability']:.2f}
        - Trigger topics: {', '.join(profile.emotional_patterns['emotional_triggers'])}
        - 触发话题：{', '.join(profile.emotional_patterns['emotional_triggers'])}
        
        Relationship status:
        关系状态：
        - Relationship depth: {profile.relationship_timeline[-1]['relationship_depth'] if profile.relationship_timeline else 'initial'}
        - 关系深度：{profile.relationship_timeline[-1]['relationship_depth'] if profile.relationship_timeline else 'initial'}
        - Shared experiences: {len(profile.relationship_timeline)} important milestones
        - 共同经历：{len(profile.relationship_timeline)} 个重要节点
        
        Recommended interaction approach:
        建议交互方式：
        - Adopt {profile.communication_style} style
        - 采用{profile.communication_style}风格
        - Focus on {', '.join(profile.knowledge_areas[:3])} topics
        - 关注{', '.join(profile.knowledge_areas[:3])}话题
        - Avoid {', '.join(profile.emotional_patterns['emotional_triggers'])} and other sensitive topics
        - 避免{', '.join(profile.emotional_patterns['emotional_triggers'])}等敏感话题
        """
        
        return prompt.strip()
    
    def update_interaction(self, user_id: str, interaction_type: str, 
                          content: str, emotional_tone: float = 0.0):
        """Update user interaction record / 更新用户交互记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_interactions 
            (user_id, interaction_type, content_summary, emotional_tone)
            VALUES (?, ?, ?, ?)
        ''', (user_id, interaction_type, content[:200], emotional_tone))
        
        conn.commit()
        conn.close()

# Global user profiler instance / 全局用户画像实例
user_profiler = AdvancedUserProfiler()

# Convenient interfaces / 便捷接口
def generate_user_profile(user_id: str) -> Dict:
    """Generate user profile / 生成用户画像"""
    return user_profiler.generate_user_profile(user_id).to_dict()

def get_user_profile_prompt(user_id: str) -> str:
    """Get user profile prompt / 获取用户画像提示词"""
    return user_profiler.get_user_profile_prompt(user_id)

def update_user_interaction(user_id: str, interaction_type: str, 
                          content: str, emotional_tone: float = 0.0):
    """Update user interaction / 更新用户交互"""
    user_profiler.update_interaction(user_id, interaction_type, content, emotional_tone)