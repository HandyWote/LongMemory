"""
Enhanced entity and relationship extraction from user messages
增强的用户消息实体和关系提取
"""
import openai
import json
from typing import Dict, List, Any
from component.config_manager import setup_system_config

# Initialize configuration / 初始化配置
config_manager = setup_system_config()

# Initialize OpenAI client / 初始化OpenAI客户端
client = openai.OpenAI(api_key=config_manager.get_api_key())

def extract_entities(message: str) -> List[Dict[str, str]]:
    """
    Extract entities from text
    从文本中提取实体
    
    Args:
        message (str): User input message / 用户输入消息
        
    Returns:
        list: Entity list, each entity contains name and type / 实体列表，每个实体包含名称和类型
    """
    sys_prompt = """
    You are a professional entity recognition expert who needs to extract all important entities from the text.

    Please output strictly in the following JSON format:
    {
        "entities": [
            {"name": "Entity Name", "type": "Entity Type"}
        ]
    }

    Entity types include: Person, Location, Object, Technology, Concept, Organization, Time, etc.
    
    Example:
    Input: "Zhang San likes basketball, he works in Beijing"
    Output: {
        "entities": [
            {"name": "Zhang San", "type": "Person"},
            {"name": "basketball", "type": "Object"},
            {"name": "Beijing", "type": "Location"}
        ]
    }

    Note:
    1. Only extract explicitly mentioned entities
    2. Entity names should be complete and accurate
    3. Entity types should be appropriately classified
    4. Do not omit important entities
    5. Return only JSON format, do not add any other text
    """
    
    try:
        response = client.chat.completions.create(
            model=config_manager.get_chat_model(),
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Please extract entities from the following text: {message}"}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        result = response.choices[0].message.content.strip()
        try:
            data = json.loads(result)
            return data.get("entities", [])
        except json.JSONDecodeError:
            return []
            
    except Exception as e:
        return []

def extract_relations(message: str, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Extract relationships from text
    从文本中提取关系
    
    Args:
        message (str): User input message / 用户输入消息
        entities (list): Extracted entity list / 已提取的实体列表
        
    Returns:
        list: Relationship list, each relationship contains subject, relation and object / 关系列表，每个关系包含主语、关系和宾语
    """
    sys_prompt = """
    You are a professional relationship extraction expert who needs to extract relationships between entities from the text.

    Please output strictly in the following JSON format:
    {
        "relations": [
            {"subject": "Subject Entity", "relation": "Relation", "object": "Object Entity"}
        ]
    }

    Relationship types include: likes, is, located in, uses, owns, belongs to, contains, creates, develops, works, etc.
    
    Example:
    Input: "Zhang San likes basketball, he works in Beijing"
    Output: {
        "relations": [
            {"subject": "Zhang San", "relation": "likes", "object": "basketball"},
            {"subject": "Zhang San", "relation": "located in", "object": "Beijing"}
        ]
    }

    Note:
    1. Only extract relationships that clearly exist
    2. The subject and object of the relationship must be entities mentioned in the text
    3. The relationship type should accurately describe the connection between entities
    4. Ensure the logic of the relationship is correct
    5. Return only JSON format, do not add any other text
    """
    
    try:
        response = client.chat.completions.create(
            model=config_manager.get_chat_model(),
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Please extract relationships from the following text: {message}"}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        result = response.choices[0].message.content.strip()
        try:
            data = json.loads(result)
            return data.get("relations", [])
        except json.JSONDecodeError:
            return []
            
    except Exception as e:
        return []

def extract_summary(message: str) -> str:
    """
    Generate text summary
    生成文本摘要
    
    Args:
        message (str): User input message / 用户输入消息
        
    Returns:
        str: Text summary / 文本摘要
    """
    sys_prompt = """
    You are a professional text summarization expert who needs to generate a concise and accurate summary of the text.

    Requirements:
    1. The summary should contain the core information of the text
    2. The language should be concise and clear, avoiding redundancy
    3. Maintain the main points and key information of the original text
    4. Control the summary length between 50-200 words
    5. Return the summary text directly, do not add any explanation or format

    Example:
    Input: "Zhang San likes basketball, he works in Beijing as a software engineer"
    Output: "Zhang San is a software engineer working in Beijing who likes basketball"
    """
    
    try:
        response = client.chat.completions.create(
            model=config_manager.get_chat_model(),
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Please generate a summary for the following text: {message}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
            
    except Exception as e:
        return ""

def extract_entity_relation(message: str) -> Dict[str, Any]:
    """
    Use three independent API calls to extract entities, relationships and summary respectively
    使用三次独立的API调用分别提取实体、关系和摘要
    
    Args:
        message (str): User input message / 用户输入消息
        
    Returns:
        dict: Dictionary containing extracted entities and relationships / 包含提取的实体和关系的字典
    """
    try:
        # Step 1: Extract entities / 第一步：提取实体
        entities = extract_entities(message)
        
        # Step 2: Extract relationships / 第二步：提取关系
        relations = extract_relations(message, entities)
        
        # Step 3: Generate summary / 第三步：生成摘要
        summary = extract_summary(message)
        
        # Combine results / 组合结果
        result = {
            "entities": entities,
            "relations": relations,
            "summary": summary
        }
        
        return result
        
    except Exception as e:
        # Handle overall exception / 处理整体异常
        return {
            "error": f"Error during extraction: {str(e)}"
        }
