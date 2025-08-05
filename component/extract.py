"""
旨在提取用户消息中的实体以及关系
如：
用户是xxx。
用户喜欢xxx
"""
import openai
import json
with open ("./config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    API_KEY = config["apiKey"]
    EMBEDDING_MODEL = config["embeddingModel"]
    CHAT_MODEL = config["chatModel"]

# 初始化OpenAI客户端
client = openai.OpenAI(api_key=API_KEY)

def extract_entities(message):
    """
    专门提取文本中的实体
    
    Args:
        message (str): 用户输入的消息
        
    Returns:
        list: 实体列表，每个实体包含name和type
    """
    sys_prompt = """
    你是一个专业的实体识别专家，需要从文本中提取所有重要的实体。

    请严格按照以下JSON格式输出：
    {
        "entities": [
            {"name": "实体名称", "type": "实体类型"}
        ]
    }

    实体类型包括：人物、地点、物品、技术、概念、组织、时间等。
    
    示例：
    输入："张三喜欢篮球，他在北京工作"
    输出：{
        "entities": [
            {"name": "张三", "type": "人物"},
            {"name": "篮球", "type": "物品"},
            {"name": "北京", "type": "地点"}
        ]
    }

    注意：
    1. 只提取明确提到的实体
    2. 实体名称要完整准确
    3. 实体类型要恰当分类
    4. 不要遗漏重要实体
    5. 只返回JSON格式，不要添加任何其他文字
    """
    
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"请从以下文本中提取实体：{message}"}
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

def extract_relations(message, entities):
    """
    专门提取文本中的关系
    
    Args:
        message (str): 用户输入的消息
        entities (list): 已提取的实体列表
        
    Returns:
        list: 关系列表，每个关系包含subject、relation和object
    """
    sys_prompt = """
    你是一个专业的关系抽取专家，需要从文本中提取实体之间的关系。

    请严格按照以下JSON格式输出：
    {
        "relations": [
            {"subject": "主语实体", "relation": "关系", "object": "宾语实体"}
        ]
    }

    关系类型包括：喜欢、是、位于、使用、拥有、属于、包含、创建、开发、工作等。
    
    示例：
    输入："张三喜欢篮球，他在北京工作"
    输出：{
        "relations": [
            {"subject": "张三", "relation": "喜欢", "object": "篮球"},
            {"subject": "张三", "relation": "位于", "object": "北京"}
        ]
    }

    注意：
    1. 只提取明确存在的关系
    2. 关系的主语和宾语必须是文本中提到的实体
    3. 关系类型要准确描述实体间的联系
    4. 确保关系的逻辑正确
    5. 只返回JSON格式，不要添加任何其他文字
    """
    
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"请从以下文本中提取关系：{message}"}
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

def extract_summary(message):
    """
    专门生成文本的摘要
    
    Args:
        message (str): 用户输入的消息
        
    Returns:
        str: 文本摘要
    """
    sys_prompt = """
    你是一个专业的文本摘要专家，需要为文本生成简洁准确的摘要。

    要求：
    1. 摘要要包含文本的核心信息
    2. 语言简洁明了，避免冗余
    3. 保持原文的主要观点和关键信息
    4. 摘要长度控制在50-200字之间
    5. 直接返回摘要文本，不要添加任何解释或格式

    示例：
    输入："张三喜欢篮球，他在北京工作，是一名软件工程师"
    输出："张三是一名在北京工作的软件工程师，喜欢篮球"
    """
    
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"请为以下文本生成摘要：{message}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
            
    except Exception as e:
        return ""

def extract_entity_relation(message):
    """
    使用三次独立的API调用分别提取实体、关系和摘要
    
    Args:
        message (str): 用户输入的消息
        
    Returns:
        dict: 包含提取的实体和关系的字典
    """
    try:
        # 第一步：提取实体
        entities = extract_entities(message)
        
        # 第二步：提取关系
        relations = extract_relations(message, entities)
        
        # 第三步：生成摘要
        summary = extract_summary(message)
        
        # 组合结果
        result = {
            "entities": entities,
            "relations": relations,
            "summary": summary
        }
        
        return result
        
    except Exception as e:
        # 处理整体异常
        return {
            "error": f"提取过程中出错: {str(e)}"
        }
    
if __name__ == "__main__":
    msg = """技术的演进，让全栈的门槛变低了
曾几何时，前端和后端是两个泾渭分明、需要完全不同技能集的领域。前端写HTML/CSS/JS，后端搞Java/PHP/Python，中间隔着一条API的银河。
但现在呢？

Node.js的出现，让JavaScript统一了前后端语言。
Next.js, Nuxt 这类元框架，把路由、数据获取、服务端渲染这些原本属于后端一部分的工作，无缝地集成到了前端的开发流程里。
tRPC 这类工具，甚至能让前后端共享类型，连写API文档都省了。
Vercel, Netlify 这类平台，把部署、CDN、Serverless函数这些复杂的运维工作，变成了一键式的傻瓜操作。

技术的发展，正在疯狂地模糊前端和后端的边界。一个熟悉JavaScript的前端，几乎可以无缝地去写服务端的逻辑。
"""
    print(extract_entity_relation(msg)['summary'])
