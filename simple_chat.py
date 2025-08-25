"""
极简AI对话脚本
"""
import openai
import chromadb
import json
from datetime import datetime
import uuid
import os

class SimpleChat:
    def __init__(self, config_file="./config.json"):
        # 加载配置
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件 {config_file} 不存在")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 从配置中获取参数
        API_KEY = config.get("apiKey")
        CHAT_MODEL = config.get("chatModel", "gpt-3.5-turbo")
        EMBEDDING_MODEL = config.get("embeddingModel", "text-embedding-ada-002")
        DB_PATH = config.get("database", "./chroma_db")
        
        # 初始化OpenAI客户端
        self.client = openai.OpenAI(api_key=API_KEY)
        self.chat_model = CHAT_MODEL
        self.embedding_model = EMBEDDING_MODEL
        
        # 初始化ChromaDB客户端
        self.db_client = chromadb.PersistentClient(path=DB_PATH)
        
        # 获取或创建对话历史集合
        self.conversation_collection = self.db_client.get_or_create_collection(
            name="conversation_history"
        )
        
        # 生成用户UUID
        self.user_uuid = str(uuid.uuid4())
        
    def get_embedding(self, text):
        """获取文本的向量表示"""
        response = self.client.embeddings.create(
            input=[text],
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    def store_message(self, message, message_type="user"):
        """存储消息到数据库"""
        timestamp = datetime.now().isoformat()
        metadata = {
            "uuid": self.user_uuid,
            "timestamp": timestamp,
            "type": message_type
        }
        
        embedding = self.get_embedding(message)
        
        self.conversation_collection.add(
            ids=[f"{message_type}_{timestamp}"],
            documents=[message],
            metadatas=[metadata],
            embeddings=[embedding]
        )
    
    def retrieve_relevant_history(self, query, n_results=5):
        """检索相关对话历史"""
        query_embedding = self.get_embedding(query)
        
        results = self.conversation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"uuid": self.user_uuid}
        )
        
        # 格式化历史记录
        history = []
        if results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                history.append({
                    "role": metadata["type"],
                    "content": doc,
                    "timestamp": metadata["timestamp"]
                })
            
            # 按时间排序
            history.sort(key=lambda x: x["timestamp"])
            
        return history
    
    def generate_response(self, user_input):
        """生成AI回复"""
        # 存储用户输入
        self.store_message(user_input, "user")
        
        # 检索相关历史
        history = self.retrieve_relevant_history(user_input)
        
        # 构建对话上下文
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Use the conversation history to provide contextually relevant responses."}
        ]
        
        # 添加历史对话（限制最近10条）
        for item in history[-10:]:
            messages.append({
                "role": item["role"],
                "content": item["content"]
            })
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})
        
        # 调用OpenAI API生成回复
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        # 存储AI回复
        self.store_message(ai_response, "assistant")
        
        return ai_response

def main():
    """主函数"""
    print("极简AI对话系统")
    print("输入 'quit' 或 'exit' 退出对话")
    print("-" * 30)
    
    try:
        # 初始化对话系统
        chat = SimpleChat()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    except Exception as e:
        print(f"初始化对话系统时出错: {e}")
        return
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n您: ").strip()
            
            # 检查退出命令
            if user_input.lower() in ['quit', 'exit']:
                print("AI: 再见！")
                break
            
            # 跳过空输入
            if not user_input:
                continue
            
            # 生成并打印AI回复
            ai_response = chat.generate_response(user_input)
            print(f"AI: {ai_response}")
            
        except KeyboardInterrupt:
            print("\n\nAI: 再见！")
            break
        except Exception as e:
            print(f"AI: 对不起，我遇到了一个错误: {str(e)}")

if __name__ == "__main__":
    main()