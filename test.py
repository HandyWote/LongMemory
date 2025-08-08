#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
查询向量数据库并使用AI生成回复的脚本
使用新的错误处理和日志系统
"""

import sys
import json
import logging
import argparse
from typing import Dict, Any, Optional

# 导入错误处理和配置管理
from component.error_handler import (
    setup_system_logging, 
    get_error_handler, 
    APIError,
    DatabaseError,
    ProcessingError
)
from component.config_manager import setup_system_config
from component.toEmbeding import TextEmbedder
from component.toDatabase import MemoryDatabase

# 设置系统日志和配置
setup_system_logging()
config_manager = setup_system_config()
error_handler = get_error_handler(__name__)

# 导入OpenAI
try:
    import openai
    from openai import OpenAI
    client = OpenAI(api_key=config_manager.get_api_key())
except ImportError:
    error_handler.logger.error("OpenAI库未安装，请运行: pip install openai")
    sys.exit(1)

class DatabaseQuerySystem:
    """数据库查询系统"""
    
    def __init__(self):
        """初始化查询系统"""
        self.db = None
        self.embedder = None
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化数据库和嵌入器组件"""
        try:
            error_handler.logger.info("初始化数据库组件")
            db_config = config_manager.get_database_config()
            self.db = MemoryDatabase(
                persist_directory=db_config['persist_directory'],
                reset_database=db_config['reset_database']
            )
            
            error_handler.logger.info("初始化文本嵌入器")
            self.embedder = TextEmbedder(api_key=config_manager.get_api_key())
            
        except Exception as e:
            error_handler.log_and_raise(
                ProcessingError(f"初始化组件失败: {str(e)}", details={"phase": "initialization"})
            )
    
    def query_database(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        使用查询文本向量搜索数据库
        
        Args:
            query_text: 查询文本
            n_results: 返回结果数量
            
        Returns:
            数据库查询结果
        """
        try:
            error_handler.logger.info(f"开始查询数据库: {query_text}")
            
            # 生成查询文本的向量
            query_embedding = self.embedder.get_embedding(query_text)
            error_handler.logger.debug(f"生成查询向量，维度: {len(query_embedding)}")
            
            # 在所有集合中搜索相似项
            results = {}
            
            # 查询实体
            try:
                entities_result = self.db.entities_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
                results['entities'] = entities_result
                error_handler.logger.info(f"查询到 {len(entities_result.get('documents', [[]])[0])} 个实体")
            except Exception as e:
                error_handler.logger.error(f"查询实体时出错: {e}")
                results['entities'] = {}
            
            # 查询关系
            try:
                relations_result = self.db.relations_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
                results['relations'] = relations_result
                error_handler.logger.info(f"查询到 {len(relations_result.get('documents', [[]])[0])} 个关系")
            except Exception as e:
                error_handler.logger.error(f"查询关系时出错: {e}")
                results['relations'] = {}
            
            # 查询摘要
            try:
                summaries_result = self.db.summaries_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
                results['summaries'] = summaries_result
                error_handler.logger.info(f"查询到 {len(summaries_result.get('documents', [[]])[0])} 个摘要")
            except Exception as e:
                error_handler.logger.error(f"查询摘要时出错: {e}")
                results['summaries'] = {}
            
            error_handler.logger.info("数据库查询完成")
            return results
            
        except Exception as e:
            error_handler.log_and_raise(
                DatabaseError(f"数据库查询失败: {str(e)}", details={"query_text": query_text})
            )
    
    def format_context(self, results: Dict[str, Any]) -> str:
        """
        格式化查询结果作为上下文
        
        Args:
            results: 数据库查询结果
            
        Returns:
            格式化后的上下文字符串
        """
        context_parts = []
        
        # 处理实体
        if 'entities' in results and results['entities'].get('documents'):
            entities = results['entities']['documents'][0] if results['entities']['documents'] else []
            if entities:
                context_parts.append("相关实体:")
                for doc in entities:
                    context_parts.append(f"- {doc}")
        
        # 处理关系
        if 'relations' in results and results['relations'].get('documents'):
            relations = results['relations']['documents'][0] if results['relations']['documents'] else []
            if relations:
                context_parts.append("\n相关关系:")
                for doc in relations:
                    context_parts.append(f"- {doc}")
        
        # 处理摘要
        if 'summaries' in results and results['summaries'].get('documents'):
            summaries = results['summaries']['documents'][0] if results['summaries']['documents'] else []
            if summaries:
                context_parts.append("\n相关摘要:")
                for doc in summaries:
                    context_parts.append(f"- {doc}")
        
        formatted_context = '\n'.join(context_parts)
        error_handler.logger.debug(f"格式化上下文，长度: {len(formatted_context)}")
        return formatted_context
    
    def generate_response(self, query_text: str, context: str) -> str:
        """
        使用AI根据查询和上下文生成回复
        
        Args:
            query_text: 用户查询
            context: 数据库查询结果上下文
            
        Returns:
            AI生成的回复
        """
        try:
            error_handler.logger.info("开始生成AI回复")
            
            # 构造提示词
            if context.strip():
                prompt = f"""
                基于以下背景信息回答用户问题：
                
                背景信息：
                {context}
                
                用户问题：
                {query_text}
                
                请根据背景信息回答问题。如果背景信息与问题无关或没有背景信息，请直接回答问题或说明无法根据提供的信息回答问题。
                """
            else:
                prompt = f"请回答以下问题：{query_text}"
            
            api_config = config_manager.get_api_config()
            response = client.chat.completions.create(
                model=config_manager.get_chat_model(),
                messages=[
                    {"role": "system", "content": "你是一个智能助手，能够根据提供的背景信息回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500,
                timeout=api_config['timeout']
            )
            
            ai_response = response.choices[0].message.content.strip()
            error_handler.logger.info(f"AI回复生成成功，长度: {len(ai_response)}")
            return ai_response
            
        except Exception as e:
            error_handler.log_and_raise(
                APIError(f"生成AI回复失败: {str(e)}", details={"query_text": query_text})
            )
    
    def process_query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        处理完整查询流程
        
        Args:
            query_text: 查询文本
            n_results: 返回结果数量
            
        Returns:
            包含查询结果和AI回复的字典
        """
        try:
            error_handler.logger.info(f"开始处理查询: {query_text}")
            
            # 查询数据库
            results = self.query_database(query_text, n_results)
            
            # 格式化上下文
            context = self.format_context(results)
            
            # 生成AI回复
            response = self.generate_response(query_text, context)
            
            result = {
                'query': query_text,
                'results': results,
                'context': context,
                'response': response,
                'timestamp': error_handler.logger.handlers[0].formatter.formatTime(
                    logging.LogRecord(
                        "", logging.INFO, "", 0, "", (), None
                    )
                )
            }
            
            error_handler.logger.info("查询处理完成")
            return result
            
        except Exception as e:
            error_response = error_handler.create_error_response(e, {"query_text": query_text})
            error_handler.logger.error(f"查询处理失败: {error_response}")
            return {
                'query': query_text,
                'error': error_response,
                'timestamp': error_handler.logger.handlers[0].formatter.formatTime(
                    logging.LogRecord(
                        "", logging.INFO, "", 0, "", (), None
                    )
                )
            }

def interactive_mode():
    """交互式模式"""
    print("向量数据库查询与AI回复系统 - 交互式模式")
    print("=" * 50)
    print("输入 'quit' 或 'exit' 退出程序")
    print("输入 'help' 查看帮助信息")
    print("=" * 50)
    
    query_system = DatabaseQuerySystem()
    
    while True:
        try:
            query_text = input("\n请输入您的查询: ").strip()
            
            if query_text.lower() in ['quit', 'exit']:
                print("感谢使用，再见！")
                break
            
            if query_text.lower() == 'help':
                print("\n帮助信息:")
                print("- 直接输入问题进行查询")
                print("- 输入 'quit' 或 'exit' 退出程序")
                print("- 输入 'help' 显示此帮助信息")
                continue
            
            if not query_text:
                print("查询不能为空")
                continue
            
            print("\n正在查询数据库...")
            result = query_system.process_query(query_text)
            
            if 'error' in result:
                print(f"查询失败: {result['error']['error']['message']}")
                continue
            
            print("\n找到以下相关信息:")
            if result['context'].strip():
                print(result['context'])
            else:
                print("未找到相关数据")
            
            print("\nAI回复:")
            print(result['response'])
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断，再见！")
            break
        except Exception as e:
            error_response = error_handler.create_error_response(e)
            print(f"发生错误: {error_response['error']['message']}")

def command_line_mode(query_text: str, n_results: int = 5, output_format: str = "text"):
    """命令行模式"""
    print(f"查询: {query_text}")
    
    query_system = DatabaseQuerySystem()
    result = query_system.process_query(query_text, n_results)
    
    if 'error' in result:
        print(f"查询失败: {result['error']['error']['message']}")
        return
    
    if output_format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("\n找到以下相关信息:")
        if result['context'].strip():
            print(result['context'])
        else:
            print("未找到相关数据")
        
        print("\nAI回复:")
        print(result['response'])

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='向量数据库查询与AI回复系统')
    parser.add_argument('query', nargs='?', help='查询文本')
    parser.add_argument('-n', '--n-results', type=int, default=5, help='返回结果数量 (默认: 5)')
    parser.add_argument('-f', '--format', choices=['text', 'json'], default='text', help='输出格式 (默认: text)')
    parser.add_argument('-i', '--interactive', action='store_true', help='交互式模式')
    
    args = parser.parse_args()
    
    if args.interactive or not args.query:
        interactive_mode()
    else:
        command_line_mode(args.query, args.n_results, args.format)

if __name__ == "__main__":
    main()
