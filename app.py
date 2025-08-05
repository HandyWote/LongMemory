"""
主应用文件，整合整个流程：接收消息 -> 提取信息 -> 向量化 -> 存储到数据库
"""
import flask
from flask import request, Flask
import json
import traceback
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入各个组件
from component.getMessage import get_message
from component.extract import extract_entity_relation
from component.toEmbeding import process_message_for_database, format_for_database
from component.toDatabase import store_knowledge_triple

app = Flask(__name__)

@app.route('/process_message', methods=['POST'])
def process_message():
    """
    处理用户消息的完整流程:
    1. 接收消息 (getMessage)
    2. 提取实体、关系和摘要 (extract)
    3. 向量化处理 (toEmbeding)
    4. 存储到数据库 (toDatabase)
    """
    try:
        logger.info("开始处理用户消息")
        
        # 1. 获取用户消息
        try:
            data = request.get_json()
            logger.info(f"接收到的请求数据: {data}")
        except Exception as json_error:
            logger.error(f"JSON解析错误: {str(json_error)}")
            return flask.jsonify({
                'status': 400,
                'message': 'JSON格式错误: ' + str(json_error)
            }), 400
    except Exception as e:
        # 记录详细的错误信息
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        logger.error(f"处理请求时发生错误: {str(e)}", extra=error_info)
        
        return flask.jsonify({
            'status': 500,
            'message': f'服务器内部错误: {str(e)}',
            'error_info': error_info
        }), 500
            
    if not data:
        logger.warning("请求数据为空")
        return flask.jsonify({
            'status': 400,
            'message': '请求数据不能为空，必须是有效的JSON格式'
        }), 400
    
    msg = data.get('msg')
    uuid = data.get('uuid')
    
    logger.info(f"提取到的消息内容 - msg: {msg}, uuid: {uuid}")
    
    if not msg or not uuid:
        logger.warning(f"缺少必需字段 - msg: {msg}, uuid: {uuid}")
        return flask.jsonify({
            'status': 400,
            'message': 'msg和uuid字段都是必需的'
        }), 400
    
    msg_data = {
        'msg': msg,
        'uuid': uuid
    }
    
    try:
        # 2. 处理消息并生成向量表示
        logger.info("开始处理消息并生成向量表示")
        processed_data = process_message_for_database(msg_data)
        logger.info(f"消息处理完成，处理后的数据: {processed_data}")
        
        # 3. 格式化为数据库存储格式
        logger.info("开始格式化为数据库存储格式")
        formatted_data = format_for_database(processed_data)
        logger.info(f"数据格式化完成，格式化后的数据: {formatted_data}")
        
        # 检查数据是否为空
        entities = formatted_data.get("entities", [])
        relations = formatted_data.get("relations", [])
        summaries = formatted_data.get("summaries", [])
        
        logger.info(f"准备存储的数据 - 实体数量: {len(entities)}, 关系数量: {len(relations)}, 摘要数量: {len(summaries)}")
        
        if not entities and not relations and not summaries:
            logger.warning("没有需要存储的数据")
            return flask.jsonify({
                'status': 200,
                'message': '消息处理完成，但未提取到任何实体、关系或摘要',
                'data': {}
            })
        
        # 4. 存储到数据库
        logger.info("开始存储到数据库")
        result = store_knowledge_triple(
            entities=formatted_data.get("entities", []),
            relations=formatted_data.get("relations", []),
            summaries=formatted_data.get("summaries", []),
            entities_metadata=formatted_data.get("entities_metadata"),
            relations_metadata=formatted_data.get("relations_metadata"),
            summaries_metadata=formatted_data.get("summaries_metadata"),
            uuid=formatted_data.get("uuid")
        )
        logger.info(f"数据存储完成，存储结果: {result}")
        
        return flask.jsonify({
            'status': 200,
            'message': '消息处理成功',
            'data': result
        })
        
    except Exception as e:
        # 记录详细的错误信息
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        logger.error(f"处理消息时发生错误: {str(e)}", extra={'error_info': error_info})
        
        return flask.jsonify({
            'status': 500,
            'message': f'服务器内部错误: {str(e)}',
            'error_info': error_info
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    logger.info("健康检查接口被调用")
    return flask.jsonify({
        'status': 200,
        'message': '服务运行正常'
    })

if __name__ == '__main__':
    logger.info("应用启动")
    app.run(debug=True, host='0.0.0.0', port=5000)