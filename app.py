"""
主应用文件，整合整个流程：接收消息 -> 提取信息 -> 向量化 -> 存储到数据库
"""
import flask
from flask import request, Flask
import json
import traceback

# 导入错误处理和配置管理
from component.error_handler import (
    setup_system_logging, 
    get_error_handler, 
    get_input_validator,
    ValidationError,
    APIError,
    ProcessingError,
    DatabaseError
)
from component.config_manager import setup_system_config

# 设置系统日志和配置
setup_system_logging()
config_manager = setup_system_config()
error_handler = get_error_handler(__name__)
input_validator = get_input_validator()

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
        error_handler.logger.info("开始处理用户消息")
        
        # 1. 获取用户消息
        try:
            data = request.get_json()
            error_handler.logger.info(f"接收到的请求数据: {data}")
        except Exception as json_error:
            validation_error = ValidationError(
                f"JSON格式错误: {str(json_error)}",
                details={"raw_request": request.get_data(as_text=True)}
            )
            error_response = error_handler.create_error_response(validation_error)
            return flask.jsonify(error_response), error_response['status']
            
    except Exception as e:
        error_response = error_handler.create_error_response(e, {"phase": "request_parsing"})
        return flask.jsonify(error_response), error_response['status']
    
    if not data:
        validation_error = ValidationError(
            "请求数据不能为空，必须是有效的JSON格式"
        )
        error_response = error_handler.create_error_response(validation_error)
        return flask.jsonify(error_response), error_response['status']
    
    # 使用输入验证器验证数据格式
    try:
        input_validator.validate_message_format(data)
        input_validator.validate_required_fields(data, ['msg', 'uuid'])
    except ValidationError as e:
        error_response = error_handler.create_error_response(e)
        return flask.jsonify(error_response), error_response['status']
    
    msg = data.get('msg')
    uuid = data.get('uuid')
    
    error_handler.logger.info(f"提取到的消息内容 - msg: {msg}, uuid: {uuid}")
    
    msg_data = {
        'msg': msg,
        'uuid': uuid
    }
    
    try:
        # 2. 处理消息并生成向量表示
        error_handler.logger.info("开始处理消息并生成向量表示")
        processed_data = process_message_for_database(msg_data)
        error_handler.logger.info(f"消息处理完成，处理后的数据: {processed_data}")
        
        # 3. 格式化为数据库存储格式
        error_handler.logger.info("开始格式化为数据库存储格式")
        formatted_data = format_for_database(processed_data)
        error_handler.logger.info(f"数据格式化完成，格式化后的数据: {formatted_data}")
        
        # 检查数据是否为空
        entities = formatted_data.get("entities", [])
        relations = formatted_data.get("relations", [])
        summaries = formatted_data.get("summaries", [])
        
        error_handler.logger.info(f"准备存储的数据 - 实体数量: {len(entities)}, 关系数量: {len(relations)}, 摘要数量: {len(summaries)}")
        
        if not entities and not relations and not summaries:
            error_handler.logger.warning("没有需要存储的数据")
            return flask.jsonify({
                'status': 200,
                'message': '消息处理完成，但未提取到任何实体、关系或摘要',
                'data': {}
            })
        
        # 4. 存储到数据库
        error_handler.logger.info("开始存储到数据库")
        result = store_knowledge_triple(
            entities=formatted_data.get("entities", []),
            relations=formatted_data.get("relations", []),
            summaries=formatted_data.get("summaries", []),
            entities_metadata=formatted_data.get("entities_metadata"),
            relations_metadata=formatted_data.get("relations_metadata"),
            summaries_metadata=formatted_data.get("summaries_metadata"),
            uuid=formatted_data.get("uuid")
        )
        error_handler.logger.info(f"数据存储完成，存储结果: {result}")
        
        return flask.jsonify({
            'status': 200,
            'message': '消息处理成功',
            'data': result
        })
        
    except ValidationError as e:
        error_response = error_handler.create_error_response(e, {"phase": "validation"})
        return flask.jsonify(error_response), error_response['status']
    except APIError as e:
        error_response = error_handler.create_error_response(e, {"phase": "api_processing"})
        return flask.jsonify(error_response), error_response['status']
    except ProcessingError as e:
        error_response = error_handler.create_error_response(e, {"phase": "data_processing"})
        return flask.jsonify(error_response), error_response['status']
    except DatabaseError as e:
        error_response = error_handler.create_error_response(e, {"phase": "database_storage"})
        return flask.jsonify(error_response), error_response['status']
    except Exception as e:
        error_response = error_handler.create_error_response(e, {"phase": "unknown", "msg_data": msg_data})
        return flask.jsonify(error_response), error_response['status']

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    error_handler.logger.info("健康检查接口被调用")
    return flask.jsonify({
        'status': 200,
        'message': '服务运行正常',
        'config': {
            'chat_model': config_manager.get_chat_model(),
            'embedding_model': config_manager.get_embedding_model(),
            'log_level': config_manager.get_logging_config()['log_level']
        }
    })

if __name__ == '__main__':
    error_handler.logger.info("应用启动")
    error_handler.logger.info(f"配置信息 - 聊天模型: {config_manager.get_chat_model()}, 嵌入模型: {config_manager.get_embedding_model()}")
    app.run(debug=True, host='0.0.0.0', port=5000)
