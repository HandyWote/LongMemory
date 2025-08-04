"""
主应用文件，整合整个流程：接收消息 -> 提取信息 -> 向量化 -> 存储到数据库
"""
import flask
from flask import request, Flask
import json
import traceback

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
        # 1. 获取用户消息
        data = request.get_json()
        if not data:
            return flask.jsonify({
                'status': 400,
                'message': '请求数据不能为空'
            }), 400
        
        msg = data.get('msg')
        uuid = data.get('uuid')
        
        if not msg or not uuid:
            return flask.jsonify({
                'status': 400,
                'message': 'msg和uuid字段都是必需的'
            }), 400
        
        msg_data = {
            'msg': msg,
            'uuid': uuid
        }
        
        # 2. 处理消息并生成向量表示
        processed_data = process_message_for_database(msg_data)
        
        # 3. 格式化为数据库存储格式
        formatted_data = format_for_database(processed_data)
        
        # 4. 存储到数据库
        result = store_knowledge_triple(
            entities=formatted_data.get("entities", []),
            relations=formatted_data.get("relations", []),
            summaries=formatted_data.get("summaries", []),
            entities_metadata=formatted_data.get("entities_metadata"),
            relations_metadata=formatted_data.get("relations_metadata"),
            summaries_metadata=formatted_data.get("summaries_metadata"),
            uuid=formatted_data.get("uuid")
        )
        
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
        
        return flask.jsonify({
            'status': 500,
            'message': f'服务器内部错误: {str(e)}',
            'error_info': error_info
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return flask.jsonify({
        'status': 200,
        'message': '服务运行正常'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)