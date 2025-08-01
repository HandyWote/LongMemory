"""
构建路由获取用户消息
"""
import flask
from flask import request, Flask

app = Flask(__name__)
@app.route('/message', methods=['POST'])
def get_message():
    """获取用户消息"""
    try:
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

        return flask.jsonify({
            'status': 200,
            'message': '获取用户消息成功',
            'data': {
                'msg': msg,
                'uuid': uuid
            }
        })
        
    except Exception as e:
        return flask.jsonify({
            'status': 500,
            'message': f'服务器内部错误: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)