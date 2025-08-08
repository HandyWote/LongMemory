#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
错误处理和日志系统测试套件
"""

import unittest
import json
import tempfile
import os
import logging
from unittest.mock import patch, MagicMock

# 导入被测试的模块
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from component.error_handler import (
    setup_system_logging,
    get_error_handler,
    get_input_validator,
    ValidationError,
    APIError,
    DatabaseError,
    ProcessingError,
    ConfigurationError,
    ErrorCode,
    ErrorType
)
from component.config_manager import ConfigManager, setup_system_config

class TestErrorHandler(unittest.TestCase):
    """错误处理器测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.error_handler = get_error_handler("test_error_handler")
        self.input_validator = get_input_validator()
        
        # 创建临时日志文件
        self.temp_log_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log')
        self.temp_log_file.close()
        
        # 设置日志系统
        setup_system_logging("DEBUG", self.temp_log_file.name)
    
    def tearDown(self):
        """测试后清理"""
        # 删除临时日志文件
        if os.path.exists(self.temp_log_file.name):
            os.unlink(self.temp_log_file.name)
    
    def test_validation_error_creation(self):
        """测试验证错误创建"""
        error = ValidationError("测试验证错误")
        
        self.assertEqual(error.message, "测试验证错误")
        self.assertEqual(error.error_code, ErrorCode.INVALID_INPUT)
        self.assertEqual(error.error_type, ErrorType.VALIDATION_ERROR)
        self.assertIsNotNone(error.timestamp)
    
    def test_api_error_creation(self):
        """测试API错误创建"""
        error = APIError("测试API错误", details={"endpoint": "/test"})
        
        self.assertEqual(error.message, "测试API错误")
        self.assertEqual(error.error_code, ErrorCode.API_TIMEOUT)
        self.assertEqual(error.error_type, ErrorType.API_ERROR)
        self.assertEqual(error.details["endpoint"], "/test")
    
    def test_database_error_creation(self):
        """测试数据库错误创建"""
        error = DatabaseError("测试数据库错误")
        
        self.assertEqual(error.message, "测试数据库错误")
        self.assertEqual(error.error_code, ErrorCode.DB_CONNECTION_ERROR)
        self.assertEqual(error.error_type, ErrorType.DATABASE_ERROR)
    
    def test_input_validator_required_fields(self):
        """测试输入验证器必需字段验证"""
        # 测试缺少必需字段
        data = {"msg": "test"}
        with self.assertRaises(ValidationError) as context:
            self.input_validator.validate_required_fields(data, ["msg", "uuid"])
        
        self.assertIn("缺少必需字段", str(context.exception))
        self.assertIn("uuid", context.exception.details["missing_fields"])
        
        # 测试所有字段都存在
        data_complete = {"msg": "test", "uuid": "12345"}
        try:
            self.input_validator.validate_required_fields(data_complete, ["msg", "uuid"])
        except ValidationError:
            self.fail("不应抛出验证错误")
    
    def test_input_validator_message_format(self):
        """测试输入验证器消息格式验证"""
        # 测试非字典输入
        with self.assertRaises(ValidationError) as context:
            self.input_validator.validate_message_format("invalid")
        
        self.assertIn("必须是字典格式", str(context.exception))
        
        # 测试无效的msg字段
        with self.assertRaises(ValidationError) as context:
            self.input_validator.validate_message_format({"msg": 123})
        
        self.assertIn("必须是字符串", str(context.exception))
        
        # 测试有效输入
        valid_data = {"msg": "test", "uuid": "12345"}
        try:
            self.input_validator.validate_message_format(valid_data)
        except ValidationError:
            self.fail("不应抛出验证错误")
    
    def test_error_response_creation(self):
        """测试错误响应创建"""
        # 测试验证错误响应
        error = ValidationError("测试验证错误")
        response = self.error_handler.create_error_response(error)
        
        self.assertEqual(response["status"], 400)
        self.assertEqual(response["error"]["type"], "validation_error")
        self.assertEqual(response["error"]["message"], "测试验证错误")
        self.assertIsNotNone(response["error"]["timestamp"])
        
        # 测试API错误响应
        api_error = APIError("测试API错误")
        api_response = self.error_handler.create_error_response(api_error)
        
        self.assertEqual(api_response["status"], 502)
        self.assertEqual(api_response["error"]["type"], "api_error")
        
        # 测试数据库错误响应
        db_error = DatabaseError("测试数据库错误")
        db_response = self.error_handler.create_error_response(db_error)
        
        self.assertEqual(db_response["status"], 503)
        self.assertEqual(db_response["error"]["type"], "database_error")
    
    def test_unknown_error_handling(self):
        """测试未知错误处理"""
        # 测试普通异常转换
        unknown_error = ValueError("未知错误")
        response = self.error_handler.create_error_response(unknown_error)
        
        self.assertEqual(response["status"], 500)
        self.assertEqual(response["error"]["type"], "unknown_error")
        self.assertEqual(response["error"]["code"], ErrorCode.UNKNOWN_ERROR.value)
        self.assertIn("ValueError", response["error"]["details"]["exception_type"])
    
    def test_error_logging(self):
        """测试错误日志记录"""
        # 创建一个错误并记录
        error = ValidationError("测试日志记录", details={"test": "data"})
        self.error_handler.handle_exception(error, {"context": "test"})
        
        # 检查日志文件是否包含错误信息
        with open(self.temp_log_file.name, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        self.assertIn("测试日志记录", log_content)
        self.assertIn("validation_error", log_content)
        self.assertIn("test", log_content)

class TestConfigManager(unittest.TestCase):
    """配置管理器测试类"""
    
    def setUp(self):
        """测试前设置"""
        # 创建临时配置文件
        self.temp_config_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.test_config = {
            "apiKey": "test_api_key",
            "chatModel": "gpt-4",
            "embeddingModel": "text-embedding-3-large",
            "log_level": "DEBUG",
            "log_file": "test.log"
        }
        json.dump(self.test_config, self.temp_config_file)
        self.temp_config_file.close()
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_config_file.name):
            os.unlink(self.temp_config_file.name)
    
    def test_config_manager_initialization(self):
        """测试配置管理器初始化"""
        config_manager = ConfigManager(self.temp_config_file.name)
        
        self.assertEqual(config_manager.get_api_key(), "test_api_key")
        self.assertEqual(config_manager.get_chat_model(), "gpt-4")
        self.assertEqual(config_manager.get_embedding_model(), "text-embedding-3-large")
    
    def test_config_manager_missing_file(self):
        """测试配置文件不存在的情况"""
        with self.assertRaises(ConfigurationError) as context:
            ConfigManager("nonexistent_config.json")
        
        self.assertIn("配置文件不存在", str(context.exception))
    
    def test_config_manager_invalid_json(self):
        """测试配置文件JSON格式错误"""
        # 创建无效的JSON文件
        with open(self.temp_config_file.name, 'w') as f:
            f.write("invalid json content")
        
        with self.assertRaises(ConfigurationError) as context:
            ConfigManager(self.temp_config_file.name)
        
        self.assertIn("配置文件格式错误", str(context.exception))
    
    def test_config_manager_missing_required_fields(self):
        """测试配置文件缺少必需字段"""
        # 创建缺少必需字段的配置
        incomplete_config = {"apiKey": "test"}
        with open(self.temp_config_file.name, 'w') as f:
            json.dump(incomplete_config, f)
        
        with self.assertRaises(ConfigurationError) as context:
            ConfigManager(self.temp_config_file.name)
        
        self.assertIn("缺少必需字段", str(context.exception))
    
    def test_config_manager_get_methods(self):
        """测试配置管理器各种get方法"""
        config_manager = ConfigManager(self.temp_config_file.name)
        
        # 测试数据库配置
        db_config = config_manager.get_database_config()
        self.assertEqual(db_config['database'], 'memory.db')
        self.assertEqual(db_config['persist_directory'], './chroma_db')
        
        # 测试日志配置
        log_config = config_manager.get_logging_config()
        self.assertEqual(log_config['log_level'], 'DEBUG')
        self.assertEqual(log_config['log_file'], 'test.log')
        
        # 测试API配置
        api_config = config_manager.get_api_config()
        self.assertEqual(api_config['timeout'], 30)
        self.assertEqual(api_config['max_retries'], 3)
        
        # 测试记忆配置
        memory_config = config_manager.get_memory_config()
        self.assertEqual(memory_config['short_term_memory_size'], 10)
        self.assertEqual(memory_config['long_term_memory_decay_rate'], 0.1)
    
    def test_config_manager_update_and_save(self):
        """测试配置更新和保存"""
        config_manager = ConfigManager(self.temp_config_file.name)
        
        # 更新配置
        config_manager.update_config("new_key", "new_value")
        self.assertEqual(config_manager.get("new_key"), "new_value")
        
        # 保存配置
        config_manager.save_config()
        
        # 重新加载验证
        new_config_manager = ConfigManager(self.temp_config_file.name)
        self.assertEqual(new_config_manager.get("new_key"), "new_value")

class TestIntegration(unittest.TestCase):
    """集成测试类"""
    
    def setUp(self):
        """测试前设置"""
        # 创建临时配置和日志文件
        self.temp_config_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.test_config = {
            "apiKey": "test_api_key",
            "chatModel": "gpt-4",
            "embeddingModel": "text-embedding-3-large",
            "log_level": "DEBUG"
        }
        json.dump(self.test_config, self.temp_config_file)
        self.temp_config_file.close()
        
        self.temp_log_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log')
        self.temp_log_file.close()
        
        # 设置系统配置和日志
        setup_system_config(self.temp_config_file.name)
        setup_system_logging("DEBUG", self.temp_log_file.name)
    
    def tearDown(self):
        """测试后清理"""
        for temp_file in [self.temp_config_file.name, self.temp_log_file.name]:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_system_setup_integration(self):
        """测试系统设置集成"""
        from component.config_manager import get_config_manager
        from component.error_handler import get_error_handler
        
        # 获取配置管理器
        config_manager = get_config_manager()
        self.assertEqual(config_manager.get_api_key(), "test_api_key")
        
        # 获取错误处理器
        error_handler = get_error_handler("integration_test")
        self.assertIsNotNone(error_handler.logger)
        
        # 测试错误处理
        error = ValidationError("集成测试错误")
        response = error_handler.create_error_response(error)
        
        self.assertEqual(response["status"], 400)
        self.assertEqual(response["error"]["message"], "集成测试错误")
        
        # 检查日志文件
        with open(self.temp_log_file.name, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        self.assertIn("集成测试错误", log_content)

def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigManager))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("开始运行错误处理和日志系统测试...")
    success = run_tests()
    
    if success:
        print("\n✅ 所有测试通过！")
    else:
        print("\n❌ 部分测试失败！")
        exit(1)
