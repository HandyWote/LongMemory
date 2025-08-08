"""
系统配置管理器
"""
import json
import os
from typing import Dict, Any, Optional
from .error_handler import ConfigurationError, get_error_handler, ErrorCode

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = "./config.json"):
        self.config_file = config_file
        self.config = {}
        self.error_handler = get_error_handler("ConfigManager")
        self.load_config()
    
    def load_config(self) -> None:
        """
        加载配置文件
        
        Raises:
            ConfigurationError: 如果配置文件不存在或格式错误
        """
        try:
            if not os.path.exists(self.config_file):
                raise ConfigurationError(
                    f"配置文件不存在: {self.config_file}",
                    ErrorCode.CONFIG_MISSING,
                    {"config_file": self.config_file}
                )
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            # 验证必需的配置项
            self._validate_config()
            
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"配置文件格式错误: {str(e)}",
                ErrorCode.CONFIG_INVALID,
                {"config_file": self.config_file, "error": str(e)}
            )
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(
                f"加载配置文件时发生错误: {str(e)}",
                ErrorCode.CONFIG_INVALID,
                {"config_file": self.config_file}
            )
    
    def _validate_config(self) -> None:
        """
        验证配置项的完整性
        
        Raises:
            ConfigurationError: 如果必需的配置项缺失
        """
        required_fields = ['apiKey', 'chatModel', 'embeddingModel']
        missing_fields = []
        
        for field in required_fields:
            if field not in self.config or not self.config[field]:
                missing_fields.append(field)
        
        if missing_fields:
            raise ConfigurationError(
                f"配置文件中缺少必需字段: {', '.join(missing_fields)}",
                ErrorCode.CONFIG_MISSING,
                {"missing_fields": missing_fields}
            )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        
        Args:
            key: 配置项键名
            default: 默认值
            
        Returns:
            配置项值
        """
        return self.config.get(key, default)
    
    def get_api_key(self) -> str:
        """
        获取API密钥
        
        Returns:
            API密钥
            
        Raises:
            ConfigurationError: 如果API密钥未配置
        """
        api_key = self.get('apiKey')
        if not api_key:
            raise ConfigurationError(
                "API密钥未配置",
                ErrorCode.CONFIG_MISSING
            )
        return api_key
    
    def get_chat_model(self) -> str:
        """
        获取聊天模型
        
        Returns:
            聊天模型名称
        """
        return self.get('chatModel', 'gpt-4')
    
    def get_embedding_model(self) -> str:
        """
        获取嵌入模型
        
        Returns:
            嵌入模型名称
        """
        return self.get('embeddingModel', 'text-embedding-3-large')
    
    def get_database_config(self) -> Dict[str, Any]:
        """
        获取数据库配置
        
        Returns:
            数据库配置字典
        """
        return {
            'database': self.get('database', 'memory.db'),
            'persist_directory': self.get('persist_directory', './chroma_db'),
            'reset_database': self.get('reset_database', False)
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        获取日志配置
        
        Returns:
            日志配置字典
        """
        return {
            'log_level': self.get('log_level', 'INFO'),
            'log_file': self.get('log_file', 'longmemory.log'),
            'log_format': self.get('log_format', None)
        }
    
    def get_api_config(self) -> Dict[str, Any]:
        """
        获取API配置
        
        Returns:
            API配置字典
        """
        return {
            'timeout': self.get('api_timeout', 30),
            'max_retries': self.get('api_max_retries', 3),
            'retry_delay': self.get('api_retry_delay', 1)
        }
    
    def get_memory_config(self) -> Dict[str, Any]:
        """
        获取记忆系统配置
        
        Returns:
            记忆系统配置字典
        """
        return {
            'short_term_memory_size': self.get('short_term_memory_size', 10),
            'long_term_memory_decay_rate': self.get('long_term_memory_decay_rate', 0.1),
            'memory_consolidation_threshold': self.get('memory_consolidation_threshold', 0.5),
            'forgetting_threshold': self.get('forgetting_threshold', 0.2)
        }
    
    def reload_config(self) -> None:
        """
        重新加载配置文件
        """
        self.load_config()
    
    def update_config(self, key: str, value: Any) -> None:
        """
        更新配置项
        
        Args:
            key: 配置项键名
            value: 配置项值
        """
        self.config[key] = value
    
    def save_config(self) -> None:
        """
        保存配置到文件
        
        Raises:
            ConfigurationError: 如果保存失败
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise ConfigurationError(
                f"保存配置文件时发生错误: {str(e)}",
                ErrorCode.CONFIG_INVALID,
                {"config_file": self.config_file}
            )

# 全局配置管理器实例
_config_manager = None

def get_config_manager(config_file: str = "./config.json") -> ConfigManager:
    """
    获取配置管理器实例
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        ConfigManager实例
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager

def setup_system_config(config_file: str = "./config.json") -> ConfigManager:
    """
    设置系统配置
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        ConfigManager实例
    """
    global _config_manager
    _config_manager = ConfigManager(config_file)
    return _config_manager

if __name__ == "__main__":
    # 测试配置管理器
    try:
        config_manager = get_config_manager()
        print("API Key:", config_manager.get_api_key())
        print("Chat Model:", config_manager.get_chat_model())
        print("Embedding Model:", config_manager.get_embedding_model())
        print("Database Config:", config_manager.get_database_config())
        print("Logging Config:", config_manager.get_logging_config())
        print("API Config:", config_manager.get_api_config())
        print("Memory Config:", config_manager.get_memory_config())
    except Exception as e:
        print(f"配置管理器测试失败: {e}")
