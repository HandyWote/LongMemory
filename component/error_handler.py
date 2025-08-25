"""
Simplified error handling and logging system
简化的错误处理和日志系统
"""
import logging
import traceback
import json
from datetime import datetime
from typing import Dict, Any, Optional, Union
from enum import Enum

class ErrorType(Enum):
    """Error type enumeration / 错误类型枚举"""
    VALIDATION_ERROR = "validation_error"
    API_ERROR = "api_error"
    DATABASE_ERROR = "database_error"
    PROCESSING_ERROR = "processing_error"
    CONFIGURATION_ERROR = "configuration_error"
    NETWORK_ERROR = "network_error"
    UNKNOWN_ERROR = "unknown_error"

class ErrorCode(Enum):
    """Error code enumeration / 错误码枚举"""
    # Validation errors (1000-1999)
    INVALID_INPUT = 1001
    MISSING_REQUIRED_FIELD = 1002
    INVALID_FORMAT = 1003
    
    # API errors (2000-2999)
    API_TIMEOUT = 2001
    API_RATE_LIMIT = 2002
    API_AUTH_ERROR = 2003
    API_QUOTA_EXCEEDED = 2004
    
    # Database errors (3000-3999)
    DB_CONNECTION_ERROR = 3001
    DB_QUERY_ERROR = 3002
    DB_INSERT_ERROR = 3003
    DB_UPDATE_ERROR = 3004
    
    # Processing errors (4000-4999)
    EXTRACTION_ERROR = 4001
    EMBEDDING_ERROR = 4002
    PROCESSING_TIMEOUT = 4003
    
    # Configuration errors (5000-5999)
    CONFIG_MISSING = 5001
    CONFIG_INVALID = 5002
    
    # Network errors (6000-6999)
    NETWORK_TIMEOUT = 6001
    NETWORK_CONNECTION_ERROR = 6002
    
    # Unknown errors (9000-9999)
    UNKNOWN_ERROR = 9001

class SystemError(Exception):
    """Base system exception class / 系统基础异常类"""
    def __init__(self, message: str, error_code: ErrorCode, error_type: ErrorType, 
                 details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.error_type = error_type
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
        super().__init__(self.message)

class ValidationError(SystemError):
    """Validation error / 验证错误"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.INVALID_INPUT, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, ErrorType.VALIDATION_ERROR, details)

class APIError(SystemError):
    """API error / API错误"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.API_TIMEOUT, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, ErrorType.API_ERROR, details)

class DatabaseError(SystemError):
    """Database error / 数据库错误"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.DB_CONNECTION_ERROR, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, ErrorType.DATABASE_ERROR, details)

class ProcessingError(SystemError):
    """Processing error / 处理错误"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.PROCESSING_TIMEOUT, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, ErrorType.PROCESSING_ERROR, details)

class ConfigurationError(SystemError):
    """Configuration error / 配置错误"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.CONFIG_MISSING, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, ErrorType.CONFIGURATION_ERROR, details)

class NetworkError(SystemError):
    """Network error / 网络错误"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.NETWORK_TIMEOUT, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, ErrorType.NETWORK_ERROR, details)

class LoggerConfig:
    """Logger configuration class / 日志配置类"""
    
    @staticmethod
    def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None, 
                     log_format: Optional[str] = None) -> None:
        """
        Setup logging configuration
        设置日志配置
        
        Args:
            log_level: Log level
            log_file: Log file path
            log_format: Log format
        """
        if log_format is None:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        
        # Create formatter / 创建格式化器
        formatter = logging.Formatter(log_format)
        
        # Configure root logger / 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers / 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add console handler / 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Add file handler if log file is specified / 如果指定了日志文件则添加文件处理器
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Set log level for third-party libraries / 设置第三方库的日志级别
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('openai').setLevel(logging.WARNING)
        logging.getLogger('chromadb').setLevel(logging.WARNING)

class ErrorHandler:
    """Error handler / 错误处理器"""
    
    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)
    
    def handle_exception(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle exception and return standardized error information
        处理异常并返回标准化错误信息
        
        Args:
            exception: Exception object
            context: Context information
            
        Returns:
            Standardized error information dictionary
        """
        context = context or {}
        
        # If it's a system error, use it directly / 如果是系统错误，直接使用
        if isinstance(exception, SystemError):
            error_info = {
                "error_code": exception.error_code.value,
                "error_type": exception.error_type.value,
                "message": exception.message,
                "details": exception.details,
                "timestamp": exception.timestamp
            }
        else:
            # Convert to system error / 转换为系统错误
            error_info = {
                "error_code": ErrorCode.UNKNOWN_ERROR.value,
                "error_type": ErrorType.UNKNOWN_ERROR.value,
                "message": str(exception),
                "details": {
                    "exception_type": type(exception).__name__,
                    "traceback": traceback.format_exc()
                },
                "timestamp": datetime.now().isoformat()
            }
        
        # Add context information / 添加上下文信息
        if context:
            error_info["context"] = context
        
        # Log error / 记录错误日志
        self.logger.error(
            f"Error occurred: {error_info['error_type']} - {error_info['message']}",
            extra={"error_info": error_info}
        )
        
        return error_info
    
    def create_error_response(self, exception: Exception, context: Optional[Dict[str, Any]] = None, 
                            http_status: int = 500) -> Dict[str, Any]:
        """
        Create HTTP error response
        创建HTTP错误响应
        
        Args:
            exception: Exception object
            context: Context information
            http_status: HTTP status code
            
        Returns:
            HTTP error response dictionary
        """
        error_info = self.handle_exception(exception, context)
        
        # Determine HTTP status code based on error type / 根据错误类型确定HTTP状态码
        if isinstance(exception, ValidationError):
            http_status = 400
        elif isinstance(exception, (APIError, NetworkError)):
            http_status = 502
        elif isinstance(exception, DatabaseError):
            http_status = 503
        
        return {
            "status": http_status,
            "error": {
                "code": error_info["error_code"],
                "type": error_info["error_type"],
                "message": error_info["message"],
                "details": error_info.get("details", {}),
                "timestamp": error_info["timestamp"]
            }
        }
    
    def log_and_raise(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log error and re-raise exception
        记录错误并重新抛出异常
        
        Args:
            exception: Exception object
            context: Context information
        """
        self.handle_exception(exception, context)
        raise exception

class InputValidator:
    """Input validator / 输入验证器"""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
    
    def validate_required_fields(self, data: Dict[str, Any], required_fields: list) -> None:
        """
        Validate required fields
        验证必需字段
        
        Args:
            data: Input data
            required_fields: Required fields list
            
        Raises:
            ValidationError: If validation fails
        """
        missing_fields = []
        for field in required_fields:
            if field not in data or not data[field]:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValidationError(
                f"Missing required fields: {', '.join(missing_fields)}",
                ErrorCode.MISSING_REQUIRED_FIELD,
                {"missing_fields": missing_fields}
            )
    
    def validate_message_format(self, data: Dict[str, Any]) -> None:
        """
        Validate message format
        验证消息格式
        
        Args:
            data: Input data
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(data, dict):
            raise ValidationError(
                "Input data must be a dictionary",
                ErrorCode.INVALID_FORMAT,
                {"received_type": type(data).__name__}
            )
        
        # Validate message field / 验证消息字段
        if 'msg' in data and not isinstance(data['msg'], str):
            raise ValidationError(
                "msg field must be a string",
                ErrorCode.INVALID_FORMAT,
                {"field": "msg", "received_type": type(data['msg']).__name__}
            )
        
        # Validate UUID field / 验证UUID字段
        if 'uuid' in data and not isinstance(data['uuid'], str):
            raise ValidationError(
                "uuid field must be a string",
                ErrorCode.INVALID_FORMAT,
                {"field": "uuid", "received_type": type(data['uuid']).__name__}
            )

# Global error handler instance / 全局错误处理器实例
error_handler = ErrorHandler("LongMemory")

# Global input validator instance / 全局输入验证器实例
input_validator = InputValidator(error_handler)

def setup_system_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup system logging configuration
    设置系统日志配置
    
    Args:
        log_level: Log level
        log_file: Log file path
    """
    LoggerConfig.setup_logging(log_level, log_file)

def get_error_handler(logger_name: str = __name__) -> ErrorHandler:
    """
    Get error handler instance
    获取错误处理器实例
    
    Args:
        logger_name: Logger name
        
    Returns:
        ErrorHandler instance
    """
    return ErrorHandler(logger_name)

def get_input_validator() -> InputValidator:
    """
    Get input validator instance
    获取输入验证器实例
    
    Returns:
        InputValidator instance
    """
    return input_validator
