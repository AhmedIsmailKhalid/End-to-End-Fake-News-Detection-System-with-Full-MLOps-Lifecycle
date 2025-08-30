# utils/structured_logger.py
# Production-ready structured logging system for MLOps grade enhancement

import logging
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time


class LogLevel(Enum):
    """Standardized log levels with numeric values for filtering"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class EventType(Enum):
    """Standardized event types for structured logging"""
    # Model lifecycle events
    MODEL_TRAINING_START = "model.training.start"
    MODEL_TRAINING_COMPLETE = "model.training.complete"
    MODEL_TRAINING_ERROR = "model.training.error"
    MODEL_VALIDATION = "model.validation"
    MODEL_PROMOTION = "model.promotion"
    MODEL_BACKUP = "model.backup"
    
    # Data processing events
    DATA_LOADING = "data.loading"
    DATA_VALIDATION = "data.validation" 
    DATA_PREPROCESSING = "data.preprocessing"
    DATA_QUALITY_CHECK = "data.quality.check"
    
    # Feature engineering events
    FEATURE_EXTRACTION = "features.extraction"
    FEATURE_SELECTION = "features.selection"
    FEATURE_VALIDATION = "features.validation"
    
    # Cross-validation and ensemble events
    CROSS_VALIDATION_START = "cv.start"
    CROSS_VALIDATION_COMPLETE = "cv.complete"
    ENSEMBLE_CREATION = "ensemble.creation"
    ENSEMBLE_VALIDATION = "ensemble.validation"
    STATISTICAL_COMPARISON = "model.statistical_comparison"
    
    # System performance events
    PERFORMANCE_METRIC = "system.performance"
    RESOURCE_USAGE = "system.resource_usage"
    CPU_CONSTRAINT_WARNING = "system.cpu_constraint"
    
    # API and application events
    API_REQUEST = "api.request"
    API_RESPONSE = "api.response"
    API_ERROR = "api.error"
    PREDICTION_REQUEST = "prediction.request"
    PREDICTION_RESPONSE = "prediction.response"
    
    # Monitoring and alerting events
    DRIFT_DETECTION = "monitor.drift"
    ALERT_TRIGGERED = "alert.triggered"
    HEALTH_CHECK = "health.check"
    
    # Security and access events
    ACCESS_GRANTED = "security.access_granted"
    ACCESS_DENIED = "security.access_denied"
    AUTHENTICATION = "security.authentication"


@dataclass
class LogEntry:
    """Structured log entry with standardized fields"""
    timestamp: str
    level: str
    event_type: str
    message: str
    component: str
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[list] = None
    environment: str = "production"
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary for JSON serialization"""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_json(self) -> str:
        """Convert log entry to JSON string"""
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)


class StructuredLogger:
    """Production-ready structured logger with performance monitoring"""
    
    def __init__(self, 
                 name: str,
                 log_level: LogLevel = LogLevel.INFO,
                 log_file: Optional[Path] = None,
                 max_file_size_mb: int = 100,
                 backup_count: int = 5,
                 enable_console: bool = True,
                 enable_json_format: bool = True):
        
        self.name = name
        self.log_level = log_level
        self.enable_json_format = enable_json_format
        self.session_id = self._generate_session_id()
        self._local = threading.local()
        
        # Setup logging infrastructure
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level.value)
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Setup file logging if specified
        if log_file:
            self._setup_file_handler(log_file, max_file_size_mb, backup_count)
        
        # Setup console logging if enabled
        if enable_console:
            self._setup_console_handler()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID for tracking related events"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        thread_id = threading.current_thread().ident
        return f"session_{timestamp}_{thread_id}"
    
    def _setup_file_handler(self, log_file: Path, max_size_mb: int, backup_count: int):
        """Setup rotating file handler"""
        from logging.handlers import RotatingFileHandler
        
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        if self.enable_json_format:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(self._create_standard_formatter())
        
        self.logger.addHandler(file_handler)
    
    def _setup_console_handler(self):
        """Setup console handler with appropriate formatting"""
        console_handler = logging.StreamHandler(sys.stdout)
        
        if self.enable_json_format:
            console_handler.setFormatter(JsonFormatter())
        else:
            console_handler.setFormatter(self._create_standard_formatter())
        
        self.logger.addHandler(console_handler)
    
    def _create_standard_formatter(self):
        """Create human-readable formatter for non-JSON output"""
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _get_trace_id(self) -> Optional[str]:
        """Get current trace ID from thread-local storage"""
        return getattr(self._local, 'trace_id', None)
    
    def set_trace_id(self, trace_id: str):
        """Set trace ID for current thread"""
        self._local.trace_id = trace_id
    
    def clear_trace_id(self):
        """Clear trace ID for current thread"""
        if hasattr(self._local, 'trace_id'):
            del self._local.trace_id
    
    def _create_log_entry(self, 
                         level: LogLevel,
                         event_type: EventType,
                         message: str,
                         component: str = None,
                         duration_ms: float = None,
                         metadata: Dict[str, Any] = None,
                         tags: list = None,
                         user_id: str = None) -> LogEntry:
        """Create structured log entry"""
        
        return LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level.name,
            event_type=event_type.value,
            message=message,
            component=component or self.name,
            session_id=self.session_id,
            trace_id=self._get_trace_id(),
            user_id=user_id,
            duration_ms=duration_ms,
            metadata=metadata or {},
            tags=tags or [],
            environment=self._detect_environment(),
            version=self._get_version()
        )
    
    def _detect_environment(self) -> str:
        """Detect current environment"""
        if any(env in str(Path.cwd()) for env in ['test', 'pytest']):
            return 'test'
        elif 'STREAMLIT_SERVER_PORT' in os.environ:
            return 'streamlit'
        elif 'SPACE_ID' in os.environ:
            return 'huggingface_spaces'
        elif 'DOCKER_CONTAINER' in os.environ:
            return 'docker'
        else:
            return 'local'
    
    def _get_version(self) -> str:
        """Get application version"""
        # Try to read from metadata or config
        try:
            metadata_path = Path("/tmp/metadata.json")
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                return metadata.get('model_version', '1.0')
        except:
            pass
        return '1.0'
    
    def log(self, 
            level: LogLevel,
            event_type: EventType,
            message: str,
            component: str = None,
            duration_ms: float = None,
            metadata: Dict[str, Any] = None,
            tags: list = None,
            user_id: str = None,
            exc_info: bool = False):
        """Core logging method"""
        
        if level.value < self.log_level.value:
            return
        
        # Create structured log entry
        log_entry = self._create_log_entry(
            level=level,
            event_type=event_type,
            message=message,
            component=component,
            duration_ms=duration_ms,
            metadata=metadata,
            tags=tags,
            user_id=user_id
        )
        
        # Add exception information if requested
        if exc_info:
            log_entry.metadata['exception'] = {
                'type': sys.exc_info()[0].__name__ if sys.exc_info()[0] else None,
                'message': str(sys.exc_info()[1]) if sys.exc_info()[1] else None,
                'traceback': traceback.format_exc() if sys.exc_info()[0] else None
            }
        
        # Log using Python's logging framework
        self.logger.log(
            level.value,
            log_entry.to_json() if self.enable_json_format else message,
            extra={'log_entry': log_entry}
        )
    
    # Convenience methods for different log levels
    def debug(self, event_type: EventType, message: str, **kwargs):
        """Log debug message"""
        self.log(LogLevel.DEBUG, event_type, message, **kwargs)
    
    def info(self, event_type: EventType, message: str, **kwargs):
        """Log info message"""
        self.log(LogLevel.INFO, event_type, message, **kwargs)
    
    def warning(self, event_type: EventType, message: str, **kwargs):
        """Log warning message"""
        self.log(LogLevel.WARNING, event_type, message, **kwargs)
    
    def error(self, event_type: EventType, message: str, **kwargs):
        """Log error message"""
        self.log(LogLevel.ERROR, event_type, message, exc_info=True, **kwargs)
    
    def critical(self, event_type: EventType, message: str, **kwargs):
        """Log critical message"""
        self.log(LogLevel.CRITICAL, event_type, message, exc_info=True, **kwargs)
    
    @contextmanager
    def operation(self, 
                  event_type: EventType,
                  operation_name: str,
                  component: str = None,
                  metadata: Dict[str, Any] = None):
        """Context manager for timing operations"""
        
        start_time = time.time()
        trace_id = f"{operation_name}_{int(start_time * 1000)}"
        
        # Set trace ID for operation
        self.set_trace_id(trace_id)
        
        # Log operation start
        self.info(
            event_type,
            f"Starting {operation_name}",
            component=component,
            metadata={**(metadata or {}), 'operation': operation_name, 'status': 'started'}
        )
        
        try:
            yield self
            
            # Log successful completion
            duration = (time.time() - start_time) * 1000
            self.info(
                event_type,
                f"Completed {operation_name}",
                component=component,
                duration_ms=duration,
                metadata={**(metadata or {}), 'operation': operation_name, 'status': 'completed'}
            )
            
        except Exception as e:
            # Log error
            duration = (time.time() - start_time) * 1000
            self.error(
                EventType.MODEL_TRAINING_ERROR,
                f"Failed {operation_name}: {str(e)}",
                component=component,
                duration_ms=duration,
                metadata={**(metadata or {}), 'operation': operation_name, 'status': 'failed'}
            )
            raise
        
        finally:
            # Clear trace ID
            self.clear_trace_id()
    
    def log_performance_metrics(self, 
                               component: str,
                               metrics: Dict[str, Union[int, float]],
                               tags: list = None):
        """Log performance metrics"""
        self.info(
            EventType.PERFORMANCE_METRIC,
            f"Performance metrics for {component}",
            component=component,
            metadata={'metrics': metrics},
            tags=tags or []
        )
    
    def log_model_metrics(self,
                         model_name: str,
                         metrics: Dict[str, float],
                         dataset_size: int = None,
                         cv_folds: int = None,
                         metadata: Dict[str, Any] = None):
        """Log model performance metrics"""
        model_metadata = {
            'model_name': model_name,
            'metrics': metrics,
            **(metadata or {})
        }
        
        if dataset_size:
            model_metadata['dataset_size'] = dataset_size
        if cv_folds:
            model_metadata['cv_folds'] = cv_folds
        
        self.info(
            EventType.MODEL_VALIDATION,
            f"Model validation completed for {model_name}",
            component="model_trainer",
            metadata=model_metadata,
            tags=['model_validation', 'metrics']
        )
    
    def log_cpu_constraint_warning(self, 
                                  component: str,
                                  operation: str,
                                  resource_usage: Dict[str, Any] = None):
        """Log CPU constraint warnings for HuggingFace Spaces"""
        self.warning(
            EventType.CPU_CONSTRAINT_WARNING,
            f"CPU constraint detected in {component} during {operation}",
            component=component,
            metadata={
                'operation': operation,
                'resource_usage': resource_usage or {},
                'optimization_applied': True,
                'environment': 'huggingface_spaces'
            },
            tags=['cpu_constraint', 'optimization', 'hfs']
        )


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        """Format log record as JSON"""
        if hasattr(record, 'log_entry'):
            return record.log_entry.to_json()
        
        # Fallback for non-structured logs
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'component': record.name,
            'environment': 'unknown'
        }
        
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }
        
        return json.dumps(log_data, default=str, ensure_ascii=False)


# Singleton logger instances for different components
class MLOpsLoggers:
    """Centralized logger management for MLOps components"""
    
    _loggers: Dict[str, StructuredLogger] = {}
    
    @classmethod
    def get_logger(cls, 
                   component: str,
                   log_level: LogLevel = LogLevel.INFO,
                   log_file: Optional[Path] = None) -> StructuredLogger:
        """Get or create logger for component"""
        if component not in cls._loggers:
            if log_file is None:
                log_file = Path("/tmp/logs") / f"{component}.log"
            
            cls._loggers[component] = StructuredLogger(
                name=component,
                log_level=log_level,
                log_file=log_file,
                enable_console=True,
                enable_json_format=True
            )
        
        return cls._loggers[component]
    
    @classmethod
    def get_model_trainer_logger(cls) -> StructuredLogger:
        """Get logger for model training components"""
        return cls.get_logger("model_trainer", LogLevel.INFO)
    
    @classmethod
    def get_retraining_logger(cls) -> StructuredLogger:
        """Get logger for retraining components"""
        return cls.get_logger("model_retrainer", LogLevel.INFO)
    
    @classmethod
    def get_api_logger(cls) -> StructuredLogger:
        """Get logger for API components"""
        return cls.get_logger("api_server", LogLevel.INFO)
    
    @classmethod
    def get_monitoring_logger(cls) -> StructuredLogger:
        """Get logger for monitoring components"""
        return cls.get_logger("monitoring", LogLevel.INFO)
    
    @classmethod
    def get_data_logger(cls) -> StructuredLogger:
        """Get logger for data processing components"""
        return cls.get_logger("data_processing", LogLevel.INFO)


# Performance monitoring utilities
class PerformanceMonitor:
    """Monitor and log performance metrics for CPU-constrained environments"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
    
    def monitor_training_performance(self, 
                                   model_name: str,
                                   dataset_size: int,
                                   training_time: float,
                                   memory_usage_mb: float = None):
        """Monitor and log training performance"""
        
        # Calculate performance metrics
        samples_per_second = dataset_size / training_time if training_time > 0 else 0
        
        performance_metrics = {
            'training_time_seconds': training_time,
            'dataset_size': dataset_size,
            'samples_per_second': samples_per_second,
            'model_name': model_name
        }
        
        if memory_usage_mb:
            performance_metrics['memory_usage_mb'] = memory_usage_mb
        
        # Log performance
        self.logger.log_performance_metrics(
            component="model_trainer",
            metrics=performance_metrics,
            tags=['training_performance', 'cpu_optimized']
        )
        
        # Check for performance issues
        if training_time > 300:  # 5 minutes
            self.logger.log_cpu_constraint_warning(
                component="model_trainer",
                operation="model_training",
                resource_usage={'training_time': training_time, 'dataset_size': dataset_size}
            )
    
    def monitor_cv_performance(self, 
                              cv_folds: int,
                              total_cv_time: float,
                              models_evaluated: int):
        """Monitor cross-validation performance"""
        
        avg_fold_time = total_cv_time / cv_folds if cv_folds > 0 else 0
        avg_model_time = total_cv_time / models_evaluated if models_evaluated > 0 else 0
        
        cv_metrics = {
            'cv_folds': cv_folds,
            'total_cv_time_seconds': total_cv_time,
            'avg_fold_time_seconds': avg_fold_time,
            'models_evaluated': models_evaluated,
            'avg_model_time_seconds': avg_model_time
        }
        
        self.logger.log_performance_metrics(
            component="cross_validation",
            metrics=cv_metrics,
            tags=['cv_performance', 'statistical_validation']
        )
    
    def monitor_ensemble_performance(self,
                                   individual_models_count: int,
                                   ensemble_training_time: float,
                                   statistical_test_time: float):
        """Monitor ensemble creation and validation performance"""
        
        ensemble_metrics = {
            'individual_models_count': individual_models_count,
            'ensemble_training_time_seconds': ensemble_training_time,
            'statistical_test_time_seconds': statistical_test_time,
            'total_ensemble_time_seconds': ensemble_training_time + statistical_test_time
        }
        
        self.logger.log_performance_metrics(
            component="ensemble_manager",
            metrics=ensemble_metrics,
            tags=['ensemble_performance', 'statistical_tests']
        )


# Integration helpers for existing codebase
def setup_mlops_logging():
    """Setup structured logging for MLOps components"""
    # Ensure log directory exists
    log_dir = Path("/tmp/logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger to avoid interference
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    return MLOpsLoggers


def get_component_logger(component_name: str) -> StructuredLogger:
    """Get logger for specific component (backwards compatibility)"""
    return MLOpsLoggers.get_logger(component_name)


# Decorators for automatic logging
def log_function_call(event_type: EventType, component: str = None):
    """Decorator to automatically log function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = MLOpsLoggers.get_logger(component or func.__module__)
            
            with logger.operation(
                event_type=event_type,
                operation_name=func.__name__,
                component=component,
                metadata={'function': func.__name__, 'args_count': len(args), 'kwargs_count': len(kwargs)}
            ):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Example usage functions for integration
def integrate_with_retrain_py():
    """Example integration with retrain.py"""
    logger = MLOpsLoggers.get_retraining_logger()
    
    # Example: Log retraining session start
    logger.info(
        EventType.MODEL_TRAINING_START,
        "Enhanced retraining session started with LightGBM and ensemble",
        component="retrain",
        metadata={
            'models': ['logistic_regression', 'random_forest', 'lightgbm'],
            'ensemble_enabled': True,
            'enhanced_features': True
        },
        tags=['retraining', 'lightgbm', 'ensemble']
    )
    
    return logger


def integrate_with_train_py():
    """Example integration with train.py"""
    logger = MLOpsLoggers.get_model_trainer_logger()
    
    # Example: Log training session start
    logger.info(
        EventType.MODEL_TRAINING_START,
        "Enhanced training session started with comprehensive CV",
        component="train",
        metadata={
            'models': ['logistic_regression', 'random_forest', 'lightgbm'],
            'cv_folds': 5,
            'ensemble_enabled': True
        },
        tags=['training', 'cv', 'ensemble']
    )
    
    return logger


# CPU constraint monitoring
import os
import psutil

def monitor_cpu_constraints():
    """Monitor CPU usage and memory for HuggingFace Spaces constraints"""
    logger = MLOpsLoggers.get_monitoring_logger()
    
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        resource_metrics = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_mb': memory.used / 1024 / 1024,
            'memory_available_mb': memory.available / 1024 / 1024,
            'process_memory_mb': process.memory_info().rss / 1024 / 1024,
            'process_cpu_percent': process.cpu_percent()
        }
        
        # Log resource usage
        logger.log_performance_metrics(
            component="system_monitor",
            metrics=resource_metrics,
            tags=['resource_monitoring', 'hfs_constraints']
        )
        
        # Alert on high usage (HFS constraints)
        if cpu_percent > 80 or memory.percent > 85:
            logger.log_cpu_constraint_warning(
                component="system_monitor", 
                operation="resource_monitoring",
                resource_usage=resource_metrics
            )
        
        return resource_metrics
        
    except Exception as e:
        logger.error(
            EventType.PERFORMANCE_METRIC,
            f"Failed to monitor CPU constraints: {str(e)}",
            component="system_monitor"
        )
        return None


if __name__ == "__main__":
    # Example usage and testing
    setup_mlops_logging()
    
    # Test structured logging
    logger = MLOpsLoggers.get_model_trainer_logger()
    
    # Test basic logging
    logger.info(
        EventType.MODEL_TRAINING_START,
        "Testing structured logging system",
        metadata={'test': True, 'version': '1.0'},
        tags=['test', 'structured_logging']
    )
    
    # Test operation timing
    with logger.operation(
        EventType.MODEL_VALIDATION,
        "test_operation",
        metadata={'test_data': 'example'}
    ):
        time.sleep(0.1)  # Simulate work
    
    # Test performance monitoring
    perf_monitor = PerformanceMonitor(logger)
    perf_monitor.monitor_training_performance(
        model_name="test_model",
        dataset_size=1000,
        training_time=5.0,
        memory_usage_mb=150.0
    )
    
    # Test CPU monitoring
    monitor_cpu_constraints()
    
    print("Structured logging system test completed successfully!")