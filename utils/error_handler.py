# utils/error_handler.py
# Production-ready error handling system for MLOps grade enhancement

import functools
import traceback
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union, Type
from contextlib import contextmanager
from enum import Enum
import json

# Import structured logger
try:
    from .structured_logger import StructuredLogger, EventType, LogLevel, MLOpsLoggers
    STRUCTURED_LOGGING_AVAILABLE = True
except ImportError:
    STRUCTURED_LOGGING_AVAILABLE = False
    # Fallback to standard logging
    import logging


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling"""
    LOW = "low"              # Non-critical errors that don't affect core functionality
    MEDIUM = "medium"        # Errors that degrade performance but allow continuation
    HIGH = "high"           # Critical errors that require immediate attention
    CRITICAL = "critical"   # System-breaking errors that require emergency response


class ErrorCategory(Enum):
    """Error categories for better classification and handling"""
    # Data-related errors
    DATA_VALIDATION = "data_validation"
    DATA_LOADING = "data_loading"  
    DATA_PREPROCESSING = "data_preprocessing"
    DATA_QUALITY = "data_quality"
    
    # Model-related errors
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    MODEL_LOADING = "model_loading"
    MODEL_PREDICTION = "model_prediction"
    
    # Feature engineering errors
    FEATURE_EXTRACTION = "feature_extraction"
    FEATURE_SELECTION = "feature_selection"
    
    # System-related errors
    RESOURCE_CONSTRAINT = "resource_constraint"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    IO_OPERATION = "io_operation"
    
    # API and service errors
    API_ERROR = "api_error"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    
    # External service errors
    EXTERNAL_SERVICE = "external_service"
    NETWORK = "network"
    
    # Unknown/uncategorized errors
    UNKNOWN = "unknown"


class MLOpsError(Exception):
    """Base exception class for MLOps-related errors"""
    
    def __init__(self, 
                 message: str,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 component: str = None,
                 metadata: Dict[str, Any] = None,
                 suggestion: str = None,
                 original_error: Exception = None):
        
        self.message = message
        self.category = category
        self.severity = severity
        self.component = component
        self.metadata = metadata or {}
        self.suggestion = suggestion
        self.original_error = original_error
        self.timestamp = datetime.now().isoformat()
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization"""
        return {
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'component': self.component,
            'metadata': self.metadata,
            'suggestion': self.suggestion,
            'timestamp': self.timestamp,
            'original_error': {
                'type': type(self.original_error).__name__ if self.original_error else None,
                'message': str(self.original_error) if self.original_error else None
            }
        }


# Specific error types for different scenarios
class DataValidationError(MLOpsError):
    """Error in data validation"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATA_VALIDATION, 
                         severity=ErrorSeverity.HIGH, **kwargs)


class ModelTrainingError(MLOpsError):
    """Error during model training"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.MODEL_TRAINING, 
                         severity=ErrorSeverity.HIGH, **kwargs)


class ResourceConstraintError(MLOpsError):
    """Error due to resource constraints (CPU/Memory)"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.RESOURCE_CONSTRAINT, 
                         severity=ErrorSeverity.MEDIUM, **kwargs)


class ConfigurationError(MLOpsError):
    """Error in configuration or setup"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.CONFIGURATION, 
                         severity=ErrorSeverity.HIGH, **kwargs)


class FeatureEngineeringError(MLOpsError):
    """Error in feature engineering process"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.FEATURE_EXTRACTION, 
                         severity=ErrorSeverity.MEDIUM, **kwargs)


class ErrorHandler:
    """Centralized error handling with logging, recovery, and monitoring"""
    
    def __init__(self, component: str, logger: Optional[StructuredLogger] = None):
        self.component = component
        self.error_count = {}  # Track error frequency
        self.recovery_strategies = {}  # Store recovery functions
        
        # Setup logger
        if STRUCTURED_LOGGING_AVAILABLE and logger is None:
            self.logger = MLOpsLoggers.get_logger(component)
        elif logger:
            self.logger = logger
        else:
            # Fallback to standard logging
            import logging
            self.logger = logging.getLogger(component)
    
    def register_recovery_strategy(self, 
                                  error_category: ErrorCategory,
                                  recovery_func: Callable):
        """Register recovery strategy for specific error category"""
        self.recovery_strategies[error_category] = recovery_func
    
    def handle_error(self, 
                    error: Exception,
                    context: Dict[str, Any] = None,
                    category: ErrorCategory = None,
                    severity: ErrorSeverity = None,
                    suggestion: str = None,
                    attempt_recovery: bool = True) -> Dict[str, Any]:
        """
        Central error handling method
        
        Returns:
            Dict with error details and recovery status
        """
        
        # Convert to MLOpsError if not already
        if not isinstance(error, MLOpsError):
            mlops_error = MLOpsError(
                message=str(error),
                category=category or self._classify_error(error),
                severity=severity or self._determine_severity(error),
                component=self.component,
                metadata=context or {},
                suggestion=suggestion,
                original_error=error
            )
        else:
            mlops_error = error
        
        # Track error frequency
        error_key = f"{mlops_error.category.value}:{type(error).__name__}"
        self.error_count[error_key] = self.error_count.get(error_key, 0) + 1
        
        # Log error
        self._log_error(mlops_error, context)
        
        # Attempt recovery if enabled
        recovery_result = None
        if attempt_recovery and mlops_error.category in self.recovery_strategies:
            try:
                recovery_result = self.recovery_strategies[mlops_error.category](mlops_error, context)
                self._log_recovery_attempt(mlops_error, recovery_result)
            except Exception as recovery_error:
                self._log_recovery_failure(mlops_error, recovery_error)
        
        return {
            'error': mlops_error.to_dict(),
            'recovery_attempted': recovery_result is not None,
            'recovery_successful': recovery_result is not None and recovery_result.get('success', False),
            'recovery_result': recovery_result,
            'error_count': self.error_count.get(error_key, 1)
        }
    
    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Automatically classify error based on type and message"""
        error_type = type(error).__name__.lower()
        error_message = str(error).lower()
        
        # Data-related errors
        if any(keyword in error_message for keyword in ['data', 'dataframe', 'csv', 'dataset']):
            if any(keyword in error_message for keyword in ['validation', 'invalid', 'format']):
                return ErrorCategory.DATA_VALIDATION
            elif any(keyword in error_message for keyword in ['load', 'read', 'file']):
                return ErrorCategory.DATA_LOADING
            else:
                return ErrorCategory.DATA_PREPROCESSING
        
        # Model-related errors
        if any(keyword in error_message for keyword in ['model', 'training', 'fit', 'predict']):
            if 'training' in error_message or 'fit' in error_message:
                return ErrorCategory.MODEL_TRAINING
            elif 'predict' in error_message:
                return ErrorCategory.MODEL_PREDICTION
            else:
                return ErrorCategory.MODEL_VALIDATION
        
        # Resource constraints
        if any(keyword in error_message for keyword in ['memory', 'cpu', 'resource', 'timeout']):
            return ErrorCategory.RESOURCE_CONSTRAINT
        
        # IO errors
        if 'ioerror' in error_type or any(keyword in error_message for keyword in ['file', 'path', 'directory']):
            return ErrorCategory.IO_OPERATION
        
        # Configuration errors
        if any(keyword in error_message for keyword in ['config', 'parameter', 'argument']):
            return ErrorCategory.CONFIGURATION
        
        # Feature engineering
        if any(keyword in error_message for keyword in ['feature', 'transform', 'vectoriz']):
            return ErrorCategory.FEATURE_EXTRACTION
        
        # API errors
        if any(keyword in error_message for keyword in ['api', 'request', 'response', 'http']):
            return ErrorCategory.API_ERROR
        
        return ErrorCategory.UNKNOWN
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on error type and context"""
        error_type = type(error).__name__.lower()
        error_message = str(error).lower()
        
        # Critical system errors
        if error_type in ['systemexit', 'keyboardinterrupt', 'memoryerror']:
            return ErrorSeverity.CRITICAL
        
        # High severity - prevents core functionality
        if any(keyword in error_message for keyword in ['training failed', 'model not found', 'critical']):
            return ErrorSeverity.HIGH
        
        # Medium severity - degrades performance
        if any(keyword in error_message for keyword in ['warning', 'timeout', 'resource']):
            return ErrorSeverity.MEDIUM
        
        # Default to medium for unknown errors
        return ErrorSeverity.MEDIUM
    
    def _log_error(self, error: MLOpsError, context: Dict[str, Any]):
        """Log error with structured logging"""
        if STRUCTURED_LOGGING_AVAILABLE:
            log_level = self._get_log_level_for_severity(error.severity)
            
            self.logger.log(
                level=log_level,
                event_type=EventType.MODEL_TRAINING_ERROR,
                message=f"Error in {self.component}: {error.message}",
                component=self.component,
                metadata={
                    'error_category': error.category.value,
                    'error_severity': error.severity.value,
                    'error_metadata': error.metadata,
                    'context': context or {},
                    'suggestion': error.suggestion,
                    'error_count': self.error_count.get(f"{error.category.value}:{type(error.original_error).__name__}", 1)
                },
                tags=[error.category.value, error.severity.value, 'error_handling']
            )
        else:
            # Fallback logging
            self.logger.error(f"Error in {self.component}: {error.message}")
    
    def _get_log_level_for_severity(self, severity: ErrorSeverity) -> LogLevel:
        """Map error severity to log level"""
        severity_to_log_level = {
            ErrorSeverity.LOW: LogLevel.WARNING,
            ErrorSeverity.MEDIUM: LogLevel.ERROR,
            ErrorSeverity.HIGH: LogLevel.ERROR,
            ErrorSeverity.CRITICAL: LogLevel.CRITICAL
        }
        return severity_to_log_level.get(severity, LogLevel.ERROR)
    
    def _log_recovery_attempt(self, error: MLOpsError, recovery_result: Dict[str, Any]):
        """Log recovery attempt results"""
        if STRUCTURED_LOGGING_AVAILABLE:
            success = recovery_result.get('success', False)
            event_type = EventType.MODEL_TRAINING_COMPLETE if success else EventType.MODEL_TRAINING_ERROR
            
            self.logger.info(
                event_type,
                f"Recovery {'succeeded' if success else 'failed'} for {error.category.value} error",
                component=self.component,
                metadata={
                    'original_error': error.message,
                    'recovery_result': recovery_result,
                    'error_category': error.category.value
                },
                tags=['error_recovery', 'automated_recovery']
            )
    
    def _log_recovery_failure(self, error: MLOpsError, recovery_error: Exception):
        """Log recovery failure"""
        if STRUCTURED_LOGGING_AVAILABLE:
            self.logger.error(
                EventType.MODEL_TRAINING_ERROR,
                f"Recovery failed for {error.category.value} error: {str(recovery_error)}",
                component=self.component,
                metadata={
                    'original_error': error.message,
                    'recovery_error': str(recovery_error),
                    'error_category': error.category.value
                },
                tags=['error_recovery', 'recovery_failure']
            )


# Decorator for automatic error handling
def handle_errors(component: str = None, 
                 category: ErrorCategory = None,
                 severity: ErrorSeverity = None,
                 attempt_recovery: bool = True,
                 reraise: bool = True):
    """Decorator for automatic error handling"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            comp_name = component or func.__module__
            error_handler = ErrorHandler(comp_name)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Handle the error
                result = error_handler.handle_error(
                    error=e,
                    context={
                        'function': func.__name__,
                        'args_count': len(args),
                        'kwargs_count': len(kwargs)
                    },
                    category=category,
                    severity=severity,
                    attempt_recovery=attempt_recovery
                )
                
                # Re-raise if specified, otherwise return error result
                if reraise:
                    raise
                else:
                    return result
        
        return wrapper
    return decorator


# Context manager for error handling
@contextmanager
def error_handling_context(component: str,
                          operation: str,
                          category: ErrorCategory = None,
                          severity: ErrorSeverity = None,
                          metadata: Dict[str, Any] = None):
    """Context manager for handling errors within a specific operation"""
    error_handler = ErrorHandler(component)
    
    try:
        yield error_handler
    except Exception as e:
        result = error_handler.handle_error(
            error=e,
            context={
                'operation': operation,
                **(metadata or {})
            },
            category=category,
            severity=severity
        )
        # Always re-raise in context manager
        raise


# Recovery strategies for common scenarios
class RecoveryStrategies:
    """Common recovery strategies for different error categories"""
    
    @staticmethod
    def data_loading_recovery(error: MLOpsError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recovery strategy for data loading errors"""
        try:
            # Try alternative data sources or fallback datasets
            if 'file_path' in context:
                # Try backup locations
                backup_paths = [
                    Path(context['file_path']).with_suffix('.backup.csv'),
                    Path('/tmp/data/fallback_dataset.csv'),
                    Path('/tmp/data/combined_dataset.csv')
                ]
                
                for backup_path in backup_paths:
                    if backup_path.exists():
                        return {
                            'success': True,
                            'recovery_method': 'fallback_data_source',
                            'fallback_path': str(backup_path)
                        }
            
            return {'success': False, 'reason': 'No fallback data sources available'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def model_training_recovery(error: MLOpsError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recovery strategy for model training errors"""
        try:
            # Common recovery strategies for training failures
            recovery_methods = []
            
            # Reduce model complexity
            if 'resource' in str(error.message).lower():
                recovery_methods.append('reduce_model_complexity')
            
            # Fallback to simpler model
            if 'lightgbm' in str(error.message).lower():
                recovery_methods.append('fallback_to_logistic_regression')
            
            # Reduce dataset size for memory issues
            if 'memory' in str(error.message).lower():
                recovery_methods.append('reduce_dataset_size')
            
            return {
                'success': len(recovery_methods) > 0,
                'recovery_methods': recovery_methods,
                'suggestion': 'Apply suggested recovery methods and retry training'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def feature_engineering_recovery(error: MLOpsError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recovery strategy for feature engineering errors"""
        try:
            # Fallback to standard TF-IDF if enhanced features fail
            if 'enhanced' in str(error.message).lower():
                return {
                    'success': True,
                    'recovery_method': 'fallback_to_standard_features',
                    'suggestion': 'Switch to standard TF-IDF features and continue training'
                }
            
            return {'success': False, 'reason': 'No applicable recovery method'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


# CPU constraint specific error handling for HuggingFace Spaces
class CPUConstraintHandler:
    """Specialized handler for CPU constraint issues in HuggingFace Spaces"""
    
    def __init__(self, component: str):
        self.component = component
        self.error_handler = ErrorHandler(component)
        
        # Register CPU-specific recovery strategies
        self.error_handler.register_recovery_strategy(
            ErrorCategory.RESOURCE_CONSTRAINT,
            self._cpu_recovery_strategy
        )
    
    def _cpu_recovery_strategy(self, error: MLOpsError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recovery strategy specifically for CPU constraints"""
        try:
            recovery_actions = []
            
            # Reduce parallel processing
            if 'n_jobs' in str(error.message) or 'parallel' in str(error.message):
                recovery_actions.append('force_single_threading')
            
            # Reduce model complexity for CPU efficiency
            if 'training' in context.get('operation', '').lower():
                recovery_actions.extend([
                    'reduce_cv_folds',
                    'simplify_hyperparameter_grid',
                    'disable_ensemble_if_slow'
                ])
            
            # Memory optimization for CPU-bound systems
            if 'memory' in str(error.message).lower():
                recovery_actions.extend([
                    'reduce_feature_dimensions',
                    'batch_processing',
                    'garbage_collection'
                ])
            
            return {
                'success': len(recovery_actions) > 0,
                'recovery_actions': recovery_actions,
                'cpu_optimizations': True,
                'environment': 'huggingface_spaces'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def monitor_and_handle_cpu_issues(self, 
                                     operation_func: Callable,
                                     *args,
                                     timeout_seconds: int = 300,
                                     **kwargs) -> Any:
        """Monitor operation for CPU issues and handle automatically"""
        import time
        import signal
        
        start_time = time.time()
        
        def timeout_handler(signum, frame):
            raise ResourceConstraintError(
                f"Operation {operation_func.__name__} exceeded CPU time limit ({timeout_seconds}s)",
                component=self.component,
                metadata={
                    'timeout_seconds': timeout_seconds,
                    'operation': operation_func.__name__,
                    'environment': 'cpu_constrained'
                },
                suggestion="Reduce model complexity or dataset size for CPU-constrained environment"
            )
        
        # Set timeout signal
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            result = operation_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log performance if slow
            if execution_time > timeout_seconds * 0.8:  # 80% of timeout
                if STRUCTURED_LOGGING_AVAILABLE:
                    logger = MLOpsLoggers.get_monitoring_logger()
                    logger.log_cpu_constraint_warning(
                        component=self.component,
                        operation=operation_func.__name__,
                        resource_usage={
                            'execution_time_seconds': execution_time,
                            'timeout_threshold': timeout_seconds,
                            'cpu_efficiency': 'low'
                        }
                    )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Handle error with CPU constraint context
            self.error_handler.handle_error(
                error=e,
                context={
                    'operation': operation_func.__name__,
                    'execution_time': execution_time,
                    'timeout_limit': timeout_seconds,
                    'environment': 'cpu_constrained'
                },
                category=ErrorCategory.RESOURCE_CONSTRAINT,
                severity=ErrorSeverity.HIGH
            )
            raise
        
        finally:
            # Clear timeout
            signal.alarm(0)


# Integration utilities for existing codebase
def setup_error_handling() -> Dict[str, ErrorHandler]:
    """Setup error handlers for all MLOps components"""
    handlers = {}
    
    components = [
        'model_trainer',
        'model_retrainer', 
        'data_processor',
        'feature_engineer',
        'api_server',
        'monitoring'
    ]
    
    for component in components:
        handler = ErrorHandler(component)
        
        # Register common recovery strategies
        handler.register_recovery_strategy(
            ErrorCategory.DATA_LOADING, 
            RecoveryStrategies.data_loading_recovery
        )
        handler.register_recovery_strategy(
            ErrorCategory.MODEL_TRAINING,
            RecoveryStrategies.model_training_recovery
        )
        handler.register_recovery_strategy(
            ErrorCategory.FEATURE_EXTRACTION,
            RecoveryStrategies.feature_engineering_recovery
        )
        
        handlers[component] = handler
    
    return handlers


def get_error_handler(component: str) -> ErrorHandler:
    """Get error handler for specific component"""
    return ErrorHandler(component)


# Example integration functions
def integrate_with_retrain_py():
    """Example integration with retrain.py for robust error handling"""
    
    # Setup error handler for retraining component
    error_handler = ErrorHandler('model_retrainer')
    
    # Register specific recovery strategies
    error_handler.register_recovery_strategy(
        ErrorCategory.MODEL_TRAINING,
        lambda error, context: {
            'success': True,
            'recovery_method': 'fallback_to_individual_models',
            'suggestion': 'Disable ensemble and use best individual model'
        }
    )
    
    return error_handler


def integrate_with_train_py():
    """Example integration with train.py for comprehensive error handling"""
    
    # Setup error handler for training component
    error_handler = ErrorHandler('model_trainer')
    
    # CPU constraint handler for HuggingFace Spaces
    cpu_handler = CPUConstraintHandler('model_trainer')
    
    return error_handler, cpu_handler


# Error reporting and analytics
class ErrorReporter:
    """Collect and report error analytics for MLOps monitoring"""
    
    def __init__(self, report_file: Path = None):
        self.report_file = report_file or Path("/tmp/logs/error_report.json")
        self.error_stats = {}
    
    def record_error(self, error_info: Dict[str, Any]):
        """Record error for analytics"""
        category = error_info.get('error', {}).get('category', 'unknown')
        severity = error_info.get('error', {}).get('severity', 'medium')
        
        key = f"{category}:{severity}"
        
        if key not in self.error_stats:
            self.error_stats[key] = {
                'count': 0,
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'recovery_attempts': 0,
                'recovery_successes': 0
            }
        
        stats = self.error_stats[key]
        stats['count'] += 1
        stats['last_seen'] = datetime.now().isoformat()
        
        if error_info.get('recovery_attempted', False):
            stats['recovery_attempts'] += 1
            if error_info.get('recovery_successful', False):
                stats['recovery_successes'] += 1
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate error analytics report"""
        total_errors = sum(stats['count'] for stats in self.error_stats.values())
        total_recovery_attempts = sum(stats['recovery_attempts'] for stats in self.error_stats.values())
        total_recovery_successes = sum(stats['recovery_successes'] for stats in self.error_stats.values())
        
        recovery_rate = (total_recovery_successes / total_recovery_attempts * 100) if total_recovery_attempts > 0 else 0
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_errors': total_errors,
                'unique_error_types': len(self.error_stats),
                'recovery_attempts': total_recovery_attempts,
                'recovery_successes': total_recovery_successes,
                'recovery_rate_percent': recovery_rate
            },
            'error_breakdown': self.error_stats,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> list:
        """Generate recommendations based on error patterns"""
        recommendations = []
        
        # High frequency errors
        high_freq_errors = {k: v for k, v in self.error_stats.items() if v['count'] > 5}
        if high_freq_errors:
            recommendations.append({
                'type': 'high_frequency_errors',
                'message': f'Address frequently occurring errors: {", ".join(high_freq_errors.keys())}',
                'priority': 'high'
            })
        
        # Low recovery rates
        low_recovery_errors = {
            k: v for k, v in self.error_stats.items() 
            if v['recovery_attempts'] > 0 and (v['recovery_successes'] / v['recovery_attempts']) < 0.5
        }
        if low_recovery_errors:
            recommendations.append({
                'type': 'low_recovery_rate',
                'message': 'Improve recovery strategies for poorly recovering error types',
                'priority': 'medium',
                'affected_errors': list(low_recovery_errors.keys())
            })
        
        # Resource constraint patterns
        resource_errors = {k: v for k, v in self.error_stats.items() if 'resource_constraint' in k}
        if resource_errors:
            recommendations.append({
                'type': 'resource_optimization',
                'message': 'Consider CPU/memory optimizations for resource constraint errors',
                'priority': 'high',
                'suggestion': 'Review HuggingFace Spaces constraints and optimize accordingly'
            })
        
        return recommendations
    
    def save_report(self):
        """Save error report to file"""
        report = self.generate_report()
        
        self.report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


# Global error reporter instance
_global_error_reporter = None

def get_global_error_reporter() -> ErrorReporter:
    """Get global error reporter instance"""
    global _global_error_reporter
    if _global_error_reporter is None:
        _global_error_reporter = ErrorReporter()
    return _global_error_reporter


if __name__ == "__main__":
    # Example usage and testing
    print("Testing error handling system...")
    
    # Test basic error handling
    error_handler = ErrorHandler('test_component')
    
    try:
        raise ValueError("Test error for demonstration")
    except Exception as e:
        result = error_handler.handle_error(
            error=e,
            context={'test': True},
            category=ErrorCategory.DATA_VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            suggestion="This is a test error for demonstration purposes"
        )
        print("Error handling result:", result)
    
    # Test decorator
    @handle_errors(component='test_decorator', category=ErrorCategory.MODEL_TRAINING)
    def test_function_with_error():
        raise ModelTrainingError("Test model training error")
    
    try:
        test_function_with_error()
    except Exception as e:
        print("Decorator handled error:", type(e).__name__)
    
    # Test CPU constraint handler
    cpu_handler = CPUConstraintHandler('test_cpu')
    
    def slow_operation():
        import time
        time.sleep(0.1)  # Simulate work
        return "completed"
    
    try:
        result = cpu_handler.monitor_and_handle_cpu_issues(slow_operation, timeout_seconds=1)
        print("CPU monitoring result:", result)
    except Exception as e:
        print("CPU constraint error:", str(e))
    
    # Test error reporting
    reporter = get_global_error_reporter()
    
    # Record some test errors
    test_error_info = {
        'error': {
            'category': 'model_training',
            'severity': 'high',
            'message': 'Test error for reporting'
        },
        'recovery_attempted': True,
        'recovery_successful': False
    }
    
    reporter.record_error(test_error_info)
    report = reporter.generate_report()
    print("Error report:", json.dumps(report, indent=2))
    
    print("Error handling system test completed successfully!")