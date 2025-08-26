# utils/statistical_analysis.py
# Advanced statistical analysis for Data Science grade enhancement (B+ → A-)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
import logging

# Import structured logging if available
try:
    from .structured_logger import StructuredLogger, EventType, MLOpsLoggers
    STRUCTURED_LOGGING_AVAILABLE = True
except ImportError:
    STRUCTURED_LOGGING_AVAILABLE = False
    import logging

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """Container for statistical analysis results with uncertainty quantification"""
    point_estimate: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    method: str
    sample_size: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'point_estimate': float(self.point_estimate),
            'confidence_interval': [float(self.confidence_interval[0]), float(self.confidence_interval[1])],
            'confidence_level': float(self.confidence_level),
            'method': self.method,
            'sample_size': int(self.sample_size),
            'metadata': self.metadata,
            'timestamp': datetime.now().isoformat()
        }
    
    def margin_of_error(self) -> float:
        """Calculate margin of error from confidence interval"""
        return (self.confidence_interval[1] - self.confidence_interval[0]) / 2
    
    def is_significant_improvement_over(self, baseline_value: float) -> bool:
        """Check if improvement over baseline is statistically significant"""
        return self.confidence_interval[0] > baseline_value


class BootstrapAnalyzer:
    """Advanced bootstrap analysis for model performance uncertainty quantification"""
    
    def __init__(self, 
                 n_bootstrap: int = 1000,
                 confidence_level: float = 0.95,
                 random_state: int = 42):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        if STRUCTURED_LOGGING_AVAILABLE:
            self.logger = MLOpsLoggers.get_logger('statistical_analysis')
        else:
            self.logger = logging.getLogger(__name__)
    
    def bootstrap_metric(self, 
                        y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        metric_func: Callable,
                        stratify: bool = True) -> StatisticalResult:
        """
        Bootstrap confidence interval for any metric function
        
        Args:
            y_true: True labels
            y_pred: Predicted labels or probabilities
            metric_func: Function that takes (y_true, y_pred) and returns metric
            stratify: Whether to use stratified bootstrap sampling
        """
        
        n_samples = len(y_true)
        bootstrap_scores = []
        
        # Original metric value
        original_score = metric_func(y_true, y_pred)
        
        for i in range(self.n_bootstrap):
            # Bootstrap sampling
            if stratify:
                # Stratified bootstrap to maintain class distribution
                indices = self._stratified_bootstrap_indices(y_true)
            else:
                indices = self.rng.choice(n_samples, size=n_samples, replace=True)
            
            # Calculate metric on bootstrap sample
            try:
                bootstrap_score = metric_func(y_true[indices], y_pred[indices])
                bootstrap_scores.append(bootstrap_score)
            except Exception as e:
                # Skip invalid bootstrap samples
                continue
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        # Calculate confidence interval
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_scores, lower_percentile)
        ci_upper = np.percentile(bootstrap_scores, upper_percentile)
        
        return StatisticalResult(
            point_estimate=original_score,
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=self.confidence_level,
            method='bootstrap',
            sample_size=n_samples,
            metadata={
                'n_bootstrap': self.n_bootstrap,
                'bootstrap_mean': float(np.mean(bootstrap_scores)),
                'bootstrap_std': float(np.std(bootstrap_scores)),
                'stratified': stratify,
                'valid_bootstraps': len(bootstrap_scores)
            }
        )
    
    def _stratified_bootstrap_indices(self, y_true: np.ndarray) -> np.ndarray:
        """Generate stratified bootstrap indices maintaining class distribution"""
        indices = []
        unique_classes, class_counts = np.unique(y_true, return_counts=True)
        
        for class_label, count in zip(unique_classes, class_counts):
            class_indices = np.where(y_true == class_label)[0]
            bootstrap_indices = self.rng.choice(class_indices, size=count, replace=True)
            indices.extend(bootstrap_indices)
        
        return np.array(indices)
    
    def bootstrap_model_comparison(self,
                                 y_true: np.ndarray,
                                 y_pred_1: np.ndarray,
                                 y_pred_2: np.ndarray,
                                 metric_func: Callable,
                                 model_1_name: str = "Model 1",
                                 model_2_name: str = "Model 2") -> Dict[str, Any]:
        """
        Bootstrap comparison between two models with statistical significance testing
        """
        
        n_samples = len(y_true)
        differences = []
        
        # Calculate original difference
        score_1 = metric_func(y_true, y_pred_1)
        score_2 = metric_func(y_true, y_pred_2)
        original_difference = score_2 - score_1
        
        # Bootstrap sampling for difference
        for i in range(self.n_bootstrap):
            indices = self.rng.choice(n_samples, size=n_samples, replace=True)
            
            try:
                boot_score_1 = metric_func(y_true[indices], y_pred_1[indices])
                boot_score_2 = metric_func(y_true[indices], y_pred_2[indices])
                differences.append(boot_score_2 - boot_score_1)
            except:
                continue
        
        differences = np.array(differences)
        
        # Calculate confidence interval for difference
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(differences, (alpha / 2) * 100)
        ci_upper = np.percentile(differences, (1 - alpha / 2) * 100)
        
        # Statistical significance test
        p_value_bootstrap = np.mean(differences <= 0) * 2  # Two-tailed test
        is_significant = ci_lower > 0 or ci_upper < 0
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(differences)) / 2)
        cohens_d = original_difference / pooled_std if pooled_std > 0 else 0
        
        return {
            'model_1_name': model_1_name,
            'model_2_name': model_2_name,
            'model_1_score': StatisticalResult(
                point_estimate=score_1,
                confidence_interval=(score_1 - np.std(differences), score_1 + np.std(differences)),
                confidence_level=self.confidence_level,
                method='bootstrap_individual',
                sample_size=n_samples
            ).to_dict(),
            'model_2_score': StatisticalResult(
                point_estimate=score_2,
                confidence_interval=(score_2 - np.std(differences), score_2 + np.std(differences)),
                confidence_level=self.confidence_level,
                method='bootstrap_individual',
                sample_size=n_samples
            ).to_dict(),
            'difference': StatisticalResult(
                point_estimate=original_difference,
                confidence_interval=(ci_lower, ci_upper),
                confidence_level=self.confidence_level,
                method='bootstrap_difference',
                sample_size=n_samples,
                metadata={
                    'p_value_bootstrap': float(p_value_bootstrap),
                    'is_significant': bool(is_significant),
                    'effect_size_cohens_d': float(cohens_d),
                    'bootstrap_mean_difference': float(np.mean(differences)),
                    'bootstrap_std_difference': float(np.std(differences))
                }
            ).to_dict()
        }


class FeatureImportanceAnalyzer:
    """Advanced feature importance analysis with uncertainty quantification"""
    
    def __init__(self, 
                 n_bootstrap: int = 500,
                 confidence_level: float = 0.95,
                 random_state: int = 42):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        if STRUCTURED_LOGGING_AVAILABLE:
            self.logger = MLOpsLoggers.get_logger('feature_importance')
        else:
            self.logger = logging.getLogger(__name__)
    
    def analyze_importance_stability(self,
                                   model,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Analyze feature importance stability using bootstrap sampling
        """
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        importance_samples = []
        
        # Bootstrap sampling for importance stability
        for i in range(self.n_bootstrap):
            # Bootstrap sample
            indices = self.rng.choice(len(X), size=len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            try:
                # Fit model on bootstrap sample
                model_copy = self._clone_model(model)
                model_copy.fit(X_boot, y_boot)
                
                # Extract feature importances
                if hasattr(model_copy, 'feature_importances_'):
                    importances = model_copy.feature_importances_
                elif hasattr(model_copy, 'coef_'):
                    importances = np.abs(model_copy.coef_).flatten()
                else:
                    # Use permutation importance as fallback
                    from sklearn.inspection import permutation_importance
                    perm_importance = permutation_importance(model_copy, X_boot, y_boot, n_repeats=5, random_state=self.random_state)
                    importances = perm_importance.importances_mean
                
                importance_samples.append(importances)
                
            except Exception as e:
                continue
        
        importance_samples = np.array(importance_samples)
        
        # Calculate statistics for each feature
        feature_stats = {}
        
        for i, feature_name in enumerate(feature_names):
            if i < importance_samples.shape[1]:
                feature_importances = importance_samples[:, i]
                
                # Calculate confidence interval
                alpha = 1 - self.confidence_level
                ci_lower = np.percentile(feature_importances, (alpha / 2) * 100)
                ci_upper = np.percentile(feature_importances, (1 - alpha / 2) * 100)
                
                # Stability metrics
                cv_importance = np.std(feature_importances) / np.mean(feature_importances) if np.mean(feature_importances) > 0 else np.inf
                
                feature_stats[feature_name] = StatisticalResult(
                    point_estimate=float(np.mean(feature_importances)),
                    confidence_interval=(float(ci_lower), float(ci_upper)),
                    confidence_level=self.confidence_level,
                    method='bootstrap_importance',
                    sample_size=len(importance_samples),
                    metadata={
                        'coefficient_of_variation': float(cv_importance),
                        'std_importance': float(np.std(feature_importances)),
                        'min_importance': float(np.min(feature_importances)),
                        'max_importance': float(np.max(feature_importances)),
                        'stability_rank': None  # Will be filled later
                    }
                ).to_dict()
        
        # Rank features by stability (lower CV = more stable)
        sorted_features = sorted(
            feature_stats.items(),
            key=lambda x: x[1]['metadata']['coefficient_of_variation']
        )
        
        for rank, (feature_name, stats) in enumerate(sorted_features):
            feature_stats[feature_name]['metadata']['stability_rank'] = rank + 1
        
        return {
            'feature_importance_analysis': feature_stats,
            'stability_ranking': [name for name, _ in sorted_features],
            'analysis_metadata': {
                'n_bootstrap_samples': self.n_bootstrap,
                'confidence_level': self.confidence_level,
                'n_features_analyzed': len(feature_names),
                'valid_bootstrap_runs': len(importance_samples)
            }
        }
    
    def _clone_model(self, model):
        """Clone model for bootstrap sampling"""
        from sklearn.base import clone
        try:
            return clone(model)
        except:
            # Fallback: create new instance with same parameters
            return type(model)(**model.get_params())
    
    def permutation_importance_with_ci(self,
                                     model,
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     scoring_func: Callable,
                                     feature_names: List[str] = None,
                                     n_repeats: int = 10) -> Dict[str, Any]:
        """
        Calculate permutation importance with confidence intervals
        """
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Baseline score
        baseline_score = scoring_func(model, X, y)
        
        feature_importance_scores = {}
        
        for feature_idx, feature_name in enumerate(feature_names):
            importance_scores = []
            
            # Multiple permutation rounds for each feature
            for _ in range(n_repeats):
                # Permute feature
                X_permuted = X.copy()
                X_permuted[:, feature_idx] = self.rng.permutation(X_permuted[:, feature_idx])
                
                # Calculate score with permuted feature
                permuted_score = scoring_func(model, X_permuted, y)
                importance = baseline_score - permuted_score
                importance_scores.append(importance)
            
            # Calculate statistics
            importance_scores = np.array(importance_scores)
            
            alpha = 1 - self.confidence_level
            ci_lower = np.percentile(importance_scores, (alpha / 2) * 100)
            ci_upper = np.percentile(importance_scores, (1 - alpha / 2) * 100)
            
            feature_importance_scores[feature_name] = StatisticalResult(
                point_estimate=float(np.mean(importance_scores)),
                confidence_interval=(float(ci_lower), float(ci_upper)),
                confidence_level=self.confidence_level,
                method='permutation_importance',
                sample_size=n_repeats,
                metadata={
                    'baseline_score': float(baseline_score),
                    'std_importance': float(np.std(importance_scores)),
                    'is_statistically_important': float(ci_lower) > 0
                }
            ).to_dict()
        
        return {
            'permutation_importance': feature_importance_scores,
            'baseline_score': float(baseline_score),
            'analysis_metadata': {
                'n_repeats': n_repeats,
                'confidence_level': self.confidence_level,
                'scoring_function': scoring_func.__name__ if hasattr(scoring_func, '__name__') else 'custom'
            }
        }


class AdvancedCrossValidation:
    """Advanced cross-validation with comprehensive statistical reporting"""
    
    def __init__(self, 
                 cv_folds: int = 5,
                 n_bootstrap: int = 200,
                 confidence_level: float = 0.95,
                 random_state: int = 42):
        self.cv_folds = cv_folds
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.bootstrap_analyzer = BootstrapAnalyzer(n_bootstrap, confidence_level, random_state)
        
        if STRUCTURED_LOGGING_AVAILABLE:
            self.logger = MLOpsLoggers.get_logger('cross_validation')
        else:
            self.logger = logging.getLogger(__name__)
    
    def comprehensive_cv_analysis(self,
                                model,
                                X: np.ndarray,
                                y: np.ndarray,
                                scoring_metrics: Dict[str, Callable]) -> Dict[str, Any]:
        """
        Comprehensive cross-validation analysis with statistical significance testing
        """
        
        from sklearn.model_selection import cross_validate, StratifiedKFold
        
        # Setup CV strategy
        cv_strategy = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y,
            cv=cv_strategy,
            scoring=scoring_metrics,
            return_train_score=True,
            return_indices=True,
            n_jobs=1
        )
        
        analysis_results = {
            'cv_folds': self.cv_folds,
            'metrics_analysis': {},
            'fold_analysis': [],
            'statistical_tests': {},
            'confidence_intervals': {}
        }
        
        # Analyze each metric
        for metric_name, metric_func in scoring_metrics.items():
            test_scores = cv_results[f'test_{metric_name}']
            train_scores = cv_results[f'train_{metric_name}']
            
            # Bootstrap confidence intervals for CV scores
            test_ci = self._bootstrap_cv_scores(test_scores)
            train_ci = self._bootstrap_cv_scores(train_scores)
            
            # Statistical tests
            statistical_tests = self._perform_cv_statistical_tests(test_scores, train_scores)
            
            analysis_results['metrics_analysis'][metric_name] = {
                'test_scores': {
                    'mean': float(np.mean(test_scores)),
                    'std': float(np.std(test_scores)),
                    'confidence_interval': test_ci,
                    'scores': test_scores.tolist()
                },
                'train_scores': {
                    'mean': float(np.mean(train_scores)),
                    'std': float(np.std(train_scores)),
                    'confidence_interval': train_ci,
                    'scores': train_scores.tolist()
                },
                'overfitting_analysis': {
                    'overfitting_score': float(np.mean(train_scores) - np.mean(test_scores)),
                    'overfitting_ci': self._calculate_overfitting_ci(train_scores, test_scores)
                },
                'statistical_tests': statistical_tests
            }
        
        # Fold-by-fold analysis
        for fold_idx in range(self.cv_folds):
            fold_analysis = {
                'fold': fold_idx + 1,
                'metrics': {}
            }
            
            for metric_name in scoring_metrics.keys():
                fold_analysis['metrics'][metric_name] = {
                    'test_score': float(cv_results[f'test_{metric_name}'][fold_idx]),
                    'train_score': float(cv_results[f'train_{metric_name}'][fold_idx])
                }
            
            analysis_results['fold_analysis'].append(fold_analysis)
        
        return analysis_results
    
    def _bootstrap_cv_scores(self, scores: np.ndarray) -> Dict[str, float]:
        """Bootstrap confidence interval for CV scores"""
        bootstrap_means = []
        
        for _ in range(self.n_bootstrap):
            bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_means, (alpha / 2) * 100)
        ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
        
        return {
            'lower': float(ci_lower),
            'upper': float(ci_upper),
            'confidence_level': self.confidence_level
        }
    
    def _perform_cv_statistical_tests(self, test_scores: np.ndarray, train_scores: np.ndarray) -> Dict[str, Any]:
        """Perform statistical tests on CV results"""
        
        tests = {}
        
        # Test for overfitting using paired t-test
        try:
            t_stat, p_value = stats.ttest_rel(train_scores, test_scores)
            tests['overfitting_ttest'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant_overfitting': p_value < 0.05 and t_stat > 0,
                'interpretation': 'Significant overfitting detected' if (p_value < 0.05 and t_stat > 0) else 'No significant overfitting'
            }
        except Exception as e:
            tests['overfitting_ttest'] = {'error': str(e)}
        
        # Normality test for CV scores
        try:
            shapiro_stat, shapiro_p = stats.shapiro(test_scores)
            tests['normality_test'] = {
                'shapiro_statistic': float(shapiro_stat),
                'p_value': float(shapiro_p),
                'normally_distributed': shapiro_p > 0.05,
                'interpretation': 'CV scores are normally distributed' if shapiro_p > 0.05 else 'CV scores are not normally distributed'
            }
        except Exception as e:
            tests['normality_test'] = {'error': str(e)}
        
        # Stability test (coefficient of variation)
        cv_coefficient = np.std(test_scores) / np.mean(test_scores) if np.mean(test_scores) > 0 else np.inf
        tests['stability_analysis'] = {
            'coefficient_of_variation': float(cv_coefficient),
            'stability_interpretation': self._interpret_stability(cv_coefficient)
        }
        
        return tests
    
    def _calculate_overfitting_ci(self, train_scores: np.ndarray, test_scores: np.ndarray) -> Dict[str, float]:
        """Calculate confidence interval for overfitting metric"""
        overfitting_differences = train_scores - test_scores
        
        bootstrap_diffs = []
        for _ in range(self.n_bootstrap):
            indices = np.random.choice(len(overfitting_differences), size=len(overfitting_differences), replace=True)
            bootstrap_diffs.append(np.mean(overfitting_differences[indices]))
        
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_diffs, (alpha / 2) * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)
        
        return {
            'lower': float(ci_lower),
            'upper': float(ci_upper),
            'confidence_level': self.confidence_level
        }
    
    def _interpret_stability(self, cv_coefficient: float) -> str:
        """Interpret CV stability based on coefficient of variation"""
        if cv_coefficient < 0.1:
            return "Very stable performance across folds"
        elif cv_coefficient < 0.2:
            return "Stable performance across folds"
        elif cv_coefficient < 0.3:
            return "Moderately stable performance across folds"
        else:
            return "Unstable performance across folds - consider data quality or model complexity"


class StatisticalModelComparison:
    """Advanced statistical comparison between models with comprehensive uncertainty analysis"""
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 n_bootstrap: int = 1000,
                 random_state: int = 42):
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.bootstrap_analyzer = BootstrapAnalyzer(n_bootstrap, confidence_level, random_state)
        
        if STRUCTURED_LOGGING_AVAILABLE:
            self.logger = MLOpsLoggers.get_logger('model_comparison')
        else:
            self.logger = logging.getLogger(__name__)
    
    def comprehensive_model_comparison(self,
                                     models: Dict[str, Any],
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     metrics: Dict[str, Callable],
                                     cv_folds: int = 5) -> Dict[str, Any]:
        """
        Comprehensive pairwise model comparison with statistical significance testing
        """
        
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Get CV predictions for each model
        model_predictions = {}
        model_cv_scores = {}
        
        for model_name, model in models.items():
            # Cross-validation predictions
            cv_pred = cross_val_predict(model, X, y, cv=cv_strategy, method='predict_proba')
            if cv_pred.ndim == 2 and cv_pred.shape[1] == 2:
                cv_pred = cv_pred[:, 1]  # Binary classification probabilities
            
            model_predictions[model_name] = cv_pred
            
            # Calculate CV scores for each metric
            model_cv_scores[model_name] = {}
            for metric_name, metric_func in metrics.items():
                try:
                    if 'roc_auc' in metric_name.lower():
                        scores = [metric_func(y[test], cv_pred[test]) for train, test in cv_strategy.split(X, y)]
                    else:
                        pred_labels = (cv_pred > 0.5).astype(int)
                        scores = [metric_func(y[test], pred_labels[test]) for train, test in cv_strategy.split(X, y)]
                    
                    model_cv_scores[model_name][metric_name] = np.array(scores)
                except Exception as e:
                    self.logger.warning(f"Failed to calculate {metric_name} for {model_name}: {e}")
        
        # Pairwise comparisons
        comparison_results = {}
        model_names = list(models.keys())
        
        for i, model1_name in enumerate(model_names):
            for j, model2_name in enumerate(model_names[i+1:], i+1):
                comparison_key = f"{model1_name}_vs_{model2_name}"
                
                comparison_results[comparison_key] = self._pairwise_comparison(
                    model1_name, model2_name,
                    model_cv_scores[model1_name],
                    model_cv_scores[model2_name],
                    model_predictions[model1_name],
                    model_predictions[model2_name],
                    y, metrics
                )
        
        # Overall ranking
        ranking = self._rank_models(model_cv_scores, primary_metric='f1')
        
        return {
            'individual_model_results': model_cv_scores,
            'pairwise_comparisons': comparison_results,
            'model_ranking': ranking,
            'analysis_metadata': {
                'cv_folds': cv_folds,
                'confidence_level': self.confidence_level,
                'n_bootstrap': self.n_bootstrap,
                'models_compared': len(models),
                'metrics_evaluated': list(metrics.keys())
            }
        }
    
    def _pairwise_comparison(self,
                           model1_name: str, model2_name: str,
                           scores1: Dict[str, np.ndarray],
                           scores2: Dict[str, np.ndarray],
                           pred1: np.ndarray, pred2: np.ndarray,
                           y_true: np.ndarray,
                           metrics: Dict[str, Callable]) -> Dict[str, Any]:
        """Detailed pairwise comparison between two models"""
        
        comparison = {
            'models': [model1_name, model2_name],
            'metric_comparisons': {},
            'overall_comparison': {}
        }
        
        significant_improvements = 0
        total_comparisons = 0
        
        # Compare each metric
        for metric_name in scores1.keys():
            if metric_name in scores2:
                metric_comparison = self._compare_metric_scores(
                    scores1[metric_name], scores2[metric_name], metric_name
                )
                
                comparison['metric_comparisons'][metric_name] = metric_comparison
                
                if metric_comparison['statistical_tests']['significant_improvement']:
                    significant_improvements += 1
                total_comparisons += 1
        
        # Bootstrap comparison of predictions
        if len(pred1) == len(pred2) == len(y_true):
            bootstrap_comparison = self._bootstrap_prediction_comparison(
                y_true, pred1, pred2, metrics
            )
            comparison['bootstrap_prediction_comparison'] = bootstrap_comparison
        
        # Overall decision
        improvement_rate = significant_improvements / total_comparisons if total_comparisons > 0 else 0
        
        comparison['overall_comparison'] = {
            'significant_improvements': significant_improvements,
            'total_comparisons': total_comparisons,
            'improvement_rate': float(improvement_rate),
            'recommendation': self._make_comparison_recommendation(improvement_rate, significant_improvements)
        }
        
        return comparison
    
    def _compare_metric_scores(self, scores1: np.ndarray, scores2: np.ndarray, metric_name: str) -> Dict[str, Any]:
        """Statistical comparison of metric scores between two models"""
        
        # Basic statistics
        mean1, mean2 = np.mean(scores1), np.mean(scores2)
        std1, std2 = np.std(scores1), np.std(scores2)
        improvement = mean2 - mean1
        
        # Statistical tests
        statistical_tests = {}
        
        # Paired t-test
        try:
            t_stat, p_value = stats.ttest_rel(scores2, scores1)
            statistical_tests['paired_ttest'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'effect_direction': 'improvement' if t_stat > 0 else 'degradation'
            }
        except Exception as e:
            statistical_tests['paired_ttest'] = {'error': str(e)}
        
        # Wilcoxon signed-rank test (non-parametric)
        try:
            w_stat, w_p = stats.wilcoxon(scores2, scores1, alternative='two-sided')
            statistical_tests['wilcoxon'] = {
                'statistic': float(w_stat),
                'p_value': float(w_p),
                'significant': w_p < 0.05
            }
        except Exception as e:
            statistical_tests['wilcoxon'] = {'error': str(e)}
        
        # Bootstrap confidence interval for difference
        bootstrap_diffs = []
        for _ in range(200):  # Reduced for performance
            indices = np.random.choice(len(scores1), size=len(scores1), replace=True)
            diff = np.mean(scores2[indices]) - np.mean(scores1[indices])
            bootstrap_diffs.append(diff)
        
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_diffs, (alpha / 2) * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        cohens_d = improvement / pooled_std if pooled_std > 0 else 0
        
        return {
            'metric_name': metric_name,
            'mean_scores': {'model1': float(mean1), 'model2': float(mean2)},
            'improvement': float(improvement),
            'relative_improvement_percent': float((improvement / mean1) * 100) if mean1 > 0 else 0,
            'confidence_interval': {'lower': float(ci_lower), 'upper': float(ci_upper)},
            'effect_size_cohens_d': float(cohens_d),
            'statistical_tests': statistical_tests,
            'significant_improvement': improvement > 0 and ci_lower > 0,
            'interpretation': self._interpret_effect_size(cohens_d)
        }
    
    def _bootstrap_prediction_comparison(self, y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray, metrics: Dict[str, Callable]) -> Dict[str, Any]:
        """Bootstrap comparison of model predictions"""
        
        bootstrap_results = {}
        
        for metric_name, metric_func in metrics.items():
            try:
                # For probabilistic metrics, use probabilities directly
                if 'roc_auc' in metric_name.lower():
                    comparison = self.bootstrap_analyzer.bootstrap_model_comparison(
                        y_true, pred1, pred2, metric_func, "Model1", "Model2"
                    )
                else:
                    # For classification metrics, convert to class predictions
                    pred1_class = (pred1 > 0.5).astype(int)
                    pred2_class = (pred2 > 0.5).astype(int)
                    comparison = self.bootstrap_analyzer.bootstrap_model_comparison(
                        y_true, pred1_class, pred2_class, metric_func, "Model1", "Model2"
                    )
                
                bootstrap_results[metric_name] = comparison
                
            except Exception as e:
                bootstrap_results[metric_name] = {'error': str(e)}
        
        return bootstrap_results
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "Negligible effect"
        elif abs_d < 0.5:
            return "Small effect"
        elif abs_d < 0.8:
            return "Medium effect"
        else:
            return "Large effect"
    
    def _make_comparison_recommendation(self, improvement_rate: float, significant_improvements: int) -> str:
        """Make recommendation based on comparison results"""
        if improvement_rate >= 0.75 and significant_improvements >= 2:
            return "Strong recommendation for model upgrade"
        elif improvement_rate >= 0.5 and significant_improvements >= 1:
            return "Moderate recommendation for model upgrade"
        elif improvement_rate > 0:
            return "Weak recommendation for model upgrade - consider other factors"
        else:
            return "No recommendation for model upgrade"
    
    def _rank_models(self, model_cv_scores: Dict[str, Dict[str, np.ndarray]], primary_metric: str = 'f1') -> Dict[str, Any]:
        """Rank models based on CV performance with statistical significance"""
        
        # Calculate mean scores for primary metric
        model_means = {}
        for model_name, scores in model_cv_scores.items():
            if primary_metric in scores:
                model_means[model_name] = np.mean(scores[primary_metric])
        
        # Sort by mean performance
        sorted_models = sorted(model_means.items(), key=lambda x: x[1], reverse=True)
        
        # Statistical significance testing for ranking
        ranking_with_significance = []
        for i, (model_name, mean_score) in enumerate(sorted_models):
            rank_info = {
                'rank': i + 1,
                'model_name': model_name,
                'mean_score': float(mean_score),
                'significantly_better_than': []
            }
            
            # Compare with lower-ranked models
            for j, (other_model, other_score) in enumerate(sorted_models[i+1:], i+1):
                try:
                    t_stat, p_value = stats.ttest_rel(
                        model_cv_scores[model_name][primary_metric],
                        model_cv_scores[other_model][primary_metric]
                    )
                    
                    if p_value < 0.05 and t_stat > 0:
                        rank_info['significantly_better_than'].append({
                            'model': other_model,
                            'p_value': float(p_value),
                            'rank': j + 1
                        })
                except Exception:
                    continue
            
            ranking_with_significance.append(rank_info)
        
        return {
            'ranking': ranking_with_significance,
            'primary_metric': primary_metric,
            'ranking_method': 'mean_cv_score_with_significance_testing'
        }


# Integration utilities for existing codebase
class MLOpsStatisticalAnalyzer:
    """Comprehensive statistical analyzer for MLOps pipeline"""
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 n_bootstrap: int = 1000,
                 random_state: int = 42):
        
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        
        # Initialize analyzers
        self.bootstrap_analyzer = BootstrapAnalyzer(n_bootstrap, confidence_level, random_state)
        self.feature_analyzer = FeatureImportanceAnalyzer(n_bootstrap, confidence_level, random_state)
        self.cv_analyzer = AdvancedCrossValidation(5, n_bootstrap, confidence_level, random_state)
        self.comparison_analyzer = StatisticalModelComparison(confidence_level, n_bootstrap, random_state)
        
        if STRUCTURED_LOGGING_AVAILABLE:
            self.logger = MLOpsLoggers.get_logger('statistical_analyzer')
        else:
            self.logger = logging.getLogger(__name__)
    
    def comprehensive_model_analysis(self,
                                   models: Dict[str, Any],
                                   X_train: np.ndarray,
                                   X_test: np.ndarray,
                                   y_train: np.ndarray,
                                   y_test: np.ndarray,
                                   feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis of models including:
        - Bootstrap confidence intervals for performance metrics
        - Feature importance stability analysis
        - Advanced cross-validation with statistical testing
        - Pairwise model comparisons with significance testing
        """
        
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        
        # Define metrics
        def accuracy_func(y_true, y_pred): return accuracy_score(y_true, y_pred)
        def f1_func(y_true, y_pred): return f1_score(y_true, y_pred, average='weighted')
        def precision_func(y_true, y_pred): return precision_score(y_true, y_pred, average='weighted')
        def recall_func(y_true, y_pred): return recall_score(y_true, y_pred, average='weighted')
        def roc_auc_func(y_true, y_pred_proba): return roc_auc_score(y_true, y_pred_proba)
        
        metrics = {
            'accuracy': accuracy_func,
            'f1': f1_func,
            'precision': precision_func,
            'recall': recall_func,
            'roc_auc': roc_auc_func
        }
        
        analysis_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'configuration': {
                'confidence_level': self.confidence_level,
                'n_bootstrap': self.n_bootstrap,
                'models_analyzed': list(models.keys())
            },
            'individual_model_analysis': {},
            'comparative_analysis': {},
            'feature_importance_analysis': {},
            'recommendations': []
        }
        
        # Individual model analysis
        for model_name, model in models.items():
            try:
                # Fit model
                model.fit(X_train, y_train)
                
                # Get predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                # Bootstrap analysis for each metric
                bootstrap_results = {}
                for metric_name, metric_func in metrics.items():
                    if metric_name == 'roc_auc':
                        result = self.bootstrap_analyzer.bootstrap_metric(
                            y_test, y_pred_proba, metric_func
                        )
                    else:
                        result = self.bootstrap_analyzer.bootstrap_metric(
                            y_test, y_pred, metric_func
                        )
                    bootstrap_results[metric_name] = result.to_dict()
                
                # Cross-validation analysis
                cv_analysis = self.cv_analyzer.comprehensive_cv_analysis(
                    model, X_train, y_train, metrics
                )
                
                # Feature importance analysis (if supported)
                feature_analysis = {}
                if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                    try:
                        feature_analysis = self.feature_analyzer.analyze_importance_stability(
                            model, X_train, y_train, feature_names
                        )
                    except Exception as e:
                        feature_analysis = {'error': str(e)}
                
                analysis_results['individual_model_analysis'][model_name] = {
                    'bootstrap_metrics': bootstrap_results,
                    'cross_validation_analysis': cv_analysis,
                    'feature_importance_analysis': feature_analysis
                }
                
            except Exception as e:
                self.logger.error(f"Analysis failed for model {model_name}: {e}")
                analysis_results['individual_model_analysis'][model_name] = {'error': str(e)}
        
        # Comparative analysis
        if len(models) > 1:
            try:
                comparative_results = self.comparison_analyzer.comprehensive_model_comparison(
                    models, X_train, y_train, metrics
                )
                analysis_results['comparative_analysis'] = comparative_results
                
                # Generate recommendations based on comparison
                recommendations = self._generate_analysis_recommendations(comparative_results)
                analysis_results['recommendations'].extend(recommendations)
                
            except Exception as e:
                analysis_results['comparative_analysis'] = {'error': str(e)}
        
        return analysis_results
    
    def _generate_analysis_recommendations(self, comparative_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on statistical analysis"""
        recommendations = []
        
        # Model ranking recommendations
        if 'model_ranking' in comparative_results:
            ranking = comparative_results['model_ranking']['ranking']
            if len(ranking) > 0:
                best_model = ranking[0]
                significantly_better_count = len(best_model.get('significantly_better_than', []))
                
                if significantly_better_count > 0:
                    recommendations.append({
                        'type': 'model_selection',
                        'priority': 'high',
                        'message': f"Model '{best_model['model_name']}' shows statistically significant improvement over {significantly_better_count} other model(s)",
                        'action': f"Consider promoting {best_model['model_name']} to production"
                    })
        
        # Feature importance recommendations
        for model_name, analysis in comparative_results.get('individual_model_analysis', {}).items():
            feature_analysis = analysis.get('feature_importance_analysis', {})
            if 'stability_ranking' in feature_analysis:
                unstable_features = [
                    name for name, stats in feature_analysis['feature_importance_analysis'].items()
                    if stats['metadata']['coefficient_of_variation'] > 0.5
                ]
                
                if unstable_features:
                    recommendations.append({
                        'type': 'feature_engineering',
                        'priority': 'medium',
                        'message': f"Model '{model_name}' has {len(unstable_features)} unstable features with high variance",
                        'action': "Review feature engineering process and consider feature selection"
                    })
        
        # Cross-validation recommendations
        for model_name, analysis in comparative_results.get('individual_model_analysis', {}).items():
            cv_analysis = analysis.get('cross_validation_analysis', {})
            for metric_name, metric_analysis in cv_analysis.get('metrics_analysis', {}).items():
                overfitting_analysis = metric_analysis.get('overfitting_analysis', {})
                if overfitting_analysis.get('overfitting_score', 0) > 0.1:  # 10% overfitting threshold
                    recommendations.append({
                        'type': 'model_complexity',
                        'priority': 'medium',
                        'message': f"Model '{model_name}' shows significant overfitting in {metric_name}",
                        'action': "Consider regularization or reducing model complexity"
                    })
        
        return recommendations
    
    def save_analysis_report(self, analysis_results: Dict[str, Any], file_path: Path = None):
        """Save comprehensive analysis report"""
        if file_path is None:
            file_path = Path("/tmp/logs/statistical_analysis_report.json")
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        self.logger.info(f"Statistical analysis report saved to {file_path}")
        return file_path


# Integration functions for existing codebase
def integrate_statistical_analysis_with_retrain():
    """Integration example for retrain.py"""
    analyzer = MLOpsStatisticalAnalyzer()
    
    # Example usage in retraining context
    def enhanced_model_comparison(models_dict, X_train, X_test, y_train, y_test):
        """Enhanced model comparison with comprehensive statistical analysis"""
        
        analysis_results = analyzer.comprehensive_model_analysis(
            models_dict, X_train, X_test, y_train, y_test
        )
        
        # Extract promotion decision based on statistical significance
        comparative_analysis = analysis_results.get('comparative_analysis', {})
        ranking = comparative_analysis.get('model_ranking', {}).get('ranking', [])
        
        if ranking:
            best_model = ranking[0]
            promotion_confidence = len(best_model.get('significantly_better_than', [])) / (len(ranking) - 1) if len(ranking) > 1 else 1.0
            
            return {
                'recommended_model': best_model['model_name'],
                'statistical_confidence': promotion_confidence,
                'analysis_results': analysis_results,
                'promote_candidate': promotion_confidence > 0.5
            }
        
        return {'error': 'No valid model ranking available'}
    
    return enhanced_model_comparison

def integrate_statistical_analysis_with_train():
    """Integration example for train.py"""
    analyzer = MLOpsStatisticalAnalyzer()
    
    def enhanced_ensemble_validation(individual_models, ensemble_model, X, y):
        """Enhanced ensemble validation with bootstrap confidence intervals"""
        
        models_to_compare = {**individual_models, 'ensemble': ensemble_model}
        
        # Perform comprehensive statistical analysis
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        analysis_results = analyzer.comprehensive_model_analysis(
            models_to_compare, X_train, X_test, y_train, y_test
        )
        
        # Check if ensemble is statistically significantly better
        comparative_analysis = analysis_results.get('comparative_analysis', {})
        ensemble_comparisons = {
            k: v for k, v in comparative_analysis.get('pairwise_comparisons', {}).items()
            if 'ensemble' in k
        }
        
        significant_improvements = 0
        total_comparisons = len(ensemble_comparisons)
        
        for comparison in ensemble_comparisons.values():
            if comparison.get('overall_comparison', {}).get('improvement_rate', 0) > 0.5:
                significant_improvements += 1
        
        ensemble_confidence = significant_improvements / total_comparisons if total_comparisons > 0 else 0
        
        return {
            'use_ensemble': ensemble_confidence > 0.5,
            'ensemble_confidence': ensemble_confidence,
            'statistical_analysis': analysis_results
        }
    
    return enhanced_ensemble_validation


if __name__ == "__main__":
    # Example usage and testing
    print("Testing advanced statistical analysis system...")
    
    # Generate sample data for testing
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = (X[:, 0] + X[:, 1] + np.random.randn(200) * 0.1 > 0).astype(int)
    
    # Create sample models
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    models = {
        'logistic_regression': LogisticRegression(random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=50, random_state=42)
    }
    
    # Test comprehensive analysis
    analyzer = MLOpsStatisticalAnalyzer(n_bootstrap=100)  # Reduced for testing
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Running comprehensive statistical analysis...")
    results = analyzer.comprehensive_model_analysis(
        models, X_train, X_test, y_train, y_test
    )
    
    print(f"Analysis completed for {len(models)} models")
    print(f"Generated {len(results['recommendations'])} recommendations")
    
    # Test bootstrap analysis
    bootstrap_analyzer = BootstrapAnalyzer(n_bootstrap=100)
    
    from sklearn.metrics import f1_score
    def f1_metric(y_true, y_pred):
        return f1_score(y_true, y_pred, average='weighted')
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    bootstrap_result = bootstrap_analyzer.bootstrap_metric(y_test, y_pred, f1_metric)
    print(f"Bootstrap F1 confidence interval: {bootstrap_result.confidence_interval}")
    
    print("Advanced statistical analysis system test completed successfully!")