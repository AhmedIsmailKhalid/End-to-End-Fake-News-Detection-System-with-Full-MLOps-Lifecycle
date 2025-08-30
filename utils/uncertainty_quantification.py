# utils/uncertainty_quantification.py
# Enhanced uncertainty quantification integration for existing MLOps pipeline

import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Callable
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass
import logging

# Import statistical analysis components
try:
    from .statistical_analysis import (
        MLOpsStatisticalAnalyzer, BootstrapAnalyzer, 
        FeatureImportanceAnalyzer, StatisticalResult
    )
    STATISTICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    STATISTICAL_ANALYSIS_AVAILABLE = False
    logging.warning("Statistical analysis components not available")

# Import structured logging
try:
    from .structured_logger import StructuredLogger, EventType, MLOpsLoggers
    STRUCTURED_LOGGING_AVAILABLE = True
except ImportError:
    STRUCTURED_LOGGING_AVAILABLE = False
    import logging


@dataclass
class UncertaintyReport:
    """Comprehensive uncertainty quantification report"""
    model_performance_uncertainty: Dict[str, Any]
    feature_importance_uncertainty: Dict[str, Any] 
    cross_validation_uncertainty: Dict[str, Any]
    prediction_uncertainty: Dict[str, Any]
    model_comparison_uncertainty: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    confidence_level: float
    analysis_timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'model_performance_uncertainty': self.model_performance_uncertainty,
            'feature_importance_uncertainty': self.feature_importance_uncertainty,
            'cross_validation_uncertainty': self.cross_validation_uncertainty,
            'prediction_uncertainty': self.prediction_uncertainty,
            'model_comparison_uncertainty': self.model_comparison_uncertainty,
            'recommendations': self.recommendations,
            'confidence_level': self.confidence_level,
            'analysis_timestamp': self.analysis_timestamp
        }
    
    def save_report(self, file_path: Path = None) -> Path:
        """Save uncertainty report to file"""
        if file_path is None:
            file_path = Path("/tmp/logs/uncertainty_report.json")
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        return file_path


class EnhancedUncertaintyQuantifier:
    """Enhanced uncertainty quantification for MLOps pipeline integration"""
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 n_bootstrap: int = 1000,
                 random_state: int = 42):
        
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        
        if STATISTICAL_ANALYSIS_AVAILABLE:
            self.statistical_analyzer = MLOpsStatisticalAnalyzer(
                confidence_level, n_bootstrap, random_state
            )
            self.bootstrap_analyzer = BootstrapAnalyzer(n_bootstrap, confidence_level, random_state)
            self.feature_analyzer = FeatureImportanceAnalyzer(n_bootstrap, confidence_level, random_state)
        else:
            raise ImportError("Statistical analysis components required for uncertainty quantification")
        
        if STRUCTURED_LOGGING_AVAILABLE:
            self.logger = MLOpsLoggers.get_logger('uncertainty_quantification')
        else:
            self.logger = logging.getLogger(__name__)
    
    def quantify_model_uncertainty(self, 
                                  model, 
                                  X_train: np.ndarray, 
                                  X_test: np.ndarray,
                                  y_train: np.ndarray, 
                                  y_test: np.ndarray,
                                  model_name: str = "model") -> Dict[str, Any]:
        """Quantify uncertainty in model performance metrics"""
        
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        
        # Fit model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Define metric functions
        metrics = {
            'accuracy': lambda y_true, y_pred: accuracy_score(y_true, y_pred),
            'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
            'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
            'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
            'roc_auc': lambda y_true, y_pred_proba: roc_auc_score(y_true, y_pred_proba)
        }
        
        # Bootstrap confidence intervals for each metric
        uncertainty_results = {}
        
        for metric_name, metric_func in metrics.items():
            try:
                if metric_name == 'roc_auc':
                    result = self.bootstrap_analyzer.bootstrap_metric(
                        y_test, y_pred_proba, metric_func
                    )
                else:
                    result = self.bootstrap_analyzer.bootstrap_metric(
                        y_test, y_pred, metric_func
                    )
                
                uncertainty_results[metric_name] = {
                    'point_estimate': result.point_estimate,
                    'confidence_interval': result.confidence_interval,
                    'margin_of_error': result.margin_of_error(),
                    'relative_uncertainty': result.margin_of_error() / result.point_estimate if result.point_estimate > 0 else np.inf,
                    'confidence_level': result.confidence_level,
                    'sample_size': result.sample_size,
                    'metadata': result.metadata
                }
                
            except Exception as e:
                uncertainty_results[metric_name] = {'error': str(e)}
        
        # Overall uncertainty assessment
        valid_uncertainties = [
            r['relative_uncertainty'] for r in uncertainty_results.values() 
            if isinstance(r, dict) and 'relative_uncertainty' in r and np.isfinite(r['relative_uncertainty'])
        ]
        
        overall_assessment = {
            'model_name': model_name,
            'average_relative_uncertainty': float(np.mean(valid_uncertainties)) if valid_uncertainties else np.inf,
            'max_relative_uncertainty': float(np.max(valid_uncertainties)) if valid_uncertainties else np.inf,
            'uncertainty_level': self._classify_uncertainty_level(np.mean(valid_uncertainties)) if valid_uncertainties else 'unknown'
        }
        
        return {
            'metric_uncertainties': uncertainty_results,
            'overall_assessment': overall_assessment,
            'analysis_metadata': {
                'confidence_level': self.confidence_level,
                'n_bootstrap': self.n_bootstrap,
                'test_size': len(y_test),
                'train_size': len(y_train)
            }
        }
    
    def quantify_feature_importance_uncertainty(self,
                                              model,
                                              X: np.ndarray,
                                              y: np.ndarray,
                                              feature_names: List[str] = None) -> Dict[str, Any]:
        """Quantify uncertainty in feature importance rankings"""
        
        try:
            # Analyze feature importance stability
            stability_results = self.feature_analyzer.analyze_importance_stability(
                model, X, y, feature_names
            )
            
            # Extract uncertainty metrics
            feature_uncertainties = {}
            unstable_features = []
            
            for feature_name, analysis in stability_results['feature_importance_analysis'].items():
                cv = analysis['metadata']['coefficient_of_variation']
                
                feature_uncertainties[feature_name] = {
                    'importance_mean': analysis['point_estimate'],
                    'importance_ci': analysis['confidence_interval'],
                    'coefficient_of_variation': cv,
                    'stability_rank': analysis['metadata']['stability_rank'],
                    'uncertainty_level': self._classify_feature_uncertainty(cv)
                }
                
                # Flag highly uncertain features
                if cv > 0.5:  # 50% coefficient of variation threshold
                    unstable_features.append({
                        'feature': feature_name,
                        'cv': cv,
                        'reason': 'High variance in importance across bootstrap samples'
                    })
            
            return {
                'feature_importance_uncertainties': feature_uncertainties,
                'stability_ranking': stability_results['stability_ranking'],
                'unstable_features': unstable_features,
                'uncertainty_summary': {
                    'total_features': len(feature_uncertainties),
                    'unstable_features_count': len(unstable_features),
                    'uncertainty_rate': len(unstable_features) / len(feature_uncertainties) if feature_uncertainties else 0
                },
                'analysis_metadata': stability_results['analysis_metadata']
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def quantify_cross_validation_uncertainty(self,
                                            model,
                                            X: np.ndarray,
                                            y: np.ndarray,
                                            cv_folds: int = 5) -> Dict[str, Any]:
        """Quantify uncertainty in cross-validation results"""
        
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import f1_score, accuracy_score
        
        try:
            # Define CV strategy
            cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            # Comprehensive CV analysis with uncertainty quantification
            metrics = {
                'accuracy': lambda y_true, y_pred: accuracy_score(y_true, y_pred),
                'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
            }
            
            cv_analysis = self.statistical_analyzer.cv_analyzer.comprehensive_cv_analysis(
                model, X, y, metrics
            )
            
            # Extract uncertainty information
            cv_uncertainties = {}
            
            for metric_name, analysis in cv_analysis['metrics_analysis'].items():
                test_scores = analysis['test_scores']
                
                # Calculate additional uncertainty metrics
                cv_coefficient = test_scores['std'] / test_scores['mean'] if test_scores['mean'] > 0 else np.inf
                
                cv_uncertainties[metric_name] = {
                    'cv_mean': test_scores['mean'],
                    'cv_std': test_scores['std'],
                    'cv_scores': test_scores['scores'],
                    'coefficient_of_variation': cv_coefficient,
                    'confidence_interval': test_scores['confidence_interval'],
                    'stability_level': self._classify_cv_stability(cv_coefficient),
                    'overfitting_analysis': analysis.get('overfitting_analysis', {}),
                    'statistical_tests': analysis.get('statistical_tests', {})
                }
            
            return {
                'cv_uncertainties': cv_uncertainties,
                'cv_metadata': {
                    'cv_folds': cv_folds,
                    'sample_size': len(X),
                    'confidence_level': self.confidence_level
                },
                'stability_assessment': self._assess_cv_stability(cv_uncertainties)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def quantify_prediction_uncertainty(self,
                                      model,
                                      X_new: np.ndarray,
                                      n_bootstrap_predictions: int = 100) -> Dict[str, Any]:
        """Quantify uncertainty in individual predictions using bootstrap"""
        
        try:
            # This requires the original training data - simplified version for demonstration
            # In practice, you'd need to store bootstrap models or use other uncertainty methods
            
            if hasattr(model, 'predict_proba'):
                # For probabilistic models, use prediction probabilities as uncertainty proxy
                probabilities = model.predict_proba(X_new)
                predictions = model.predict(X_new)
                
                # Calculate prediction uncertainty metrics
                prediction_uncertainties = []
                
                for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
                    max_proba = np.max(proba)
                    entropy = -np.sum(proba * np.log(proba + 1e-8))  # Add small constant for numerical stability
                    
                    uncertainty_info = {
                        'prediction': int(pred),
                        'prediction_probability': float(max_proba),
                        'entropy': float(entropy),
                        'uncertainty_level': self._classify_prediction_uncertainty(max_proba),
                        'all_class_probabilities': proba.tolist()
                    }
                    
                    prediction_uncertainties.append(uncertainty_info)
                
                # Overall prediction uncertainty summary
                avg_entropy = np.mean([p['entropy'] for p in prediction_uncertainties])
                avg_confidence = np.mean([p['prediction_probability'] for p in prediction_uncertainties])
                
                uncertain_predictions = sum(1 for p in prediction_uncertainties if p['uncertainty_level'] in ['high', 'very_high'])
                
                return {
                    'individual_predictions': prediction_uncertainties,
                    'uncertainty_summary': {
                        'total_predictions': len(prediction_uncertainties),
                        'uncertain_predictions': uncertain_predictions,
                        'uncertainty_rate': uncertain_predictions / len(prediction_uncertainties),
                        'average_entropy': float(avg_entropy),
                        'average_confidence': float(avg_confidence)
                    }
                }
            else:
                return {
                    'error': 'Model does not support probability predictions - uncertainty quantification limited'
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    def comprehensive_uncertainty_analysis(self,
                                         models: Dict[str, Any],
                                         X_train: np.ndarray,
                                         X_test: np.ndarray,
                                         y_train: np.ndarray,
                                         y_test: np.ndarray,
                                         feature_names: List[str] = None) -> UncertaintyReport:
        """Perform comprehensive uncertainty analysis across all components"""
        
        # Model performance uncertainty
        model_uncertainties = {}
        for model_name, model in models.items():
            model_uncertainties[model_name] = self.quantify_model_uncertainty(
                model, X_train, X_test, y_train, y_test, model_name
            )
        
        # Feature importance uncertainty (using best model)
        best_model_name = min(model_uncertainties.keys(), 
                             key=lambda k: model_uncertainties[k]['overall_assessment']['average_relative_uncertainty'])
        best_model = models[best_model_name]
        
        feature_uncertainty = self.quantify_feature_importance_uncertainty(
            best_model, X_train, y_train, feature_names
        )
        
        # Cross-validation uncertainty
        cv_uncertainty = self.quantify_cross_validation_uncertainty(
            best_model, X_train, y_train
        )
        
        # Prediction uncertainty on test set
        prediction_uncertainty = self.quantify_prediction_uncertainty(
            best_model, X_test
        )
        
        # Model comparison uncertainty
        if len(models) > 1:
            comparison_uncertainty = self._quantify_model_comparison_uncertainty(
                models, X_train, y_train
            )
        else:
            comparison_uncertainty = {'single_model': 'No comparison available'}
        
        # Generate recommendations
        recommendations = self._generate_uncertainty_recommendations(
            model_uncertainties, feature_uncertainty, cv_uncertainty, prediction_uncertainty
        )
        
        return UncertaintyReport(
            model_performance_uncertainty=model_uncertainties,
            feature_importance_uncertainty=feature_uncertainty,
            cross_validation_uncertainty=cv_uncertainty,
            prediction_uncertainty=prediction_uncertainty,
            model_comparison_uncertainty=comparison_uncertainty,
            recommendations=recommendations,
            confidence_level=self.confidence_level,
            analysis_timestamp=datetime.now().isoformat()
        )
    
    def _quantify_model_comparison_uncertainty(self,
                                             models: Dict[str, Any],
                                             X: np.ndarray,
                                             y: np.ndarray) -> Dict[str, Any]:
        """Quantify uncertainty in model comparisons"""
        
        try:
            # Use comprehensive model comparison with statistical analysis
            from sklearn.metrics import f1_score, accuracy_score
            
            metrics = {
                'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
                'accuracy': lambda y_true, y_pred: accuracy_score(y_true, y_pred)
            }
            
            comparison_results = self.statistical_analyzer.comparison_analyzer.comprehensive_model_comparison(
                models, X, y, metrics
            )
            
            # Extract uncertainty information from comparisons
            comparison_uncertainties = {}
            
            for comparison_name, comparison_data in comparison_results.get('pairwise_comparisons', {}).items():
                overall_comp = comparison_data.get('overall_comparison', {})
                
                comparison_uncertainties[comparison_name] = {
                    'improvement_rate': overall_comp.get('improvement_rate', 0),
                    'significant_improvements': overall_comp.get('significant_improvements', 0),
                    'total_comparisons': overall_comp.get('total_comparisons', 0),
                    'recommendation': overall_comp.get('recommendation', 'No recommendation'),
                    'uncertainty_level': self._classify_comparison_uncertainty(overall_comp.get('improvement_rate', 0))
                }
            
            # Overall comparison uncertainty
            ranking = comparison_results.get('model_ranking', {})
            ranking_uncertainty = self._assess_ranking_uncertainty(ranking)
            
            return {
                'pairwise_comparison_uncertainties': comparison_uncertainties,
                'ranking_uncertainty': ranking_uncertainty,
                'comparison_metadata': comparison_results.get('analysis_metadata', {})
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _classify_uncertainty_level(self, relative_uncertainty: float) -> str:
        """Classify overall uncertainty level"""
        if relative_uncertainty < 0.05:
            return 'very_low'
        elif relative_uncertainty < 0.1:
            return 'low'
        elif relative_uncertainty < 0.2:
            return 'medium'
        elif relative_uncertainty < 0.5:
            return 'high'
        else:
            return 'very_high'
    
    def _classify_feature_uncertainty(self, cv: float) -> str:
        """Classify feature importance uncertainty"""
        if cv < 0.2:
            return 'stable'
        elif cv < 0.5:
            return 'moderately_stable'
        elif cv < 1.0:
            return 'unstable'
        else:
            return 'very_unstable'
    
    def _classify_cv_stability(self, cv_coefficient: float) -> str:
        """Classify cross-validation stability"""
        if cv_coefficient < 0.1:
            return 'very_stable'
        elif cv_coefficient < 0.2:
            return 'stable'
        elif cv_coefficient < 0.3:
            return 'moderately_stable'
        else:
            return 'unstable'
    
    def _classify_prediction_uncertainty(self, max_probability: float) -> str:
        """Classify individual prediction uncertainty"""
        if max_probability > 0.95:
            return 'very_low'
        elif max_probability > 0.8:
            return 'low'
        elif max_probability > 0.6:
            return 'medium'
        elif max_probability > 0.5:
            return 'high'
        else:
            return 'very_high'
    
    def _classify_comparison_uncertainty(self, improvement_rate: float) -> str:
        """Classify model comparison uncertainty"""
        if improvement_rate > 0.8:
            return 'very_confident'
        elif improvement_rate > 0.6:
            return 'confident'
        elif improvement_rate > 0.4:
            return 'moderate'
        elif improvement_rate > 0.2:
            return 'uncertain'
        else:
            return 'very_uncertain'
    
    def _assess_cv_stability(self, cv_uncertainties: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall cross-validation stability"""
        
        stability_levels = [info.get('stability_level', 'unknown') for info in cv_uncertainties.values()]
        
        stable_count = sum(1 for level in stability_levels if level in ['very_stable', 'stable'])
        
        return {
            'stable_metrics': stable_count,
            'total_metrics': len(stability_levels),
            'stability_rate': stable_count / len(stability_levels) if stability_levels else 0,
            'overall_stability': 'stable' if stable_count / len(stability_levels) > 0.6 else 'unstable'
        }
    
    def _assess_ranking_uncertainty(self, ranking: Dict[str, Any]) -> Dict[str, Any]:
        """Assess uncertainty in model ranking"""
        
        if not ranking or 'ranking' not in ranking:
            return {'uncertainty': 'unknown', 'reason': 'No ranking data available'}
        
        ranking_data = ranking['ranking']
        
        if len(ranking_data) < 2:
            return {'uncertainty': 'low', 'reason': 'Only one model'}
        
        # Check if top model is significantly better than others
        top_model = ranking_data[0]
        significantly_better_count = len(top_model.get('significantly_better_than', []))
        total_other_models = len(ranking_data) - 1
        
        if significantly_better_count == total_other_models:
            return {
                'uncertainty': 'low',
                'reason': 'Top model significantly better than all others',
                'confidence': 'high'
            }
        elif significantly_better_count > total_other_models / 2:
            return {
                'uncertainty': 'medium',
                'reason': 'Top model significantly better than some others',
                'confidence': 'medium'
            }
        else:
            return {
                'uncertainty': 'high',
                'reason': 'No clear statistical winner among models',
                'confidence': 'low'
            }
    
    def _generate_uncertainty_recommendations(self,
                                            model_uncertainties: Dict[str, Any],
                                            feature_uncertainty: Dict[str, Any],
                                            cv_uncertainty: Dict[str, Any],
                                            prediction_uncertainty: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on uncertainty analysis"""
        
        recommendations = []
        
        # Model performance uncertainty recommendations
        for model_name, uncertainty in model_uncertainties.items():
            overall_assessment = uncertainty.get('overall_assessment', {})
            uncertainty_level = overall_assessment.get('uncertainty_level', 'unknown')
            
            if uncertainty_level in ['high', 'very_high']:
                recommendations.append({
                    'type': 'model_performance',
                    'priority': 'high',
                    'model': model_name,
                    'issue': f'High performance uncertainty ({uncertainty_level})',
                    'action': 'Collect more training data or consider model regularization',
                    'details': {
                        'avg_relative_uncertainty': overall_assessment.get('average_relative_uncertainty', 0),
                        'max_relative_uncertainty': overall_assessment.get('max_relative_uncertainty', 0)
                    }
                })
        
        # Feature importance uncertainty recommendations
        unstable_features = feature_uncertainty.get('unstable_features', [])
        if unstable_features:
            recommendations.append({
                'type': 'feature_importance',
                'priority': 'medium',
                'issue': f'{len(unstable_features)} features have unstable importance rankings',
                'action': 'Review feature engineering and consider feature selection',
                'details': {
                    'unstable_features': [f['feature'] for f in unstable_features],
                    'uncertainty_rate': feature_uncertainty.get('uncertainty_summary', {}).get('uncertainty_rate', 0)
                }
            })
        
        # Cross-validation stability recommendations
        cv_stability = cv_uncertainty.get('stability_assessment', {})
        if cv_stability.get('overall_stability') == 'unstable':
            recommendations.append({
                'type': 'cross_validation',
                'priority': 'medium',
                'issue': 'Unstable cross-validation performance',
                'action': 'Check data quality, consider stratified sampling, or increase CV folds',
                'details': {
                    'stability_rate': cv_stability.get('stability_rate', 0),
                    'stable_metrics': cv_stability.get('stable_metrics', 0),
                    'total_metrics': cv_stability.get('total_metrics', 0)
                }
            })
        
        # Prediction uncertainty recommendations
        pred_summary = prediction_uncertainty.get('uncertainty_summary', {})
        uncertainty_rate = pred_summary.get('uncertainty_rate', 0)
        
        if uncertainty_rate > 0.2:  # More than 20% uncertain predictions
            recommendations.append({
                'type': 'prediction_uncertainty',
                'priority': 'high',
                'issue': f'{uncertainty_rate:.1%} of predictions have high uncertainty',
                'action': 'Consider implementing prediction confidence thresholds or human review for uncertain cases',
                'details': {
                    'uncertain_predictions': pred_summary.get('uncertain_predictions', 0),
                    'total_predictions': pred_summary.get('total_predictions', 0),
                    'average_confidence': pred_summary.get('average_confidence', 0)
                }
            })
        
        return recommendations


# Integration functions for existing codebase
def integrate_uncertainty_quantification_with_retrain():
    """Integration function for retrain.py"""
    
    def enhanced_model_comparison_with_uncertainty(models_dict, X_train, X_test, y_train, y_test):
        """Enhanced model comparison with comprehensive uncertainty quantification"""
        
        try:
            quantifier = EnhancedUncertaintyQuantifier()
            
            # Perform comprehensive uncertainty analysis
            uncertainty_report = quantifier.comprehensive_uncertainty_analysis(
                models_dict, X_train, X_test, y_train, y_test
            )
            
            # Save uncertainty report
            report_path = uncertainty_report.save_report()
            
            # Extract promotion decision based on uncertainty analysis
            model_uncertainties = uncertainty_report.model_performance_uncertainty
            
            # Find model with lowest uncertainty
            best_model_name = min(
                model_uncertainties.keys(),
                key=lambda k: model_uncertainties[k]['overall_assessment']['average_relative_uncertainty']
            )
            
            best_uncertainty = model_uncertainties[best_model_name]['overall_assessment']['average_relative_uncertainty']
            uncertainty_level = model_uncertainties[best_model_name]['overall_assessment']['uncertainty_level']
            
            # Decision logic incorporating uncertainty
            promote_candidate = (
                uncertainty_level in ['very_low', 'low', 'medium'] and
                len(uncertainty_report.recommendations) <= 2
            )
            
            return {
                'recommended_model': best_model_name,
                'uncertainty_level': uncertainty_level,
                'average_uncertainty': best_uncertainty,
                'uncertainty_report': uncertainty_report.to_dict(),
                'report_path': str(report_path),
                'promote_candidate': promote_candidate,
                'recommendations': uncertainty_report.recommendations
            }
            
        except Exception as e:
            return {'error': f'Uncertainty quantification failed: {str(e)}'}
    
    return enhanced_model_comparison_with_uncertainty

def integrate_uncertainty_quantification_with_train():
    """Integration function for train.py"""
    
    def enhanced_ensemble_validation_with_uncertainty(individual_models, ensemble_model, X, y):
        """Enhanced ensemble validation with uncertainty quantification"""
        
        try:
            from sklearn.model_selection import train_test_split
            
            quantifier = EnhancedUncertaintyQuantifier()
            
            # Prepare models for analysis
            models_to_analyze = {**individual_models, 'ensemble': ensemble_model}
            
            # Split data for uncertainty analysis
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Perform uncertainty analysis
            uncertainty_report = quantifier.comprehensive_uncertainty_analysis(
                models_to_analyze, X_train, X_test, y_train, y_test
            )
            
            # Determine ensemble recommendation based on uncertainty
            ensemble_uncertainty = uncertainty_report.model_performance_uncertainty.get('ensemble', {})
            ensemble_uncertainty_level = ensemble_uncertainty.get('overall_assessment', {}).get('uncertainty_level', 'unknown')
            
            # Compare ensemble uncertainty with individual models
            individual_uncertainties = [
                uncertainty_report.model_performance_uncertainty[name]['overall_assessment']['average_relative_uncertainty']
                for name in individual_models.keys()
                if name in uncertainty_report.model_performance_uncertainty
            ]
            
            ensemble_avg_uncertainty = ensemble_uncertainty.get('overall_assessment', {}).get('average_relative_uncertainty', np.inf)
            best_individual_uncertainty = min(individual_uncertainties) if individual_uncertainties else np.inf
            
            # Decision logic
            use_ensemble = (
                ensemble_uncertainty_level in ['very_low', 'low', 'medium'] and
                ensemble_avg_uncertainty <= best_individual_uncertainty * 1.1  # Allow 10% increase in uncertainty
            )
            
            return {
                'use_ensemble': use_ensemble,
                'ensemble_uncertainty_level': ensemble_uncertainty_level,
                'ensemble_avg_uncertainty': ensemble_avg_uncertainty,
                'best_individual_uncertainty': best_individual_uncertainty,
                'uncertainty_analysis': uncertainty_report.to_dict(),
                'recommendations': uncertainty_report.recommendations
            }
            
        except Exception as e:
            return {'error': f'Uncertainty quantification failed: {str(e)}'}
    
    return enhanced_ensemble_validation_with_uncertainty


if __name__ == "__main__":
    # Example usage and testing
    print("Testing enhanced uncertainty quantification system...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(300, 15)
    y = (X[:, 0] + X[:, 1] + np.random.randn(300) * 0.2 > 0).astype(int)
    
    # Create sample models
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    models = {
        'logistic_regression': LogisticRegression(random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=50, random_state=42)
    }
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Test comprehensive uncertainty analysis
    if STATISTICAL_ANALYSIS_AVAILABLE:
        quantifier = EnhancedUncertaintyQuantifier(n_bootstrap=100)  # Reduced for testing
        
        print("Running comprehensive uncertainty analysis...")
        uncertainty_report = quantifier.comprehensive_uncertainty_analysis(
            models, X_train, X_test, y_train, y_test
        )
        
        print(f"Generated {len(uncertainty_report.recommendations)} uncertainty-based recommendations")
        print(f"Overall confidence level: {uncertainty_report.confidence_level}")
        
        # Save report
        report_path = uncertainty_report.save_report()
        print(f"Uncertainty report saved to: {report_path}")
        
        print("Enhanced uncertainty quantification system test completed successfully!")
        
    else:
        print("Statistical analysis components not available - skipping test")