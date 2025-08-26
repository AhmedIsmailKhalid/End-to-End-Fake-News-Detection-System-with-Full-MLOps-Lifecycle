# tests/test_retrain.py
# Comprehensive test suite for enhanced retraining pipeline with LightGBM + ensemble

import pytest
import numpy as np
import pandas as pd
import joblib
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.retrain import (
    EnhancedModelRetrainer, CVModelComparator, EnsembleManager,
    preprocess_text_function, AutomatedRetrainingManager
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import lightgbm as lgb


class TestPreprocessing:
    """Test preprocessing and data handling functionality"""
    
    def test_preprocess_text_function_basic(self):
        """Test basic text preprocessing functionality"""
        texts = [
            "Check out this link: https://example.com and email me@test.com",
            "Multiple!!! question marks??? and dots...",
            "Mixed123 characters456 and symbols@#$",
            ""
        ]
        
        processed = preprocess_text_function(texts)
        
        # Should remove URLs and emails
        assert "https://example.com" not in processed[0]
        assert "me@test.com" not in processed[0]
        
        # Should normalize punctuation
        assert "!!!" not in processed[1]
        assert "???" not in processed[1]
        
        # Should remove non-alphabetic chars except basic punctuation
        assert "123" not in processed[2]
        assert "@#$" not in processed[2]
        
        # Should handle empty strings
        assert processed[3] == ""
    
    def test_preprocess_text_function_edge_cases(self):
        """Test preprocessing with edge cases"""
        edge_cases = [None, 123, [], {"text": "test"}]
        
        # Should convert all inputs to strings without crashing
        processed = preprocess_text_function(edge_cases)
        assert len(processed) == 4
        for result in processed:
            assert isinstance(result, str)


class TestCVModelComparator:
    """Test cross-validation and model comparison functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    @pytest.fixture
    def cv_comparator(self):
        """Create CV comparator instance"""
        return CVModelComparator(cv_folds=3, random_state=42)
    
    def test_create_cv_strategy(self, cv_comparator, sample_data):
        """Test CV strategy creation with different data sizes"""
        X, y = sample_data
        
        # Normal case
        cv_strategy = cv_comparator.create_cv_strategy(X, y)
        assert cv_strategy.n_splits <= 3
        assert cv_strategy.n_splits >= 2
        
        # Small dataset case
        X_small = X[:8]
        y_small = y[:8]
        cv_strategy_small = cv_comparator.create_cv_strategy(X_small, y_small)
        assert cv_strategy_small.n_splits >= 2
        assert cv_strategy_small.n_splits <= len(np.unique(y_small))
    
    def test_perform_model_cv_evaluation(self, cv_comparator, sample_data):
        """Test CV evaluation of individual models"""
        X, y = sample_data
        
        # Create simple pipeline for testing
        model = Pipeline([
            ('model', LogisticRegression(random_state=42, max_iter=100))
        ])
        
        results = cv_comparator.perform_model_cv_evaluation(model, X, y)
        
        # Should return comprehensive CV results
        assert 'test_scores' in results
        assert 'train_scores' in results
        assert 'fold_results' in results
        assert 'n_splits' in results
        
        # Should have all metrics
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in expected_metrics:
            assert metric in results['test_scores']
            assert 'mean' in results['test_scores'][metric]
            assert 'std' in results['test_scores'][metric]
            assert 'scores' in results['test_scores'][metric]
    
    def test_compare_models_with_cv(self, cv_comparator, sample_data):
        """Test statistical comparison between two models"""
        X, y = sample_data
        
        # Create two different models
        model1 = Pipeline([('model', LogisticRegression(random_state=42, max_iter=100))])
        model2 = Pipeline([('model', RandomForestClassifier(random_state=42, n_estimators=10))])
        
        comparison = cv_comparator.compare_models_with_cv(model1, model2, X, y)
        
        # Should return comprehensive comparison
        assert 'metric_comparisons' in comparison
        assert 'promotion_decision' in comparison
        assert 'feature_engineering_comparison' in comparison
        
        # Should have statistical tests for each metric
        for metric in ['accuracy', 'f1', 'precision', 'recall']:
            if metric in comparison['metric_comparisons']:
                metric_comp = comparison['metric_comparisons'][metric]
                assert 'improvement' in metric_comp
                assert 'tests' in metric_comp
                if 'paired_ttest' in metric_comp['tests']:
                    assert 'p_value' in metric_comp['tests']['paired_ttest']
                    assert 'significant' in metric_comp['tests']['paired_ttest']
    
    def test_feature_upgrade_assessment(self, cv_comparator):
        """Test feature engineering upgrade detection"""
        # Mock results with different feature types
        results1 = {'feature_engineering_type': 'standard_tfidf'}
        results2 = {'feature_engineering_type': 'enhanced'}
        
        upgrade = cv_comparator._assess_feature_upgrade(results1, results2)
        
        assert upgrade['is_upgrade'] == True
        assert upgrade['upgrade_type'] == 'standard_to_enhanced'
        assert 'upgrade' in upgrade['description'].lower()


class TestEnsembleManager:
    """Test ensemble creation and validation"""
    
    @pytest.fixture
    def ensemble_manager(self):
        """Create ensemble manager instance"""
        return EnsembleManager(random_state=42)
    
    @pytest.fixture
    def individual_models(self, sample_data):
        """Create individual trained models"""
        X, y = sample_data
        
        models = {
            'logistic_regression': Pipeline([
                ('model', LogisticRegression(random_state=42, max_iter=100))
            ]),
            'random_forest': Pipeline([
                ('model', RandomForestClassifier(random_state=42, n_estimators=10))
            ])
        }
        
        # Fit models
        for model in models.values():
            model.fit(X, y)
            
        return models
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for ensemble testing"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    def test_create_ensemble(self, ensemble_manager, individual_models):
        """Test ensemble creation from individual models"""
        ensemble = ensemble_manager.create_ensemble(individual_models)
        
        assert isinstance(ensemble, VotingClassifier)
        assert len(ensemble.estimators) == len(individual_models)
        assert ensemble.voting == 'soft'
        
        # Check estimator names match
        estimator_names = [name for name, _ in ensemble.estimators]
        assert set(estimator_names) == set(individual_models.keys())
    
    def test_evaluate_ensemble_vs_individuals(self, ensemble_manager, individual_models, sample_data):
        """Test ensemble performance comparison"""
        X, y = sample_data
        X_train, X_test, y_train, y_test = X[:80], X[80:], y[:80], y[80:]
        
        # Create and fit ensemble
        ensemble = ensemble_manager.create_ensemble(individual_models)
        ensemble.fit(X_train, y_train)
        
        # Evaluate
        results = ensemble_manager.evaluate_ensemble_vs_individuals(
            ensemble, individual_models, X_test, y_test
        )
        
        # Should have results for all models plus ensemble
        expected_keys = set(individual_models.keys()) | {'ensemble', 'ensemble_analysis'}
        assert set(results.keys()) == expected_keys
        
        # Should have all metrics for each model
        for model_name in individual_models.keys():
            assert 'accuracy' in results[model_name]
            assert 'f1' in results[model_name]
            assert 'precision' in results[model_name]
            assert 'recall' in results[model_name]
            assert 'roc_auc' in results[model_name]
        
        # Should have ensemble analysis
        assert 'best_individual_f1' in results['ensemble_analysis']
        assert 'ensemble_f1' in results['ensemble_analysis']
        assert 'improvement' in results['ensemble_analysis']
    
    def test_statistical_ensemble_comparison(self, ensemble_manager, individual_models, sample_data):
        """Test statistical comparison for ensemble recommendation"""
        X, y = sample_data
        cv_manager = CVModelComparator(cv_folds=3, random_state=42)
        
        ensemble = ensemble_manager.create_ensemble(individual_models)
        
        results = ensemble_manager.statistical_ensemble_comparison(
            ensemble, individual_models, X, y, cv_manager
        )
        
        # Should have comprehensive statistical comparison
        assert 'ensemble_recommendation' in results
        assert 'statistical_comparisons' in results
        
        recommendation = results['ensemble_recommendation']
        assert 'use_ensemble' in recommendation
        assert 'confidence' in recommendation
        assert 'significantly_better_than' in recommendation


class TestEnhancedModelRetrainer:
    """Test main retraining functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def retrainer(self, temp_dir):
        """Create retrainer instance with temporary paths"""
        retrainer = EnhancedModelRetrainer()
        
        # Override paths to use temp directory
        retrainer.base_dir = temp_dir
        retrainer.data_dir = temp_dir / "data"
        retrainer.model_dir = temp_dir / "model"
        retrainer.logs_dir = temp_dir / "logs"
        retrainer.backup_dir = temp_dir / "backups"
        retrainer.features_dir = temp_dir / "features"
        
        # Recreate paths
        for dir_path in [retrainer.data_dir, retrainer.model_dir, retrainer.logs_dir, 
                         retrainer.backup_dir, retrainer.features_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Update file paths
        retrainer.combined_data_path = retrainer.data_dir / "combined_dataset.csv"
        retrainer.metadata_path = temp_dir / "metadata.json"
        retrainer.prod_pipeline_path = retrainer.model_dir / "pipeline.pkl"
        
        return retrainer
    
    @pytest.fixture
    def sample_dataset(self, temp_dir):
        """Create sample dataset for testing"""
        data = {
            'text': [
                'This is a real news article about politics and government.',
                'Fake news alert: celebrities do crazy things for attention.',
                'Scientific breakthrough in renewable energy technology announced.',
                'Conspiracy theory about secret government mind control programs.',
                'Local weather update: sunny skies expected this weekend.',
                'Breaking: major financial market crash predicted by experts.'
            ] * 20,  # Repeat to get enough samples
            'label': [0, 1, 0, 1, 0, 1] * 20  # 0=real, 1=fake
        }
        
        df = pd.DataFrame(data)
        dataset_path = temp_dir / "data" / "combined_dataset.csv"
        dataset_path.parent.mkdir(exist_ok=True)
        df.to_csv(dataset_path, index=False)
        
        return dataset_path, df
    
    def test_setup_models(self, retrainer):
        """Test model configuration setup"""
        # Should have all three models configured
        expected_models = {'logistic_regression', 'random_forest', 'lightgbm'}
        assert set(retrainer.models.keys()) == expected_models
        
        # Should have LightGBM properly configured
        lgb_config = retrainer.models['lightgbm']
        assert isinstance(lgb_config['model'], lgb.LGBMClassifier)
        assert lgb_config['model'].n_jobs == 1  # CPU optimization
        assert 'param_grid' in lgb_config
        
        # All models should have CPU-friendly settings
        for model_config in retrainer.models.values():
            model = model_config['model']
            if hasattr(model, 'n_jobs'):
                assert model.n_jobs == 1
    
    def test_load_new_data(self, retrainer, sample_dataset):
        """Test data loading and validation"""
        dataset_path, expected_df = sample_dataset
        
        success, df, message = retrainer.load_new_data()
        
        assert success == True
        assert df is not None
        assert len(df) == len(expected_df)
        assert 'text' in df.columns
        assert 'label' in df.columns
        assert set(df['label'].unique()) == {0, 1}
    
    def test_clean_and_validate_data(self, retrainer):
        """Test data cleaning and validation"""
        # Create test data with various issues
        dirty_data = pd.DataFrame({
            'text': [
                'Valid text sample',
                'Short',  # Too short
                '',  # Empty
                None,  # Null
                'Valid longer text sample for testing',
                'x' * 15000,  # Too long
                'Another valid text sample'
            ],
            'label': [0, 1, 0, 2, 1, 1, 0]  # Invalid label (2)
        })
        
        clean_df = retrainer.clean_and_validate_data(dirty_data)
        
        # Should filter out problematic rows
        assert len(clean_df) < len(dirty_data)
        assert all(clean_df['text'].str.len() > 10)
        assert all(clean_df['text'].str.len() < 10000)
        assert set(clean_df['label'].unique()).issubset({0, 1})
        assert not clean_df.isnull().any().any()
    
    def test_create_preprocessing_pipeline_standard(self, retrainer):
        """Test standard TF-IDF pipeline creation"""
        retrainer.use_enhanced_features = False
        
        pipeline = retrainer.create_preprocessing_pipeline()
        
        assert isinstance(pipeline, Pipeline)
        step_names = [name for name, _ in pipeline.steps]
        
        # Should have standard pipeline steps
        expected_steps = ['preprocess', 'vectorize', 'feature_select', 'model']
        assert step_names == expected_steps
        
        # Model step should be None (set later)
        assert pipeline.named_steps['model'] is None
    
    @patch('model.retrain.ENHANCED_FEATURES_AVAILABLE', True)
    def test_create_preprocessing_pipeline_enhanced(self, retrainer):
        """Test enhanced feature pipeline creation (mocked)"""
        retrainer.use_enhanced_features = True
        
        with patch('model.retrain.AdvancedFeatureEngineer') as mock_fe:
            pipeline = retrainer.create_preprocessing_pipeline()
            
            assert isinstance(pipeline, Pipeline)
            step_names = [name for name, _ in pipeline.steps]
            
            # Should have enhanced pipeline steps
            expected_steps = ['enhanced_features', 'model']
            assert step_names == expected_steps
            
            # Should create feature engineer with correct parameters
            mock_fe.assert_called_once()
            call_kwargs = mock_fe.call_args[1]
            assert call_kwargs['feature_selection_k'] == retrainer.feature_selection_k
            assert call_kwargs['tfidf_max_features'] == retrainer.max_features
    
    def test_hyperparameter_tuning_small_dataset(self, retrainer):
        """Test hyperparameter tuning with very small dataset"""
        # Create minimal dataset that should skip tuning
        X = np.random.randn(15, 5)
        y = np.random.randint(0, 2, 15)
        
        pipeline = retrainer.create_preprocessing_pipeline()
        
        best_model, results = retrainer.hyperparameter_tuning_with_cv(
            pipeline, X, y, 'logistic_regression'
        )
        
        # Should skip tuning and use default parameters
        assert 'note' in results
        assert 'skipped' in results['note'].lower()
        assert results['best_params'] == 'default_parameters'
        assert best_model is not None
    
    def test_detect_production_feature_type(self, retrainer, temp_dir):
        """Test production model feature type detection"""
        # Test with no existing model
        feature_type = retrainer.detect_production_feature_type()
        assert feature_type in ['standard_tfidf', 'unknown']
        
        # Test with metadata indicating enhanced features
        metadata = {
            'feature_engineering': {
                'type': 'enhanced'
            }
        }
        with open(retrainer.metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        feature_type = retrainer.detect_production_feature_type()
        assert feature_type == 'enhanced'
    
    def test_error_handling_invalid_data(self, retrainer, temp_dir):
        """Test error handling with invalid data scenarios"""
        # Test with no data files
        success, df, message = retrainer.load_new_data()
        assert success == False
        assert 'No data files found' in message
        
        # Test with empty dataset
        empty_df = pd.DataFrame({'text': [], 'label': []})
        empty_path = temp_dir / "data" / "combined_dataset.csv"
        empty_path.parent.mkdir(exist_ok=True)
        empty_df.to_csv(empty_path, index=False)
        
        success, df, message = retrainer.load_new_data()
        assert success == False
        assert 'Insufficient data' in message


class TestIntegration:
    """Integration tests for complete retraining workflow"""
    
    @pytest.fixture
    def complete_setup(self):
        """Set up complete testing environment"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create retrainer
            retrainer = EnhancedModelRetrainer()
            retrainer.base_dir = temp_path
            retrainer.setup_paths()
            
            # Create sample data
            data = pd.DataFrame({
                'text': [
                    f'Real news article number {i} with substantial content for testing.'
                    for i in range(30)
                ] + [
                    f'Fake news article number {i} with misleading information and content.'
                    for i in range(30)
                ],
                'label': [0] * 30 + [1] * 30
            })
            
            data.to_csv(retrainer.combined_data_path, index=False)
            
            # Create mock production model
            mock_model = Pipeline([
                ('vectorize', Mock()),
                ('model', LogisticRegression(random_state=42))
            ])
            joblib.dump(mock_model, retrainer.prod_pipeline_path)
            
            yield retrainer, data
    
    def test_end_to_end_retraining_workflow(self, complete_setup):
        """Test complete retraining workflow"""
        retrainer, data = complete_setup
        
        # Disable ensemble for faster testing
        retrainer.enable_ensemble = False
        retrainer.use_enhanced_features = False
        
        # Should complete without errors
        success, message = retrainer.retrain_model()
        
        # Should either promote or keep current model
        assert success == True
        assert 'enhanced' in message.lower() or 'keeping' in message.lower() or 'promoted' in message.lower()
        
        # Should create proper logs
        assert retrainer.retraining_log_path.exists()
    
    @patch('model.retrain.ENHANCED_FEATURES_AVAILABLE', True)
    def test_ensemble_selection_workflow(self, complete_setup):
        """Test ensemble selection in complete workflow"""
        retrainer, data = complete_setup
        
        # Enable ensemble and enhanced features (mocked)
        retrainer.enable_ensemble = True
        retrainer.use_enhanced_features = False  # Keep False to avoid import issues
        
        with patch.object(retrainer, 'train_and_evaluate_models') as mock_train:
            # Mock successful training with ensemble selection
            mock_results = {
                'logistic_regression': {
                    'model': Mock(),
                    'tuning_results': {
                        'cross_validation': {
                            'test_scores': {'f1': {'mean': 0.75}}
                        }
                    }
                },
                'random_forest': {
                    'model': Mock(), 
                    'tuning_results': {
                        'cross_validation': {
                            'test_scores': {'f1': {'mean': 0.77}}
                        }
                    }
                },
                'lightgbm': {
                    'model': Mock(),
                    'tuning_results': {
                        'cross_validation': {
                            'test_scores': {'f1': {'mean': 0.76}}
                        }
                    }
                },
                'ensemble': {
                    'model': Mock(),
                    'statistical_comparison': {
                        'ensemble_recommendation': {'use_ensemble': True, 'confidence': 0.85}
                    }
                }
            }
            mock_train.return_value = mock_results
            
            # Test model selection
            best_name, best_model, best_metrics = retrainer.select_best_model(mock_results)
            
            # Should select ensemble when recommended
            assert best_name == 'ensemble'
            assert best_model == mock_results['ensemble']['model']


class TestAutomatedRetrainingManager:
    """Test automated retraining management"""
    
    @pytest.fixture
    def automation_manager(self):
        """Create automation manager for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AutomatedRetrainingManager(base_dir=Path(temp_dir))
            yield manager
    
    def test_initialization(self, automation_manager):
        """Test automation manager initialization"""
        assert automation_manager.enhanced_features_available is not None
        assert automation_manager.automation_dir.exists()
        assert hasattr(automation_manager, 'drift_monitor')
    
    def test_manual_retraining_trigger(self, automation_manager):
        """Test manual retraining trigger functionality"""
        with patch.object(EnhancedModelRetrainer, 'automated_retrain_with_validation') as mock_retrain:
            mock_retrain.return_value = (True, "Retraining completed successfully")
            
            result = automation_manager.trigger_manual_retraining("test_reason")
            
            assert result['success'] == True
            assert 'enhanced' in result['message'].lower()
            mock_retrain.assert_called_once()


# Performance and Resource Tests
class TestPerformanceConstraints:
    """Test performance under CPU constraints (HuggingFace Spaces)"""
    
    def test_cpu_optimization_settings(self):
        """Test all models use CPU-friendly settings"""
        retrainer = EnhancedModelRetrainer()
        
        for model_name, config in retrainer.models.items():
            model = config['model']
            
            # Check n_jobs setting for models that support it
            if hasattr(model, 'n_jobs'):
                assert model.n_jobs == 1, f"{model_name} should use n_jobs=1 for CPU optimization"
            
            # Check LightGBM specific settings
            if isinstance(model, lgb.LGBMClassifier):
                assert model.n_estimators <= 100, "LightGBM should use reasonable n_estimators for CPU"
                assert model.num_leaves <= 31, "LightGBM should use reasonable num_leaves for CPU"
                assert model.verbose == -1, "LightGBM should suppress verbose output"
    
    def test_memory_efficient_processing(self):
        """Test memory-efficient data processing"""
        retrainer = EnhancedModelRetrainer()
        
        # Test with reasonably sized dataset
        large_data = pd.DataFrame({
            'text': ['Sample text for testing memory efficiency'] * 1000,
            'label': np.random.randint(0, 2, 1000)
        })
        
        # Should handle without memory issues
        cleaned_data = retrainer.clean_and_validate_data(large_data)
        assert len(cleaned_data) <= len(large_data)
        
        # Check feature selection limits
        assert retrainer.feature_selection_k <= retrainer.max_features
        assert retrainer.max_features <= 7500  # Reasonable limit for CPU constraints


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])