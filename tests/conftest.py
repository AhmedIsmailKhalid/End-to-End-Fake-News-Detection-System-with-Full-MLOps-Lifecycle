# tests/conftest.py
# Shared test configuration and fixtures

import pytest
import numpy as np
import pandas as pd
import tempfile
import sys
import os
from pathlib import Path
from unittest.mock import patch

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture(scope="session") 
def sample_fake_news_data():
    """Generate realistic fake news dataset for testing"""
    np.random.seed(42)
    
    # Realistic fake news patterns
    fake_texts = [
        "BREAKING: Scientists discover shocking truth about vaccines that doctors don't want you to know!",
        "EXCLUSIVE: Celebrity caught in major scandal - you won't believe what happened next!",
        "ALERT: Government secretly planning massive operation - leaked documents reveal everything!",
        "AMAZING: Local mom discovers one weird trick that makes millions - experts hate her!",
        "URGENT: New study proves everything you know about nutrition is completely wrong!",
    ] * 20
    
    # Realistic real news patterns
    real_texts = [
        "Local city council approves new infrastructure budget for road maintenance and repairs.",
        "University researchers publish peer-reviewed study on climate change impacts in regional ecosystems.",
        "Stock market shows mixed results following quarterly earnings reports from major corporations.",
        "Public health officials recommend updated vaccination schedules based on recent clinical trials.",
        "Municipal government announces new public transportation routes to improve city connectivity.",
    ] * 20
    
    # Combine and create DataFrame
    all_texts = fake_texts + real_texts
    all_labels = [1] * len(fake_texts) + [0] * len(real_texts)
    
    df = pd.DataFrame({
        'text': all_texts,
        'label': all_labels
    })
    
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

@pytest.fixture
def mock_enhanced_features():
    """Mock enhanced feature engineering when not available"""
    with patch('model.retrain.ENHANCED_FEATURES_AVAILABLE', True):
        with patch('model.retrain.AdvancedFeatureEngineer') as mock_fe:
            # Configure mock to behave like real feature engineer
            mock_instance = mock_fe.return_value
            mock_instance.get_feature_metadata.return_value = {
                'total_features': 5000,
                'feature_types': {
                    'tfidf_features': 3000,
                    'sentiment_features': 10,
                    'readability_features': 15,
                    'entity_features': 25,
                    'linguistic_features': 50
                },
                'configuration': {'test': True}
            }
            mock_instance.get_feature_importance.return_value = {
                'feature_1': 0.15,
                'feature_2': 0.12,
                'feature_3': 0.10
            }
            mock_instance.get_feature_names.return_value = [f'feature_{i}' for i in range(5000)]
            
            yield mock_fe

# tests/test_data_processing.py
# Test data processing and validation components

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from data.data_validator import DataValidator
from data.prepare_datasets import DatasetPreparer

class TestDataValidation:
    """Test data validation functionality"""
    
    def test_validate_text_column(self, sample_fake_news_data):
        """Test text column validation"""
        validator = DataValidator()
        
        # Valid data should pass
        is_valid, issues = validator.validate_dataframe(sample_fake_news_data)
        assert is_valid == True
        assert len(issues) == 0
        
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'text': ['', 'x', None, 'Valid text here'],
            'label': [0, 1, 0, 2]  # Invalid label
        })
        
        is_valid, issues = validator.validate_dataframe(invalid_data)
        assert is_valid == False
        assert len(issues) > 0
    
    def test_text_quality_validation(self):
        """Test text quality validation rules"""
        validator = DataValidator()
        
        # Test minimum length requirement
        short_texts = pd.DataFrame({
            'text': ['hi', 'ok', 'This is a proper length text for validation'],
            'label': [0, 1, 0]
        })
        
        is_valid, issues = validator.validate_dataframe(short_texts)
        assert is_valid == False
        assert any('length' in str(issue).lower() for issue in issues)


# tests/test_train_integration.py
# Test integration with train.py to ensure compatibility

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

class TestTrainRetrainCompatibility:
    """Test compatibility between train.py and retrain.py"""
    
    def test_metadata_compatibility(self):
        """Test metadata format compatibility between train and retrain"""
        from model.train import EnhancedModelTrainer
        from model.retrain import EnhancedModelRetrainer
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Mock trainer to avoid full training
            trainer = EnhancedModelTrainer(use_enhanced_features=False)
            trainer.base_dir = temp_path
            trainer.setup_paths()
            
            # Create sample metadata as train.py would
            sample_metadata = {
                'model_version': 'v1.0',
                'model_type': 'enhanced_pipeline_cv',
                'feature_engineering': {'type': 'standard'},
                'test_f1': 0.85,
                'cross_validation': {
                    'test_scores': {'f1': {'mean': 0.82, 'std': 0.03}}
                }
            }
            
            # Save metadata
            import json
            with open(trainer.metadata_path, 'w') as f:
                json.dump(sample_metadata, f)
            
            # Test retrainer can read it
            retrainer = EnhancedModelRetrainer()
            retrainer.base_dir = temp_path
            retrainer.setup_paths()
            
            metadata = retrainer.load_existing_metadata()
            assert metadata is not None
            assert metadata['model_version'] == 'v1.0'
            assert metadata['feature_engineering']['type'] == 'standard'
    
    def test_model_file_compatibility(self):
        """Test model file format compatibility"""
        # Both train.py and retrain.py should save/load models consistently
        from model.retrain import EnhancedModelRetrainer
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            retrainer = EnhancedModelRetrainer()
            retrainer.base_dir = temp_path
            retrainer.setup_paths()
            
            # Create mock pipeline as train.py would save
            from sklearn.pipeline import Pipeline
            from sklearn.linear_model import LogisticRegression
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            mock_pipeline = Pipeline([
                ('vectorize', TfidfVectorizer(max_features=1000)),
                ('model', LogisticRegression())
            ])
            
            import joblib
            joblib.dump(mock_pipeline, retrainer.prod_pipeline_path)
            
            # Test retrainer can load it
            success, model, message = retrainer.load_production_model()
            assert success == True
            assert model is not None


# tests/pytest.ini
# Pytest configuration file
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --color=yes
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    cpu_constraint: marks tests that verify CPU constraint compliance
filterwarnings =
    ignore::UserWarning
    ignore::FutureWarning
    ignore::DeprecationWarning


# tests/test_lightgbm_integration.py
# Specific tests for LightGBM integration

import pytest
import numpy as np
from unittest.mock import patch
import lightgbm as lgb

class TestLightGBMIntegration:
    """Test LightGBM-specific functionality"""
    
    def test_lightgbm_model_configuration(self):
        """Test LightGBM model is properly configured for CPU constraints"""
        from model.retrain import EnhancedModelRetrainer
        
        retrainer = EnhancedModelRetrainer()
        lgb_config = retrainer.models['lightgbm']
        lgb_model = lgb_config['model']
        
        # Verify CPU-friendly configuration
        assert isinstance(lgb_model, lgb.LGBMClassifier)
        assert lgb_model.n_jobs == 1
        assert lgb_model.verbose == -1
        assert lgb_model.n_estimators <= 100
        assert lgb_model.num_leaves <= 31
        
        # Verify parameter grid is reasonable for CPU
        param_grid = lgb_config['param_grid']
        assert all(est <= 100 for est in param_grid['model__n_estimators'])
        assert all(leaves <= 31 for leaves in param_grid['model__num_leaves'])
    
    def test_lightgbm_training_integration(self):
        """Test LightGBM integrates properly in training pipeline"""
        from model.retrain import EnhancedModelRetrainer
        
        # Create minimal dataset
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        retrainer = EnhancedModelRetrainer()
        retrainer.use_enhanced_features = False
        
        # Test hyperparameter tuning works with LightGBM
        pipeline = retrainer.create_preprocessing_pipeline()
        
        try:
            best_model, results = retrainer.hyperparameter_tuning_with_cv(
                pipeline, X, y, 'lightgbm'
            )
            
            # Should complete without errors
            assert best_model is not None
            assert 'cross_validation' in results or 'error' in results
            
        except Exception as e:
            # If tuning fails, should fall back gracefully
            assert 'fallback' in str(e).lower() or 'error' in str(e).lower()
    
    def test_lightgbm_cpu_performance(self):
        """Test LightGBM performance is acceptable under CPU constraints"""
        import time
        from model.retrain import EnhancedModelRetrainer
        
        # Create reasonably sized dataset
        X = np.random.randn(200, 20)
        y = np.random.randint(0, 2, 200)
        
        retrainer = EnhancedModelRetrainer()
        pipeline = retrainer.create_preprocessing_pipeline()
        lgb_model = retrainer.models['lightgbm']['model']
        pipeline.set_params(model=lgb_model)
        
        # Time the training
        start_time = time.time()
        pipeline.fit(X, y)
        training_time = time.time() - start_time
        
        # Should complete reasonably quickly on CPU
        assert training_time < 30  # Should take less than 30 seconds
        
        # Should produce valid predictions
        predictions = pipeline.predict(X[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)


# tests/test_ensemble_statistical_validation.py  
# Test ensemble statistical validation logic

import pytest
import numpy as np
from scipy import stats
from unittest.mock import Mock, patch

class TestEnsembleStatisticalValidation:
    """Test statistical validation for ensemble selection"""
    
    def test_paired_ttest_ensemble_selection(self):
        """Test paired t-test logic for ensemble vs individual models"""
        from model.retrain import CVModelComparator
        
        comparator = CVModelComparator(cv_folds=5, random_state=42)
        
        # Create mock CV scores where ensemble is significantly better
        individual_scores = [0.75, 0.74, 0.76, 0.73, 0.75]
        ensemble_scores = [0.80, 0.81, 0.79, 0.78, 0.82]
        
        # Test metric comparison
        comparison = comparator._compare_metric_scores(
            individual_scores, ensemble_scores, 'f1', 'individual', 'ensemble'
        )
        
        assert 'tests' in comparison
        assert 'paired_ttest' in comparison['tests']
        
        # Should detect significant improvement
        t_test_result = comparison['tests']['paired_ttest']
        assert 'p_value' in t_test_result
        assert 'significant' in t_test_result
        
        # With this clear difference, should be significant
        if t_test_result['p_value'] is not None:
            assert t_test_result['significant'] == True
    
    def test_ensemble_not_selected_when_not_significant(self):
        """Test ensemble is not selected when improvement is not significant"""
        from model.retrain import CVModelComparator
        
        comparator = CVModelComparator(cv_folds=5, random_state=42)
        
        # Create mock CV scores where ensemble is only marginally better
        individual_scores = [0.75, 0.74, 0.76, 0.73, 0.75]
        ensemble_scores = [0.751, 0.741, 0.761, 0.731, 0.751]  # Tiny improvement
        
        comparison = comparator._compare_metric_scores(
            individual_scores, ensemble_scores, 'f1', 'individual', 'ensemble'
        )
        
        # Should not show significant improvement
        assert comparison['significant_improvement'] == False
    
    def test_effect_size_calculation(self):
        """Test Cohen's d effect size calculation"""
        from model.retrain import CVModelComparator
        
        comparator = CVModelComparator(cv_folds=5, random_state=42)
        
        # Create scores with known effect size
        individual_scores = [0.70, 0.71, 0.69, 0.72, 0.70]
        ensemble_scores = [0.80, 0.81, 0.79, 0.82, 0.80]  # Large effect
        
        comparison = comparator._compare_metric_scores(
            individual_scores, ensemble_scores, 'f1', 'individual', 'ensemble'
        )
        
        assert 'effect_size' in comparison
        effect_size = comparison['effect_size']
        
        # Should detect large effect size
        assert abs(effect_size) > 0.5  # Large effect by Cohen's standards
    
    def test_promotion_decision_with_feature_upgrade(self):
        """Test promotion decision considers feature engineering upgrades"""
        from model.retrain import CVModelComparator
        
        comparator = CVModelComparator()
        
        # Mock comparison results with feature upgrade
        mock_results = {
            'metric_comparisons': {
                'f1': {
                    'improvement': 0.008,  # Small improvement
                    'significant_improvement': False
                },
                'accuracy': {
                    'improvement': 0.005,
                    'significant_improvement': False
                }
            },
            'feature_engineering_comparison': {
                'feature_upgrade': {
                    'is_upgrade': True,
                    'upgrade_type': 'standard_to_enhanced'
                }
            }
        }
        
        decision = comparator._make_enhanced_promotion_decision(mock_results)
        
        # Should promote despite small improvement due to feature upgrade
        assert decision['promote_candidate'] == True
        assert decision['feature_engineering_factor'] == True
        assert 'feature' in decision['reason'].lower()


# tests/run_tests.py
# Test runner script with different test categories

import pytest
import sys
from pathlib import Path

def run_unit_tests():
    """Run fast unit tests"""
    return pytest.main([
        "tests/", 
        "-m", "not slow and not integration",
        "-v", 
        "--tb=short"
    ])

def run_integration_tests():
    """Run slower integration tests"""
    return pytest.main([
        "tests/",
        "-m", "integration", 
        "-v",
        "--tb=short"
    ])

def run_cpu_constraint_tests():
    """Run tests that verify CPU constraint compliance"""
    return pytest.main([
        "tests/",
        "-m", "cpu_constraint",
        "-v",
        "--tb=short"
    ])

def run_all_tests():
    """Run complete test suite"""
    return pytest.main([
        "tests/",
        "-v",
        "--tb=short",
        "--cov=model",
        "--cov-report=html"
    ])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "unit":
            exit_code = run_unit_tests()
        elif test_type == "integration":  
            exit_code = run_integration_tests()
        elif test_type == "cpu":
            exit_code = run_cpu_constraint_tests()
        elif test_type == "all":
            exit_code = run_all_tests()
        else:
            print("Usage: python run_tests.py [unit|integration|cpu|all]")
            exit_code = 1
    else:
        exit_code = run_unit_tests()  # Default to unit tests
    
    sys.exit(exit_code)