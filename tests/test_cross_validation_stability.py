import pytest
import numpy as np
import json
from pathlib import Path
from model.train import ModelTrainer

def test_model_performance_thresholds():
    """Test that model meets minimum performance requirements"""
    trainer = ModelTrainer()
    
    # Load test results
    metadata = trainer.load_metadata()
    
    assert metadata['test_f1'] >= 0.75, "F1 score below threshold"
    assert metadata['test_accuracy'] >= 0.75, "Accuracy below threshold"

def test_cross_validation_stability():
    """Test CV results show acceptable stability (low variance)"""
    # Load CV results
    try:
        from path_config import path_manager
        cv_results_path = path_manager.get_logs_path("cv_results.json")
        
        if not cv_results_path.exists():
            pytest.skip("No CV results available")
            
        with open(cv_results_path, 'r') as f:
            cv_data = json.load(f)
        
        # Test CV stability - standard deviation should be reasonable
        test_scores = cv_data.get('test_scores', {})
        
        if 'f1' in test_scores:
            f1_std = test_scores['f1'].get('std', 0)
            f1_mean = test_scores['f1'].get('mean', 0)
            
            # CV coefficient of variation should be < 0.15 (15%)
            cv_coefficient = f1_std / f1_mean if f1_mean > 0 else 1
            assert cv_coefficient < 0.15, f"CV results too unstable: CV={cv_coefficient:.3f}"
            
        if 'accuracy' in test_scores:
            acc_std = test_scores['accuracy'].get('std', 0)
            acc_mean = test_scores['accuracy'].get('mean', 0)
            
            cv_coefficient = acc_std / acc_mean if acc_mean > 0 else 1
            assert cv_coefficient < 0.15, f"Accuracy CV too unstable: CV={cv_coefficient:.3f}"
            
    except FileNotFoundError:
        pytest.skip("CV results file not found")

def test_model_overfitting_indicators():
    """Test that model doesn't show signs of severe overfitting"""
    try:
        from path_config import path_manager
        cv_results_path = path_manager.get_logs_path("cv_results.json")
        
        if not cv_results_path.exists():
            pytest.skip("No CV results available")
            
        with open(cv_results_path, 'r') as f:
            cv_data = json.load(f)
        
        # Check overfitting score if available
        perf_indicators = cv_data.get('performance_indicators', {})
        overfitting_score = perf_indicators.get('overfitting_score')
        
        if overfitting_score is not None and overfitting_score != 'Unknown':
            # Overfitting score should be reasonable (< 0.1 difference)
            assert overfitting_score < 0.1, f"Model shows overfitting: {overfitting_score}"
            
    except (FileNotFoundError, KeyError):
        pytest.skip("Performance indicators not available")