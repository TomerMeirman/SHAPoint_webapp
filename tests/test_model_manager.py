"""
Comprehensive tests for ModelManager functionality.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the components to test
from shapoint_webapp.model_manager import ModelManager


class TestModelManagerInit:
    """Test ModelManager initialization and configuration."""
    
    def test_init_with_valid_config(self):
        """Test ModelManager initializes with valid configuration."""
        config = {
            'model': {
                'task': 'Classification',
                'k_variables': 5,
                'max_leaves': 3,
                'score_scale': 10,
                'use_optuna': False,
                'n_random_features': 3,
                'params_path': '',
                'model_path': '',
                'risk_bins': 8
            },
            'data': {
                'default_dataset': 'cardiovascular',
                'target_column': 'cardio',
                'exclude_columns': ['id'],
                'feature_types': {},
                'feature_metadata': {}
            }
        }
        manager = ModelManager(config)
        
        assert manager.config == config
        assert manager.model is None
        assert manager.feature_names is None
        assert manager.feature_types == {}
        assert manager.population_stats == {}
    
    def test_init_with_minimal_config(self):
        """Test ModelManager works with minimal configuration."""
        config = {
            'model': {'task': 'Classification'},
            'data': {'target_column': 'target'}
        }
        manager = ModelManager(config)
        assert manager.config == config
    
    def _get_base_config(self):
        """Get a standard configuration for testing."""
        return {
            'model': {
                'task': 'Classification',
                'k_variables': 5,
                'max_leaves': 3,
                'score_scale': 10,
                'use_optuna': False,
                'n_random_features': 3,
                'params_path': '',
                'model_path': '',
                'risk_bins': 8
            },
            'data': {
                'default_dataset': 'cardiovascular',
                'data_path': '../SHAPoint/examples/data/cardio_train.csv',
                'data_separator': ';',
                'target_column': 'cardio',
                'exclude_columns': ['id'],
                'feature_types': {},
                'feature_metadata': {}
            }
        }


class TestFeatureDetection:
    """Test automatic feature detection functionality."""
    
    def setup_method(self):
        """Set up test data for each test."""
        self.config = {
            'model': {'task': 'Classification', 'k_variables': 5},
            'data': {'target_column': 'target', 'feature_types': {}, 'feature_metadata': {}}
        }
        self.manager = ModelManager(self.config)
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        self.sample_data = pd.DataFrame({
            'age': np.random.normal(50, 15, n_samples),
            'gender': np.random.choice([1, 2], n_samples),
            'blood_pressure': np.random.normal(120, 20, n_samples),
            'cholesterol': np.random.choice([1, 2, 3], n_samples),
            'smoker': np.random.choice([0, 1], n_samples),
            'target': np.random.choice([0, 1], n_samples)
        })
        
        # Set up the manager with this data
        self.manager.X_train = self.sample_data.drop('target', axis=1)
        self.manager.y_train = self.sample_data['target']
        self.manager.feature_names = self.manager.X_train.columns.tolist()
        self.manager._detect_feature_types(self.manager.X_train)
    
    def test_detect_feature_types(self):
        """Test automatic feature type detection."""
        assert self.manager.feature_types['age'] == 'continuous'
        assert self.manager.feature_types['gender'] == 'categorical'
        assert self.manager.feature_types['blood_pressure'] == 'continuous'
        assert self.manager.feature_types['cholesterol'] == 'categorical'
        assert self.manager.feature_types['smoker'] == 'categorical'
    
    def test_auto_detect_continuous_metadata(self):
        """Test auto-detection of continuous feature metadata."""
        metadata = self.manager._auto_detect_continuous_metadata('age')
        
        assert 'min_value' in metadata
        assert 'max_value' in metadata
        assert 'default_value' in metadata
        assert 'step' in metadata
        
        # Check values are reasonable
        assert metadata['min_value'] < metadata['default_value'] < metadata['max_value']
        assert metadata['step'] > 0
    
    def test_auto_detect_categorical_metadata(self):
        """Test auto-detection of categorical feature metadata."""
        metadata = self.manager._auto_detect_categorical_metadata('gender')
        
        assert 'options' in metadata
        assert 'default_value' in metadata
        
        # Check options structure
        options = metadata['options']
        assert isinstance(options, dict)
        assert 1 in options
        assert 2 in options
        assert options[1]['label'] == 'Female'
        assert options[2]['label'] == 'Male'
    
    def test_generate_categorical_labels(self):
        """Test intelligent categorical label generation."""
        # Test known mappings
        assert self.manager._generate_categorical_label('gender', 1) == 'Female'
        assert self.manager._generate_categorical_label('gender', 2) == 'Male'
        assert self.manager._generate_categorical_label('smoke', 0) == 'No'
        assert self.manager._generate_categorical_label('smoke', 1) == 'Yes'
        
        # Test unknown feature
        assert self.manager._generate_categorical_label('unknown_feature', 5) == '5'
    
    def test_get_feature_metadata_with_override(self):
        """Test feature metadata with user overrides."""
        # Set up override in config
        self.manager.config['data']['feature_metadata'] = {
            'age': {
                'label': 'Patient Age',
                'min_value': 18,
                'max_value': 100
            }
        }
        
        metadata = self.manager.get_feature_metadata('age')
        
        # Check override values are used
        assert metadata['label'] == 'Patient Age'
        assert metadata['min_value'] == 18
        assert metadata['max_value'] == 100
        
        # Check auto-detected values are still present
        assert 'default_value' in metadata
        assert 'step' in metadata


class TestRuleEvaluation:
    """Test rule evaluation functionality."""
    
    def setup_method(self):
        """Set up manager for rule evaluation tests."""
        self.config = {'model': {}, 'data': {}}
        self.manager = ModelManager(self.config)
    
    def test_evaluate_numeric_rules(self):
        """Test evaluation of numeric comparison rules."""
        # Test >= rule
        assert self.manager._evaluate_rule('age >= 50', 60) == True
        assert self.manager._evaluate_rule('age >= 50', 40) == False
        
        # Test <= rule
        assert self.manager._evaluate_rule('age <= 30', 25) == True
        assert self.manager._evaluate_rule('age <= 30', 35) == False
        
        # Test > rule
        assert self.manager._evaluate_rule('score > 5', 6) == True
        assert self.manager._evaluate_rule('score > 5', 5) == False
        
        # Test < rule
        assert self.manager._evaluate_rule('value < 100', 90) == True
        assert self.manager._evaluate_rule('value < 100', 100) == False
        
        # Test == rule
        assert self.manager._evaluate_rule('category == 2', 2) == True
        assert self.manager._evaluate_rule('category == 2', 3) == False
    
    def test_evaluate_range_rules(self):
        """Test evaluation of range rules."""
        assert self.manager._evaluate_rule('10 < age <= 20', 15) == True
        assert self.manager._evaluate_rule('10 < age <= 20', 10) == False
        assert self.manager._evaluate_rule('10 < age <= 20', 20) == True
        assert self.manager._evaluate_rule('10 < age <= 20', 25) == False
    
    def test_evaluate_categorical_rules(self):
        """Test evaluation of categorical rules."""
        assert self.manager._evaluate_rule('gender = 1', 1) == True
        assert self.manager._evaluate_rule('gender = 1', 2) == False
        
        # Test string matching
        assert self.manager._evaluate_rule('status = active', 'active') == True
        assert self.manager._evaluate_rule('status = active', 'inactive') == False
    
    def test_evaluate_invalid_rules(self):
        """Test handling of invalid or malformed rules."""
        # Empty rule
        assert self.manager._evaluate_rule('', 10) == False
        
        # Malformed rule
        assert self.manager._evaluate_rule('invalid rule format', 10) == False
        
        # Rule with invalid syntax
        assert self.manager._evaluate_rule('age ><= 30', 25) == False


class TestModelOperations:
    """Test model training, loading, and prediction operations."""
    
    def setup_method(self):
        """Set up test environment."""
        self.config = {
            'model': {
                'task': 'Classification',
                'k_variables': 3,
                'max_leaves': 2,
                'score_scale': 10,
                'use_optuna': False,
                'n_random_features': 0,
                'params_path': '',
                'model_path': '',
                'risk_bins': 5
            },
            'data': {
                'default_dataset': 'breast_cancer',
                'target_column': 'target',
                'exclude_columns': [],
                'feature_types': {},
                'feature_metadata': {}
            }
        }
    
    @patch('shapoint_webapp.model_manager.SHAPointModel')
    def test_train_new_model(self, mock_shapoint_model):
        """Test training a new model."""
        # Mock the SHAPoint model
        mock_model_instance = MagicMock()
        mock_shapoint_model.return_value = mock_model_instance
        
        manager = ModelManager(self.config)
        
        # Mock data loading
        with patch.object(manager, '_load_data') as mock_load_data:
            # Create sample data with enough samples for stratified split
            X = pd.DataFrame({
                'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'feature3': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            })
            y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Balanced classes
            mock_load_data.return_value = (X, y)
            
            # Train model
            with patch('streamlit.spinner'), patch('streamlit.empty'), patch('threading.Thread'):
                result_model = manager._train_new_model()
            
            # Verify model was created and trained
            mock_shapoint_model.assert_called_once()
            mock_model_instance.fit.assert_called_once()
            assert result_model == mock_model_instance
            assert manager.feature_names == ['feature1', 'feature2', 'feature3']
    
    def test_get_used_features(self):
        """Test getting features used by the model."""
        manager = ModelManager(self.config)
        
        # Test with no model
        assert manager.get_used_features() == []
        
        # Test with mock model that has top_features
        mock_model = MagicMock()
        mock_model.top_features = ['feature1', 'feature2', 'feature3']
        manager.model = mock_model
        
        assert manager.get_used_features() == ['feature1', 'feature2', 'feature3']
        
        # Test fallback to feature_names
        manager.model = MagicMock()
        manager.model.top_features = None
        manager.model.feature_names_ = None  # Explicitly set to None
        manager.feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']
        manager.config['model']['k_variables'] = 3
        
        result = manager.get_used_features()
        assert len(result) == 3
        assert result == ['f1', 'f2', 'f3']
    
    @patch('streamlit.warning')
    def test_get_feature_rule_score(self, mock_warning):
        """Test getting rule-based scores for features."""
        manager = ModelManager(self.config)
        
        # Test with no model
        assert manager.get_feature_rule_score('age', 50) is None
        
        # Test with mock model
        mock_model = MagicMock()
        mock_model.get_model_summary_with_nulls.return_value = {
            'feature_summary': {
                'age': {
                    'levels_detail': [
                        {'rule': 'age >= 50', 'scaled_score': 3.0},
                        {'rule': 'age < 50', 'scaled_score': 1.0}
                    ]
                }
            }
        }
        manager.model = mock_model
        
        # Test rule matching
        assert manager.get_feature_rule_score('age', 60) == 3.0
        assert manager.get_feature_rule_score('age', 30) == 1.0
        
        # Test feature not found
        assert manager.get_feature_rule_score('unknown_feature', 10) is None


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_config_handling(self):
        """Test handling of invalid configurations."""
        # Test empty config
        with pytest.raises(KeyError):
            manager = ModelManager({})
            manager.config['model']['task']  # Should raise KeyError
    
    def test_missing_data_handling(self):
        """Test handling when data files are missing."""
        config = {
            'model': {'task': 'Classification'},
            'data': {
                'default_dataset': 'custom',
                'data_path': '/nonexistent/path.csv',
                'target_column': 'target'
            }
        }
        manager = ModelManager(config)
        
        # This should handle the missing file gracefully
        # (The actual implementation may vary based on your error handling)
        try:
            X, y = manager._load_data()
        except Exception as e:
            # Should handle file not found errors gracefully
            assert isinstance(e, (FileNotFoundError, ValueError))


class TestDataTypes:
    """Test handling of different data types and edge cases."""
    
    def setup_method(self):
        """Set up test data with various data types."""
        self.config = {
            'model': {'task': 'Classification'},
            'data': {'target_column': 'target', 'feature_types': {}, 'feature_metadata': {}}
        }
        self.manager = ModelManager(self.config)
    
    def test_mixed_data_types(self):
        """Test handling of mixed data types."""
        # Create data with various types
        mixed_data = pd.DataFrame({
            'int_feature': [25, 30, 35, 40, 45],  # Ages, clearly continuous
            'float_feature': [1.1, 2.2, 3.3, 4.4, 5.5],
            'string_category': ['A', 'B', 'A', 'C', 'B'],
            'boolean_feature': [True, False, True, False, True],
            'target': [0, 1, 0, 1, 0]
        })
        
        X = mixed_data.drop('target', axis=1)
        self.manager.X_train = X
        self.manager.feature_names = X.columns.tolist()
        self.manager._detect_feature_types(X)
        
        # Check type detection
        assert self.manager.feature_types['int_feature'] == 'continuous'
        assert self.manager.feature_types['float_feature'] == 'continuous'
        assert self.manager.feature_types['string_category'] == 'categorical'
        assert self.manager.feature_types['boolean_feature'] == 'categorical'
    
    def test_extreme_values(self):
        """Test handling of extreme values in data."""
        extreme_data = pd.DataFrame({
            'small_values': [0.001, 0.002, 0.003],
            'large_values': [1e6, 2e6, 3e6],
            'negative_values': [-100, -50, -10],
            'zero_values': [0, 0, 0],
            'target': [0, 1, 0]
        })
        
        X = extreme_data.drop('target', axis=1)
        self.manager.X_train = X
        self.manager.feature_names = X.columns.tolist()
        self.manager._detect_feature_types(X)
        
        # Test metadata generation doesn't crash
        for feature in X.columns:
            metadata = self.manager._auto_detect_continuous_metadata(feature)
            assert 'min_value' in metadata
            assert 'max_value' in metadata
            assert 'default_value' in metadata
    
    def test_single_value_features(self):
        """Test handling of features with single unique values."""
        constant_data = pd.DataFrame({
            'constant_feature': [1, 1, 1, 1, 1],
            'normal_feature': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
        
        X = constant_data.drop('target', axis=1)
        self.manager.X_train = X
        self.manager.feature_names = X.columns.tolist()
        self.manager._detect_feature_types(X)
        
        # Should handle constant features gracefully
        metadata = self.manager._auto_detect_continuous_metadata('constant_feature')
        assert metadata['min_value'] <= metadata['max_value']


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 