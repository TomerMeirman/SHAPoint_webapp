"""Integration tests for app functionality."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from shapoint_webapp.model_manager import ModelManager


class TestFeatureTypeDetection:
    """Test feature type detection."""
    
    def test_mixed_data_types(self):
        """Test feature type detection with mixed data types."""
        config = {
            'model': {'task': 'Classification'},
            'data': {'target_column': 'target', 'feature_types': {}, 'feature_metadata': {}}
        }
        manager = ModelManager(config)
        
        # Create test data
        mixed_data = pd.DataFrame({
            'continuous_var': np.random.normal(100, 20, 50),
            'categorical_var': np.random.choice([0, 1], 50),
            'target': np.random.choice([0, 1], 50)
        })
        
        X = mixed_data.drop('target', axis=1)
        manager.X_train = X
        manager.feature_names = X.columns.tolist()
        manager._detect_feature_types(X)
        
        # Verify type detection
        assert manager.feature_types['continuous_var'] == 'continuous'
        assert manager.feature_types['categorical_var'] == 'categorical'


class TestRuleEvaluation:
    """Test rule evaluation."""
    
    def test_rule_scoring(self):
        """Test rule-based scoring."""
        config = {'model': {}, 'data': {}}
        manager = ModelManager(config)
        
        # Mock model
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
        
        # Test rule evaluation
        assert manager.get_feature_rule_score('age', 60) == 3.0
        assert manager.get_feature_rule_score('age', 30) == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 