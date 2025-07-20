"""
Tests for null safety issues that were encountered during development.
These tests ensure that the application handles None/empty config values gracefully.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from shapoint_webapp.model_manager import ModelManager
from shapoint_webapp.visualizations import create_feature_explanation_table


class TestNullSafetyIssues:
    """Test cases for null safety issues that caused AttributeError: 'NoneType' object has no attribute 'get'"""
    
    def test_empty_feature_types_config(self):
        """Test that empty feature_types in config doesn't crash the app."""
        # This was the original issue: feature_types: (empty) in YAML became None
        config = {
            'model': {'k_variables': 3, 'max_leaves': 3, 'task': 'Classification'},
            'data': {
                'default_dataset': 'breast_cancer',
                'target_column': 'target',
                'exclude_columns': [],
                'feature_types': None  # This was causing the crash
            }
        }
        
        # Should not raise an exception
        model_manager = ModelManager(config)
        
        # Create sample data to test _detect_feature_types
        sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 1, 2],
            'feature2': [10.5, 20.1, 30.2, 15.3, 25.4],
            'feature3': [0, 1, 0, 1, 1]
        })
        
        # This should not crash
        model_manager._detect_feature_types(sample_data)
        
        # Verify feature types were auto-detected
        assert len(model_manager.feature_types) == 3
        assert all(ft in ['categorical', 'continuous'] for ft in model_manager.feature_types.values())
    
    def test_empty_feature_metadata_config(self):
        """Test that empty feature_metadata in config doesn't crash get_feature_metadata."""
        config = {
            'model': {'k_variables': 3},
            'data': {
                'feature_metadata': None  # This was causing the second crash
            }
        }
        
        model_manager = ModelManager(config)
        
        # This should not crash
        metadata = model_manager.get_feature_metadata('test_feature')
        
        # Should return auto-detected metadata
        assert isinstance(metadata, dict)
        assert 'label' in metadata
        assert 'help' in metadata
    
    def test_missing_data_section_in_config(self):
        """Test that missing 'data' section doesn't crash the app."""
        config = {
            'model': {'k_variables': 3}
            # Missing 'data' section entirely
        }
        
        model_manager = ModelManager(config)
        
        # These should not crash and should use fallback values
        metadata = model_manager.get_feature_metadata('test_feature')
        assert isinstance(metadata, dict)
        
        # Auto-detect on empty dataframe should not crash
        empty_df = pd.DataFrame()
        model_manager._detect_feature_types(empty_df)
    
    def test_none_used_features_handling(self):
        """Test that None used_features doesn't cause iteration errors."""
        config = {
            'model': {'k_variables': 3},
            'data': {'default_dataset': 'breast_cancer'}
        }
        
        model_manager = ModelManager(config)
        
        # Mock the model to return None for used features
        with patch.object(model_manager, 'get_used_features', return_value=None):
            # Create a sample dataframe
            df = pd.DataFrame({'feature1': [1], 'feature2': [2]})
            
            # This should not crash with "NoneType is not iterable"
            contributions = model_manager._get_feature_contributions(df)
            
            # Should return a list (possibly empty)
            assert isinstance(contributions, list)
    
    def test_empty_model_summary_handling(self):
        """Test that empty or None model_summary doesn't crash explanation table."""
        # Test with None
        result1 = create_feature_explanation_table(None)
        assert isinstance(result1, pd.DataFrame)
        assert not result1.empty  # Should have fallback data
        
        # Test with empty dict
        result2 = create_feature_explanation_table({})
        assert isinstance(result2, pd.DataFrame)
        assert not result2.empty  # Should have fallback data
        
        # Test with wrong structure
        result3 = create_feature_explanation_table({'unexpected': 'structure'})
        assert isinstance(result3, pd.DataFrame)
        assert not result3.empty  # Should have fallback data
    
    def test_config_access_patterns(self):
        """Test safe config access patterns that prevent 'NoneType' has no attribute 'get' errors."""
        # Test various problematic config structures
        test_configs = [
            {},  # Empty config
            {'data': None},  # data is None
            {'data': {}},  # data is empty
            {'data': {'feature_types': None}},  # feature_types is None
            {'data': {'feature_metadata': None}},  # feature_metadata is None
        ]
        
        for config in test_configs:
            model_manager = ModelManager(config)
            
            # These should all work without crashing
            metadata = model_manager.get_feature_metadata('test_feature')
            assert isinstance(metadata, dict)
            
            # Safe config access (demonstrating the proper pattern)
            data_config = config.get('data', {}) if config else {}
            if data_config is None:
                data_config = {}
                
            feature_types = data_config.get('feature_types', {}) if data_config else {}
            if feature_types is None:
                feature_types = {}
            assert isinstance(feature_types, dict)
            
            feature_metadata = data_config.get('feature_metadata', {}) if data_config else {}
            if feature_metadata is None:
                feature_metadata = {}
            assert isinstance(feature_metadata, dict)


class TestRobustErrorHandling:
    """Test that the app handles various error conditions gracefully."""
    
    def test_malformed_yaml_sections(self):
        """Test handling of malformed YAML sections that become None."""
        config = {
            'model': {'k_variables': 3},
            'data': {
                'feature_types': None,
                'feature_metadata': None,
                'default_dataset': 'breast_cancer'
            },
            'ui': None,  # This could also be None
            'visualization': None
        }
        
        model_manager = ModelManager(config)
        
        # Should handle None sections gracefully
        assert model_manager.config.get('ui') is None
        assert model_manager.config.get('visualization') is None
        
        # Should still work with fallbacks
        metadata = model_manager.get_feature_metadata('test_feature')
        assert isinstance(metadata, dict)
    
    def test_feature_contributions_with_empty_data(self):
        """Test feature contributions with various edge cases."""
        config = {
            'model': {'k_variables': 1},
            'data': {'default_dataset': 'breast_cancer'}
        }
        
        model_manager = ModelManager(config)
        
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        contributions = model_manager._get_feature_contributions(empty_df)
        assert isinstance(contributions, list)
        
        # Test with dataframe but no model
        df_with_data = pd.DataFrame({'feature1': [1, 2, 3]})
        contributions = model_manager._get_feature_contributions(df_with_data)
        assert isinstance(contributions, list)
    
    def test_default_value_handling(self):
        """Test default value handling with various config states."""
        configs = [
            {},  # Empty
            {'data': None},  # None data
            {'data': {}},  # Empty data
            {'data': {'feature_metadata': None}},  # None metadata
        ]
        
        for config in configs:
            model_manager = ModelManager(config)
            
            # Should not crash and should return reasonable defaults
            default_val = model_manager._get_default_value('test_feature')
            assert default_val is not None
            assert isinstance(default_val, (int, float, str))


if __name__ == "__main__":
    pytest.main([__file__]) 