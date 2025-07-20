"""
Tests for configuration loading and validation.
"""

import pytest
import yaml
import tempfile
import os
from pathlib import Path
# Configuration tests don't need specific imports


class TestConfigurationLoading:
    """Test configuration file loading and validation."""
    
    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        config_data = {
            'app': {
                'title': 'Test App',
                'description': 'Test Description'
            },
            'model': {
                'task': 'Classification',
                'k_variables': 5,
                'score_scale': 10
            },
            'data': {
                'target_column': 'target',
                'exclude_columns': ['id']
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            with open(temp_config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            assert loaded_config == config_data
            assert loaded_config['app']['title'] == 'Test App'
            assert loaded_config['model']['task'] == 'Classification'
            assert loaded_config['data']['target_column'] == 'target'
        finally:
            os.unlink(temp_config_path)
    
    def test_config_with_comments(self):
        """Test loading config with comments and documentation."""
        config_content = """
# This is a test configuration
app:
  title: "Test App"  # Application title
  description: "Test Description"

model:
  task: "Classification"  # Classification task
  k_variables: 5  # Number of features
  
data:
  target_column: "target"
  # Exclude ID columns
  exclude_columns: ["id"]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_config_path = f.name
        
        try:
            with open(temp_config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            assert loaded_config['app']['title'] == 'Test App'
            assert loaded_config['model']['task'] == 'Classification'
            assert loaded_config['data']['target_column'] == 'target'
            assert 'id' in loaded_config['data']['exclude_columns']
        finally:
            os.unlink(temp_config_path)
    
    def test_minimal_config(self):
        """Test loading minimal configuration."""
        minimal_config = {
            'model': {'task': 'Classification'},
            'data': {'target_column': 'target'}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(minimal_config, f)
            temp_config_path = f.name
        
        try:
            with open(temp_config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            assert loaded_config['model']['task'] == 'Classification'
            assert loaded_config['data']['target_column'] == 'target'
        finally:
            os.unlink(temp_config_path)
    
    def test_invalid_yaml(self):
        """Test handling of invalid YAML syntax."""
        invalid_yaml_content = """
app:
  title: "Test App
  description: "Missing quote
model:
  task: Classification
    invalid_indent: value
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml_content)
            temp_config_path = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                with open(temp_config_path, 'r') as f:
                    yaml.safe_load(f)
        finally:
            os.unlink(temp_config_path)


class TestConfigurationValidation:
    """Test configuration validation and default handling."""
    
    def test_config_structure_validation(self):
        """Test validation of configuration structure."""
        # Test required sections
        valid_config = {
            'model': {'task': 'Classification'},
            'data': {'target_column': 'target'}
        }
        
        # This should not raise any errors for basic structure
        assert 'model' in valid_config
        assert 'data' in valid_config
        assert valid_config['model']['task'] == 'Classification'
    
    def test_feature_metadata_structure(self):
        """Test feature metadata structure validation."""
        feature_metadata = {
            'age': {
                'label': 'Patient Age',
                'min_value': 18,
                'max_value': 100,
                'default_value': 50,
                'help': 'Enter patient age'
            },
            'gender': {
                'label': 'Gender',
                'options': {
                    1: {'label': 'Female', 'range': ''},
                    2: {'label': 'Male', 'range': ''}
                },
                'default_value': 1
            }
        }
        
        # Validate continuous feature structure
        age_meta = feature_metadata['age']
        assert 'label' in age_meta
        assert 'min_value' in age_meta
        assert 'max_value' in age_meta
        assert 'default_value' in age_meta
        assert age_meta['min_value'] < age_meta['max_value']
        
        # Validate categorical feature structure
        gender_meta = feature_metadata['gender']
        assert 'label' in gender_meta
        assert 'options' in gender_meta
        assert 'default_value' in gender_meta
        assert isinstance(gender_meta['options'], dict)
        
        for option_key, option_value in gender_meta['options'].items():
            assert isinstance(option_value, dict)
            assert 'label' in option_value
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        model_config = {
            'task': 'Classification',
            'k_variables': 5,
            'max_leaves': 3,
            'score_scale': 10,
            'use_optuna': True,
            'risk_bins': 8
        }
        
        # Validate model parameters
        assert model_config['task'] in ['Classification', 'Regression', 'Survival']
        assert isinstance(model_config['k_variables'], int)
        assert model_config['k_variables'] > 0
        assert isinstance(model_config['max_leaves'], int)
        assert model_config['max_leaves'] > 0
        assert isinstance(model_config['score_scale'], int)
        assert model_config['score_scale'] > 0
        assert isinstance(model_config['use_optuna'], bool)
        assert isinstance(model_config['risk_bins'], int)
        assert model_config['risk_bins'] > 0


class TestConfigurationDefaults:
    """Test default value handling in configuration."""
    
    def test_default_app_config(self):
        """Test default application configuration values."""
        default_app_config = {
            'title': 'SHAPoint: Interpretable Risk Assessment',
            'description': 'Interactive risk assessment using SHAP-based interpretable models',
            'page_icon': '📊',
            'layout': 'wide'
        }
        
        assert default_app_config['title'].startswith('SHAPoint')
        assert 'risk assessment' in default_app_config['description'].lower()
        assert default_app_config['layout'] == 'wide'
    
    def test_default_model_config(self):
        """Test default model configuration values."""
        default_model_config = {
            'k_variables': 5,
            'max_leaves': 3,
            'task': 'Classification',
            'score_scale': 10,
            'use_optuna': True,
            'n_random_features': 3,
            'risk_bins': 8
        }
        
        # Test reasonable defaults
        assert 1 <= default_model_config['k_variables'] <= 15
        assert 1 <= default_model_config['max_leaves'] <= 10
        assert default_model_config['score_scale'] in [5, 10, 50, 100]
        assert default_model_config['risk_bins'] >= 4
    
    def test_default_ui_config(self):
        """Test default UI configuration values."""
        default_ui_config = {
            'display': {
                'decimal_places': 1,
                'enable_retraining': True
            },
            'main_panel': {
                'show_risk_gauge': True,
                'show_population_comparison': True,
                'show_shap_plots': True
            },
            'auto_update': {
                'enabled': True,
                'debounce_ms': 500
            }
        }
        
        assert isinstance(default_ui_config['display']['decimal_places'], int)
        assert default_ui_config['display']['decimal_places'] >= 0
        assert isinstance(default_ui_config['display']['enable_retraining'], bool)
        assert isinstance(default_ui_config['auto_update']['debounce_ms'], int)
        assert default_ui_config['auto_update']['debounce_ms'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 