"""
Tests for visualization components and UI functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from shapoint_webapp.visualizations import RiskVisualizer, UIComponents


class TestRiskVisualizer:
    """Test RiskVisualizer functionality."""
    
    def setup_method(self):
        """Set up test configuration and visualizer."""
        self.config = {
            'visualization': {
                'risk_gauge': {
                    'title': 'Test Risk Gauge',
                    'units': '%',
                    'max_value': 100
                },
                'population_chart': {
                    'title': 'Population Comparison',
                    'chart_type': 'histogram'
                }
            }
        }
        self.visualizer = RiskVisualizer(self.config)
    
    def test_create_risk_gauge_low_risk(self):
        """Test risk gauge creation for low risk values."""
        fig = self.visualizer.create_risk_gauge(20.0, 2.5)
        
        # Check that figure is created
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0
        
        # Check gauge properties
        gauge_data = fig.data[0]
        assert gauge_data.value == 20.0
        assert 'Low' in gauge_data.title['text']
    
    def test_create_risk_gauge_medium_risk(self):
        """Test risk gauge creation for medium risk values."""
        fig = self.visualizer.create_risk_gauge(50.0, 5.0)
        
        gauge_data = fig.data[0]
        assert gauge_data.value == 50.0
        assert 'Medium' in gauge_data.title['text']
    
    def test_create_risk_gauge_high_risk(self):
        """Test risk gauge creation for high risk values."""
        fig = self.visualizer.create_risk_gauge(80.0, 8.0)
        
        gauge_data = fig.data[0]
        assert gauge_data.value == 80.0
        assert 'High' in gauge_data.title['text']
    
    def test_create_population_comparison_valid_data(self):
        """Test population comparison with valid data."""
        # Create mock population statistics
        np.random.seed(42)
        population_stats = {
            'distribution': np.random.normal(5, 2, 1000),
            'risk_bins': {
                0: {'bin_center': 1, 'risk_percentage': 10, 'valid': True},
                1: {'bin_center': 3, 'risk_percentage': 25, 'valid': True},
                2: {'bin_center': 5, 'risk_percentage': 50, 'valid': True},
                3: {'bin_center': 7, 'risk_percentage': 75, 'valid': True},
                4: {'bin_center': 9, 'risk_percentage': 90, 'valid': True}
            },
            'bin_edges': np.array([0, 2, 4, 6, 8, 10])
        }
        
        fig = self.visualizer.create_population_comparison(6.5, population_stats)
        
        # Check that figure is created with data
        assert fig is not None
        assert len(fig.data) >= 2  # Should have histogram and line
        assert 'Population Distribution' in fig.layout.title.text
    
    def test_create_population_comparison_missing_data(self):
        """Test population comparison with missing data."""
        fig = self.visualizer.create_population_comparison(5.0, {})
        
        # Should create a placeholder figure
        assert fig is not None
        assert len(fig.layout.annotations) > 0
        assert 'not available' in fig.layout.annotations[0].text
    
    def test_create_feature_importance_plot_valid_data(self):
        """Test feature importance plot with valid model summary."""
        model_summary = {
            'feature_summary': {
                'age': {'score_range': {'min': 0, 'max': 5}},
                'cholesterol': {'score_range': {'min': 0, 'max': 3}},
                'blood_pressure': {'score_range': {'min': 0, 'max': 4}}
            }
        }
        
        fig = self.visualizer.create_feature_importance_plot(model_summary)
        
        # Check that figure is created
        assert fig is not None
        assert len(fig.data) > 0
        
        # Check that features are included
        bar_data = fig.data[0]
        assert len(bar_data.y) == 3  # Should have 3 features
        assert 'age' in bar_data.y
        assert 'cholesterol' in bar_data.y
        assert 'blood_pressure' in bar_data.y
    
    def test_create_feature_importance_plot_missing_data(self):
        """Test feature importance plot with missing model summary."""
        fig = self.visualizer.create_feature_importance_plot({})
        
        # Should create a placeholder figure
        assert fig is not None
        assert len(fig.layout.annotations) > 0
        assert 'not available' in fig.layout.annotations[0].text


class TestUIComponents:
    """Test UI component functionality."""
    
    @patch('streamlit.metric')
    def test_create_metric_card(self, mock_metric):
        """Test metric card creation."""
        UIComponents.create_metric_card("Test Metric", "42", "5", "normal")
        
        # Verify streamlit.metric was called with correct parameters
        mock_metric.assert_called_once_with(
            label="Test Metric",
            value="42",
            delta="5",
            delta_color="normal"
        )
    
    @patch('streamlit.info')
    @patch('streamlit.warning')
    @patch('streamlit.success')
    @patch('streamlit.error')
    def test_create_info_box(self, mock_error, mock_success, mock_warning, mock_info):
        """Test info box creation with different types."""
        # Test info box
        UIComponents.create_info_box("Title", "Content", "info")
        mock_info.assert_called_once()
        
        # Test warning box
        UIComponents.create_info_box("Title", "Content", "warning")
        mock_warning.assert_called_once()
        
        # Test success box
        UIComponents.create_info_box("Title", "Content", "success")
        mock_success.assert_called_once()
        
        # Test error box
        UIComponents.create_info_box("Title", "Content", "error")
        mock_error.assert_called_once()
    
    @patch('streamlit.selectbox')
    def test_create_feature_input_categorical(self, mock_selectbox):
        """Test categorical feature input creation."""
        mock_selectbox.return_value = "Female"
        
        feature_metadata = {
            'label': 'Gender',
            'help': 'Select gender',
            'default_value': 1,
            'options': {
                1: {'label': 'Female', 'range': ''},
                2: {'label': 'Male', 'range': ''}
            }
        }
        
        result = UIComponents.create_feature_input(
            'gender', feature_metadata, 'categorical', 'test_key'
        )
        
        # Check that selectbox was called
        mock_selectbox.assert_called_once()
        
        # Check the call arguments
        call_args = mock_selectbox.call_args
        assert call_args[1]['label'] == 'Gender'
        assert call_args[1]['help'] == 'Select gender'
        assert call_args[1]['key'] == 'test_key'
        assert 'Female' in call_args[1]['options']
        assert 'Male' in call_args[1]['options']
        
        # Should return the key corresponding to the selected label
        assert result == 1
    
    @patch('streamlit.number_input')
    def test_create_feature_input_continuous(self, mock_number_input):
        """Test continuous feature input creation."""
        mock_number_input.return_value = 25.5
        
        feature_metadata = {
            'label': 'Age',
            'help': 'Enter age',
            'default_value': 30.0,
            'min_value': 18.0,
            'max_value': 100.0,
            'step': 1.0
        }
        
        result = UIComponents.create_feature_input(
            'age', feature_metadata, 'continuous', 'test_key'
        )
        
        # Check that number_input was called
        mock_number_input.assert_called_once()
        
        # Check the call arguments
        call_args = mock_number_input.call_args
        assert call_args[1]['label'] == 'Age'
        assert call_args[1]['help'] == 'Enter age'
        assert call_args[1]['key'] == 'test_key'
        assert call_args[1]['min_value'] == 18.0
        assert call_args[1]['max_value'] == 100.0
        assert call_args[1]['value'] == 30.0
        assert call_args[1]['step'] == 1.0
        
        assert result == 25.5
    
    @patch('streamlit.sidebar')
    def test_create_model_info_sidebar(self, mock_sidebar):
        """Test model info sidebar creation."""
        mock_sidebar.markdown = MagicMock()
        mock_sidebar.metric = MagicMock()
        
        model_info = {
            'task': 'Classification',
            'features_used': ['age', 'gender', 'cholesterol'],
            'training_samples': 1000
        }
        
        UIComponents.create_model_info_sidebar(model_info)
        
        # Check that sidebar methods were called
        assert mock_sidebar.markdown.call_count >= 1
        assert mock_sidebar.metric.call_count >= 1
    
    @patch('streamlit.markdown')
    def test_create_risk_interpretation_low_risk(self, mock_markdown):
        """Test risk interpretation for low risk."""
        UIComponents.create_risk_interpretation(20.0, 25.5, 1)
        
        # Check that markdown was called
        mock_markdown.assert_called_once()
        
        # Check that the content includes risk level
        call_args = mock_markdown.call_args[0][0]
        assert 'Low' in call_args
        assert 'green' in call_args
        assert '25.5' in call_args
    
    @patch('streamlit.markdown')
    def test_create_risk_interpretation_high_risk(self, mock_markdown):
        """Test risk interpretation for high risk."""
        UIComponents.create_risk_interpretation(80.0, 85.2, 1)
        
        call_args = mock_markdown.call_args[0][0]
        assert 'High' in call_args
        assert 'red' in call_args
        assert '85.2' in call_args


class TestVisualizationEdgeCases:
    """Test edge cases and error handling in visualizations."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.config = {
            'visualization': {
                'risk_gauge': {'title': 'Test', 'units': '%'},
                'population_chart': {'title': 'Test'}
            }
        }
        self.visualizer = RiskVisualizer(self.config)
    
    def test_risk_gauge_extreme_values(self):
        """Test risk gauge with extreme values."""
        # Test very low risk
        fig_low = self.visualizer.create_risk_gauge(0.0, 0.0)
        assert fig_low.data[0].value == 0.0
        
        # Test very high risk
        fig_high = self.visualizer.create_risk_gauge(100.0, 10.0)
        assert fig_high.data[0].value == 100.0
        
        # Test negative values (should handle gracefully)
        fig_negative = self.visualizer.create_risk_gauge(-5.0, -1.0)
        assert fig_negative is not None
    
    def test_population_comparison_empty_distribution(self):
        """Test population comparison with empty distribution."""
        population_stats = {
            'distribution': np.array([]),
            'risk_bins': {},
            'bin_edges': np.array([])
        }
        
        fig = self.visualizer.create_population_comparison(5.0, population_stats)
        
        # Should handle empty data gracefully
        assert fig is not None
    
    def test_feature_importance_empty_summary(self):
        """Test feature importance with empty or malformed summary."""
        # Test completely empty summary
        fig_empty = self.visualizer.create_feature_importance_plot({})
        assert fig_empty is not None
        
        # Test summary without feature_summary
        fig_no_features = self.visualizer.create_feature_importance_plot({'other_data': 'value'})
        assert fig_no_features is not None
        
        # Test summary with empty feature_summary
        fig_empty_features = self.visualizer.create_feature_importance_plot({'feature_summary': {}})
        assert fig_empty_features is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 