"""
SHAPoint WebApp - Main Application
Interactive risk assessment using SHAP-based interpretable models.
"""

import streamlit as st
import yaml
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Any

# Import custom modules
try:
    # Try relative imports first (when running as installed package)
    from .model_manager import ModelManager
    from .visualizations import RiskVisualizer, UIComponents, create_feature_explanation_table
except ImportError:
    # Fall back to absolute imports (when running directly)
    from model_manager import ModelManager
    from visualizations import RiskVisualizer, UIComponents, create_feature_explanation_table


# Page configuration
st.set_page_config(
    page_title="SHAPoint: Interpretable Risk Assessment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_config() -> Dict:
    """Load configuration from YAML file."""
    # Check for config path from environment variable (for CLI usage)
    import os
    config_file = os.environ.get("SHAPOINT_CONFIG", "config.yaml")
    config_path = Path(config_file)
    
    # If relative path, make it relative to the app directory
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if config is None:
                st.error(f"Configuration file is empty or invalid: {config_path}")
                st.stop()
            return config
    except FileNotFoundError:
        st.error(f"Configuration file not found: {config_path}")
        st.stop()
    except yaml.YAMLError as e:
        st.error(f"Error parsing configuration file: {e}")
        st.stop()

@st.cache_resource
def initialize_model_manager(config: Dict) -> ModelManager:
    """Initialize and cache the model manager."""
    return ModelManager(config)

@st.cache_data
def get_model_and_setup(_model_manager: ModelManager):
    """Load or train model and get feature information."""
    # Debug: Check if model manager is valid
    if _model_manager is None:
        st.error("Model manager passed to get_model_and_setup is None")
        return None
    
    try:
        # Debug: Check ModelManager configuration
        config = getattr(_model_manager, 'config', None)
        if config is None:
            st.error("ModelManager.config is None")
            return None
        
        # Load or train the model with specific error handling
        try:
            model = _model_manager.load_or_train_model()
            if model is None:
                st.error("Model loading returned None")
                return None
        except Exception as e:
            st.error(f"Error in load_or_train_model: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None
        
        # Get all required information with error checking
        feature_names = getattr(_model_manager, 'feature_names', None)
        feature_types = getattr(_model_manager, 'feature_types', None)
        population_stats = getattr(_model_manager, 'population_stats', None)
        
        try:
            model_info = _model_manager.get_model_info()
        except Exception as e:
            st.error(f"Error getting model info: {e}")
            import traceback
            st.code(traceback.format_exc())
            model_info = {"error": str(e)}
        
        return {
            'feature_names': feature_names,
            'feature_types': feature_types,
            'population_stats': population_stats,
            'model_info': model_info
        }
    except Exception as e:
        st.error(f"Error in get_model_and_setup: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

def create_input_form(model_manager: ModelManager, feature_names: list, feature_types: dict) -> Dict:
    """Create the input form for features."""
    feature_values = {}
    
    st.sidebar.markdown("### 📝 Patient Information")
    st.sidebar.markdown("Enter the patient's information below. The risk calculation updates automatically.")
    
    # Group features for better organization
    basic_info = ['age', 'gender', 'height', 'weight']
    vital_signs = ['ap_hi', 'ap_lo']
    lab_values = ['cholesterol', 'gluc']
    lifestyle = ['smoke', 'alco', 'active']
    
    # Basic Information
    if any(f in feature_names for f in basic_info):
        st.sidebar.markdown("**👤 Basic Information**")
        for feature in basic_info:
            if feature in feature_names:
                metadata = model_manager.get_feature_metadata(feature)
                feature_type = feature_types.get(feature, 'continuous')
                value = UIComponents.create_feature_input(
                    feature, metadata, feature_type, key=f"input_{feature}"
                )
                feature_values[feature] = value
    
    # Vital Signs
    if any(f in feature_names for f in vital_signs):
        st.sidebar.markdown("**🩺 Vital Signs**")
        for feature in vital_signs:
            if feature in feature_names:
                metadata = model_manager.get_feature_metadata(feature)
                feature_type = feature_types.get(feature, 'continuous')
                value = UIComponents.create_feature_input(
                    feature, metadata, feature_type, key=f"input_{feature}"
                )
                feature_values[feature] = value
    
    # Laboratory Values
    if any(f in feature_names for f in lab_values):
        st.sidebar.markdown("**🧪 Laboratory Values**")
        for feature in lab_values:
            if feature in feature_names:
                metadata = model_manager.get_feature_metadata(feature)
                feature_type = feature_types.get(feature, 'continuous')
                value = UIComponents.create_feature_input(
                    feature, metadata, feature_type, key=f"input_{feature}"
                )
                feature_values[feature] = value
    
    # Lifestyle Factors
    if any(f in feature_names for f in lifestyle):
        st.sidebar.markdown("**🏃‍♂️ Lifestyle Factors**")
        for feature in lifestyle:
            if feature in feature_names:
                metadata = model_manager.get_feature_metadata(feature)
                feature_type = feature_types.get(feature, 'continuous')
                value = UIComponents.create_feature_input(
                    feature, metadata, feature_type, key=f"input_{feature}"
                )
                feature_values[feature] = value
    
    # Handle any remaining features
    remaining_features = [f for f in feature_names if f not in basic_info + vital_signs + lab_values + lifestyle]
    if remaining_features:
        st.sidebar.markdown("**📊 Additional Features**")
        for feature in remaining_features:
            metadata = model_manager.get_feature_metadata(feature)
            feature_type = feature_types.get(feature, 'continuous')
            value = UIComponents.create_feature_input(
                feature, metadata, feature_type, key=f"input_{feature}"
            )
            feature_values[feature] = value
    
    return feature_values

def main():
    """Main application function."""
    # Load configuration
    config = load_config()
    
    # App title and description
    st.title(config.get('app', {}).get('title', 'SHAPoint: Interpretable Risk Assessment'))
    st.markdown(config.get('app', {}).get('description', 'Interactive risk assessment using SHAP-based interpretable models'))
    
    # Add cache clearing button in the top right corner
    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        if st.button("🧹 Clear Cache", help="Clear all caches and reload"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.session_state.clear()
            st.success("Cache cleared!")
            st.rerun()
    with col3:
        if st.button("🔄 Refresh", help="Reload the page"):
            st.rerun()
    
    # Initialize model manager
    model_manager = initialize_model_manager(config)
    
    # Store default bins in session state for dynamic adjustment later
    default_bins_cfg = config.get('model', {}).get('risk_bins', 8)
    if 'risk_bins' not in st.session_state:
        st.session_state['risk_bins'] = default_bins_cfg

    # Load model and setup
    try:
        # Debug: Check if model_manager is valid
        if model_manager is None:
            st.error("Model manager is None - configuration issue")
            st.stop()
        
        # Debug: Get model setup with detailed error handling
        model_setup = get_model_and_setup(model_manager)
        if model_setup is None:
            st.error("Model setup returned None")
            st.stop()
        
        # Debug: Check each component of model_setup
        all_feature_names = model_setup.get('feature_names')
        if all_feature_names is None:
            st.error("Feature names is None")
            st.stop()
            
        feature_types = model_setup.get('feature_types')
        if feature_types is None:
            st.error("Feature types is None")
            st.stop()
            
        model_info = model_setup.get('model_info')
        if model_info is None:
            st.error("Model info is None")
            st.stop()
        
        # Debug: Check population stats
        population_stats = getattr(model_manager, 'population_stats', None)
        if population_stats is None:
            st.error("Population stats is None")
            st.stop()
        
        # Get only the features actually used by the model
        try:
            used_features = model_manager.get_used_features()
            feature_names = used_features if used_features else all_feature_names
        except AttributeError:
            # Fallback if method doesn't exist (cached version issue)
            st.warning("🔄 Clearing cache to load new features. Please refresh the page.")
            st.cache_resource.clear()
            feature_names = all_feature_names
        
        # Allow user to adjust number of risk bins on-the-fly (does NOT retrain the model)
        with st.sidebar.expander("⚙️ Risk Calculation Settings", expanded=False):
            selected_bins = st.slider(
                "Number of Risk Bins",
                min_value=4,
                max_value=20,
                value=st.session_state['risk_bins'],
                help="Change the granularity of risk calculation without retraining the model"
            )
            if selected_bins != st.session_state['risk_bins']:
                st.session_state['risk_bins'] = selected_bins
                model_manager.recalculate_risk_bins(selected_bins)
                st.cache_data.clear()
                st.rerun()
        
        # Get decimal places configuration
        decimal_places = config.get('ui', {}).get('display', {}).get('decimal_places', 1)
        
    except Exception as e:
        import traceback
        st.error(f"Error loading model: {e}")
        st.error("Full traceback:")
        st.code(traceback.format_exc())
        st.stop()
    
    # Initialize visualizer
    visualizer = RiskVisualizer(config)
    
    # Create sidebar with model info
    UIComponents.create_model_info_sidebar(model_info)
    
    # Add retraining interface if enabled
    if config.get('ui', {}).get('display', {}).get('enable_retraining', False):
        with st.sidebar.expander("🔄 Retrain Model", expanded=False):
            st.markdown("**Quick Retrain with Different Settings**")
            
            # Max features selection
            current_k = len(feature_names)
            max_available = len(all_feature_names)
            
            new_k_features = st.slider(
                "Maximum Number of Features:",
                min_value=1,
                max_value=min(max_available, 15),
                value=current_k,
                help="Select the maximum number of top features to use in the model"
            )
            
            # Score rescaling
            current_scale = config['model']['score_scale']
            new_score_scale = st.selectbox(
                "Score Scale:",
                options=[5, 10, 50, 100],
                index=[5, 10, 50, 100].index(current_scale) if current_scale in [5, 10, 50, 100] else 1,
                help="Maximum possible risk score (higher = more granular scoring)"
            )
            
            # Risk bins are now adjusted live (see sidebar settings); not part of retraining
            current_bins = st.session_state.get('risk_bins', config['model'].get('risk_bins', 8))

            st.markdown(f"Current risk-bin setting: **{current_bins} bins** (adjust in sidebar).")
            
            # Feature exclusion
            st.markdown("**Optional: Exclude Features**")
            excluded_features = st.multiselect(
                "Features to Ignore:",
                options=all_feature_names,
                default=[],
                help="Select features to exclude from model training"
            )
            
            col_retrain1, col_retrain2 = st.columns(2)
            
            with col_retrain1:
                if st.button("🚀 Retrain"):
                    with st.spinner("Retraining model..."):
                        success = model_manager.retrain_with_settings(
                            max_features=new_k_features,
                            score_scale=new_score_scale,
                            risk_bins=current_bins,  # keep current bin setting
                            excluded_features=excluded_features
                        )
                        if success:
                            st.success("✅ Model retrained! Reloading …")
                            st.cache_resource.clear()
                            st.cache_data.clear()
                            st.rerun()
            
            with col_retrain2:
                if st.button("🔄 Reload Page"):
                    st.rerun()
            
            st.markdown(f"**Current Features:** {current_k}")
            st.markdown(f"**New Features:** {new_k_features}")
            st.markdown(f"**Score Scale:** {current_scale} → {new_score_scale}")
    
    # Feature input form
    if feature_names:
        feature_values = create_input_form(model_manager, feature_names, feature_types)
        
        # Calculate risk prediction
        try:
            prediction_result = model_manager.predict_risk(feature_values)
            
            # Main content area
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Risk gauge
                if config['ui']['main_panel']['show_risk_gauge']:
                    st.markdown("### 🎯 Risk Assessment")
                    gauge_fig = visualizer.create_risk_gauge(
                        prediction_result['risk_percentage'],
                        prediction_result['risk_score']
                    )
                    st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Risk interpretation
                UIComponents.create_risk_interpretation(
                    prediction_result['risk_percentage'],
                    prediction_result['population_percentile'],
                    decimal_places
                )
            
            with col2:
                # Key metrics
                st.markdown("### 📈 Key Metrics")
                
                st.metric(
                    "Risk Score",
                    f"{prediction_result['risk_score']:.{decimal_places}f}",
                    help="Raw risk score from the model"
                )
                
                st.metric(
                    "Risk Percentage",
                    f"{prediction_result['risk_percentage']:.{decimal_places}f}%",
                    help="Risk as percentage of maximum possible risk"
                )
                
                st.metric(
                    "Population Percentile",
                    f"{prediction_result['population_percentile']:.{decimal_places}f}",
                    help="Your risk compared to the population"
                )
            
            # Population comparison with risk-percentage overlay
            if config['ui']['main_panel']['show_population_comparison'] and population_stats:
                st.markdown("### 📊 Population Distribution & Risk per Bin")
                pop_fig = visualizer.create_population_comparison(
                    prediction_result['risk_score'],
                    population_stats
                )
                st.plotly_chart(pop_fig, use_container_width=True)
            
            # Feature importance
            if config['ui']['main_panel']['show_shap_plots']:
                st.markdown("### 🔍 Feature Importance")
                importance_fig = visualizer.create_feature_importance_plot(
                    prediction_result['model_summary']
                )
                st.plotly_chart(importance_fig, use_container_width=True)
            
            # Model explanation
            st.markdown("### 📋 Model Explanation")
            
            explanation_tab1, explanation_tab2 = st.tabs(["Feature Rules", "Current Values"])
            
            with explanation_tab1:
                st.markdown("The model uses the following rules to calculate risk:")
                explanation_df = create_feature_explanation_table(prediction_result['model_summary'])
                if not explanation_df.empty:
                    st.dataframe(explanation_df, use_container_width=True)
                else:
                    st.info("Feature explanation not available.")
            
            with explanation_tab2:
                st.markdown("Current input values and their rule-based scores:")
                
                # Get the rule-based scores for current values
                current_values_data = []
                used_features = model_manager.get_used_features() or feature_names
                
                for feature in used_features:
                    if feature in feature_values:
                        feature_value = feature_values[feature]
                        
                        # Get the rule-based score for this feature/value combination
                        rule_score = model_manager.get_feature_rule_score(feature, feature_value)
                        
                        current_values_data.append({
                            "Feature": feature,
                            "Value": feature_value,
                            "Score": f"{rule_score:.1f}" if rule_score is not None else "N/A",
                            "Type": feature_types.get(feature, 'Unknown')
                        })
                
                if current_values_data:
                    current_values_df = pd.DataFrame(current_values_data)
                    st.dataframe(current_values_df, use_container_width=True)
                    
                    # Show total score
                    total_score = sum([float(row["Score"]) for row in current_values_data if row["Score"] != "N/A"])
                    st.info(f"📊 Total Score: {total_score:.1f}")
                else:
                    st.info("No feature values to display.")
            
        except Exception as e:
            st.error(f"Error calculating risk: {e}")
            st.info("Please check your input values and try again.")
    
    else:
        st.error("No features available. Please check the model configuration.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
                    <p>SHAPoint WebApp - Interpretable Risk Assessment</p>
        <p>Powered by XGBoost and SHAP | Built with Streamlit</p>
        <p><small>⚠️ This tool is for educational and research purposes only. 
        Always consult healthcare professionals for medical decisions.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 