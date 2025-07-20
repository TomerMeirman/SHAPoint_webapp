"""
Model Manager for SHAPoint WebApp
Handles model loading, training, and prediction operations.
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import math
import re

try:
    from shapoint.shapoint_model import SHAPointModel
except ImportError:
    st.error("SHAPoint package not found. Please run setup_environment.py first.")
    st.stop()


class ModelManager:
    """Manages SHAPoint model operations for the webapp."""
    
    def __init__(self, config: Dict):
        """Initialize with configuration."""
        self.config = config
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        self.feature_types = {}
        self.population_stats = {}
        # NEW: keep full vector of risk scores for fast re-binning without retraining
        self._risk_scores: Optional[pd.Series] = None  # Store computed risk scores on test set
        
    def load_or_train_model(self) -> SHAPointModel:
        """Load existing model or train a new one."""
        model_path = self.config['model'].get('model_path', '')
        
        if model_path and os.path.exists(model_path):
            return self._load_model(model_path)
        else:
            return self._train_new_model()
    
    def _load_model(self, model_path: str) -> SHAPointModel:
        """Load a pre-trained model from file."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Handle both old format (just model) and new format (full state)
            if isinstance(model_data, dict):
                self.model = model_data['model']
                self.feature_names = model_data.get('feature_names', [])
                self.feature_types = model_data.get('feature_types', {})
                self.population_stats = model_data.get('population_stats', {})
                self.X_train = model_data.get('X_train')
                self.y_train = model_data.get('y_train')
                self.X_test = model_data.get('X_test')
                self.y_test = model_data.get('y_test')
            else:
                # Old format - just the model
                self.model = model_data
                st.warning("Loaded model in old format. Some features may not work properly.")
            
            # Model loaded successfully - no UI messages
            return self.model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.info("Training a new model instead...")
            return self._train_new_model()
    
    def _train_new_model(self) -> SHAPointModel:
        """Train a new SHAPoint model."""
        # Load data
        X, y = self._load_data()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if self.config['model']['task'] == 'Classification' else None
        )
        
        # Store feature information
        self.feature_names = X.columns.tolist()
        self._detect_feature_types(X)
        
        # Create and train model
        model_params = {
            'k_variables': self.config['model']['k_variables'],
            'max_leaves': self.config['model']['max_leaves'],
            'task': self.config['model']['task'],
            'score_scale': self.config['model']['score_scale'],
            'use_optuna': self.config['model']['use_optuna'],
            'n_random_features': self.config['model']['n_random_features'],
            'base_model_optuna_params_path': self.config['model']['params_path']
        }
        
        self.model = SHAPointModel(**model_params)
        
        # Train with progress bar
        with st.spinner('Training SHAPoint model...'):
            self.model.fit(self.X_train, self.y_train)
        
        # Show success message with auto-removal
        success_placeholder = st.empty()
        with success_placeholder:
            st.success("✅ Model trained successfully!")
        
        # Auto-remove the success message after 5 seconds (with safe error handling)
        import threading
        def remove_message():
            import time
            time.sleep(5)
            try:
                success_placeholder.empty()
            except Exception:
                # Ignore threading/context errors when clearing the placeholder
                pass
        
        thread = threading.Thread(target=remove_message)
        thread.daemon = True
        thread.start()
        
        # Calculate population statistics after model training
        self._calculate_population_stats()
        
        # Always save model for future use
        self._save_model()
        
        return self.model
    
    def _load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load data based on configuration."""
        dataset_type = self.config['data']['default_dataset']
        
        if dataset_type == 'cardiovascular':
            return self._load_cardiovascular_data()
        elif dataset_type == 'breast_cancer':
            return self._load_breast_cancer_data()
        elif dataset_type == 'custom':
            return self._load_custom_data()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def _load_cardiovascular_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load cardiovascular disease dataset."""
        data_path = self.config['data']['data_path']
        separator = self.config['data'].get('data_separator', ',')  # Default to comma
        
        try:
            df = pd.read_csv(data_path, sep=separator)
        except (FileNotFoundError, ValueError, KeyError) as e:
            # Handle file not found, parsing errors, or missing config keys
            if hasattr(st, 'error'):  # Only show UI messages if in Streamlit context
                st.error(f"Data file error: {e}")
                st.info("Using breast cancer dataset instead...")
            return self._load_breast_cancer_data()
        
        # Prepare features and target
        target_col = self.config['data']['target_column']
        exclude_cols = self.config['data']['exclude_columns']
        
        feature_cols = [col for col in df.columns if col not in [target_col] + exclude_cols]
        
        X = df[feature_cols]
        y = df[target_col]
        
        return X, y
    
    def _load_breast_cancer_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load breast cancer dataset as fallback."""
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        return X, y
    
    def _load_custom_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load custom dataset from uploaded file."""
        # This would be implemented with Streamlit file uploader
        # For now, fallback to cardiovascular data
        return self._load_cardiovascular_data()
    
    def _detect_feature_types(self, X: pd.DataFrame):
        """Detect feature types (categorical vs continuous)."""
        config_types = {}
        
        # Safe config access with null safety
        if self.config is not None:
            data_config = self.config.get('data', {})
            if data_config is not None:
                config_types = data_config.get('feature_types', {})
                
                # Handle case where feature_types is None (empty in YAML)
                if config_types is None:
                    config_types = {}
        
        for col in X.columns:
            if col in config_types.get('categorical', []):
                self.feature_types[col] = 'categorical'
            elif col in config_types.get('continuous', []):
                self.feature_types[col] = 'continuous'
            else:
                # Auto-detect based on unique values and data type
                unique_vals = X[col].nunique()
                col_dtype = X[col].dtype
                
                # Improved heuristics for feature type detection
                if col_dtype == 'object':
                    # String/object columns are usually categorical
                    self.feature_types[col] = 'categorical'
                elif col_dtype in ['bool', 'boolean']:
                    # Boolean columns are categorical
                    self.feature_types[col] = 'categorical'
                elif unique_vals <= 3 and col_dtype in ['int64', 'int32']:
                    # Integers with very few unique values (≤3) are likely categorical
                    self.feature_types[col] = 'categorical'
                elif unique_vals <= 10 and col_dtype in ['int64', 'int32']:
                    # Integers with few unique values (6-10) - check if they look like IDs/codes
                    min_val, max_val = X[col].min(), X[col].max()
                    if min_val >= 0 and max_val == unique_vals - 1:
                        # Looks like 0-indexed categories (0,1,2,3...)
                        self.feature_types[col] = 'categorical'
                    elif min_val >= 1 and max_val == unique_vals:
                        # Looks like 1-indexed categories (1,2,3,4...)
                        self.feature_types[col] = 'categorical'
                    else:
                        # Likely continuous (like ages, scores, etc.)
                        self.feature_types[col] = 'continuous'
                else:
                    # Many unique values or float types are usually continuous
                    self.feature_types[col] = 'continuous'
    
    def _calculate_population_stats(self):
        """Calculate population statistics and risk bins for comparison plots."""
        if self.model is None or self.X_test is None or self.y_test is None:
            return
        
        try:
            # Calculate risk scores for test data (for unbiased risk calculation)
            risk_scores_raw = self.model.predict_risk_score(self.X_test)

            # Handle different return formats from SHAPoint
            if isinstance(risk_scores_raw, pd.DataFrame):
                if 'risk_score' in risk_scores_raw.columns:
                    scores = risk_scores_raw['risk_score']
                else:
                    scores = risk_scores_raw.iloc[:, 0]
            elif isinstance(risk_scores_raw, pd.Series):
                scores = risk_scores_raw
            else:
                scores = pd.Series(risk_scores_raw)

            # Store for future fast re-binning
            self._risk_scores = scores

            # Compute risk bins using helper (may be reused later)
            n_bins = self.config['model'].get('risk_bins', 8)
            risk_bins, bin_edges = self._compute_risk_bins(scores, self.y_test, n_bins)

            # Calculate percentiles and distribution (for plotting)
            self.population_stats = {
                'mean_risk': scores.mean(),
                'std_risk': scores.std(),
                'min_risk': scores.min(),
                'max_risk': scores.max(),
                'percentiles': {
                    '10th': scores.quantile(0.1),
                    '25th': scores.quantile(0.25),
                    '50th': scores.quantile(0.5),
                    '75th': scores.quantile(0.75),
                    '90th': scores.quantile(0.9)
                },
                'distribution': scores.values,
                'risk_bins': risk_bins,
                'bin_edges': bin_edges
            }
            
        except Exception as e:
            st.warning(f"Could not calculate population statistics: {e}")
            self.population_stats = {}
    
    def _compute_risk_bins(self, scores: pd.Series, outcomes: pd.Series, n_bins: int) -> Tuple[Dict[int, Dict[str, Any]], np.ndarray]:
        """Utility to compute bin-level risk percentages. Exposed so we can recalc without retraining."""
        min_score = int(scores.min())
        max_score = int(scores.max())

        if max_score == min_score:
            max_score += 1  # avoid zero width

        # Determine integer step so we have at most n_bins bins spanning full range
        step = math.ceil((max_score - min_score + 1) / n_bins)
        bin_edges = np.arange(min_score, max_score + step, step)

        # Ensure we have exactly n_bins+1 edges (extend or trim)
        if len(bin_edges) < n_bins + 1:
            # extend
            last = bin_edges[-1]
            while len(bin_edges) < n_bins + 1:
                last += step
                bin_edges = np.append(bin_edges, last)
        elif len(bin_edges) > n_bins + 1:
            bin_edges = bin_edges[: n_bins + 1]

        bin_indices = np.digitize(scores, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        risk_bins: Dict[int, Dict[str, Any]] = {}
        for i in range(n_bins):
            mask = bin_indices == i
            total_cases = int(mask.sum())
            if total_cases > 0:
                positive_cases = int(outcomes[mask].sum())
                risk_percentage = (positive_cases / total_cases) * 100
            else:
                positive_cases = 0
                risk_percentage = 0.0

            is_valid = total_cases >= max(1, int(len(scores) * 0.0001))  # keep if >=0.01% of population (rounded up)

            risk_bins[i] = {
                'min_score': bin_edges[i],
                'max_score': bin_edges[i + 1],
                'positive_cases': positive_cases,
                'total_cases': total_cases,
                'risk_percentage': risk_percentage,
                'bin_center': (bin_edges[i] + bin_edges[i + 1]) / 2,
                'valid': is_valid
            }
        return risk_bins, bin_edges

    def recalculate_risk_bins(self, n_bins: int):
        """Recompute risk bin statistics on the stored test risk scores without retraining the model."""
        if self._risk_scores is None or self.y_test is None:
            # If we somehow lost the raw scores, fall back to full computation
            self._calculate_population_stats()
            return

        try:
            risk_bins, bin_edges = self._compute_risk_bins(self._risk_scores, self.y_test, n_bins)
            # Update population stats in-place so downstream uses new bins immediately
            if not self.population_stats:
                self.population_stats = {}
            self.population_stats.update({
                'risk_bins': risk_bins,
                'bin_edges': bin_edges
            })
            # Also reflect in config so that cached values match UI
            self.config['model']['risk_bins'] = n_bins
        except Exception as e:
            st.warning(f"Could not recalculate risk bins: {e}")
    
    def _save_model(self):
        """Save the trained model and all associated data."""
        model_path = self.config['model']['model_path']
        if not model_path:
            return
            
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        try:
            # Save complete model state
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'feature_types': self.feature_types,
                'population_stats': self.population_stats,
                'X_train': self.X_train,
                'y_train': self.y_train,
                'X_test': self.X_test,
                'y_test': self.y_test,
                'config': self.config
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            # Model saved successfully - no UI messages
        except Exception as e:
            st.warning(f"Could not save model: {e}")
    
    def predict_risk(self, feature_values: Dict) -> Dict:
        """Predict risk for given feature values."""
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        try:
            # Create dataframe from feature values
            df = pd.DataFrame([feature_values])
            
            # Ensure all expected features are present
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = self._get_default_value(col)
            
            # Reorder columns to match training data
            df = df[self.feature_names]
            
            # Get predictions
            risk_scores = self.model.predict_risk_score(df)
            
            # Handle different return formats from SHAPoint
            if isinstance(risk_scores, pd.DataFrame):
                if 'risk_score' in risk_scores.columns:
                    risk_score = risk_scores['risk_score'].iloc[0]
                else:
                    # Take the first numeric column
                    risk_score = risk_scores.iloc[0, 0]
            elif isinstance(risk_scores, pd.Series):
                risk_score = risk_scores.iloc[0]
            else:
                # Handle numpy array or single value
                risk_score = float(risk_scores[0]) if hasattr(risk_scores, '__getitem__') else float(risk_scores)
            
            # Calculate percentage risk using bin-based approach
            if self.population_stats and 'risk_bins' in self.population_stats:
                # Find which bin this score falls into
                risk_bins = self.population_stats['risk_bins']
                bin_edges = self.population_stats['bin_edges']
                
                # Find the appropriate bin
                bin_index = np.digitize([risk_score], bin_edges)[0] - 1
                bin_index = np.clip(bin_index, 0, len(risk_bins) - 1)
                
                if bin_index in risk_bins:
                    bin_info = risk_bins[bin_index]
                    risk_percentage = bin_info['risk_percentage']
                    
                    # Show calculation info in sidebar
                    with st.sidebar:
                        st.info(f"""
                        **🎯 Bin-Based Risk Calculation**
                        - Score: {risk_score:.1f}
                        - Bin: {bin_index} ({bin_info['min_score']:.1f} - {bin_info['max_score']:.1f})
                        - Outcomes: {bin_info['positive_cases']}/{bin_info['total_cases']}
                        - **Risk: {risk_percentage:.1f}%**
                        """)
                else:
                    # Fallback if bin not found
                    risk_percentage = 50.0
                    st.sidebar.warning(f"⚠️ Bin {bin_index} not found. Using 50% fallback.")
                
            else:
                # Fallback to configured scale if no population stats
                max_score = self.config['model']['score_scale']
                risk_percentage = (risk_score / max_score) * 100
                
                # Show fallback calculation
                with st.sidebar:
                    if not self.population_stats:
                        st.warning("⚠️ No population stats - using simple calculation")
                    elif 'risk_bins' not in self.population_stats:
                        st.warning("⚠️ No risk bins - using simple calculation")
                    st.info(f"""
                    **📊 Simple Risk Calculation**
                    - Score: {risk_score:.1f}
                    - Max Score: {max_score}
                    - **Risk: {risk_percentage:.1f}%**
                    """)
            
            # Ensure percentage is within reasonable bounds
            risk_percentage = max(0.0, min(risk_percentage, 100.0))
            
            # Get model summary for explanation
            summary = self.model.get_model_summary_with_nulls()
            
            return {
                'risk_score': risk_score,
                'risk_percentage': risk_percentage,
                'model_summary': summary,
                'feature_contributions': self._get_feature_contributions(df),
                'population_percentile': self._calculate_percentile(risk_score)
            }
            
        except Exception as e:
            st.error(f"Error predicting risk: {e}")
            # Return safe defaults
            return {
                'risk_score': 0,
                'risk_percentage': 0,
                'model_summary': {},
                'feature_contributions': [],
                'population_percentile': 50.0
            }
    
    def _get_default_value(self, feature_name: str) -> Any:
        """Get default value for a feature."""
        feature_metadata = {}
        
        # Safe config access with null safety
        if self.config is not None:
            data_config = self.config.get('data', {})
            if data_config is not None:
                feature_metadata = data_config.get('feature_metadata', {}) or {}
        
        if feature_name in feature_metadata:
            return feature_metadata[feature_name].get('default_value', 0)
        
        # Fallback defaults
        if self.feature_types.get(feature_name) == 'categorical':
            return 0
        else:
            return 0.0
    
    def _get_feature_contributions(self, df: pd.DataFrame) -> List[Dict]:
        """Get individual feature contributions and scores to risk score."""
        contributions: List[Dict] = []
        
        try:
            # Features actually used by current model
            used_features = self.get_used_features() or self.feature_names or []
            
            # Ensure used_features is always a list
            if used_features is None:
                used_features = []

            # Get model summary to extract feature scoring logic (if available)
            summary = self.model.get_model_summary_with_nulls() if hasattr(self.model, "get_model_summary_with_nulls") else {}

            feature_scores: Dict[str, float] = {}
            feature_contributions: Dict[str, float] = {}
            
            # Try different ways to extract feature scores
            if isinstance(summary, dict):
                # Method 1: Look for rules or scoring information
                if 'rules' in summary:
                    for rule in summary['rules']:
                        # Extract features and scores from rules
                        if 'feature' in rule and 'score' in rule:
                            feature = rule['feature']
                            score = rule['score']
                            
                            # Check if this rule applies to current input
                            if feature in df.columns:
                                feature_value = df[feature].iloc[0]
                                
                                # Try to match rule conditions to current value
                                condition_match = True  # Simplified - assume rule applies
                                if 'condition' in rule:
                                    condition = rule['condition']
                                    # Could add more sophisticated condition matching here
                                
                                if condition_match:
                                    feature_scores[feature] = score
                                    feature_contributions[feature] = score  # Use score as contribution for now
                
                # Method 2: Look for feature importance or scoring breakdown
                elif 'feature_scores' in summary:
                    feature_scores = summary['feature_scores']
                    feature_contributions = feature_scores
                
                # Method 3: Look for SHAP-like explanations
                elif 'feature_contributions' in summary:
                    feature_contributions = summary['feature_contributions']
                
                # Method 4: Extract from model coefficients or tree structure if available
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'feature_importances_'):
                    importances = self.model.model.feature_importances_
                    for i, feature in enumerate(self.feature_names):
                        if i < len(importances):
                            feature_scores[feature] = importances[i] * 10  # Scale up for visibility
                            feature_contributions[feature] = importances[i] * 5
            
            # If rule-based details exist, compute per-feature score using input value
            if isinstance(summary, dict) and 'feature_summary' in summary:
                for feature in used_features:
                    if feature in summary['feature_summary']:
                        levels = summary['feature_summary'][feature].get('levels_detail', [])
                        val = df[feature].iloc[0]
                        for lvl in levels:
                            rule_str = str(lvl.get('rule', ''))
                            score_val = lvl.get('scaled_score', None)
                            if score_val is None:
                                continue

                            # Basic rule parsing: replace feature name with value; convert single '=' to '=='
                            safe_rule = rule_str.replace(feature, 'val')
                            # Convert single '=' to '==' when not part of >=, <=, !=
                            safe_rule = re.sub(r'(?<![<>!])=(?!=)', '==', safe_rule)
                            try:
                                if eval(safe_rule, {"val": val}):
                                    feature_scores[feature] = score_val
                                    feature_contributions[feature] = score_val
                                    break
                            except Exception:
                                # Fallback: string equality check
                                if str(val) in rule_str:
                                    feature_scores[feature] = score_val
                                    feature_contributions[feature] = score_val
                                    break
            
            # Drop SHAP-based explanations as per requirements – rely solely on model rules/summary

            # Build contributions list with actual values
            total_score = sum(feature_scores.values()) if feature_scores else 1
            
            for feature in used_features:
                feature_value = df[feature].iloc[0]
                feature_score = feature_scores.get(feature, 0)
                feature_contribution = feature_contributions.get(feature, 0)
                
                # Calculate contribution percentage
                if total_score > 0:
                    contribution_pct = (abs(feature_contribution) / total_score) * 100
                else:
                    contribution_pct = 0
                
                contributions.append({
                    'feature': feature,
                    'value': feature_value,
                    'score': feature_score,  # Individual feature score from model
                    'contribution': feature_contribution,  # Feature's contribution to total score
                    'contribution_percentage': contribution_pct,  # Percentage of total contribution
                    'magnitude': abs(feature_contribution) if feature_contribution !=0 else abs(feature_score)
                })
                
        except Exception as e:
            st.error(f"Error extracting feature contributions: {e}")
            # Fallback: return basic info without scores
            safe_features = used_features or self.feature_names or []
            if safe_features is None:
                safe_features = []
                
            for feature in safe_features:
                if feature in df.columns:
                    contributions.append({
                        'feature': feature,
                        'value': df[feature].iloc[0],
                        'score': 0,  # No score available
                        'contribution': 0,
                        'contribution_percentage': 0,
                        'magnitude': 0
                    })
        
        return contributions
    
    def _calculate_percentile(self, risk_score: float) -> float:
        """Calculate what percentile this risk score represents."""
        if not self.population_stats:
            return 50.0
        
        distribution = self.population_stats['distribution']
        percentile = (np.sum(distribution <= risk_score) / len(distribution)) * 100
        return percentile
    
    def get_feature_metadata(self, feature_name: str) -> Dict:
        """Get metadata for a feature for UI generation with auto-detection."""
        # Start with automatically detected metadata
        default_metadata = self._auto_detect_feature_metadata(feature_name)
        
        # Override with any user-specified metadata from config (with null safety)
        if self.config is not None:
            data_config = self.config.get('data', {})
            if data_config is not None:
                feature_metadata_config = data_config.get('feature_metadata', {})
                
                # Handle case where feature_metadata is None (empty in YAML)
                if feature_metadata_config is None:
                    feature_metadata_config = {}
                    
                config_metadata = feature_metadata_config.get(feature_name, {})
                default_metadata.update(config_metadata)
        
        return default_metadata
    
    def _auto_detect_feature_metadata(self, feature_name: str) -> Dict:
        """Automatically detect feature metadata from the dataset."""
        # Basic metadata
        metadata = {
            'label': feature_name.replace('_', ' ').title(),
            'help': f"Enter value for {feature_name}",
            'default_value': self._get_default_value_from_data(feature_name)
        }
        
        # Add type-specific metadata based on actual data
        if self.feature_types.get(feature_name) == 'categorical':
            metadata.update(self._auto_detect_categorical_metadata(feature_name))
        else:
            metadata.update(self._auto_detect_continuous_metadata(feature_name))
        
        return metadata
    
    def _auto_detect_categorical_metadata(self, feature_name: str) -> Dict:
        """Auto-detect metadata for categorical features."""
        metadata = {}
        
        if self.X_train is not None and feature_name in self.X_train.columns:
            unique_vals = sorted(self.X_train[feature_name].unique())
            
            # Generate intelligent options based on feature name and values
            options = {}
            for val in unique_vals:
                label = self._generate_categorical_label(feature_name, val)
                options[val] = {"label": label, "range": ""}
            
            metadata['options'] = options
            
            # Set default to most common value
            most_common = self.X_train[feature_name].mode().iloc[0] if len(self.X_train[feature_name].mode()) > 0 else unique_vals[0]
            metadata['default_value'] = most_common
        else:
            # Fallback for binary features
            metadata['options'] = {0: {"label": "No", "range": ""}, 1: {"label": "Yes", "range": ""}}
            metadata['default_value'] = 0
        
        return metadata
    
    def _auto_detect_continuous_metadata(self, feature_name: str) -> Dict:
        """Auto-detect metadata for continuous features."""
        metadata = {}
        
        if self.X_train is not None and feature_name in self.X_train.columns:
            data_col = self.X_train[feature_name]
            
            # Calculate reasonable min/max with some padding
            data_min = data_col.min()
            data_max = data_col.max()
            data_range = data_max - data_min
            padding = data_range * 0.1  # 10% padding
            
            metadata['min_value'] = max(0, data_min - padding) if data_min >= 0 else data_min - padding
            metadata['max_value'] = data_max + padding
            
            # Set default to median for better user experience
            metadata['default_value'] = data_col.median()
            
            # Set appropriate step size
            if data_col.dtype in ['int64', 'int32']:
                metadata['step'] = 1
            else:
                # For float data, use a reasonable step based on the range
                if data_range < 10:
                    metadata['step'] = 0.1
                elif data_range < 100:
                    metadata['step'] = 1.0
                else:
                    metadata['step'] = 5.0
        else:
            # Fallback defaults
            metadata.update({
                'min_value': 0,
                'max_value': 1000,
                'default_value': 0.0,
                'step': 1.0
            })
        
        return metadata
    
    def _generate_categorical_label(self, feature_name: str, value: Any) -> str:
        """Generate intelligent labels for categorical values."""
        # Known mappings for common medical features
        label_mappings = {
            'gender': {1: 'Female', 2: 'Male'},
            'cholesterol': {1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'},
            'gluc': {1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'},
            'smoke': {0: 'No', 1: 'Yes'},
            'alco': {0: 'No', 1: 'Yes'},
            'active': {0: 'No', 1: 'Yes'}
        }
        
        if feature_name.lower() in label_mappings and value in label_mappings[feature_name.lower()]:
            return label_mappings[feature_name.lower()][value]
        
        # Fallback to string representation
        return str(value)
    
    def _get_default_value_from_data(self, feature_name: str) -> Any:
        """Get a sensible default value from the actual data."""
        if self.X_train is not None and feature_name in self.X_train.columns:
            data_col = self.X_train[feature_name]
            
            if self.feature_types.get(feature_name) == 'categorical':
                # Use most common value for categorical
                return data_col.mode().iloc[0] if len(data_col.mode()) > 0 else 0
            else:
                # Use median for continuous
                return data_col.median()
        
        # Fallback
        return 0
    
    def get_used_features(self) -> List[str]:
        """Get only the features that are actually used by the trained model."""
        if self.model is None:
            return []
        
        # Try to get features from the trained model
        if hasattr(self.model, 'top_features') and self.model.top_features is not None:
            return list(self.model.top_features)
        elif hasattr(self.model, 'feature_names_') and self.model.feature_names_ is not None:
            return list(self.model.feature_names_)
        elif self.feature_names:
            # Fallback to training feature names, but limit to k_variables
            k_vars = self.config.get('model', {}).get('k_variables', 5)
            return self.feature_names[:k_vars]
        else:
            return []
    
    def get_feature_rule_score(self, feature_name: str, feature_value: Any) -> Optional[float]:
        """Get the rule-based score for a specific feature value."""
        if self.model is None:
            return None
        
        try:
            # Get model summary to extract rule information
            model_summary = self.model.get_model_summary_with_nulls()
            
            if not model_summary or 'feature_summary' not in model_summary:
                return None
            
            feature_info = model_summary['feature_summary'].get(feature_name)
            if not feature_info:
                return None
            
            # Look through the levels to find the matching rule
            levels_detail = feature_info.get('levels_detail', [])
            
            for level in levels_detail:
                rule_text = level.get('rule', '')
                scaled_score = level.get('scaled_score', 0)
                
                # Parse and evaluate the rule against the current value
                if self._evaluate_rule(rule_text, feature_value):
                    return float(scaled_score)
            
            # If no rule matched, return 0 (default score)
            return 0.0
            
        except Exception as e:
            st.warning(f"Could not get rule score for {feature_name}: {e}")
            return None
    
    def _evaluate_rule(self, rule_text: str, value: Any) -> bool:
        """Evaluate if a rule applies to the given value."""
        if not rule_text:
            return False
        
        try:
            # Clean up the rule text and make it safer for evaluation
            import re
            
            # Convert value to appropriate type for comparison
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    pass
            
            # IMPORTANT: Check range patterns FIRST before simple patterns
            # Handle range patterns: "X < feature <= Y" or "X <= feature < Y"
            range_patterns = [
                r'([0-9.]+)\s*<\s*.*?\s*<=?\s*([0-9.]+)',  # "10 < age <= 20"
                r'([0-9.]+)\s*<=?\s*.*?\s*<\s*([0-9.]+)'   # "10 <= age < 20"
            ]
            
            for range_pattern in range_patterns:
                range_match = re.search(range_pattern, rule_text)
                if range_match:
                    lower = float(range_match.group(1))
                    upper = float(range_match.group(2))
                    
                    if '<=' in rule_text and '<=' in rule_text.split('<')[1]:
                        # "X <= feature <= Y" 
                        return lower <= value <= upper
                    elif '<' in rule_text and '<=' in rule_text:
                        # "X < feature <= Y"
                        return lower < value <= upper
                    elif '<=' in rule_text and '<' in rule_text.split('<=')[1]:
                        # "X <= feature < Y"
                        return lower <= value < upper
                    else:
                        # "X < feature < Y"
                        return lower < value < upper
            
            # Handle simple comparison patterns (only after range patterns fail)
            # Pattern: "feature <= X" or "feature > X" etc.
            # Use negative lookbehind to avoid matching malformed syntax like "><="
            simple_patterns = [
                (r'(?<!>)<=\s*([0-9.]+)', lambda v, t: v <= t),  # Not preceded by >
                (r'(?<!<)>=\s*([0-9.]+)', lambda v, t: v >= t),  # Not preceded by <
                (r'(?<!>)(?<!<)(?<!=)<\s*([0-9.]+)', lambda v, t: v < t),  # Not preceded by >, <, or =
                (r'(?<!<)(?<!>)(?<!=)>\s*([0-9.]+)', lambda v, t: v > t),  # Not preceded by <, >, or =
                (r'==?\s*([0-9.]+)', lambda v, t: v == t),
                (r'(?<!>)(?<!<)(?<!=)=\s*([0-9.]+)', lambda v, t: v == t)  # Not preceded by >, <, or =
            ]
            
            for pattern, comparison_func in simple_patterns:
                match = re.search(pattern, rule_text)
                if match:
                    threshold = float(match.group(1))
                    return comparison_func(value, threshold)
            
            # Handle categorical equality (exact match)
            if isinstance(value, (int, float)) and str(int(value)) in rule_text:
                return True
            elif str(value) in rule_text:
                return True
            
            return False
            
        except Exception:
            # If rule evaluation fails, assume it doesn't match
            return False
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model."""
        if self.model is None:
            return {}
        
        info = {
            'task': self.config['model']['task'],
            'k_variables': self.config['model']['k_variables'],
            'max_leaves': self.config['model']['max_leaves'],
            'score_scale': self.config['model']['score_scale'],
            'features_used': self.model.top_features if hasattr(self.model, 'top_features') else [],
            'training_samples': len(self.X_train) if self.X_train is not None else 0,
            'test_samples': len(self.X_test) if self.X_test is not None else 0
        }
        
        return info
    
    def retrain_model(self, selected_features: List[str]) -> bool:
        """Retrain the model with selected features (no Optuna optimization)."""
        try:
            # Load data
            X, y = self._load_data()
            
            # Filter to selected features only
            if selected_features:
                missing_features = [f for f in selected_features if f not in X.columns]
                if missing_features:
                    st.error(f"Features not found in data: {missing_features}")
                    return False
                X = X[selected_features]
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Update feature names and types
            self.feature_names = list(X.columns)
            self._detect_feature_types(X)
            
            # Initialize model with config parameters (no Optuna)
            model_params = {
                'k_variables': len(selected_features) if selected_features else self.config['model']['k_variables'],
                'max_leaves': self.config['model']['max_leaves'],
                'task': self.config['model']['task'],
                'score_scale': self.config['model']['score_scale'],
                'use_optuna': False,  # Disable Optuna for quick retrain
                'n_random_features': self.config['model']['n_random_features'],
                'base_model_optuna_params_path': self.config['model']['params_path']
            }
            
            self.model = SHAPointModel(**model_params)
            
            # Train the model
            self.model.fit(self.X_train, self.y_train)
            
            # Calculate population statistics
            self._calculate_population_stats()
            
            # Save the retrained model
            self._save_model()
            
            return True
            
        except Exception as e:
            st.error(f"Error retraining model: {e}")
            return False
    
    def retrain_with_settings(self, max_features: int, score_scale: int, risk_bins: int = 8, excluded_features: List[str] = None) -> bool:
        """Retrain the model with new settings (max features, score scale, exclusions)."""
        try:
            # Load data
            X, y = self._load_data()
            
            # Remove excluded features
            if excluded_features:
                available_features = [f for f in X.columns if f not in excluded_features]
                X = X[available_features]
                st.info(f"Excluded {len(excluded_features)} features: {', '.join(excluded_features)}")
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Update feature names and types
            self.feature_names = list(X.columns)
            self._detect_feature_types(X)
            
            # Update config with new settings
            self.config['model']['k_variables'] = max_features
            self.config['model']['score_scale'] = score_scale
            self.config['model']['risk_bins'] = risk_bins
            
            # Initialize model with new parameters (no Optuna)
            model_params = {
                'k_variables': max_features,
                'max_leaves': self.config['model']['max_leaves'],
                'task': self.config['model']['task'],
                'score_scale': score_scale,
                'use_optuna': False,  # Disable Optuna for quick retrain
                'n_random_features': self.config['model']['n_random_features'],
                'base_model_optuna_params_path': self.config['model']['params_path']
            }
            
            self.model = SHAPointModel(**model_params)
            
            # Train the model
            self.model.fit(self.X_train, self.y_train)
            
            # Calculate population statistics
            self._calculate_population_stats()
            
            # Save the retrained model
            self._save_model()
            
            st.info(f"✅ Model retrained with {max_features} features, score scale {score_scale}, {risk_bins} risk bins")
            
            return True
            
        except Exception as e:
            st.error(f"Error retraining model: {e}")
            return False 