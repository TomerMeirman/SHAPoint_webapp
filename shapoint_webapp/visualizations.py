"""
Visualization components for SHAPoint WebApp
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import io
import base64


class RiskVisualizer:
    """Handles all risk visualization components."""
    
    def __init__(self, config: Dict):
        """Initialize with configuration."""
        self.config = config
    
    def create_risk_gauge(self, risk_percentage: float, risk_score: float) -> go.Figure:
        """Create a risk gauge/meter visualization."""
        gauge_config = self.config['visualization']['risk_gauge']
        
        # Determine risk level and color
        if risk_percentage < 30:
            color = "green"
            risk_level = "Low"
        elif risk_percentage < 70:
            color = "yellow"
            risk_level = "Medium"
        else:
            color = "red"
            risk_level = "High"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {
                'text': f"{gauge_config['title']}<br><span style='font-size:0.8em;color:gray'>Score: {risk_score:.1f}, Risk Level: {risk_level}</span>",
                'font': {'size': 20}
            },
            delta = {'reference': 50, 'position': "top"},
            gauge = {
                'axis': {
                    'range': [None, 100],  # Always use 0-100 for percentage
                    'tickwidth': 1, 
                    'tickcolor': "darkgreen",
                    'tick0': 0,
                    'dtick': 10  # Show ticks every 10%
                },
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'lightgreen'},
                    {'range': [30, 70], 'color': 'lightyellow'},
                    {'range': [70, 100], 'color': 'lightcoral'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            font={'color': "darkgreen", 'family': "Arial"}
        )
        
        return fig
    
    def create_population_comparison(self, user_risk: float, population_stats: Dict) -> go.Figure:
        """Create population comparison visualization."""
        if not population_stats or 'distribution' not in population_stats or 'risk_bins' not in population_stats:
            # Placeholder if stats missing
            fig = go.Figure()
            fig.add_annotation(
                text="Population statistics not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(title="Population Risk Distribution", height=350)
            return fig
        
        distribution = population_stats['distribution']
        risk_bins = population_stats['risk_bins']  # dict keyed by bin index
        bin_edges = population_stats['bin_edges']

        # Check for empty data and return placeholder
        if len(distribution) == 0 or len(risk_bins) == 0 or len(bin_edges) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="Population distribution is empty",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(title="Population Risk Distribution", height=350)
            return fig

        # Build list for histogram bins to align with risk bins
        n_bins = len(risk_bins)
        hist = go.Histogram(
            x=distribution,
            xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=(bin_edges[-1]-bin_edges[0])/n_bins),
            name='Population',
            marker_color='rgba(173,216,230,0.6)',
            opacity=0.7,
            yaxis='y1'
        )

        # Risk percentage per bin (line chart, secondary y-axis)
        valid_centers = []
        valid_rates = []
        for i in range(n_bins):
            if risk_bins[i].get('valid', True):
                valid_centers.append(risk_bins[i]['bin_center'])
                valid_rates.append(risk_bins[i]['risk_percentage'])
        line = go.Scatter(
            x=valid_centers,
            y=valid_rates,
            mode='lines+markers',
            name='Observed Case Rate (%)',
            marker=dict(color='darkorange'),
            yaxis='y2'
        )

        # Create figure with two y-axes
        fig = go.Figure(data=[hist, line])

        # User risk vline
        user_percentile = (sum(1 for x in distribution if x <= user_risk) / len(distribution)) * 100
        fig.add_vline(
            x=user_risk,
            line_dash='dash',
            line_color='red',
            line_width=3,
            annotation_text=f"Your risk: {user_risk:.1f}<br>{user_percentile:.1f} pct", annotation_position='top'
        )

        # Update layout with twin axis
        fig.update_layout(
            title="Population Distribution & Observed Case Rate",
            xaxis_title="Risk Score",
            yaxis=dict(title='Number of People', rangemode='tozero'),
            yaxis2=dict(title='Case Rate (%)', overlaying='y', side='right', rangemode='tozero'),
            bargap=0.05,
            height=450,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        # Set x-axis ticks at bin edges for clarity
        fig.update_xaxes(tickmode='array', tickvals=valid_centers, ticktext=[f"{int(edge_left)}-{int(edge_right-1)}" for edge_left, edge_right in zip(bin_edges[:-1], bin_edges[1:])])

        return fig
    
    def create_feature_importance_plot(self, model_summary: Dict) -> go.Figure:
        """Create feature importance visualization."""
        if not model_summary or 'feature_summary' not in model_summary:
            # Placeholder
            fig = go.Figure()
            fig.add_annotation(
                text="Feature importance not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        features = []
        importance_scores = []
        
        for feature, info in model_summary['feature_summary'].items():
            features.append(feature)
            # Use score range as proxy for importance
            score_range = info['score_range']['max'] - info['score_range']['min']
            importance_scores.append(score_range)
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=importance_scores,
            y=features,
            orientation='h',
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title="Feature Importance (Score Range)",
            xaxis_title="Score Range",
            yaxis_title="Features",
            height=300,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    

    
    def create_risk_distribution_box(self, population_stats: Dict, user_risk: float) -> go.Figure:
        """Create box plot showing risk distribution."""
        if not population_stats or 'distribution' not in population_stats:
            # Create fallback plot
            fig = go.Figure()
            fig.add_annotation(
                text="Box plot not available<br>Population statistics missing",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(
                title="Risk Distribution Analysis",
                height=500
            )
            return fig
        
        try:
            # Use the full distribution data for proper box plot
            distribution = population_stats['distribution']
            
            # Ensure distribution is valid
            if len(distribution) == 0:
                raise ValueError("Empty distribution data")
            
            # Calculate percentile position of user's risk
            user_percentile = (np.sum(distribution <= user_risk) / len(distribution)) * 100
            
            fig = go.Figure()
            
            # Create box plot using actual distribution data
            fig.add_trace(go.Box(
                y=distribution,
                name="Population Risk Distribution",
                boxpoints=False,
                fillcolor='lightblue',
                line=dict(color='darkblue'),
                width=0.4,  # Make box narrower
                showlegend=True
            ))
            
            # Add user's risk point slightly to the right of the box to ensure visibility
            fig.add_trace(go.Scatter(
                x=[0.25],
                y=[user_risk],
                mode='markers+text',
                marker=dict(color='red', size=15, symbol='diamond'),
                name=f'Your Risk ({user_percentile:.1f} percentile)',
                text=[f'{user_risk:.1f}'],
                textposition="middle right",
                textfont=dict(size=12, color='red'),
                showlegend=True
            ))
            
            fig.update_layout(
                title="Risk Distribution Analysis<br><sub>Box plot shows population quartiles</sub>",
                yaxis_title="Risk Score",
                xaxis=dict(showticklabels=False, range=[-0.6, 0.8]),  # Hide x-axis and allow space for marker
                height=500,  # Make taller
                showlegend=True,
                margin=dict(l=50, r=100, t=80, b=50),  # Add margin for text
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
        except Exception as e:
            # Error fallback
            fig = go.Figure()
            fig.add_annotation(
                text=f"Box plot error:<br>{str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=14, color='red')
            )
            fig.update_layout(
                title="Risk Distribution Analysis",
                height=500
            )
        
        return fig


class UIComponents:
    """Additional UI components and utilities."""
    
    @staticmethod
    def create_metric_card(title: str, value: str, delta: str = None, delta_color: str = "normal"):
        """Create a metric display card."""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric(
                label=title,
                value=value,
                delta=delta,
                delta_color=delta_color
            )
    
    @staticmethod
    def create_info_box(title: str, content: str, box_type: str = "info"):
        """Create an information box."""
        if box_type == "info":
            st.info(f"**{title}**\n\n{content}")
        elif box_type == "warning":
            st.warning(f"**{title}**\n\n{content}")
        elif box_type == "success":
            st.success(f"**{title}**\n\n{content}")
        elif box_type == "error":
            st.error(f"**{title}**\n\n{content}")
    
    @staticmethod
    def create_feature_input(feature_name: str, feature_metadata: Dict, feature_type: str, key: str = None):
        """Create appropriate input widget for a feature."""
        if key is None:
            key = f"input_{feature_name}"
        
        label = feature_metadata.get('label', feature_name)
        help_text = feature_metadata.get('help', '')
        default_value = feature_metadata.get('default_value', 0)
        
        if feature_type == 'categorical':
            options = feature_metadata.get('options', {0: {"label": "No", "range": ""}, 1: {"label": "Yes", "range": ""}})
            
            # Handle both old format (simple values) and new format (with ranges)
            option_labels = []
            option_values = []
            
            for key_val, option_data in options.items():
                if isinstance(option_data, dict):
                    # New format with label and range
                    label_text = option_data.get('label', str(key_val))
                    range_text = option_data.get('range', '')
                    if range_text:
                        display_label = f"{label_text} ({range_text})"
                    else:
                        display_label = label_text
                else:
                    # Old format - just the label
                    display_label = str(option_data)
                
                option_labels.append(display_label)
                option_values.append(key_val)
            
            try:
                default_index = option_values.index(default_value)
            except ValueError:
                default_index = 0
            
            selected_label = st.selectbox(
                label=label,
                options=option_labels,
                index=default_index,
                help=help_text,
                key=key
            )
            
            # Return the actual value (key) corresponding to the selected label
            selected_index = option_labels.index(selected_label)
            return option_values[selected_index]
        
        else:  # continuous
            min_val = feature_metadata.get('min_value', 0)
            max_val = feature_metadata.get('max_value', 1000)
            step = feature_metadata.get('step', 1 if isinstance(default_value, int) else 0.1)
            
            return st.number_input(
                label=label,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_value),
                step=float(step),
                help=help_text,
                key=key
            )
    
    @staticmethod
    def create_model_info_sidebar(model_info: Dict):
        """Create model information in sidebar."""
        if not model_info:
            return
        
        st.sidebar.markdown("### 📊 Model Information")
        
        st.sidebar.metric("Task Type", model_info.get('task', 'Unknown'))
        st.sidebar.metric("Features Used", len(model_info.get('features_used', [])))
        st.sidebar.metric("Training Samples", model_info.get('training_samples', 0))
        
        if model_info.get('features_used'):
            st.sidebar.markdown("**Selected Features:**")
            for feature in model_info['features_used']:
                st.sidebar.markdown(f"• {feature}")
    
    @staticmethod
    def create_risk_interpretation(risk_percentage: float, population_percentile: float, decimal_places: int = 1):
        """Create risk interpretation text."""
        if risk_percentage < 30:
            risk_level = "Low"
            risk_color = "green"
            interpretation = "Your risk is below average. Continue maintaining healthy habits."
        elif risk_percentage < 70:
            risk_level = "Medium"
            risk_color = "orange"
            interpretation = "Your risk is moderate. Consider discussing prevention strategies with a healthcare provider."
        else:
            risk_level = "High"
            risk_color = "red"
            interpretation = "Your risk is elevated. It's recommended to consult with a healthcare provider."
        
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 0.5rem; border: 2px solid {risk_color}; background-color: rgba(255, 255, 255, 0.8);">
            <h4 style="color: {risk_color}; margin: 0;">Risk Level: {risk_level}</h4>
            <p style="margin: 0.5rem 0;">{interpretation}</p>
            <p style="margin: 0; font-size: 0.9em; color: #666;">
                You are in the {population_percentile:.{decimal_places}f} percentile compared to the population.
            </p>
        </div>
        """, unsafe_allow_html=True)


def create_feature_explanation_table(model_summary: Dict) -> pd.DataFrame:
    """Create a table explaining the model features and their levels."""
    explanation_data = []
    
    # Try to extract feature information from different possible structures
    if model_summary and isinstance(model_summary, dict):
        # Method 1: Check for feature_summary
        if 'feature_summary' in model_summary:
            for feature, info in model_summary['feature_summary'].items():
                levels_detail = info.get('levels_detail', [])
                
                for level in levels_detail:
                    explanation_data.append({
                        'Feature': feature,
                        'Rule': level.get('rule', ''),
                        'Score Contribution': level.get('scaled_score', 0),
                        'Risk Impact': 'Increases' if level.get('scaled_score', 0) > 0 else 'Decreases'
                    })
        
        # Method 2: Check for rules structure
        elif 'rules' in model_summary:
            for rule in model_summary['rules']:
                if isinstance(rule, dict):
                    explanation_data.append({
                        'Feature': rule.get('feature', 'Unknown'),
                        'Rule': rule.get('condition', rule.get('rule', '')),
                        'Score Contribution': rule.get('score', 0),
                        'Risk Impact': 'Increases' if rule.get('score', 0) > 0 else 'Decreases'
                    })
        
        # Method 3: Check for any feature-related information
        elif 'features' in model_summary:
            features = model_summary['features']
            if isinstance(features, dict):
                for feature, info in features.items():
                    explanation_data.append({
                        'Feature': feature,
                        'Rule': str(info.get('description', 'Feature used in model')),
                        'Score Contribution': info.get('importance', 0),
                        'Risk Impact': 'Increases' if info.get('importance', 0) > 0 else 'Decreases'
                    })
            elif isinstance(features, list):
                for feature in features:
                    explanation_data.append({
                        'Feature': str(feature),
                        'Rule': 'Feature used in model',
                        'Score Contribution': 0,
                        'Risk Impact': 'Contributes to prediction'
                    })
    
    # If no data found, create a basic explanation
    if not explanation_data:
        explanation_data.append({
            'Feature': 'Model Features',
            'Rule': 'Model uses multiple features to calculate risk score',
            'Score Contribution': 'Variable',
            'Risk Impact': 'Based on feature values'
        })
    
    return pd.DataFrame(explanation_data) 