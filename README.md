# SHAPoint WebApp

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A **Streamlit web application** for interactive risk assessment using **SHAPoint** - an interpretable, task-agnostic risk scoring framework based on SHAP and decision trees.

## Overview

SHAPoint WebApp provides an intuitive web interface for the [SHAPoint](https://github.com/TomerMeirman/SHAPoint) package, enabling users to:

- **Build interpretable risk models** from healthcare or other tabular datasets
- **Visualize feature importance** and model explanations in real-time
- **Calculate individual risk scores** with population-based comparisons
- **Retrain models** with different parameters through the web interface
- **Export and reuse trained models** for production deployments

This webapp serves as both a **research tool** and **clinical decision support interface**, making advanced machine learning interpretability accessible to domain experts without programming experience.

**Important Note**: This webapp currently supports **Classification tasks only**. Support for Survival analysis and Regression tasks is planned for future releases.

## Relationship to SHAPoint

This package is a **web interface wrapper** for the core [SHAPoint](https://github.com/TomerMeirman/SHAPoint) library (v0.1.0). The webapp handles:

- **User Interface**: Streamlit-based interactive components
- **Visualization**: Plotly charts for risk assessment and feature analysis  
- **Configuration Management**: YAML-based settings with auto-detection
- **Model Persistence**: Saving and loading trained models
- **Real-time Updates**: Live risk calculation as inputs change

The core machine learning functionality is provided by **SHAPoint v0.1.0**.

## Quick Start

### Installation

```bash
# Install from PyPI (when available)
pip install shapoint-webapp

# Or install from source
git clone https://github.com/TomerMeirman/SHAPoint_webapp.git
cd SHAPoint_webapp
pip install -e .
```

### Dependencies

The webapp requires **SHAPoint v0.1.0** and will automatically install it:

```bash
# Core dependencies
shapoint>=0.1.0
streamlit>=1.46.0
plotly>=6.0.0
pyyaml>=6.0
pandas>=2.0.0
numpy>=1.20.0
```

### Running the Application

**Option 1: Using CLI (recommended after dev install)**
```bash
# Activate virtual environment & install in dev mode
venv\Scripts\activate  # Windows
pip install -e .

# Run the app with CLI
shapoint-webapp
```

**Option 2: Direct package run**
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Run the app
streamlit run shapoint_webapp/app.py
```

**Option 2: Using the provided scripts**
```bash
# Windows PowerShell
.\run_webapp.ps1

# Windows Command Prompt  
.\run_app.bat

# Or use the included Python launcher
python start_app.py
```

**Option 3: Using the command-line interface (after installation)**
```bash
shapoint-webapp
```

The application will open in your browser at `http://localhost:8501`.

## 📊 Features

### Core Functionality

- **🤖 Model Training**: Automated SHAPoint model training with Optuna optimization
- **📈 Risk Assessment**: Real-time individual risk scoring with percentile rankings
- **🔍 Model Interpretation**: Interactive feature importance and rule explanation
- **📊 Population Analytics**: Risk distribution analysis and bin-based statistics
- **⚙️ Dynamic Configuration**: Auto-detection of feature types and metadata
- **💾 Model Persistence**: Save and reload trained models

### User Interface

- **📱 Responsive Design**: Works on desktop and mobile devices
- **🎨 Clean Interface**: Intuitive layout with organized feature input forms
- **📋 Real-time Updates**: Instant risk calculation as inputs change
- **📈 Interactive Charts**: Plotly-based visualizations with hover details
- **⚡ Performance Optimized**: Caching for fast model predictions

### Advanced Features

- **🔄 Model Retraining**: Adjust parameters without leaving the interface
- **📋 Risk Bins Configuration**: Dynamic risk binning for population analysis
- **📊 Feature Selection**: Include/exclude features from model training
- **🎯 Rule-based Scoring**: Transparent score calculation based on decision rules

## 📋 Configuration

The webapp uses a YAML configuration file that automatically detects most settings:

```yaml
# config.yaml
model:
  k_variables: 5              # Number of top features
  score_scale: 10             # Maximum risk score
  risk_bins: 8                # Population analysis bins

data:
  target_column: "outcome"    # Prediction target
  exclude_columns: ["id"]     # Columns to ignore
  
  # Optional: Override auto-detection
  feature_metadata:
    age:
      label: "Patient Age"
      min_value: 18
      max_value: 100
```

Most feature properties (types, ranges, labels) are **automatically detected** from your dataset.

## 🏥 Use Cases

### Healthcare Applications
- **Cardiovascular Risk Assessment**: Predict heart disease risk from clinical factors
- **Cancer Prognosis**: Estimate survival probability from biomarkers
- **Treatment Response**: Predict therapy effectiveness from patient characteristics

### General Risk Modeling
- **Financial Risk**: Credit scoring and loan default prediction
- **Operational Risk**: Equipment failure and maintenance scheduling  
- **Security Risk**: Fraud detection and anomaly identification

## 📖 Example Usage

```python
# For programmatic access (optional)
from shapoint_webapp import ModelManager
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize model manager
manager = ModelManager(config)

# Train model
model = manager.load_or_train_model()

# Make predictions
risk_result = manager.predict_risk({
    'age': 65,
    'cholesterol': 2,
    'blood_pressure': 140
})

print(f"Risk Score: {risk_result['risk_score']}")
print(f"Risk Percentage: {risk_result['risk_percentage']}%")
```

## 🔬 Research & Citation

This webapp implements the methodology described in:

> **SHAPoint: Task‑Agnostic, Efficient, and Interpretable Point‑Based Risk Scoring via Shapley Values**  
> Tomer D. Meirman, Bracha Shapira, Noa Dagan, Lior Rokach  
> *[Paper details to be added upon publication]*

If you use SHAPoint WebApp in your research, please cite the core methodology.

## 🛠️ Development

### Setup Development Environment

```bash
git clone https://github.com/TomerMeirman/SHAPoint_webapp.git
cd SHAPoint_webapp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Run tests
pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

**Copyright 2025 Tomer D. Meirman and SHAPoint WebApp Contributors**

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## ⚠️ Disclaimer

This tool is intended for **educational and research purposes only**. For clinical or financial applications, always consult with domain experts and validate results thoroughly. The developers are not responsible for decisions made based on model outputs.

## 🔗 Links

- **Core Library**: [SHAPoint](https://github.com/TomerMeirman/SHAPoint) v0.1.0
- **Documentation**: [Coming Soon]
- **Issues**: [GitHub Issues](https://github.com/TomerMeirman/SHAPoint_webapp/issues)
- **PyPI Package**: [Coming Soon]

## 📞 Support

For questions about the webapp interface, please open an issue in this repository.  
For questions about the core SHAPoint methodology, please refer to the [main SHAPoint repository](https://github.com/TomerMeirman/SHAPoint).

---

**Made with ❤️ for interpretable machine learning** 
