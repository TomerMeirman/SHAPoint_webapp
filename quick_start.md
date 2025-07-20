# Quick Start Guide

## ✅ Simplest Way to Run the App

### **For Development Mode (Recommended):**

1. **Activate virtual environment & install in dev mode:**
   ```bash
   # Windows
   venv\Scripts\activate
   pip install -e .
   ```

2. **Run the app using CLI:**
   ```bash
   shapoint-webapp
   ```

### **Alternative (Direct run):**
1. **Activate the virtual environment:**
   ```bash
   venv\Scripts\activate
   ```

2. **Run the app:**
   ```bash
   streamlit run shapoint_webapp/app.py
   ```

3. **Open your browser** to `http://localhost:8501`

## 🚀 Alternative Methods

### Using provided scripts:
- **PowerShell:** `.\run_webapp.ps1`
- **Command Prompt:** `.\run_app.bat`
- **Python launcher:** `python start_app.py`

### Using CLI (after pip install):
```bash
shapoint-webapp
```

## 🔧 Troubleshooting

If you get import errors, make sure you're in the virtual environment:
```bash
# Check if virtual environment is active
echo $env:VIRTUAL_ENV  # PowerShell
echo %VIRTUAL_ENV%     # Command Prompt
```

If you see module import errors, try:
```bash
pip install -e .
``` 