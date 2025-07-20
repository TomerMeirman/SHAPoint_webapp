@echo off
echo Starting SHAPoint WebApp...
call venv\Scripts\activate.bat
streamlit run shapoint_webapp\app.py
pause 