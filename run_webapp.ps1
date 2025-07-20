# SHAPoint WebApp PowerShell Launcher
Write-Host "🚀 Starting SHAPoint WebApp..." -ForegroundColor Green

# Check if virtual environment exists
if (!(Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "❌ Virtual environment not found. Please run setup_environment.py first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Check if activation was successful
if ($env:VIRTUAL_ENV -eq $null) {
    Write-Host "❌ Failed to activate virtual environment." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✅ Virtual environment activated" -ForegroundColor Green

# Check if streamlit is available
try {
    $streamlitVersion = & streamlit --version 2>$null
    Write-Host "✅ Streamlit found: $streamlitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Streamlit not found. Installing..." -ForegroundColor Red
    pip install streamlit plotly
}

# Clear cache first
Write-Host "🧹 Clearing cache for fresh start..." -ForegroundColor Cyan
if (Test-Path "__pycache__") { Remove-Item -Recurse -Force "__pycache__" -ErrorAction SilentlyContinue }
if (Test-Path ".streamlit") { Remove-Item -Recurse -Force ".streamlit" -ErrorAction SilentlyContinue }
Write-Host "✅ Cache cleared" -ForegroundColor Green

# Start the webapp
Write-Host "🌐 Starting webapp at http://localhost:8501" -ForegroundColor Green
Write-Host "📊 Model will use 5 features with 3 max leaves" -ForegroundColor Cyan
Write-Host "🔄 Press Ctrl+C to stop the webapp" -ForegroundColor Yellow
Write-Host ""

# Run streamlit
streamlit run shapoint_webapp\app.py 