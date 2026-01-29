# Claude Shell App - Startup Checklist

## Prerequisites
- [ ] Python 3 installed
- [ ] Virtual environment created (`python -m venv venv`)

## Configuration
- [ ] Copy `.env.example` to `.env`
- [ ] Set `APP_DATA_DIR` if non-default location needed
- [ ] Set `ANTHROPIC_API_KEY` if using Anthropic features

## Start the App
- [ ] Activate virtual environment (`.\venv\Scripts\Activate.ps1`)
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Run the app (`.\start.ps1` or `python -m uvicorn app:app --host 0.0.0.0 --port 8000`)

## Verify
- [ ] App is accessible at http://localhost:8000
- [ ] No errors in console output
