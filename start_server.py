#!/usr/bin/env python3
"""
Startup script for the Math Routing Agent API.
This script sets up the Python path and starts the uvicorn server.
"""
import sys
import os

# Add the backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

# Load environment variables from backend/.env
from dotenv import load_dotenv
env_path = os.path.join(backend_path, '.env')
load_dotenv(env_path)

# Now import and run uvicorn
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
