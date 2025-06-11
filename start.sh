#!/bin/bash

# Activate virtual environment (using relative path)
source venv/bin/activate

# Check for environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

# Add error handling
if ! command -v uvicorn &> /dev/null; then
    echo "Error: uvicorn is not installed"
    exit 1
fi

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
