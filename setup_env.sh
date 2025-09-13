#!/bin/bash

# Setup script for creating and configuring the Python environment

echo "🔧 Setting up Python environment for Krisp Meeting Assistant..."
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install backend requirements
echo ""
echo "Installing backend dependencies..."
cd backend
pip install -r requirements.txt
cd ..

# Install frontend requirements
echo ""
echo "Installing frontend dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp env.example .env
    echo "✅ Created .env file"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your OPENAI_API_KEY"
else
    echo ""
    echo "✅ .env file already exists"
fi

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the application:"
echo "  1. With Docker: ./start.sh"
echo "  2. Without Docker:"
echo "     - Terminal 1: cd backend && python -m uvicorn main:app --reload"
echo "     - Terminal 2: streamlit run app.py"
echo ""
