#!/bin/bash

# Quick start script for Krisp Meeting Assistant

echo "🚀 Starting Krisp Meeting Assistant..."
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from template..."
    cp env.example .env
    echo "✅ Created .env file. Please edit it with your OpenAI API key!"
    echo ""
    echo "Edit .env file and add your OPENAI_API_KEY, then run this script again."
    exit 1
fi

# Check if OpenAI API key is set
if grep -q "your-openai-api-key-here" .env; then
    echo "❌ Please set your OPENAI_API_KEY in the .env file!"
    exit 1
fi

# Function to check if a service is running
check_service() {
    local service=$1
    local port=$2
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "✅ $service is running on port $port"
        return 0
    else
        echo "❌ $service is not running on port $port"
        return 1
    fi
}

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Start services with Docker Compose
echo "Starting services with Docker Compose..."
docker-compose up -d

# Wait for services to start
echo ""
echo "Waiting for services to start..."
sleep 10

# Check service status
echo ""
echo "Checking service status:"
check_service "Qdrant" 6333
check_service "Redis" 6379
check_service "Backend API" 8000
check_service "Streamlit" 8501

echo ""
echo "📊 Services Status:"
docker-compose ps

echo ""
echo "🎉 Krisp Meeting Assistant is ready!"
echo ""
echo "📌 Access the application at:"
echo "   - Streamlit UI: http://localhost:8501"
echo "   - Backend API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "📝 To ingest sample data, run:"
echo "   python scripts/ingest_data.py --source data/transcriptions --type transcript"
echo ""
echo "🛑 To stop all services, run:"
echo "   docker-compose down"
echo ""
