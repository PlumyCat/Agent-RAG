
# 📚 MCP-RAG 

MCP-RAG system built with the Model Context Protocol (MCP) that handles large files (up to 200MB) using intelligent chunking strategies, multi-format document support, and enterprise-grade reliability.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://github.com/modelcontextprotocol)

## 🌟 Features

### 📄 **Multi-Format Document Support**
- **PDF**: Intelligent page-by-page processing with table detection
- **DOCX**: Paragraph and table extraction with formatting preservation  
- **Excel**: Sheet-aware processing with column context (.xlsx/.xls)
- **CSV**: Smart row batching with header preservation
- **PPTX**: Support for PPTX
- **IMAGE**: Suppport for jpeg , png , webp , gif etc and OCR

### 🚀 **Large File Processing**
- **Adaptive chunking**: Different strategies based on file size
- **Memory management**: Streaming processing for 50MB+ files
- **Progress tracking**: Real-time progress indicators
- **Timeout handling**: Graceful handling of long-running operations

### 🧠 **Advanced RAG Capabilities**
- **Semantic search**: Vector similarity with confidence scores
- **Cross-document queries**: Search across multiple documents simultaneously
- **Source attribution**: Citations with similarity scores
- **Hybrid retrieval**: Combine semantic and keyword search

### 🔌 **Model Context Protocol (MCP) Integration**
- **Universal tool interface**: Standardized AI-to-tool communication
- **Auto-discovery**: LangChain agents automatically find and use tools
- **Secure communication**: Built-in permission controls
- **Extensible architecture**: Easy to add new document processors

### 🏢 **Enterprise Ready**
- **Azure OpenAI Support**: Full integration with Azure OpenAI deployments
- **PostgreSQL + pgvector**: Production-ready vector database with advanced search
- **Redis Caching**: Semantic query caching for improved performance
- **Batch processing**: Handles API rate limits and batch size constraints
- **Error recovery**: Retry logic and graceful degradation

## 🏗️ Architecture

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   LangChain      │    │   MCP Server    │
│   Frontend      │◄──►│   Agent          │◄──►│   (Tools)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
│
┌────────────────────────┼────────────────────────┐
│                        ▼                        │
┌───────▼────────┐    ┌─────────────────┐    ┌──────▼──────┐
│ Document       │    │ PostgreSQL      │    │ Azure OpenAI│
│ Processors     │    │ + pgvector      │    │ or OpenAI   │
└────────────────┘    └─────────────────┘    └─────────────┘

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Azure OpenAI or OpenAI API key
- PostgreSQL with pgvector extension
- Redis (optional, for caching)
- 8GB+ RAM (for large file processing)

### Installation
**Clone the repository**
```bash
git clone https://github.com/yourusername/rag-large-file-processor.git
cd rag-large-file-processor

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

# Create .env file
cat > .env << EOF
# Azure OpenAI (recommended)
AZURE_OPENAI_API_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1-mini

# OR Standard OpenAI
# OPENAI_API_KEY=your_openai_api_key_here

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mcp_rag
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Redis (optional)
REDIS_HOST=localhost
REDIS_PORT=6379


streamlit run streamlit_app.py
