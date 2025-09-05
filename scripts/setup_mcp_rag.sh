#!/bin/bash

# Script de configuration pour MCP-RAG avec Claude Desktop
# Usage: ./setup_mcp_rag.sh

echo "=== Configuration de MCP-RAG pour Claude Desktop ==="

# 1. Vérifier Python
echo "→ Vérification de Python..."
if ! command -v python &> /dev/null; then
    echo "❌ Python n'est pas installé. Installez Python 3.9+ d'abord."
    exit 1
fi

python_version=$(python --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "✅ Python $python_version détecté"

# 2. Créer l'environnement virtuel si nécessaire
if [ ! -d ".venv" ]; then
    echo "→ Création de l'environnement virtuel..."
    python -m venv .venv
    echo "✅ Environnement virtuel créé"
else
    echo "✅ Environnement virtuel existant détecté"
fi

# 3. Activer l'environnement virtuel
echo "→ Activation de l'environnement virtuel..."
source .venv/bin/activate

# 4. Installer les dépendances
echo "→ Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Créer le fichier .env si nécessaire
if [ ! -f ".env" ]; then
    echo "→ Création du fichier .env..."
    cp .env.example .env
    echo "⚠️  IMPORTANT: Configurez vos clés API dans le fichier .env"
else
    echo "✅ Fichier .env existant détecté"
fi

# 6. Vérifier PostgreSQL
echo "→ Vérification de PostgreSQL..."
if command -v psql &> /dev/null; then
    echo "✅ PostgreSQL est installé"
    echo "   Assurez-vous que:"
    echo "   - PostgreSQL est en cours d'exécution"
    echo "   - La base de données 'mcp_rag' existe"
    echo "   - L'extension pgvector est installée"
else
    echo "⚠️  PostgreSQL n'est pas détecté. Installation requise pour le stockage vectoriel."
fi

# 7. Vérifier Redis (optionnel)
echo "→ Vérification de Redis (optionnel)..."
if command -v redis-cli &> /dev/null; then
    echo "✅ Redis est installé (cache activé)"
else
    echo "ℹ️  Redis non détecté (le système fonctionnera sans cache)"
fi

# 8. Vérifier Tesseract (pour OCR)
echo "→ Vérification de Tesseract OCR..."
if command -v tesseract &> /dev/null; then
    echo "✅ Tesseract OCR est installé"
else
    echo "⚠️  Tesseract non détecté. Installation recommandée pour l'OCR:"
    echo "   Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-fra"
    echo "   macOS: brew install tesseract"
fi

echo ""
echo "=== Configuration terminée ==="
echo ""
echo "📋 Prochaines étapes:"
echo "1. Configurez vos clés API dans le fichier .env"
echo "2. Créez la base PostgreSQL avec pgvector:"
echo "   psql -U postgres -c 'CREATE DATABASE mcp_rag;'"
echo "   psql -U postgres -d mcp_rag -c 'CREATE EXTENSION IF NOT EXISTS vector;'"
echo "3. Copiez la configuration Claude Desktop:"
echo "   - Windows: %APPDATA%/Claude/claude_desktop_config.json"
echo "   - macOS: ~/Library/Application Support/Claude/claude_desktop_config.json"
echo "   - Linux: ~/.config/Claude/claude_desktop_config.json"
echo "4. Redémarrez Claude Desktop"
echo ""
echo "🚀 Pour tester le serveur MCP:"
echo "   python mcp_server.py"
echo ""
echo "📱 Pour lancer l'interface Streamlit:"
echo "   streamlit run streamlit_app.py"