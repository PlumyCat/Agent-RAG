-- Initialisation simple Better Chatbot + MCP RAG
CREATE EXTENSION IF NOT EXISTS vector;

-- Créer l'utilisateur MCP
CREATE USER mcp_service_user WITH PASSWORD 'secure_mcp_password_here';
GRANT ALL PRIVILEGES ON DATABASE better_chatbot TO mcp_service_user;

-- Tables MCP pour documents et embeddings
CREATE TABLE mcp_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50),
    content_hash VARCHAR(64) UNIQUE,
    original_content TEXT,
    status VARCHAR(50) DEFAULT 'processed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table pour chunks avec embeddings
CREATE TABLE mcp_document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES mcp_documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Permissions pour MCP user
GRANT ALL ON ALL TABLES IN SCHEMA public TO mcp_service_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO mcp_service_user;

-- Document de test
INSERT INTO mcp_documents (filename, file_type, content_hash, original_content) 
VALUES ('test.txt', 'text', 'test123', 'Document de test MCP-RAG');