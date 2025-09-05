-- =====================================================
-- Initialisation complete Better Chatbot + MCP RAG
-- =====================================================

-- Créer l'extension pgvector
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Créer l'utilisateur MCP avec permissions limitées
CREATE USER mcp_service_user WITH PASSWORD 'secure_mcp_password_here';
GRANT CONNECT ON DATABASE better_chatbot TO mcp_service_user;
GRANT USAGE ON SCHEMA public TO mcp_service_user;

-- Tables MCP RAG pour stockage de documents et embeddings
CREATE TABLE IF NOT EXISTS mcp_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50),
    file_size BIGINT,
    content_hash VARCHAR(64) UNIQUE,
    original_content TEXT,
    processed_content TEXT,
    metadata JSONB DEFAULT '{}',
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'processed',
    error_message TEXT
);

-- Table pour les chunks avec embeddings vectoriels
CREATE TABLE IF NOT EXISTS mcp_document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES mcp_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64),
    embedding vector(1536),  -- Dimension pour text-embedding-3-small
    embedding_model VARCHAR(100) DEFAULT 'text-embedding-3-small',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table pour le cache sémantique
CREATE TABLE IF NOT EXISTS mcp_semantic_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    query_hash VARCHAR(64) UNIQUE,
    query_embedding vector(1536),
    results JSONB,
    hit_count INTEGER DEFAULT 1,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index pour performance
CREATE INDEX IF NOT EXISTS idx_mcp_documents_filename ON mcp_documents(filename);
CREATE INDEX IF NOT EXISTS idx_mcp_documents_status ON mcp_documents(status);
CREATE INDEX IF NOT EXISTS idx_mcp_document_chunks_document_id ON mcp_document_chunks(document_id);

-- Index vectoriel pour recherche sémantique
CREATE INDEX IF NOT EXISTS idx_mcp_document_chunks_embedding ON mcp_document_chunks 
USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_mcp_semantic_cache_embedding ON mcp_semantic_cache 
USING hnsw (query_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Permissions pour l'utilisateur MCP
GRANT SELECT, INSERT, UPDATE, DELETE ON mcp_documents TO mcp_service_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON mcp_document_chunks TO mcp_service_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON mcp_semantic_cache TO mcp_service_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO mcp_service_user;

-- Fonction utilitaire pour recherche sémantique
CREATE OR REPLACE FUNCTION search_similar_documents(
    query_embedding vector(1536),
    similarity_threshold float DEFAULT 0.7,
    max_results int DEFAULT 10
)
RETURNS TABLE(
    document_id uuid,
    chunk_id uuid,
    content text,
    similarity float,
    filename varchar(255),
    metadata jsonb
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.document_id,
        c.id as chunk_id,
        c.content,
        1 - (c.embedding <=> query_embedding) as similarity,
        d.filename,
        c.metadata
    FROM mcp_document_chunks c
    JOIN mcp_documents d ON c.document_id = d.id
    WHERE 1 - (c.embedding <=> query_embedding) > similarity_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Accorder permission sur la fonction
GRANT EXECUTE ON FUNCTION search_similar_documents(vector, float, int) TO mcp_service_user;

-- Données de test
INSERT INTO mcp_documents (filename, file_type, content_hash, original_content, status) 
VALUES ('readme.txt', 'text', md5('Welcome to MCP RAG'), 'Welcome to MCP RAG system', 'processed')
ON CONFLICT (content_hash) DO NOTHING;

-- Trigger pour mise à jour automatique
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_modified = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_mcp_documents_modtime 
    BEFORE UPDATE ON mcp_documents 
    FOR EACH ROW 
    EXECUTE FUNCTION update_modified_column();

COMMIT;