-- =====================================================
-- Initialisation MCP RAG dans Better Chatbot Database
-- =====================================================

-- Tables pour le stockage des documents et embeddings
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

-- Index pour améliorer les performances
CREATE INDEX IF NOT EXISTS idx_mcp_documents_filename ON mcp_documents(filename);
CREATE INDEX IF NOT EXISTS idx_mcp_documents_file_type ON mcp_documents(file_type);
CREATE INDEX IF NOT EXISTS idx_mcp_documents_status ON mcp_documents(status);
CREATE INDEX IF NOT EXISTS idx_mcp_documents_upload_timestamp ON mcp_documents(upload_timestamp);

-- Table pour les chunks de documents avec embeddings
CREATE TABLE IF NOT EXISTS mcp_document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES mcp_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64),
    -- Stockage des embeddings en JSONB (en attendant pgvector)
    embedding JSONB,
    embedding_model VARCHAR(100) DEFAULT 'text-embedding-3-small',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index pour les chunks
CREATE INDEX IF NOT EXISTS idx_mcp_document_chunks_document_id ON mcp_document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_mcp_document_chunks_chunk_index ON mcp_document_chunks(chunk_index);
CREATE INDEX IF NOT EXISTS idx_mcp_document_chunks_embedding_model ON mcp_document_chunks(embedding_model);

-- Table pour les requêtes et leurs résultats (cache sémantique)
CREATE TABLE IF NOT EXISTS mcp_semantic_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    query_hash VARCHAR(64) UNIQUE,
    query_embedding JSONB,
    results JSONB,
    hit_count INTEGER DEFAULT 1,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index pour le cache sémantique
CREATE INDEX IF NOT EXISTS idx_mcp_semantic_cache_query_hash ON mcp_semantic_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_mcp_semantic_cache_last_accessed ON mcp_semantic_cache(last_accessed);

-- Table pour les statistiques et métriques
CREATE TABLE IF NOT EXISTS mcp_statistics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operation_type VARCHAR(50) NOT NULL,
    operation_data JSONB DEFAULT '{}',
    execution_time_ms INTEGER,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index pour les statistiques
CREATE INDEX IF NOT EXISTS idx_mcp_statistics_operation_type ON mcp_statistics(operation_type);
CREATE INDEX IF NOT EXISTS idx_mcp_statistics_timestamp ON mcp_statistics(timestamp);
CREATE INDEX IF NOT EXISTS idx_mcp_statistics_success ON mcp_statistics(success);

-- Fonction pour calculer la similarité cosinus (simulation sans pgvector)
CREATE OR REPLACE FUNCTION cosine_similarity(a JSONB, b JSONB) 
RETURNS FLOAT AS $$
DECLARE
    dot_product FLOAT := 0;
    norm_a FLOAT := 0;
    norm_b FLOAT := 0;
    i INTEGER;
    val_a FLOAT;
    val_b FLOAT;
BEGIN
    -- Calcul approximatif de similarité cosinus sur les premiers éléments
    FOR i IN 0..LEAST(jsonb_array_length(a), jsonb_array_length(b), 100) - 1 LOOP
        val_a := (a->>i)::FLOAT;
        val_b := (b->>i)::FLOAT;
        dot_product := dot_product + (val_a * val_b);
        norm_a := norm_a + (val_a * val_a);
        norm_b := norm_b + (val_b * val_b);
    END LOOP;
    
    IF norm_a = 0 OR norm_b = 0 THEN
        RETURN 0;
    END IF;
    
    RETURN dot_product / (sqrt(norm_a) * sqrt(norm_b));
END;
$$ LANGUAGE plpgsql;

-- Vue pour avoir un résumé des documents
CREATE OR REPLACE VIEW mcp_documents_summary AS
SELECT 
    COUNT(*) as total_documents,
    COUNT(DISTINCT file_type) as unique_file_types,
    SUM(file_size) as total_size_bytes,
    AVG(file_size) as avg_size_bytes,
    MAX(upload_timestamp) as last_upload,
    COUNT(CASE WHEN status = 'processed' THEN 1 END) as processed_documents,
    COUNT(CASE WHEN status = 'error' THEN 1 END) as error_documents
FROM mcp_documents;

-- Permissions pour l'utilisateur MCP
GRANT SELECT, INSERT, UPDATE, DELETE ON mcp_documents TO mcp_service_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON mcp_document_chunks TO mcp_service_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON mcp_semantic_cache TO mcp_service_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON mcp_statistics TO mcp_service_user;
GRANT SELECT ON mcp_documents_summary TO mcp_service_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO mcp_service_user;

-- Trigger pour mise à jour automatique des timestamps
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

-- Insérer des données de test
INSERT INTO mcp_documents (filename, file_type, file_size, content_hash, original_content, processed_content, status) 
VALUES 
    ('test_document.txt', 'text', 1024, md5('test content'), 'Test document content', 'Test document content', 'processed'),
    ('sample.pdf', 'pdf', 2048, md5('sample pdf content'), 'Sample PDF content', 'Sample PDF content', 'processed')
ON CONFLICT (content_hash) DO NOTHING;

-- Insérer des chunks de test
INSERT INTO mcp_document_chunks (document_id, chunk_index, content, content_hash, embedding, embedding_model)
SELECT 
    d.id,
    0,
    'First chunk of ' || d.filename,
    md5('First chunk of ' || d.filename),
    '[]'::jsonb,
    'text-embedding-3-small'
FROM mcp_documents d
WHERE NOT EXISTS (
    SELECT 1 FROM mcp_document_chunks c WHERE c.document_id = d.id
);

COMMIT;