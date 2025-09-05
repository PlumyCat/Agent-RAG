-- =====================================================
-- MCP RAG Advanced - PostgreSQL Schema avec pgvector
-- =====================================================

-- Extensions nécessaires
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- =====================================================
-- Tables principales
-- =====================================================

-- Sessions de conversation Claude Code
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id TEXT NOT NULL,
    title TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    -- Colonne d'extension pour intégration avec projet existant
    user_id UUID,
    -- Index
    CONSTRAINT idx_conversations_session UNIQUE(session_id)
);

CREATE INDEX idx_conversations_created ON conversations(created_at DESC);
CREATE INDEX idx_conversations_updated ON conversations(updated_at DESC);
CREATE INDEX idx_conversations_user ON conversations(user_id) WHERE user_id IS NOT NULL;

-- Documents ingérés par conversation
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL,
    filename TEXT NOT NULL,
    original_path TEXT,
    content_hash VARCHAR(64) NOT NULL, -- SHA-256 du contenu
    file_type VARCHAR(50) NOT NULL,
    file_size BIGINT NOT NULL,
    mime_type TEXT,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    full_text TEXT, -- Texte complet pour recherche full-text
    summary TEXT, -- Résumé automatique
    -- Métadonnées d'extraction
    extraction_method VARCHAR(100), -- 'standard', 'ocr', 'hybrid'
    processing_time_ms INTEGER,
    chunk_count INTEGER DEFAULT 0,
    
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

-- Index pour les documents
CREATE INDEX idx_documents_conversation ON documents(conversation_id);
CREATE INDEX idx_documents_hash ON documents(content_hash);
CREATE INDEX idx_documents_type ON documents(file_type);
CREATE INDEX idx_documents_processed ON documents(processed_at DESC);

-- Index full-text search avec support français
CREATE INDEX idx_documents_fulltext ON documents USING GIN (to_tsvector('french', COALESCE(full_text, '')));
CREATE INDEX idx_documents_summary ON documents USING GIN (to_tsvector('french', COALESCE(summary, '')));

-- Chunks de documents avec embeddings vectoriels
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL,
    content TEXT NOT NULL,
    -- Embeddings vectoriels (dimension configurable)
    embedding VECTOR(1536), -- OpenAI text-embedding-3-small par défaut
    -- Position et métadonnées
    chunk_index INTEGER NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    chunk_type VARCHAR(50) DEFAULT 'text', -- 'text', 'code', 'table', 'image'
    -- Scoring et qualité
    extraction_confidence REAL DEFAULT 1.0,
    semantic_density REAL, -- Densité sémantique du chunk
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Index pour les chunks
CREATE INDEX idx_chunks_document ON document_chunks(document_id);
CREATE INDEX idx_chunks_index ON document_chunks(document_id, chunk_index);
CREATE INDEX idx_chunks_type ON document_chunks(chunk_type);

-- Index vectoriel HNSW pour recherche de similarité
CREATE INDEX idx_chunks_embedding ON document_chunks 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- =====================================================
-- Knowledge Graph - Entités et Relations
-- =====================================================

-- Entités extraites des documents
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL, -- Version normalisée pour matching
    entity_type VARCHAR(100) NOT NULL, -- PERSON, ORG, CONCEPT, LOCATION, etc.
    description TEXT,
    confidence REAL DEFAULT 1.0,
    -- Embedding de l'entité
    embedding VECTOR(1536),
    -- Métadonnées
    aliases TEXT[], -- Aliases/synonymes
    properties JSONB DEFAULT '{}',
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    occurrence_count INTEGER DEFAULT 1,
    
    CONSTRAINT unique_entity_name_type UNIQUE(normalized_name, entity_type)
);

CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_entities_name_gin ON entities USING GIN (normalized_name gin_trgm_ops);
CREATE INDEX idx_entities_embedding ON entities USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_entities_confidence ON entities(confidence DESC);

-- Relations entre entités
CREATE TABLE relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_entity_id UUID NOT NULL,
    target_entity_id UUID NOT NULL,
    relationship_type VARCHAR(100) NOT NULL, -- "works_at", "related_to", "mentions", etc.
    confidence REAL DEFAULT 1.0,
    context_text TEXT, -- Contexte où la relation a été trouvée
    source_chunk_id UUID, -- Chunk source de la relation
    -- Métadonnées
    extraction_method VARCHAR(50), -- 'rule_based', 'ml_model', 'llm'
    properties JSONB DEFAULT '{}',
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    occurrence_count INTEGER DEFAULT 1,
    
    FOREIGN KEY (source_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
    FOREIGN KEY (target_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
    FOREIGN KEY (source_chunk_id) REFERENCES document_chunks(id) ON DELETE SET NULL,
    
    CONSTRAINT unique_relationship UNIQUE(source_entity_id, target_entity_id, relationship_type)
);

CREATE INDEX idx_relationships_source ON relationships(source_entity_id);
CREATE INDEX idx_relationships_target ON relationships(target_entity_id);
CREATE INDEX idx_relationships_type ON relationships(relationship_type);
CREATE INDEX idx_relationships_confidence ON relationships(confidence DESC);

-- =====================================================
-- Cache et Optimisations
-- =====================================================

-- Cache sémantique des requêtes fréquentes
CREATE TABLE semantic_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_hash VARCHAR(64) NOT NULL, -- Hash de la requête normalisée
    query_text TEXT NOT NULL,
    query_embedding VECTOR(1536),
    conversation_id UUID,
    -- Résultats cachés
    results JSONB NOT NULL,
    result_count INTEGER,
    -- Statistiques
    hit_count INTEGER DEFAULT 1,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE, -- TTL pour invalidation
    
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    
    CONSTRAINT unique_query_hash UNIQUE(query_hash)
);

CREATE INDEX idx_cache_hash ON semantic_cache(query_hash);
CREATE INDEX idx_cache_conversation ON semantic_cache(conversation_id);
CREATE INDEX idx_cache_embedding ON semantic_cache USING hnsw (query_embedding vector_cosine_ops);
CREATE INDEX idx_cache_expires ON semantic_cache(expires_at) WHERE expires_at IS NOT NULL;

-- =====================================================
-- Tables d'audit et monitoring
-- =====================================================

-- Log des recherches pour analytics
CREATE TABLE search_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID,
    query_text TEXT NOT NULL,
    query_type VARCHAR(50), -- 'vector', 'fulltext', 'hybrid', 'graph'
    results_count INTEGER,
    execution_time_ms INTEGER,
    embedding_time_ms INTEGER,
    search_params JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE SET NULL
);

CREATE INDEX idx_search_logs_conversation ON search_logs(conversation_id);
CREATE INDEX idx_search_logs_timestamp ON search_logs(timestamp DESC);
CREATE INDEX idx_search_logs_type ON search_logs(query_type);

-- =====================================================
-- Fonctions utilitaires
-- =====================================================

-- Fonction pour mettre à jour le timestamp updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers pour updated_at
CREATE TRIGGER update_conversations_updated_at 
    BEFORE UPDATE ON conversations 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Fonction de recherche vectorielle optimisée
CREATE OR REPLACE FUNCTION vector_search(
    query_embedding VECTOR(1536),
    conversation_id_filter UUID DEFAULT NULL,
    similarity_threshold REAL DEFAULT 0.7,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    filename TEXT,
    content TEXT,
    similarity REAL,
    chunk_index INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        dc.id as chunk_id,
        dc.document_id,
        d.filename,
        dc.content,
        (1 - (dc.embedding <=> query_embedding)) as similarity,
        dc.chunk_index
    FROM document_chunks dc
    JOIN documents d ON dc.document_id = d.id
    WHERE 
        (conversation_id_filter IS NULL OR d.conversation_id = conversation_id_filter)
        AND (1 - (dc.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY dc.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- Vues utiles
-- =====================================================

-- Vue pour les statistiques par conversation
CREATE VIEW conversation_stats AS
SELECT 
    c.id,
    c.session_id,
    c.title,
    COUNT(DISTINCT d.id) as document_count,
    COUNT(dc.id) as chunk_count,
    SUM(d.file_size) as total_file_size,
    MAX(d.processed_at) as last_document_added,
    c.created_at,
    c.updated_at
FROM conversations c
LEFT JOIN documents d ON c.id = d.conversation_id
LEFT JOIN document_chunks dc ON d.id = dc.document_id
GROUP BY c.id, c.session_id, c.title, c.created_at, c.updated_at;

-- Vue pour les entités les plus fréquentes
CREATE VIEW entity_popularity AS
SELECT 
    e.name,
    e.entity_type,
    e.occurrence_count,
    COUNT(r1.id) + COUNT(r2.id) as relationship_count,
    e.confidence,
    e.last_seen_at
FROM entities e
LEFT JOIN relationships r1 ON e.id = r1.source_entity_id
LEFT JOIN relationships r2 ON e.id = r2.target_entity_id
GROUP BY e.id, e.name, e.entity_type, e.occurrence_count, e.confidence, e.last_seen_at
ORDER BY e.occurrence_count DESC, relationship_count DESC;

-- =====================================================
-- Configuration et paramètres
-- =====================================================

-- Table de configuration du système
CREATE TABLE system_config (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Configuration par défaut
INSERT INTO system_config (key, value, description) VALUES
('embedding_model', 'text-embedding-3-small', 'Modèle d''embedding utilisé'),
('embedding_dimension', '1536', 'Dimension des vecteurs d''embedding'),
('chunk_size', '500', 'Taille des chunks en caractères'),
('chunk_overlap', '100', 'Chevauchement entre chunks'),
('similarity_threshold', '0.7', 'Seuil de similarité pour les recherches'),
('max_search_results', '10', 'Nombre maximum de résultats par recherche'),
('cache_ttl_hours', '24', 'TTL du cache sémantique en heures'),
('enable_entity_extraction', 'true', 'Activer l''extraction d''entités'),
('search_language', 'french', 'Langue pour la recherche full-text');