-- =====================================================
-- Configuration initiale post-installation
-- =====================================================

-- Optimisation des index vectoriels
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET max_parallel_maintenance_workers = 4;

-- Configuration pour l'analyse des requêtes
SELECT pg_stat_statements_reset();

-- Création d'un utilisateur en lecture seule pour monitoring
CREATE ROLE mcp_readonly WITH LOGIN PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE mcp_rag TO mcp_readonly;
GRANT USAGE ON SCHEMA public TO mcp_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO mcp_readonly;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO mcp_readonly;

-- Configuration de la langue française pour le full-text search
CREATE TEXT SEARCH CONFIGURATION fr (COPY = french);

-- Log de l'installation réussie
INSERT INTO system_config (key, value, description) VALUES 
('database_initialized', 'true', 'Indique si la base a été initialisée correctement'),
('installation_timestamp', EXTRACT(EPOCH FROM NOW())::TEXT, 'Timestamp d''installation'),
('pgvector_version', (SELECT extversion FROM pg_extension WHERE extname = 'vector'), 'Version de l''extension pgvector');

-- Affichage des statistiques post-installation
DO $$
BEGIN
    RAISE NOTICE '=== MCP RAG Database Initialization Complete ===';
    RAISE NOTICE 'Extensions installed: %', 
        (SELECT string_agg(extname, ', ') FROM pg_extension WHERE extname IN ('vector', 'pg_trgm', 'uuid-ossp'));
    RAISE NOTICE 'Tables created: %', 
        (SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public');
    RAISE NOTICE 'Indexes created: %', 
        (SELECT count(*) FROM pg_indexes WHERE schemaname = 'public');
    RAISE NOTICE 'Functions created: %', 
        (SELECT count(*) FROM pg_proc WHERE pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public'));
END $$;