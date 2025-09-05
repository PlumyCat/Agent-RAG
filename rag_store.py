"""
PostgreSQL + pgvector RAG Document Store
Advanced RAG implementation with Knowledge Graph and hybrid search
"""

import asyncio
import asyncpg
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import hashlib
import logging
from contextlib import asynccontextmanager

from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import spacy

from config import Config
import redis.asyncio as redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostgreSQLRAGStore:
    """Advanced PostgreSQL + pgvector RAG Store with Knowledge Graph"""
    
    def __init__(self, conversation_id: Optional[str] = None):
        self.config = Config()
        self.conversation_id = conversation_id
        self.connection_pool = None
        self.redis_client = None
        self.embedding_model = None
        self.text_splitter = None
        self.nlp_model = None
        self.embedding_dim = self.config.EMBEDDING_DIMENSION
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure async components are initialized"""
        if not self._initialized:
            await self._initialize_async_components()
            self._initialized = True
    
    async def _initialize_async_components(self):
        """Initialize async components"""
        await self._initialize_database_pool()
        await self._initialize_redis()
        await self._initialize_embeddings()
        await self._initialize_nlp()
        self._initialize_text_splitter()
        
        # Ensure conversation exists if conversation_id is set
        if self.conversation_id and self.conversation_id != "None":
            await self._ensure_conversation_exists()
        logger.info("PostgreSQL RAG Store initialized successfully")
    
    async def _ensure_conversation_exists(self):
        """Ensure the conversation exists in the database"""
        try:
            async with self.get_connection() as conn:
                # Check if conversation exists
                existing = await conn.fetchrow(
                    "SELECT id FROM conversations WHERE id = $1",
                    self.conversation_id
                )
                
                if not existing:
                    # Create new conversation
                    await conn.execute("""
                        INSERT INTO conversations (id, session_id, title, metadata)
                        VALUES ($1, $2, $3, $4)
                    """, self.conversation_id, self.conversation_id, "Default Session", json.dumps({}))
                    logger.info(f"Created default conversation: {self.conversation_id}")
        except Exception as e:
            logger.error(f"Failed to ensure conversation exists: {e}")
            # Don't raise - this is not critical
    
    async def _initialize_database_pool(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.connection_pool = await asyncpg.create_pool(
                host=self.config.POSTGRES_HOST,
                port=self.config.POSTGRES_PORT,
                user=self.config.POSTGRES_USER,
                password=self.config.POSTGRES_PASSWORD,
                database=self.config.POSTGRES_DB,
                min_size=self.config.POSTGRES_MIN_CONNECTIONS,
                max_size=self.config.POSTGRES_MAX_CONNECTIONS,
                command_timeout=self.config.POSTGRES_CONNECTION_TIMEOUT,
                server_settings={
                    'search_path': self.config.POSTGRES_SCHEMA,
                    'application_name': 'mcp_rag_store'
                }
            )
            
            # Database pool created (pgvector not available, using JSONB for embeddings)
            logger.info(f"Database pool created with {self.config.POSTGRES_MAX_CONNECTIONS} connections")
            logger.info("Using JSONB storage for embeddings (pgvector extension not available)")
            
            # Create cosine similarity function for JSONB arrays
            async with self.connection_pool.acquire() as conn:
                await conn.execute("""
                    CREATE OR REPLACE FUNCTION cosine_similarity(a JSONB, b JSONB)
                    RETURNS FLOAT AS $$
                    DECLARE
                        dot_product FLOAT := 0;
                        norm_a FLOAT := 0;
                        norm_b FLOAT := 0;
                        i INTEGER;
                        val_a FLOAT;
                        val_b FLOAT;
                        len_a INTEGER;
                        len_b INTEGER;
                    BEGIN
                        len_a := jsonb_array_length(a);
                        len_b := jsonb_array_length(b);
                        
                        IF len_a != len_b OR len_a = 0 THEN
                            RETURN 0;
                        END IF;
                        
                        FOR i IN 0..len_a-1 LOOP
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
                    $$ LANGUAGE plpgsql IMMUTABLE;
                """)
                
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    async def _initialize_redis(self):
        """Initialize Redis cache client"""
        if not self.config.REDIS_HOST:
            logger.warning("Redis not configured, caching disabled")
            return
            
        try:
            self.redis_client = redis.from_url(
                self.config.REDIS_URL,
                decode_responses=True,
                max_connections=10
            )
            await self.redis_client.ping()
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.warning(f"Redis initialization failed, caching disabled: {e}")
            self.redis_client = None
    
    async def _initialize_embeddings(self):
        """Initialize embedding model"""
        try:
            # Initialize both models to None first
            self.embedding_model = None
            self.sentence_model = None
            
            if self.config.USE_AZURE_OPENAI:
                self.embedding_model = AzureOpenAIEmbeddings(
                    api_key=self.config.AZURE_OPENAI_API_KEY,
                    azure_endpoint=self.config.AZURE_OPENAI_ENDPOINT,
                    api_version=self.config.AZURE_OPENAI_API_VERSION,
                    azure_deployment=self.config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                    chunk_size=self.config.EMBEDDING_BATCH_SIZE
                )
                logger.info(f"Using Azure OpenAI embeddings: {self.config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT}")
            elif self.config.OPENAI_API_KEY:
                self.embedding_model = OpenAIEmbeddings(
                    api_key=self.config.OPENAI_API_KEY,
                    model=self.config.EMBEDDING_MODEL,
                    chunk_size=self.config.EMBEDDING_BATCH_SIZE
                )
                logger.info(f"Using OpenAI embeddings: {self.config.EMBEDDING_MODEL}")
            else:
                # Fallback to local model
                from sentence_transformers import SentenceTransformer
                self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
                logger.info("Using local SentenceTransformer embeddings")
                
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    async def _initialize_nlp(self):
        """Initialize spaCy NLP model for entity extraction"""
        if not self.config.ENABLE_ENTITY_EXTRACTION:
            return
            
        try:
            self.nlp_model = spacy.load(self.config.SPACY_MODEL)
            logger.info(f"NLP model loaded: {self.config.SPACY_MODEL}")
        except IOError:
            logger.warning(f"spaCy model {self.config.SPACY_MODEL} not found, entity extraction disabled")
            self.nlp_model = None
    
    def _initialize_text_splitter(self):
        """Initialize text splitter"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        async with self.connection_pool.acquire() as conn:
            yield conn
    
    async def create_conversation(self, session_id: str, title: Optional[str] = None, 
                                metadata: Dict[str, Any] = None) -> str:
        """Create or get conversation"""
        try:
            async with self.get_connection() as conn:
                # Try to get existing conversation
                existing = await conn.fetchrow(
                    "SELECT id FROM conversations WHERE session_id = $1",
                    session_id
                )
                
                if existing:
                    return str(existing['id'])
                
                # Create new conversation
                conversation_id = str(uuid.uuid4())
                await conn.execute("""
                    INSERT INTO conversations (id, session_id, title, metadata)
                    VALUES ($1, $2, $3, $4)
                """, conversation_id, session_id, title, json.dumps(metadata or {}))
                
                self.conversation_id = conversation_id
                logger.info(f"Created conversation: {conversation_id}")
                return conversation_id
                
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise
    
    async def add_document(self, filename: str, file_type: str, content: str,
                         metadata: Dict[str, Any] = None, original_path: str = None) -> Dict[str, Any]:
        """Add document with embeddings and entity extraction"""
        await self._ensure_initialized()
        try:
            # Generate content hash for deduplication
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            async with self.get_connection() as conn:
                # Check if document already exists
                existing = await conn.fetchrow(
                    "SELECT id FROM documents WHERE content_hash = $1 AND conversation_id = $2",
                    content_hash, self.conversation_id
                )
                
                if existing:
                    logger.info(f"Document {filename} already exists")
                    return {"doc_id": str(existing['id']), "status": "exists"}
                
                # Create document record
                doc_id = str(uuid.uuid4())
                processing_start = datetime.now()
                
                # Split content into chunks
                chunks = self.text_splitter.split_text(content)
                logger.info(f"Created {len(chunks)} chunks for {filename}")
                
                # Create embeddings for chunks
                if self.embedding_model:
                    embeddings = await self.embedding_model.aembed_documents(chunks)
                elif self.sentence_model:
                    embeddings = self.sentence_model.encode(chunks).tolist()
                else:
                    raise Exception("No embedding model available")
                
                # Begin transaction
                async with conn.transaction():
                    # Insert document
                    file_size_mb = len(content.encode()) / (1024 * 1024)  # Convert bytes to MB
                    processing_time_seconds = (datetime.now() - processing_start).total_seconds()
                    
                    await conn.execute("""
                        INSERT INTO documents (
                            id, conversation_id, filename, original_path, content_hash,
                            file_type, file_size_mb, metadata, processing_time_seconds,
                            total_chunks, processing_status
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """, 
                    doc_id, self.conversation_id, filename, original_path, content_hash,
                    file_type, file_size_mb, json.dumps(metadata or {}), processing_time_seconds,
                    len(chunks), "completed")
                    
                    # Insert chunks with embeddings
                    chunk_data = []
                    embedding_model_name = self.config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or self.config.EMBEDDING_MODEL or "unknown"
                    
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        chunk_id = str(uuid.uuid4())
                        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
                        token_count = max(1, len(chunk) // 4)
                        
                        chunk_data.append((
                            chunk_id,                           # id
                            doc_id,                            # document_id  
                            self.conversation_id,              # conversation_id
                            chunk,                             # content
                            json.dumps(embedding),             # embedding as JSONB
                            i,                                 # chunk_index
                            len(chunk),                        # char_count
                            token_count,                       # token_count
                            embedding_model_name,              # embedding_model
                            json.dumps({})                     # chunk_metadata
                        ))
                    
                    await conn.executemany("""
                        INSERT INTO document_chunks (
                            id, document_id, conversation_id, content, embedding, 
                            chunk_index, char_count, token_count, embedding_model, chunk_metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """, chunk_data)
                
                # Extract entities if enabled
                if self.config.ENABLE_ENTITY_EXTRACTION and self.nlp_model:
                    await self._extract_entities_from_document(doc_id, content, conn)
                
                processing_time = int((datetime.now() - processing_start).total_seconds() * 1000)
                logger.info(f"Document {filename} processed in {processing_time}ms with {len(chunks)} chunks")
                
                return {
                    "doc_id": doc_id,
                    "chunks_created": len(chunks),
                    "processing_time_ms": processing_time,
                    "status": "success"
                }
                
        except Exception as e:
            logger.error(f"Failed to add document {filename}: {e}")
            return {"error": str(e)}
    
    async def _extract_entities_from_document(self, doc_id: str, content: str, conn):
        """Extract entities and relationships from document content"""
        try:
            doc = self.nlp_model(content)
            entities_found = {}
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                    entity_name = ent.text.strip()
                    if len(entity_name) > 2:  # Filter out very short entities
                        if entity_name not in entities_found:
                            entities_found[entity_name] = {
                                'type': ent.label_,
                                'confidence': 0.9,  # spaCy default
                                'mentions': []
                            }
                        entities_found[entity_name]['mentions'].append(ent.start_char)
            
            # Store entities
            for entity_name, entity_data in entities_found.items():
                await self._store_entity(entity_name, entity_data['type'], 
                                       entity_data['confidence'], conn)
            
            logger.info(f"Extracted {len(entities_found)} entities from document {doc_id}")
            
        except Exception as e:
            logger.error(f"Entity extraction failed for document {doc_id}: {e}")
    
    async def _store_entity(self, name: str, entity_type: str, confidence: float, conn):
        """Store or update entity in database"""
        try:
            normalized_name = name.lower().strip()
            
            # Try to insert or update entity
            await conn.execute("""
                INSERT INTO entities (name, normalized_name, entity_type, confidence, occurrence_count)
                VALUES ($1, $2, $3, $4, 1)
                ON CONFLICT (normalized_name, entity_type) 
                DO UPDATE SET 
                    occurrence_count = entities.occurrence_count + 1,
                    last_seen_at = CURRENT_TIMESTAMP,
                    confidence = GREATEST(entities.confidence, $4)
            """, name, normalized_name, entity_type, confidence)
            
        except Exception as e:
            logger.error(f"Failed to store entity {name}: {e}")
    
    async def semantic_search(self, query: str, n_results: int = 10,
                            conversation_id: Optional[str] = None,
                            similarity_threshold: float = None) -> List[Dict[str, Any]]:
        """Perform semantic vector search"""
        try:
            # Use cache key for frequent queries
            cache_key = None
            if self.redis_client and self.config.ENABLE_SEMANTIC_CACHE:
                query_hash = hashlib.md5(f"{query}_{conversation_id}_{n_results}".encode()).hexdigest()
                cache_key = f"search:{query_hash}"
                
                cached_result = await self.redis_client.get(cache_key)
                if cached_result:
                    logger.info("Returning cached search result")
                    return json.loads(cached_result)
            
            # Create query embedding
            if self.embedding_model:
                query_embedding = await self.embedding_model.aembed_query(query)
            elif self.sentence_model:
                query_embedding = self.sentence_model.encode([query])[0].tolist()
            else:
                raise Exception("No embedding model available")
            
            # Search in database
            conv_id = conversation_id or self.conversation_id
            threshold = similarity_threshold or self.config.SIMILARITY_THRESHOLD
            
            async with self.get_connection() as conn:
                # Convert embedding to JSONB format for PostgreSQL
                query_embedding_jsonb = json.dumps(query_embedding)
                
                results = await conn.fetch("""
                    SELECT 
                        dc.id as chunk_id,
                        dc.document_id,
                        dc.content,
                        dc.chunk_index,
                        d.filename,
                        d.file_type,
                        cosine_similarity(dc.embedding, $1::jsonb) as similarity_score
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.conversation_id = $2
                        AND cosine_similarity(dc.embedding, $1::jsonb) >= $3
                    ORDER BY cosine_similarity(dc.embedding, $1::jsonb) DESC
                    LIMIT $4
                """, query_embedding_jsonb, conv_id, threshold, n_results)
            
            # Format results
            formatted_results = []
            for row in results:
                formatted_results.append({
                    "chunk_id": str(row['chunk_id']),
                    "document_id": str(row['document_id']),
                    "content": row['content'],
                    "filename": row['filename'],
                    "file_type": row['file_type'],
                    "chunk_index": row['chunk_index'],
                    "similarity_score": float(row['similarity_score'])
                })
            
            # Cache result
            if cache_key and formatted_results:
                await self.redis_client.setex(
                    cache_key, 
                    self.config.CACHE_TTL_HOURS * 3600,
                    json.dumps(formatted_results)
                )
            
            logger.info(f"Semantic search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def fulltext_search(self, query: str, n_results: int = 10,
                            conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform PostgreSQL full-text search over chunks.content"""
        try:
            conv_id = conversation_id or self.conversation_id
            
            async with self.get_connection() as conn:
                results = await conn.fetch("""
                    SELECT 
                        d.id as document_id,
                        d.filename,
                        d.file_type,
                        dc.id as chunk_id,
                        dc.chunk_index,
                        ts_rank(to_tsvector('french', dc.content), query) as text_rank,
                        ts_headline('french', dc.content, query, 
                                   'MaxWords=50, MinWords=10, ShortWord=3, HighlightAll=false, MaxFragments=3') as highlight
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id,
                         plainto_tsquery('french', $1) query
                    WHERE d.conversation_id = $2
                      AND to_tsvector('french', dc.content) @@ query
                    ORDER BY text_rank DESC
                    LIMIT $3
                """, query, conv_id, n_results)
            
            formatted_results = []
            for row in results:
                formatted_results.append({
                    "document_id": str(row['document_id']),
                    "chunk_id": str(row['chunk_id']),
                    "filename": row['filename'],
                    "file_type": row['file_type'],
                    "chunk_index": row['chunk_index'],
                    "text_rank": float(row['text_rank']),
                    "highlight": row['highlight'],
                    "search_type": "fulltext"
                })
            
            logger.info(f"Full-text search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Full-text search failed: {e}")
            return []
    
    async def hybrid_search(self, query: str, n_results: int = 10,
                          vector_weight: float = 0.7, text_weight: float = 0.3,
                          conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and full-text search"""
        try:
            # Run both searches in parallel
            vector_results_task = self.semantic_search(query, n_results * 2, conversation_id)
            text_results_task = self.fulltext_search(query, n_results * 2, conversation_id)
            
            vector_results, text_results = await asyncio.gather(
                vector_results_task, text_results_task
            )
            
            # Combine and score results
            combined_scores = {}
            
            # Add vector search results
            for result in vector_results:
                doc_id = result['document_id']
                combined_scores[doc_id] = {
                    'result': result,
                    'vector_score': result['similarity_score'] * vector_weight,
                    'text_score': 0
                }
            
            # Add text search results
            for result in text_results:
                doc_id = result['document_id']
                if doc_id in combined_scores:
                    combined_scores[doc_id]['text_score'] = result['text_rank'] * text_weight
                else:
                    combined_scores[doc_id] = {
                        'result': result,
                        'vector_score': 0,
                        'text_score': result['text_rank'] * text_weight
                    }
            
            # Calculate final scores and sort
            final_results = []
            for doc_id, scores in combined_scores.items():
                final_score = scores['vector_score'] + scores['text_score']
                result = scores['result']
                result['final_score'] = final_score
                result['vector_score'] = scores['vector_score']
                result['text_score'] = scores['text_score']
                result['search_type'] = 'hybrid'
                final_results.append(result)
            
            # Sort by final score and limit results
            final_results.sort(key=lambda x: x['final_score'], reverse=True)
            return final_results[:n_results]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    async def get_document_summary(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of documents in conversation"""
        await self._ensure_initialized()
        try:
            conv_id = conversation_id or self.conversation_id
            
            async with self.get_connection() as conn:
                # Get document statistics
                stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(DISTINCT d.id) as document_count,
                        COUNT(dc.id) as chunk_count,
                        SUM(d.file_size_mb) as total_file_size,
                        MAX(d.created_at) as last_document_added
                    FROM documents d
                    LEFT JOIN document_chunks dc ON d.id = dc.document_id
                    WHERE d.conversation_id = $1
                """, conv_id)
                
                # Get file type distribution
                file_types = await conn.fetch("""
                    SELECT file_type, COUNT(*) as count
                    FROM documents 
                    WHERE conversation_id = $1
                    GROUP BY file_type
                    ORDER BY count DESC
                """, conv_id)
                
                return {
                    "conversation_id": conv_id,
                    "total_documents": stats['document_count'] or 0,
                    "total_chunks": stats['chunk_count'] or 0,
                    "total_file_size": stats['total_file_size'] or 0,
                    "last_document_added": stats['last_document_added'].isoformat() if stats['last_document_added'] else None,
                    "file_type_distribution": {row['file_type']: row['count'] for row in file_types},
                    "embedding_model": self.config.EMBEDDING_MODEL,
                    "vector_db_status": "postgresql_pgvector"
                }
                
        except Exception as e:
            logger.error(f"Failed to get document summary: {e}")
            return {"error": str(e)}
    
    async def clear_conversation_documents(self, conversation_id: Optional[str] = None):
        """Clear all documents from a conversation"""
        try:
            conv_id = conversation_id or self.conversation_id
            
            async with self.get_connection() as conn:
                # Delete documents (cascades to chunks due to FK)
                result = await conn.execute(
                    "DELETE FROM documents WHERE conversation_id = $1",
                    conv_id
                )
                
                deleted_count = int(result.split()[-1]) if result else 0
                logger.info(f"Cleared {deleted_count} documents from conversation {conv_id}")
                
        except Exception as e:
            logger.error(f"Failed to clear documents: {e}")
            raise
    
    async def close(self):
        """Close connections and cleanup"""
        if self.connection_pool:
            await self.connection_pool.close()
        if self.redis_client:
            await self.redis_client.close()
        logger.info("PostgreSQL RAG Store connections closed")
