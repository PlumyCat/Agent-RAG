#!/usr/bin/env python3
"""
Migration script: ChromaDB -> PostgreSQL + pgvector
Migrates existing ChromaDB data to new PostgreSQL schema
"""

import asyncio
import asyncpg
import chromadb
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List
import logging
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import Config
from rag_store import RAGDocumentStore  # Old ChromaDB store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDBToPostgreSQLMigrator:
    """Migrate data from ChromaDB to PostgreSQL"""
    
    def __init__(self):
        self.config = Config()
        self.pg_connection = None
        self.chroma_store = None
        
    async def connect_postgresql(self):
        """Connect to PostgreSQL database"""
        try:
            self.pg_connection = await asyncpg.connect(
                host=self.config.POSTGRES_HOST,
                port=self.config.POSTGRES_PORT,
                user=self.config.POSTGRES_USER,
                password=self.config.POSTGRES_PASSWORD,
                database=self.config.POSTGRES_DB
            )
            logger.info("Connected to PostgreSQL")
            
            # Ensure extensions are available
            await self.pg_connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await self.pg_connection.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            await self.pg_connection.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def connect_chromadb(self):
        """Connect to existing ChromaDB store"""
        try:
            self.chroma_store = RAGDocumentStore()
            logger.info("Connected to ChromaDB")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
    
    async def check_schema_exists(self) -> bool:
        """Check if PostgreSQL schema exists"""
        try:
            result = await self.pg_connection.fetchrow(
                "SELECT table_name FROM information_schema.tables WHERE table_name = 'conversations'"
            )
            return result is not None
        except Exception as e:
            logger.error(f"Failed to check schema: {e}")
            return False
    
    async def run_schema_setup(self):
        """Run schema setup if needed"""
        if not await self.check_schema_exists():
            logger.info("PostgreSQL schema not found, setting up...")
            
            # Read and execute schema file
            schema_file = project_root / "database" / "schema.sql"
            if not schema_file.exists():
                raise FileNotFoundError(f"Schema file not found: {schema_file}")
            
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            
            # Execute schema (split by ';' and execute each statement)
            statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
            for statement in statements:
                if statement and not statement.startswith('--'):
                    try:
                        await self.pg_connection.execute(statement)
                    except Exception as e:
                        logger.warning(f"Schema statement failed (may be expected): {e}")
            
            logger.info("Schema setup completed")
        else:
            logger.info("PostgreSQL schema already exists")
    
    async def migrate_documents(self):
        """Migrate documents and chunks from ChromaDB"""
        try:
            # Get all documents from ChromaDB
            collection_info = await self.chroma_store.get_document_summary()
            
            if collection_info.get("total_documents", 0) == 0:
                logger.info("No documents found in ChromaDB to migrate")
                return
            
            logger.info(f"Found {collection_info['total_documents']} documents to migrate")
            
            # Create a default conversation for migration
            conversation_id = str(uuid.uuid4())
            await self.pg_connection.execute("""
                INSERT INTO conversations (id, session_id, title, metadata)
                VALUES ($1, $2, $3, $4)
            """, conversation_id, "chromadb_migration", "Migrated from ChromaDB", 
            json.dumps({"migration_date": datetime.now().isoformat()}))
            
            # Get all data from ChromaDB
            chroma_data = self.chroma_store.collection.get(
                include=["documents", "metadatas", "embeddings", "ids"]
            )
            
            if not chroma_data["documents"]:
                logger.warning("No documents found in ChromaDB collection")
                return
            
            # Group chunks by document
            documents_data = {}
            
            for i, chunk_id in enumerate(chroma_data["ids"]):
                metadata = chroma_data["metadatas"][i]
                doc_id = metadata.get("doc_id", "unknown")
                
                if doc_id not in documents_data:
                    documents_data[doc_id] = {
                        "filename": metadata.get("filename", "unknown"),
                        "file_type": metadata.get("file_type", "unknown"),
                        "chunks": [],
                        "metadata": metadata,
                        "file_size": metadata.get("file_size_bytes", 0)
                    }
                
                documents_data[doc_id]["chunks"].append({
                    "content": chroma_data["documents"][i],
                    "embedding": chroma_data["embeddings"][i] if chroma_data["embeddings"] else None,
                    "chunk_index": metadata.get("chunk_index", i),
                    "metadata": metadata
                })
            
            # Migrate each document
            migrated_count = 0
            
            async with self.pg_connection.transaction():
                for old_doc_id, doc_data in documents_data.items():
                    try:
                        new_doc_id = str(uuid.uuid4())
                        
                        # Reconstruct full text from chunks
                        full_text = "\n".join([chunk["content"] for chunk in doc_data["chunks"]])
                        
                        # Insert document
                        await self.pg_connection.execute("""
                            INSERT INTO documents (
                                id, conversation_id, filename, content_hash,
                                file_type, file_size, metadata, full_text,
                                chunk_count, extraction_method, processed_at
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        """, 
                        new_doc_id, conversation_id, doc_data["filename"],
                        old_doc_id,  # Use old doc_id as content hash for migration
                        doc_data["file_type"], doc_data["file_size"],
                        json.dumps(doc_data["metadata"]), full_text,
                        len(doc_data["chunks"]), "chromadb_migration",
                        datetime.now())
                        
                        # Insert chunks
                        for chunk in doc_data["chunks"]:
                            chunk_id = str(uuid.uuid4())
                            
                            await self.pg_connection.execute("""
                                INSERT INTO document_chunks (
                                    id, document_id, content, embedding, chunk_index,
                                    start_char, end_char, chunk_type, extraction_confidence, metadata
                                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                            """, 
                            chunk_id, new_doc_id, chunk["content"],
                            chunk["embedding"], chunk["chunk_index"],
                            0, len(chunk["content"]), "text", 1.0,
                            json.dumps(chunk["metadata"]))
                        
                        migrated_count += 1
                        logger.info(f"Migrated document {doc_data['filename']} ({migrated_count}/{len(documents_data)})")
                        
                    except Exception as e:
                        logger.error(f"Failed to migrate document {old_doc_id}: {e}")
                        raise
            
            logger.info(f"Successfully migrated {migrated_count} documents to PostgreSQL")
            
            # Update conversation with final statistics
            await self.pg_connection.execute("""
                UPDATE conversations 
                SET metadata = $2, updated_at = CURRENT_TIMESTAMP
                WHERE id = $1
            """, conversation_id, json.dumps({
                "migration_date": datetime.now().isoformat(),
                "migrated_documents": migrated_count,
                "source": "chromadb"
            }))
            
        except Exception as e:
            logger.error(f"Document migration failed: {e}")
            raise
    
    async def verify_migration(self):
        """Verify migration was successful"""
        try:
            # Check document count
            doc_count = await self.pg_connection.fetchval(
                "SELECT COUNT(*) FROM documents"
            )
            
            chunk_count = await self.pg_connection.fetchval(
                "SELECT COUNT(*) FROM document_chunks"
            )
            
            logger.info(f"Migration verification:")
            logger.info(f"- Documents in PostgreSQL: {doc_count}")
            logger.info(f"- Chunks in PostgreSQL: {chunk_count}")
            
            # Test a simple vector search
            if chunk_count > 0:
                test_result = await self.pg_connection.fetchrow("""
                    SELECT content, (embedding <=> embedding) as distance
                    FROM document_chunks 
                    WHERE embedding IS NOT NULL 
                    LIMIT 1
                """)
                
                if test_result:
                    logger.info(f"✅ Vector search test successful")
                else:
                    logger.warning("⚠️  No embeddings found for vector search test")
            
            return doc_count, chunk_count
            
        except Exception as e:
            logger.error(f"Migration verification failed: {e}")
            return 0, 0
    
    async def create_backup(self):
        """Create backup of ChromaDB before migration"""
        try:
            backup_dir = project_root / "data" / "chromadb_backup"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Export ChromaDB data to JSON
            collection_data = self.chroma_store.collection.get(
                include=["documents", "metadatas", "embeddings", "ids"]
            )
            
            backup_file = backup_dir / f"chromadb_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(backup_file, 'w') as f:
                json.dump({
                    "export_date": datetime.now().isoformat(),
                    "collection_name": "documents",
                    "data": collection_data,
                    "documents_metadata": self.chroma_store.documents_metadata
                }, f, indent=2, default=str)
            
            logger.info(f"ChromaDB backup created: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return None
    
    async def run_migration(self):
        """Run complete migration process"""
        try:
            logger.info("Starting ChromaDB to PostgreSQL migration...")
            
            # Step 1: Connect to databases
            await self.connect_postgresql()
            self.connect_chromadb()
            
            # Step 2: Setup schema if needed
            await self.run_schema_setup()
            
            # Step 3: Create backup
            backup_file = await self.create_backup()
            if backup_file:
                logger.info(f"Backup created: {backup_file}")
            
            # Step 4: Migrate documents
            await self.migrate_documents()
            
            # Step 5: Verify migration
            doc_count, chunk_count = await self.verify_migration()
            
            logger.info("✅ Migration completed successfully!")
            logger.info(f"📊 Migrated {doc_count} documents with {chunk_count} chunks")
            
            if backup_file:
                logger.info(f"💾 Backup available at: {backup_file}")
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            raise
        finally:
            # Cleanup connections
            if self.pg_connection:
                await self.pg_connection.close()

async def main():
    """Main migration entry point"""
    migrator = ChromaDBToPostgreSQLMigrator()
    
    try:
        # Check if user wants to proceed
        print("🔄 ChromaDB to PostgreSQL Migration")
        print("=" * 50)
        print("This will migrate all data from ChromaDB to PostgreSQL + pgvector")
        print("A backup of ChromaDB will be created before migration.")
        print("")
        
        response = input("Do you want to proceed with the migration? (y/N): ")
        if response.lower() != 'y':
            print("Migration cancelled.")
            return
        
        await migrator.run_migration()
        
        print("\n🎉 Migration completed successfully!")
        print("You can now use the new PostgreSQL-based RAG system.")
        
    except KeyboardInterrupt:
        print("\n❌ Migration cancelled by user")
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())