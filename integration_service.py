"""
Integration service for MCP-RAG with Archive System
Provides unified metadata standards and cross-system operations
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

import asyncpg
from config import Config

class SourceType(Enum):
    ARCHIVE = "archive"
    DOCUMENT = "document" 
    CONVERSATION = "conversation"

class CreatedBy(Enum):
    MCP_SERVER = "mcp_server"
    USER = "user"
    SYSTEM = "system"

@dataclass
class AnalysisMetrics:
    topics: List[str]
    sentiment: float  # -1 to 1
    complexity_score: float  # 0 to 1
    semantic_clusters: List[str]

@dataclass
class ProcessingMetrics:
    token_count: int
    char_count: int
    processing_time_ms: float
    confidence_score: float

@dataclass
class Relations:
    related_archives: Optional[List[str]] = None
    related_documents: Optional[List[str]] = None
    semantic_neighbors: Optional[List[str]] = None

@dataclass 
class UnifiedMetadata:
    """Unified metadata standard for both Archive and RAG systems"""
    source_type: SourceType
    created_by: CreatedBy
    version: str
    analysis: AnalysisMetrics
    metrics: ProcessingMetrics
    relations: Relations
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON for storage in JSONB columns"""
        return {
            "source_type": self.source_type.value,
            "created_by": self.created_by.value, 
            "version": self.version,
            "analysis": {
                "topics": self.analysis.topics,
                "sentiment": self.analysis.sentiment,
                "complexity_score": self.analysis.complexity_score,
                "semantic_clusters": self.analysis.semantic_clusters
            },
            "metrics": {
                "token_count": self.metrics.token_count,
                "char_count": self.metrics.char_count,
                "processing_time_ms": self.metrics.processing_time_ms,
                "confidence_score": self.metrics.confidence_score
            },
            "relations": {
                "related_archives": self.relations.related_archives,
                "related_documents": self.relations.related_documents,
                "semantic_neighbors": self.relations.semantic_neighbors
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'UnifiedMetadata':
        """Create from JSON stored in database"""
        return cls(
            source_type=SourceType(data["source_type"]),
            created_by=CreatedBy(data["created_by"]),
            version=data["version"],
            analysis=AnalysisMetrics(
                topics=data["analysis"]["topics"],
                sentiment=data["analysis"]["sentiment"],
                complexity_score=data["analysis"]["complexity_score"],
                semantic_clusters=data["analysis"]["semantic_clusters"]
            ),
            metrics=ProcessingMetrics(
                token_count=data["metrics"]["token_count"],
                char_count=data["metrics"]["char_count"],
                processing_time_ms=data["metrics"]["processing_time_ms"],
                confidence_score=data["metrics"]["confidence_score"]
            ),
            relations=Relations(
                related_archives=data["relations"].get("related_archives"),
                related_documents=data["relations"].get("related_documents"),
                semantic_neighbors=data["relations"].get("semantic_neighbors")
            ),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )

class ArchiveRAGIntegrationService:
    """Service for integrating MCP-RAG with Archive system"""
    
    def __init__(self):
        self.config = Config()
        self.archive_pool = None
        
    async def _get_archive_connection(self) -> asyncpg.Connection:
        """Get connection to archive database"""
        if self.archive_pool is None:
            self.archive_pool = await asyncpg.create_pool(
                host=self.config.POSTGRES_HOST,
                port=self.config.POSTGRES_PORT,
                user=self.config.POSTGRES_USER,
                password=self.config.POSTGRES_PASSWORD,
                database=self.config.POSTGRES_DB,
                min_size=2,
                max_size=10
            )
        return self.archive_pool.acquire()
    
    async def get_archives_for_indexing(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get archives that haven't been indexed in RAG yet"""
        async with await self._get_archive_connection() as conn:
            query = """
                SELECT 
                    a.id as archive_id,
                    a.title,
                    a.mcp_metadata,
                    ai.content,
                    ai.mcp_analysis,
                    ai.content_summary,
                    ai.tokens_count,
                    ai.id as item_id
                FROM archive a
                JOIN archive_item ai ON a.id = ai.archive_id
                WHERE a.mcp_metadata->>'indexed_in_rag' IS NULL
                   OR a.mcp_metadata->>'indexed_in_rag' = 'false'
                ORDER BY a.created_at DESC
                LIMIT $1
            """
            results = await conn.fetch(query, limit)
            return [dict(row) for row in results]
    
    async def mark_archive_as_indexed(self, archive_id: str, document_id: str):
        """Mark archive as indexed in RAG system"""
        async with await self._get_archive_connection() as conn:
            await conn.execute("""
                UPDATE archive 
                SET mcp_metadata = jsonb_set(
                    COALESCE(mcp_metadata, '{}'),
                    '{indexed_in_rag}',
                    'true'
                ) || jsonb_set(
                    COALESCE(mcp_metadata, '{}'),
                    '{rag_document_id}',
                    $2::jsonb
                )
                WHERE id = $1
            """, archive_id, json.dumps(document_id))
    
    async def enrich_archive_with_rag_analysis(
        self, 
        archive_id: str, 
        rag_analysis: Dict[str, Any]
    ):
        """Enrich archive with RAG analysis results"""
        async with await self._get_archive_connection() as conn:
            await conn.execute("""
                UPDATE archive_item
                SET mcp_analysis = jsonb_set(
                    COALESCE(mcp_analysis, '{}'),
                    '{rag_analysis}',
                    $2::jsonb
                )
                WHERE archive_id = $1
            """, archive_id, json.dumps(rag_analysis))
    
    async def find_similar_archives(
        self, 
        content: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar archives using full-text search"""
        async with await self._get_archive_connection() as conn:
            query = """
                SELECT 
                    ai.id,
                    ai.content,
                    ai.content_summary,
                    a.title,
                    a.id as archive_id,
                    ts_rank(
                        to_tsvector('english', ai.content), 
                        plainto_tsquery('english', $1)
                    ) as similarity_score
                FROM archive_item ai
                JOIN archive a ON ai.archive_id = a.id
                WHERE to_tsvector('english', ai.content) @@ plainto_tsquery('english', $1)
                ORDER BY similarity_score DESC
                LIMIT $2
            """
            results = await conn.fetch(query, content, limit)
            return [dict(row) for row in results]
    
    async def get_archive_statistics(self) -> Dict[str, Any]:
        """Get statistics about archive system"""
        async with await self._get_archive_connection() as conn:
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(DISTINCT a.id) as total_archives,
                    COUNT(ai.id) as total_items,
                    COUNT(CASE WHEN a.mcp_metadata->>'indexed_in_rag' = 'true' 
                          THEN 1 END) as indexed_archives,
                    AVG(ai.tokens_count) as avg_tokens_per_item,
                    SUM(ai.tokens_count) as total_tokens
                FROM archive a
                LEFT JOIN archive_item ai ON a.id = ai.archive_id
            """)
            return dict(stats) if stats else {}
    
    async def create_unified_metadata(
        self,
        content: str,
        source_type: SourceType,
        created_by: CreatedBy = CreatedBy.MCP_SERVER,
        additional_analysis: Dict[str, Any] = None
    ) -> UnifiedMetadata:
        """Create unified metadata for content"""
        
        # Basic metrics
        char_count = len(content)
        # Rough token estimation (1 token ≈ 4 characters for English)
        token_count = max(1, char_count // 4)
        
        # Placeholder analysis (would be replaced by actual AI analysis)
        topics = self._extract_topics(content)
        sentiment = self._analyze_sentiment(content)
        complexity_score = min(1.0, char_count / 10000)  # Simple heuristic
        
        analysis = AnalysisMetrics(
            topics=topics,
            sentiment=sentiment,
            complexity_score=complexity_score,
            semantic_clusters=[]
        )
        
        metrics = ProcessingMetrics(
            token_count=token_count,
            char_count=char_count,
            processing_time_ms=0.0,  # Would be measured
            confidence_score=0.8  # Placeholder
        )
        
        relations = Relations()
        
        return UnifiedMetadata(
            source_type=source_type,
            created_by=created_by,
            version="1.0",
            analysis=analysis,
            metrics=metrics,
            relations=relations
        )
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract topics from content (placeholder implementation)"""
        # This would use proper NLP libraries like spaCy, transformers etc.
        common_words = content.lower().split()
        # Simple keyword extraction
        topics = []
        if any(word in common_words for word in ['ai', 'artificial', 'intelligence']):
            topics.append('AI')
        if any(word in common_words for word in ['code', 'programming', 'development']):
            topics.append('Development')
        if any(word in common_words for word in ['data', 'database', 'analytics']):
            topics.append('Data')
        return topics[:5]  # Limit to 5 topics
    
    def _analyze_sentiment(self, content: str) -> float:
        """Analyze sentiment of content (placeholder implementation)"""
        # This would use proper sentiment analysis models
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'wrong']
        
        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if positive_count + negative_count == 0:
            return 0.0  # Neutral
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    async def close(self):
        """Close database connections"""
        if self.archive_pool:
            await self.archive_pool.close()

# Example usage functions
async def index_archives_in_rag():
    """Example: Index archives in RAG system"""
    integration_service = ArchiveRAGIntegrationService()
    
    try:
        archives = await integration_service.get_archives_for_indexing(limit=10)
        print(f"Found {len(archives)} archives to index")
        
        for archive in archives:
            # Create unified metadata
            metadata = await integration_service.create_unified_metadata(
                content=archive['content'],
                source_type=SourceType.ARCHIVE
            )
            
            print(f"Archive {archive['archive_id']}: {len(archive['content'])} chars")
            print(f"Topics: {metadata.analysis.topics}")
            print(f"Sentiment: {metadata.analysis.sentiment}")
            
            # Here you would add the archive to RAG system
            # await rag_store.add_document(...)
            
            # Mark as indexed
            # await integration_service.mark_archive_as_indexed(
            #     archive['archive_id'], 
            #     document_id
            # )
            
    finally:
        await integration_service.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(index_archives_in_rag())