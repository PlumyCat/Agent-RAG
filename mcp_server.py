import asyncio
import os
import uuid
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP
from rag_store import PostgreSQLRAGStore
from document_processors import ContentExtractor
from config import Config
import logging

import sys
import signal
import traceback

# Setup signal handlers for graceful shutdown
def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("RAGLargeFileProcessor")

# Global RAG store with default conversation (generate UUID)
import uuid
default_conversation_id = str(uuid.uuid4())
rag_store = PostgreSQLRAGStore(conversation_id=default_conversation_id)

# @mcp.tool()
# async def upload_documents_with_rag(file_paths: List[str]) -> Dict[str, Any]:
#     """
#     Upload documents and create vector embeddings for RAG with large file support.
    
#     Args:
#         file_paths: List of file paths to process
    
#     Returns:
#         Upload results with RAG processing status
#     """
#     results = {
#         "processed_documents": [],
#         "failed_documents": [],
#         "rag_summary": {
#             "total_chunks_created": 0,
#             "total_documents": 0,
#             "processing_time_ms": 0
#         }
#     }
    
#     import time
#     start_time = time.time()
    
#     for file_path in file_paths:
#         try:
#             logger.info(f"Processing file: {file_path}")
            
#             if not os.path.exists(file_path):
#                 results["failed_documents"].append({
#                     "file_path": file_path,
#                     "error": "File not found"
#                 })
#                 continue
            
#             # Check file size
#             file_info = await ContentExtractor.get_file_info(file_path)
#             if file_info.get("file_size_mb", 0) > Config.MAX_FILE_SIZE_MB:
#                 results["failed_documents"].append({
#                     "file_path": file_path,
#                     "error": f"File too large. Maximum size: {Config.MAX_FILE_SIZE_MB}MB"
#                 })
#                 continue
            
#             # Generate document ID
#             doc_id = str(uuid.uuid4())
#             filename = os.path.basename(file_path)
            
#             # Extract content based on file type
#             try:
#                 content, file_type = await ContentExtractor.extract_content(file_path)
#             except ValueError as e:
#                 results["failed_documents"].append({
#                     "file_path": file_path,
#                     "error": str(e)
#                 })
#                 continue
            
#             if not content or content.startswith("Error"):
#                 results["failed_documents"].append({
#                     "file_path": file_path,
#                     "error": "Failed to extract content or content is empty"
#                 })
#                 continue
            
#             # Prepare metadata
#             metadata = {
#                 "file_size_bytes": file_info["file_size_bytes"],
#                 "file_size_mb": file_info["file_size_mb"],
#                 "file_extension": os.path.splitext(file_path)[1].lower(),
#                 "processing_strategy": file_info["processing_strategy"]
#             }
            
#             # Add to RAG store
#             rag_result = await rag_store.add_document(doc_id, filename, file_type, content, metadata)
            
#             if "error" in rag_result:
#                 results["failed_documents"].append({
#                     "file_path": file_path,
#                     "error": rag_result["error"]
#                 })
#                 continue
            
#             results["processed_documents"].append({
#                 "doc_id": doc_id,
#                 "filename": filename,
#                 "file_type": file_type,
#                 "chunks_created": rag_result["chunks_created"],
#                 "content_length": len(content),
#                 "file_size_mb": file_info["file_size_mb"],
#                 "processing_strategy": file_info["processing_strategy"],
#                 "status": "success"
#             })
            
#             results["rag_summary"]["total_chunks_created"] += rag_result["chunks_created"]
#             results["rag_summary"]["total_documents"] += 1
            
#             logger.info(f"Successfully processed {filename}: {rag_result['chunks_created']} chunks")
            
#         except Exception as e:
#             logger.error(f"Error processing {file_path}: {e}")
#             results["failed_documents"].append({
#                 "file_path": file_path,
#                 "error": str(e)
#             })
    
#     results["rag_summary"]["processing_time_ms"] = int((time.time() - start_time) * 1000)
    
#     return results

@mcp.tool()
async def upload_documents_with_rag(file_paths: List[str]) -> Dict[str, Any]:
    """Upload documents with comprehensive error handling"""
    try:
        logger.info(f"Starting document upload for {len(file_paths)} files")
        
        results = {
            "processed_documents": [],
            "failed_documents": [],
            "rag_summary": {
                "total_chunks_created": 0,
                "total_documents": 0,
                "processing_time_ms": 0
            }
        }
        
        import time
        start_time = time.time()
        
        for file_path in file_paths:
            try:
                logger.info(f"Processing file: {file_path}")
                
                if not os.path.exists(file_path):
                    error_msg = f"File not found: {file_path}"
                    logger.error(error_msg)
                    results["failed_documents"].append({
                        "file_path": file_path,
                        "error": error_msg
                    })
                    continue
                
                # Check file size
                try:
                    file_info = await ContentExtractor.get_file_info(file_path)
                    if "error" in file_info:
                        results["failed_documents"].append({
                            "file_path": file_path,
                            "error": file_info["error"]
                        })
                        continue
                        
                except Exception as e:
                    error_msg = f"Failed to get file info: {str(e)}"
                    logger.error(error_msg)
                    results["failed_documents"].append({
                        "file_path": file_path,
                        "error": error_msg
                    })
                    continue
                
                if file_info.get("file_size_mb", 0) > Config.MAX_FILE_SIZE_MB:
                    error_msg = f"File too large. Maximum size: {Config.MAX_FILE_SIZE_MB}MB"
                    logger.error(error_msg)
                    results["failed_documents"].append({
                        "file_path": file_path,
                        "error": error_msg
                    })
                    continue
                
                filename = os.path.basename(file_path)
                
                # Extract content based on file type
                try:
                    content, file_type = await ContentExtractor.extract_content(file_path)
                except Exception as e:
                    error_msg = f"Content extraction failed: {str(e)}"
                    logger.error(error_msg)
                    results["failed_documents"].append({
                        "file_path": file_path,
                        "error": error_msg
                    })
                    continue
                
                if not content or content.startswith("Error"):
                    error_msg = "Failed to extract content or content is empty"
                    logger.error(error_msg)
                    results["failed_documents"].append({
                        "file_path": file_path,
                        "error": error_msg
                    })
                    continue
                
                # Prepare metadata
                metadata = {
                    "file_size_bytes": file_info["file_size_bytes"],
                    "file_size_mb": file_info["file_size_mb"],
                    "file_extension": os.path.splitext(file_path)[1].lower(),
                    "processing_strategy": file_info["processing_strategy"]
                }
                
                # Add to RAG store
                try:
                    rag_result = await rag_store.add_document(filename, file_type, content, metadata, file_path)
                except Exception as e:
                    error_msg = f"RAG store addition failed: {str(e)}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    results["failed_documents"].append({
                        "file_path": file_path,
                        "error": error_msg
                    })
                    continue
                
                if "error" in rag_result:
                    results["failed_documents"].append({
                        "file_path": file_path,
                        "error": rag_result["error"]
                    })
                    continue
                
                results["processed_documents"].append({
                    "doc_id": rag_result.get("doc_id"),
                    "filename": filename,
                    "file_type": file_type,
                    "chunks_created": rag_result["chunks_created"],
                    "content_length": len(content),
                    "file_size_mb": file_info["file_size_mb"],
                    "processing_strategy": file_info["processing_strategy"],
                    "status": "success"
                })
                
                results["rag_summary"]["total_chunks_created"] += rag_result["chunks_created"]
                results["rag_summary"]["total_documents"] += 1
                
                logger.info(f"Successfully processed {filename}: {rag_result['chunks_created']} chunks")
                
            except Exception as e:
                error_msg = f"Unexpected error processing {file_path}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                results["failed_documents"].append({
                    "file_path": file_path,
                    "error": error_msg
                })
        
        results["rag_summary"]["processing_time_ms"] = int((time.time() - start_time) * 1000)
        
        logger.info(f"Upload completed: {results['rag_summary']['total_documents']} successful, {len(results['failed_documents'])} failed")
        return results
        
    except Exception as e:
        error_msg = f"Critical error in upload_documents_with_rag: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg}

@mcp.tool()
async def semantic_search_documents(query: str, n_results: int = 10, 
                                  document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Perform semantic search across uploaded documents using RAG.
    
    Args:
        query: Search query
        n_results: Number of results to return
        document_ids: Optional list of specific document IDs to search
    
    Returns:
        Semantically similar content chunks
    """
    try:
        # Note: document_ids filtering is not supported yet in rag_store;
        # call semantic_search without attempting to pass them as conversation_id.
        results = await rag_store.semantic_search(query, n_results)
        
        return {
            "query": query,
            "total_results": len(results),
            "results": results,
            "search_type": "semantic_similarity"
        }
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return {"error": f"Semantic search failed: {str(e)}"}

@mcp.tool()
async def retrieve_context_for_query(query: str, max_context_length: int = 4000) -> Dict[str, Any]:
    """
    Retrieve relevant context for a query using RAG (optimized for LLM consumption).
    
    Args:
        query: Query to find relevant context for
        max_context_length: Maximum length of combined context
    
    Returns:
        Retrieved context optimized for LLM consumption
    """
    try:
        # Get semantic search results
        search_results = await rag_store.semantic_search(query, n_results=8)
        
        # Combine most relevant chunks
        context_parts = []
        current_length = 0
        sources_used = set()
        
        for result in search_results:
            content = result.get("content", "")
            similarity = result.get("similarity_score", 0.0)
            
            # Skip if similarity is too low
            if similarity < 0.3:
                continue
            
            # Add source information (metadata fields are top-level in current results)
            filename = result.get("filename", "unknown")
            file_type = result.get("file_type", "unknown")
            source_info = f"\n[Source: {filename} ({file_type}) - Similarity: {similarity:.2f}]\n"
            full_content = source_info + content.strip()
            
            if current_length + len(full_content) <= max_context_length:
                context_parts.append(full_content)
                current_length += len(full_content)
                sources_used.add(filename)
            else:
                break
        
        combined_context = "\n\n".join(context_parts)
        
        return {
            "query": query,
            "retrieved_context": combined_context,
            "context_length": len(combined_context),
            "sources_used": list(sources_used),
            "chunks_retrieved": len(context_parts),
            "retrieval_method": "semantic_similarity"
        }
    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        return {"error": f"Context retrieval failed: {str(e)}"}

@mcp.tool()
async def get_rag_document_summary() -> Dict[str, Any]:
    """
    Get summary of RAG-enabled document store.
    
    Returns:
        Summary of documents and vector database status
    """
    return await rag_store.get_document_summary()

@mcp.tool()
async def analyze_document_collection() -> Dict[str, Any]:
    """
    Analyze the collection of uploaded documents.
    
    Returns:
        Analysis of document collection
    """
    try:
        summary = await rag_store.get_document_summary()
        
        if summary.get("total_documents", 0) == 0:
            return {"error": "No documents in collection"}
        
        # Analyze by file type using summary-provided distribution
        file_types = summary.get("file_type_distribution", {}) or {}
        total_size_mb = summary.get("total_file_size", 0) or 0
        # Large files count is not directly exposed; set to 0 as placeholder
        large_files = 0
        
        return {
            "collection_analysis": {
                "total_documents": summary.get("total_documents", 0),
                "total_chunks": summary.get("total_chunks", 0),
                "file_type_distribution": file_types,
                "total_size_mb": round(total_size_mb, 2),
                "large_files_count": large_files,
                "average_chunks_per_doc": round((summary.get("total_chunks", 0) or 0) / summary.get("total_documents", 1), 1) if summary.get("total_documents", 0) > 0 else 0
            },
            "embedding_info": {
                "model": summary.get("embedding_model", "unknown"),
                "vector_db_status": summary.get("vector_db_status", "unknown")
            }
        }
    except Exception as e:
        logger.error(f"Document collection analysis failed: {e}")
        return {"error": str(e)}
    
@mcp.tool()
async def process_images_with_ocr(image_paths: List[str], enhancement_level: str = "standard") -> Dict[str, Any]:
    """
    Process multiple images with OCR using specified enhancement level.
    
    Args:
        image_paths: List of image file paths
        enhancement_level: OCR enhancement level (light, standard, aggressive)
    
    Returns:
        OCR processing results
    """
    try:
        logger.info(f"Processing {len(image_paths)} images with OCR (level: {enhancement_level})")
        
        results = {
            "processed_images": [],
            "failed_images": [],
            "batch_summary": {
                "total_images": len(image_paths),
                "successful_extractions": 0,
                "total_characters": 0,
                "processing_time_ms": 0
            }
        }
        
        import time
        start_time = time.time()
        
        for image_path in image_paths:
            try:
                if not os.path.exists(image_path):
                    results["failed_images"].append({
                        "image_path": image_path,
                        "error": "File not found"
                    })
                    continue
                
                # Check file size
                file_info = await ContentExtractor.get_file_info(image_path)
                if file_info.get("file_size_mb", 0) > Config.MAX_FILE_SIZE_MB:
                    results["failed_images"].append({
                        "image_path": image_path,
                        "error": f"File too large. Maximum size: {Config.MAX_FILE_SIZE_MB}MB"
                    })
                    continue
                
                # Extract content from image
                content = await ContentExtractor.extract_image_content(image_path)
                
                if content and not content.startswith("Error") and not content.startswith("No text"):
                    results["processed_images"].append({
                        "image_path": image_path,
                        "filename": os.path.basename(image_path),
                        "content_length": len(content),
                        "file_size_mb": file_info["file_size_mb"],
                        "status": "success",
                        "content_preview": content[:200] + "..." if len(content) > 200 else content
                    })
                    
                    results["batch_summary"]["successful_extractions"] += 1
                    results["batch_summary"]["total_characters"] += len(content)
                else:
                    results["failed_images"].append({
                        "image_path": image_path,
                        "error": content or "No text extracted"
                    })
            
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                results["failed_images"].append({
                    "image_path": image_path,
                    "error": str(e)
                })
        
        results["batch_summary"]["processing_time_ms"] = int((time.time() - start_time) * 1000)
        
        logger.info(f"Image processing complete: {results['batch_summary']['successful_extractions']}/{len(image_paths)} successful")
        return results
        
    except Exception as e:
        logger.error(f"Image batch processing failed: {e}")
        return {"error": str(e)}

@mcp.tool()
async def clear_document_store() -> Dict[str, Any]:
    """
    Clear all documents from the RAG store.
    
    Returns:
        Confirmation of clearing operation
    """
    try:
        await rag_store.clear_conversation_documents()
        return {
            "status": "success",
            "message": "All documents cleared from RAG store",
            "timestamp": str(asyncio.get_event_loop().time())
        }
    except Exception as e:
        logger.error(f"Failed to clear document store: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info("Starting RAG Large File MCP Server...")
    mcp.run(transport="stdio")
