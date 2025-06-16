-- Supabase Migration for Clerk Legal AI System
-- Includes vector storage and full-text search setup

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm; -- For trigram similarity
CREATE EXTENSION IF NOT EXISTS unaccent; -- For accent-insensitive search

-- Document registry for deduplication
CREATE TABLE IF NOT EXISTS document_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_hash VARCHAR(64) UNIQUE NOT NULL,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    case_name TEXT NOT NULL,
    first_seen_at TIMESTAMPTZ DEFAULT NOW(),
    last_duplicate_found TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    duplicate_locations JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for document registry
CREATE INDEX IF NOT EXISTS idx_document_hash ON document_registry(document_hash);
CREATE INDEX IF NOT EXISTS idx_doc_case_name ON document_registry(case_name);
CREATE INDEX IF NOT EXISTS idx_doc_created_at ON document_registry(created_at);

-- Main vector storage table with full-text search
CREATE TABLE IF NOT EXISTS case_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_name TEXT NOT NULL,
    document_id TEXT NOT NULL,
    content TEXT NOT NULL,
    search_text TEXT NOT NULL, -- Preprocessed text for full-text search
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', search_text)) STORED,
    embedding vector(1536) NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for case documents
CREATE INDEX IF NOT EXISTS idx_case_documents_case ON case_documents(case_name);
CREATE INDEX IF NOT EXISTS idx_case_documents_doc ON case_documents(document_id);
CREATE INDEX IF NOT EXISTS idx_case_created_at ON case_documents(created_at);

-- Vector similarity index (IVFFlat)
CREATE INDEX IF NOT EXISTS idx_embeddings ON case_documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100); -- Adjust lists parameter based on data size

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_search_vector ON case_documents 
USING GIN (search_vector);

-- Trigram index for fuzzy search
CREATE INDEX IF NOT EXISTS idx_content_trgm ON case_documents 
USING GIN (content gin_trgm_ops);

-- Function for vector search with case isolation
CREATE OR REPLACE FUNCTION search_case_vectors(
    case_name_filter TEXT,
    query_embedding vector(1536),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    case_name TEXT,
    document_id TEXT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cd.id,
        cd.case_name,
        cd.document_id,
        cd.content,
        cd.metadata,
        1 - (cd.embedding <=> query_embedding) AS similarity
    FROM case_documents cd
    WHERE cd.case_name = case_name_filter
        AND 1 - (cd.embedding <=> query_embedding) > match_threshold
    ORDER BY cd.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Function for full-text search with case isolation
CREATE OR REPLACE FUNCTION search_case_fulltext(
    case_name_filter TEXT,
    text_query TEXT,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    case_name TEXT,
    document_id TEXT,
    content TEXT,
    metadata JSONB,
    rank FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cd.id,
        cd.case_name,
        cd.document_id,
        cd.content,
        cd.metadata,
        ts_rank_cd(cd.search_vector, plainto_tsquery('english', text_query)) AS rank
    FROM case_documents cd
    WHERE cd.case_name = case_name_filter
        AND cd.search_vector @@ plainto_tsquery('english', text_query)
    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;

-- Function for hybrid search combining vector and full-text
CREATE OR REPLACE FUNCTION hybrid_search_case_documents(
    case_name_filter TEXT,
    text_query TEXT,
    embedding_query vector(1536),
    match_count INT DEFAULT 20,
    vector_weight FLOAT DEFAULT 0.7,
    text_weight FLOAT DEFAULT 0.3
)
RETURNS TABLE (
    id UUID,
    case_name TEXT,
    document_id TEXT,
    content TEXT,
    metadata JSONB,
    vector_similarity FLOAT,
    text_rank FLOAT,
    combined_score FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT 
            cd.id,
            cd.case_name,
            cd.document_id,
            cd.content,
            cd.metadata,
            1 - (cd.embedding <=> embedding_query) AS vector_similarity
        FROM case_documents cd
        WHERE cd.case_name = case_name_filter
        ORDER BY cd.embedding <=> embedding_query
        LIMIT match_count * 2  -- Get more results for merging
    ),
    text_results AS (
        SELECT 
            cd.id,
            cd.case_name,
            cd.document_id,
            cd.content,
            cd.metadata,
            ts_rank_cd(cd.search_vector, plainto_tsquery('english', text_query)) AS text_rank
        FROM case_documents cd
        WHERE cd.case_name = case_name_filter
            AND cd.search_vector @@ plainto_tsquery('english', text_query)
        ORDER BY text_rank DESC
        LIMIT match_count * 2  -- Get more results for merging
    ),
    combined AS (
        SELECT 
            COALESCE(v.id, t.id) AS id,
            COALESCE(v.case_name, t.case_name) AS case_name,
            COALESCE(v.document_id, t.document_id) AS document_id,
            COALESCE(v.content, t.content) AS content,
            COALESCE(v.metadata, t.metadata) AS metadata,
            COALESCE(v.vector_similarity, 0) AS vector_similarity,
            COALESCE(t.text_rank, 0) AS text_rank,
            (COALESCE(v.vector_similarity, 0) * vector_weight + 
             COALESCE(t.text_rank, 0) * text_weight) AS combined_score
        FROM vector_results v
        FULL OUTER JOIN text_results t ON v.id = t.id
    )
    SELECT * FROM combined
    ORDER BY combined_score DESC
    LIMIT match_count;
END;
$$;

-- Function to update search index for a document
CREATE OR REPLACE FUNCTION update_document_search_index(
    case_name_filter TEXT,
    document_id_filter TEXT
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    -- Force update of search vector for specified document
    UPDATE case_documents
    SET updated_at = NOW()
    WHERE case_name = case_name_filter
        AND document_id = document_id_filter;
END;
$$;

-- Trigger to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_document_registry_updated_at 
BEFORE UPDATE ON document_registry
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_case_documents_updated_at 
BEFORE UPDATE ON case_documents
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Create text search configuration for legal documents
CREATE TEXT SEARCH CONFIGURATION IF NOT EXISTS legal_english (COPY = english);

-- Add custom dictionary for legal terms (optional)
-- This would require creating a custom dictionary file
-- ALTER TEXT SEARCH CONFIGURATION legal_english
-- ALTER MAPPING FOR word WITH legal_dictionary, english_stem;

-- Grant necessary permissions (adjust based on your Supabase setup)
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO authenticated;
-- GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO authenticated;

-- Create a view for easy case statistics
CREATE OR REPLACE VIEW case_statistics AS
SELECT 
    case_name,
    COUNT(DISTINCT document_id) as document_count,
    COUNT(*) as chunk_count,
    MIN(created_at) as first_document_added,
    MAX(created_at) as last_document_added,
    pg_size_pretty(SUM(octet_length(content))::bigint) as total_content_size
FROM case_documents
GROUP BY case_name;

-- Comments for documentation
COMMENT ON TABLE case_documents IS 'Main storage for document chunks with vector embeddings and full-text search';
COMMENT ON COLUMN case_documents.search_text IS 'Preprocessed text optimized for full-text search';
COMMENT ON COLUMN case_documents.search_vector IS 'PostgreSQL tsvector for full-text search';
COMMENT ON COLUMN case_documents.embedding IS 'OpenAI text-embedding-3-small 1536-dimensional vector';
COMMENT ON FUNCTION hybrid_search_case_documents IS 'Performs hybrid search combining vector similarity and full-text search with weighted scoring';