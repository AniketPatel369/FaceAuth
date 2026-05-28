-- FaceAuth Database Schema
-- PostgreSQL 15+

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    identifier VARCHAR(255) NOT NULL UNIQUE,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Face embeddings table — stores 512-D ArcFace vectors
CREATE TABLE IF NOT EXISTS face_embeddings (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    embedding float8[] NOT NULL,
    image_path TEXT,
    quality_score FLOAT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Index for fast user lookup
CREATE INDEX IF NOT EXISTS idx_face_embeddings_user_id ON face_embeddings(user_id);

-- Index for user identifier lookup
CREATE INDEX IF NOT EXISTS idx_users_identifier ON users(identifier);
