package com.faceauth.repository;

import com.faceauth.entity.FaceEmbedding;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

// Data access for face embedding vectors
@Repository
public interface FaceEmbeddingRepository extends JpaRepository<FaceEmbedding, Long> {

    // Get all embeddings for a specific user (1-4 per user)
    List<FaceEmbedding> findByUserId(Long userId);

    // Count how many embeddings a user has enrolled (max 4)
    int countByUserId(Long userId);

    // Get all embeddings in the system (for matching loop)
    // For small scale (<10k users), loading all is fine
    List<FaceEmbedding> findAll();
}
