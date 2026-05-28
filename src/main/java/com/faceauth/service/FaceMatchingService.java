package com.faceauth.service;

import com.faceauth.config.FaceAuthProperties;
import com.faceauth.entity.FaceEmbedding;
import com.faceauth.repository.FaceEmbeddingRepository;
import com.faceauth.util.CosineSimilarityUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.List;

// Compares a query embedding against all stored embeddings
// Uses brute-force cosine similarity (efficient for <10k users)
@Service
public class FaceMatchingService {

    private static final Logger log = LoggerFactory.getLogger(FaceMatchingService.class);

    private final FaceEmbeddingRepository embeddingRepository;
    private final FaceAuthProperties properties;

    public FaceMatchingService(FaceEmbeddingRepository embeddingRepository,
                                FaceAuthProperties properties) {
        this.embeddingRepository = embeddingRepository;
        this.properties = properties;
    }

    // Find the best matching user for the given query embedding
    // Returns MatchResult with userId, similarity, and match decision
    public MatchResult findBestMatch(float[] queryEmbedding) {
        List<FaceEmbedding> allEmbeddings = embeddingRepository.findAll();

        if (allEmbeddings.isEmpty()) {
            return new MatchResult(null, null, 0f, false);
        }

        Long bestUserId = null;
        String bestUserName = null;
        float bestSimilarity = -1f;

        // Brute-force comparison against every stored embedding
        for (FaceEmbedding stored : allEmbeddings) {
            float[] storedVector = CosineSimilarityUtil.toFloatArray(stored.getEmbedding());

            // Dot product = cosine similarity (both vectors are L2-normalized)
            float similarity = CosineSimilarityUtil.dotProduct(queryEmbedding, storedVector);

            if (similarity > bestSimilarity) {
                bestSimilarity = similarity;
                bestUserId = stored.getUser().getId();
                bestUserName = stored.getUser().getName();
            }
        }

        float acceptThreshold = properties.getMatching().getAcceptThreshold();
        boolean matched = bestSimilarity >= acceptThreshold;

        log.info("Best match: userId={}, similarity={}, matched={}",
                bestUserId, String.format("%.4f", bestSimilarity), matched);

        return new MatchResult(bestUserId, bestUserName, bestSimilarity, matched);
    }

    // Result of a matching operation
    public record MatchResult(Long userId, String userName, float similarity, boolean matched) {}
}
