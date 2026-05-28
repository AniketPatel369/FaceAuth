package com.faceauth.dto;

import lombok.Builder;
import lombok.Getter;

// Response DTO returned after a successful face enrollment
@Getter
@Builder
public class UploadFaceResponse {

    private boolean success;
    private Long userId;
    private boolean faceDetected;
    private float detectionConfidence;
    private boolean embeddingGenerated;
    private int embeddingDimensions;
    private boolean stored;

    // Total embeddings stored for this user (out of max 4)
    private int totalEnrollments;

    // Error message if enrollment failed
    private String error;

    // Factory method for successful enrollment
    public static UploadFaceResponse success(Long userId, float confidence, int totalEnrollments) {
        return UploadFaceResponse.builder()
                .success(true)
                .userId(userId)
                .faceDetected(true)
                .detectionConfidence(confidence)
                .embeddingGenerated(true)
                .embeddingDimensions(512)
                .stored(true)
                .totalEnrollments(totalEnrollments)
                .build();
    }

    // Factory method for failed enrollment
    public static UploadFaceResponse failure(String error) {
        return UploadFaceResponse.builder()
                .success(false)
                .error(error)
                .build();
    }
}
