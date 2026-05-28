package com.faceauth.dto;

import lombok.Builder;
import lombok.Getter;

// Response DTO returned after a face matching/recognition attempt
@Getter
@Builder
public class MatchFaceResponse {

    private boolean success;

    // Whether a match was found above the accept threshold
    private boolean matched;

    // Matched user's ID (null if no match)
    private Long userId;

    // Matched user's name (null if no match)
    private String userName;

    // Highest similarity score found
    private float similarity;

    // Threshold used for accept decision
    private float threshold;

    // Error message if matching failed
    private String error;

    // Factory method for a successful match
    public static MatchFaceResponse matched(Long userId, String userName, float similarity, float threshold) {
        return MatchFaceResponse.builder()
                .success(true)
                .matched(true)
                .userId(userId)
                .userName(userName)
                .similarity(similarity)
                .threshold(threshold)
                .build();
    }

    // Factory method for no match found
    public static MatchFaceResponse noMatch(float bestSimilarity, float threshold) {
        return MatchFaceResponse.builder()
                .success(true)
                .matched(false)
                .similarity(bestSimilarity)
                .threshold(threshold)
                .build();
    }

    // Factory method for processing failure
    public static MatchFaceResponse failure(String error) {
        return MatchFaceResponse.builder()
                .success(false)
                .matched(false)
                .error(error)
                .build();
    }
}
