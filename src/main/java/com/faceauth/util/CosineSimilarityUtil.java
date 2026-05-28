package com.faceauth.util;

// Computes cosine similarity between two vectors
// Used to compare face embeddings for identity matching
public class CosineSimilarityUtil {

    private CosineSimilarityUtil() {}

    // Cosine similarity for L2-normalized vectors = dot product
    // Since we always L2-normalize before storing, this is equivalent to cosine similarity
    public static float dotProduct(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "Vector length mismatch: " + a.length + " vs " + b.length);
        }

        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    // Full cosine similarity (for non-normalized vectors)
    public static float cosineSimilarity(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "Vector length mismatch: " + a.length + " vs " + b.length);
        }

        float dot = 0f, normA = 0f, normB = 0f;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        float denominator = (float) (Math.sqrt(normA) * Math.sqrt(normB));
        if (denominator == 0f) return 0f;

        return dot / denominator;
    }

    // Convert double[] (from DB) to float[] (for computation)
    public static float[] toFloatArray(double[] doubles) {
        float[] floats = new float[doubles.length];
        for (int i = 0; i < doubles.length; i++) {
            floats[i] = (float) doubles[i];
        }
        return floats;
    }
}
