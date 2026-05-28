package com.faceauth.util;

// L2 normalization for embedding vectors
// Must be applied before storage and comparison
public class VectorNormalizationUtil {

    private VectorNormalizationUtil() {}

    // L2-normalize a float vector in place
    // After this: cosine similarity = simple dot product
    public static float[] l2Normalize(float[] vector) {
        float sumSquares = 0f;
        for (float v : vector) {
            sumSquares += v * v;
        }

        float norm = (float) Math.sqrt(sumSquares);
        if (norm == 0f) return vector;

        float[] normalized = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            normalized[i] = vector[i] / norm;
        }
        return normalized;
    }

    // Convert float[] to double[] for PostgreSQL float8[] storage
    public static double[] toDoubleArray(float[] floats) {
        double[] doubles = new double[floats.length];
        for (int i = 0; i < floats.length; i++) {
            doubles[i] = floats[i];
        }
        return doubles;
    }
}
