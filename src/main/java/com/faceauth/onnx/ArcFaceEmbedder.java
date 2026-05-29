package com.faceauth.onnx;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.faceauth.config.FaceAuthProperties;
import com.faceauth.util.ImageUtils;
import com.faceauth.util.VectorNormalizationUtil;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

// Runs ArcFace embedding model via ONNX Runtime
// Input: aligned 112x112 BGR face → Output: L2-normalized 512-D embedding
// Supports configurable enhancements via feature flags
@Component
public class ArcFaceEmbedder {

    private static final Logger log = LoggerFactory.getLogger(ArcFaceEmbedder.class);

    // ArcFace input dimensions
    private static final int INPUT_SIZE = 112;

    // Expected output embedding dimension
    private static final int EMBEDDING_DIM = 512;

    private final OnnxModelManager modelManager;
    private final FaceAuthProperties properties;

    public ArcFaceEmbedder(OnnxModelManager modelManager, FaceAuthProperties properties) {
        this.modelManager = modelManager;
        this.properties = properties;
    }

    // Generate face embedding from an aligned 112x112 BGR face image
    // Applies configured enhancements: CLAHE, flip-ensemble, normalization divisor
    // Returns L2-normalized 512-D float vector
    public float[] generateEmbedding(Mat alignedFace) {
        FaceAuthProperties.Enhancements enhancements = properties.getEnhancements();

        // Step 1: Optionally apply CLAHE contrast enhancement
        Mat processedFace = alignedFace;
        boolean claheApplied = false;
        if (enhancements.isClahe()) {
            processedFace = ImageUtils.applyCLAHE(alignedFace);
            claheApplied = true;
            log.debug("CLAHE enhancement applied");
        }

        try {
            // Step 2: Generate base embedding
            float[] embedding = generateRawEmbedding(processedFace, enhancements.getNormalizationDivisor());

            // Step 3: Optionally apply flip-ensemble
            if (enhancements.isFlipEnsemble()) {
                Mat flippedFace = new Mat();
                org.bytedeco.opencv.global.opencv_core.flip(processedFace, flippedFace, 1);

                float[] flippedEmbedding = generateRawEmbedding(flippedFace, enhancements.getNormalizationDivisor());
                flippedFace.close();

                // Average the two embeddings
                float[] combined = new float[EMBEDDING_DIM];
                for (int i = 0; i < EMBEDDING_DIM; i++) {
                    combined[i] = embedding[i] + flippedEmbedding[i];
                }

                // L2-normalize the combined vector
                embedding = VectorNormalizationUtil.l2Normalize(combined);
                log.debug("Flip-ensemble applied");
            }

            log.info("Generated {}-D embedding (L2-normalized) [enhancements: {}]",
                    embedding.length, getActiveEnhancementNames());
            return embedding;

        } finally {
            // Cleanup CLAHE intermediate Mat if we created one
            if (claheApplied && processedFace != null) {
                processedFace.close();
            }
        }
    }

    // Generate a single raw L2-normalized embedding (no enhancements)
    // This is the core inference step used by both original and flipped paths
    private float[] generateRawEmbedding(Mat bgrFace, float normalizationDivisor) {
        try {
            OrtEnvironment env = modelManager.getEnvironment();
            OrtSession session = modelManager.getEmbeddingSession();

            // Convert aligned face BGR → RGB (ArcFace is trained on RGB)
            Mat rgbFace = ImageUtils.bgrToRgb(bgrFace);
            OnnxTensor inputTensor = OnnxTensorUtils.matToTensor(
                    env, rgbFace, INPUT_SIZE, INPUT_SIZE, normalizationDivisor);
            rgbFace.close();

            // Get the input name from model metadata
            String inputName = session.getInputNames().iterator().next();

            // Run inference
            OrtSession.Result output = session.run(
                    Collections.singletonMap(inputName, inputTensor));

            // Extract raw embedding
            float[][] rawOutput = (float[][]) output.get(0).getValue();
            float[] embedding = rawOutput[0];

            // Validate output dimensions
            if (embedding.length != EMBEDDING_DIM) {
                log.warn("Unexpected embedding dimension: {} (expected {})",
                        embedding.length, EMBEDDING_DIM);
            }

            // L2-normalize the embedding
            float[] normalized = VectorNormalizationUtil.l2Normalize(embedding);

            // Cleanup
            inputTensor.close();
            output.close();

            return normalized;

        } catch (OrtException e) {
            log.error("ArcFace embedding generation failed", e);
            throw new RuntimeException("Face embedding generation failed", e);
        }
    }

    // Get list of currently active enhancement names (for logging and responses)
    public List<String> getActiveEnhancementNames() {
        FaceAuthProperties.Enhancements enhancements = properties.getEnhancements();
        List<String> active = new ArrayList<>();

        active.add("umeyama-alignment");
        active.add("bgr-to-rgb");

        if (enhancements.isFlipEnsemble()) {
            active.add("flip-ensemble");
        }
        if (enhancements.isClahe()) {
            active.add("clahe");
        }

        float divisor = enhancements.getNormalizationDivisor();
        active.add("norm-divisor-" + String.format("%.1f", divisor));

        return active;
    }

    // Get the embedding dimension (always 512 for ArcFace)
    public int getEmbeddingDimension() {
        return EMBEDDING_DIM;
    }
}
