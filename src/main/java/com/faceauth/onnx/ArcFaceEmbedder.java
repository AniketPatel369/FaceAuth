package com.faceauth.onnx;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.faceauth.util.VectorNormalizationUtil;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.Collections;

// Runs ArcFace embedding model via ONNX Runtime
// Input: aligned 112x112 BGR face → Output: L2-normalized 512-D embedding
@Component
public class ArcFaceEmbedder {

    private static final Logger log = LoggerFactory.getLogger(ArcFaceEmbedder.class);

    // ArcFace input dimensions
    private static final int INPUT_SIZE = 112;

    // Expected output embedding dimension
    private static final int EMBEDDING_DIM = 512;

    private final OnnxModelManager modelManager;

    public ArcFaceEmbedder(OnnxModelManager modelManager) {
        this.modelManager = modelManager;
    }

    // Generate face embedding from an aligned 112x112 BGR face image
    // Returns L2-normalized 512-D float vector
    public float[] generateEmbedding(Mat alignedFace) {
        try {
            OrtEnvironment env = modelManager.getEnvironment();
            OrtSession session = modelManager.getEmbeddingSession();

            // Convert aligned face to input tensor
            // OpenCV loads as BGR, but ArcFace is trained on RGB. We must convert it.
            // Normalization: (pixel - 127.5) / 127.5 → [-1, 1]
            Mat rgbFace = com.faceauth.util.ImageUtils.bgrToRgb(alignedFace);
            OnnxTensor inputTensor = OnnxTensorUtils.matToTensor(
                    env, rgbFace, INPUT_SIZE, INPUT_SIZE);
            rgbFace.close(); // Clean up intermediate Mat

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

    // Get the embedding dimension (always 512 for ArcFace)
    public int getEmbeddingDimension() {
        return EMBEDDING_DIM;
    }
}
