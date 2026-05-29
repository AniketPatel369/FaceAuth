package com.faceauth.service;

import com.faceauth.onnx.ArcFaceEmbedder;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.List;

// Generates 512-D face embedding from an aligned face image
// Wraps ArcFaceEmbedder with logging and validation
// All configured enhancements (CLAHE, flip-ensemble, etc.) are applied automatically
@Service
public class FaceEmbeddingService {

    private static final Logger log = LoggerFactory.getLogger(FaceEmbeddingService.class);

    private final ArcFaceEmbedder embedder;

    public FaceEmbeddingService(ArcFaceEmbedder embedder) {
        this.embedder = embedder;
    }

    // Generate L2-normalized 512-D embedding from aligned 112x112 face
    // Automatically applies all enabled enhancements from feature flags
    public float[] generateEmbedding(Mat alignedFace) {
        if (alignedFace.cols() != 112 || alignedFace.rows() != 112) {
            throw new IllegalArgumentException(
                    "Expected 112x112 aligned face, got: " +
                    alignedFace.cols() + "x" + alignedFace.rows());
        }

        float[] embedding = embedder.generateEmbedding(alignedFace);

        log.info("Generated {}-D embedding (L2-normalized)", embedding.length);
        return embedding;
    }

    // Get the list of currently active enhancement names
    // Used to include in API responses for transparency
    public List<String> getActiveEnhancements() {
        return embedder.getActiveEnhancementNames();
    }
}
