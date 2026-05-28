package com.faceauth.onnx;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.faceauth.config.FaceAuthProperties;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.nio.file.Path;
import java.nio.file.Paths;

// Loads and manages ONNX Runtime sessions for detection and embedding models
// Sessions are created once at startup and reused for all requests (thread-safe)
@Component
public class OnnxModelManager {

    private static final Logger log = LoggerFactory.getLogger(OnnxModelManager.class);

    private final FaceAuthProperties properties;

    private OrtEnvironment environment;
    private OrtSession detectionSession;
    private OrtSession embeddingSession;

    public OnnxModelManager(FaceAuthProperties properties) {
        this.properties = properties;
    }

    // Load both ONNX models at startup
    @PostConstruct
    public void init() throws OrtException {
        environment = OrtEnvironment.getEnvironment();

        String basePath = properties.getModels().getBasePath();
        int threads = properties.getOnnx().getIntraOpThreads();

        // Configure session options
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setIntraOpNumThreads(threads);

        // Load face detection model (SCRFD)
        String detectionModelPath = Paths.get(basePath, properties.getModels().getDetectionModel()).toString();
        log.info("Loading detection model: {}", detectionModelPath);
        detectionSession = environment.createSession(detectionModelPath, opts);
        log.info("Detection model loaded successfully");

        // Load face embedding model (ArcFace)
        String embeddingModelPath = Paths.get(basePath, properties.getModels().getEmbeddingModel()).toString();
        log.info("Loading embedding model: {}", embeddingModelPath);
        embeddingSession = environment.createSession(embeddingModelPath, opts);
        log.info("Embedding model loaded successfully");
    }

    // Clean up native resources on shutdown
    @PreDestroy
    public void cleanup() throws OrtException {
        if (detectionSession != null) detectionSession.close();
        if (embeddingSession != null) embeddingSession.close();
        if (environment != null) environment.close();
        log.info("ONNX sessions closed");
    }

    public OrtEnvironment getEnvironment() {
        return environment;
    }

    public OrtSession getDetectionSession() {
        return detectionSession;
    }

    public OrtSession getEmbeddingSession() {
        return embeddingSession;
    }
}
