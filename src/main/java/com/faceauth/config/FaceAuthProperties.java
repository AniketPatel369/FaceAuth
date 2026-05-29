package com.faceauth.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

// Binds all "faceauth.*" properties from application.yml to typed Java fields
@Getter
@Setter
@Configuration
@ConfigurationProperties(prefix = "faceauth")
public class FaceAuthProperties {

    private Models models = new Models();
    private Onnx onnx = new Onnx();
    private Detection detection = new Detection();
    private Matching matching = new Matching();
    private Image image = new Image();
    private Storage storage = new Storage();
    private Enrollment enrollment = new Enrollment();
    private Enhancements enhancements = new Enhancements();

    // ONNX model file paths
    @Getter @Setter
    public static class Models {
        private String detectionModel = "det_2.5g.onnx";
        private String embeddingModel = "w600k_r50.onnx";
        private String basePath = "models";
    }

    // ONNX Runtime thread settings
    @Getter @Setter
    public static class Onnx {
        private int intraOpThreads = 4;
    }

    // Face detection thresholds
    @Getter @Setter
    public static class Detection {
        private float minConfidence = 0.5f;
        private int minFaceSize = 80;
    }

    // Face matching thresholds
    @Getter @Setter
    public static class Matching {
        private float acceptThreshold = 0.45f;
        private float rejectThreshold = 0.30f;
    }

    // Image validation constraints
    @Getter @Setter
    public static class Image {
        private int minWidth = 300;
        private int minHeight = 300;
        private int maxWidth = 5000;
        private int maxHeight = 5000;
        private String allowedTypes = "image/jpeg,image/png";
    }

    // File storage settings
    @Getter @Setter
    public static class Storage {
        private String faceDir = "faces";
    }

    // Enrollment limits
    @Getter @Setter
    public static class Enrollment {
        private int maxImagesPerUser = 4;
    }

    // Enhancement feature flags — toggle individual improvements
    // WARNING: Changing embedding-affecting flags requires re-enrollment!
    @Getter @Setter
    public static class Enhancements {
        // Flip-Ensemble: average original + mirrored face embedding (was V3)
        private boolean flipEnsemble = true;

        // CLAHE: Apply contrast enhancement on luminance channel before embedding
        private boolean clahe = false;

        // Pixel normalization divisor (127.5 = legacy, 128.0 = InsightFace standard)
        private float normalizationDivisor = 127.5f;
    }
}
