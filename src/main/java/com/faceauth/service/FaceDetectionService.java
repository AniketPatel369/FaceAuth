package com.faceauth.service;

import com.faceauth.config.FaceAuthProperties;
import com.faceauth.dto.DetectionResult;
import com.faceauth.onnx.SCRFDDetector;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.List;

// Validates and runs face detection on uploaded images
// Enforces: exactly 1 face, sufficient confidence, minimum face size
@Service
public class FaceDetectionService {

    private static final Logger log = LoggerFactory.getLogger(FaceDetectionService.class);

    private final SCRFDDetector detector;
    private final FaceAuthProperties properties;

    public FaceDetectionService(SCRFDDetector detector, FaceAuthProperties properties) {
        this.detector = detector;
        this.properties = properties;
    }

    // Detect exactly one face in the image, return its detection result
    // Throws if: no face, multiple faces, face too small, low confidence
    public DetectionResult detectSingleFace(Mat bgrImage) {
        float minConfidence = properties.getDetection().getMinConfidence();
        int minFaceSize = properties.getDetection().getMinFaceSize();

        // Run SCRFD detection
        List<DetectionResult> detections = detector.detect(bgrImage, minConfidence);

        // Validate: at least one face found
        if (detections.isEmpty()) {
            throw new IllegalArgumentException("No face detected in the image");
        }

        // Validate: only one face allowed
        if (detections.size() > 1) {
            throw new IllegalArgumentException(
                    "Multiple faces detected (" + detections.size() +
                    "). Please upload an image with only one face");
        }

        DetectionResult face = detections.get(0);

        // Validate: face is large enough
        if (face.getFaceWidth() < minFaceSize || face.getFaceHeight() < minFaceSize) {
            throw new IllegalArgumentException(
                    "Face too small: " + (int) face.getFaceWidth() + "x" +
                    (int) face.getFaceHeight() + " pixels. Minimum: " +
                    minFaceSize + "x" + minFaceSize);
        }

        log.info("Face detected: confidence={}, size={}x{}",
                String.format("%.3f", face.getConfidence()),
                (int) face.getFaceWidth(),
                (int) face.getFaceHeight());

        return face;
    }
}
