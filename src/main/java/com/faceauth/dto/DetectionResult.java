package com.faceauth.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;

// Internal DTO — carries SCRFD detection results between services
// Not exposed via API
@Getter
@AllArgsConstructor
public class DetectionResult {

    // Bounding box of detected face [x1, y1, x2, y2]
    private final float[] boundingBox;

    // Detection confidence score (0.0 to 1.0)
    private final float confidence;

    // 5 landmark points: [leftEye, rightEye, nose, leftMouth, rightMouth]
    // Each point is [x, y], so this is float[5][2]
    private final float[][] landmarks;

    // Convenience: face width in pixels
    public float getFaceWidth() {
        return boundingBox[2] - boundingBox[0];
    }

    // Convenience: face height in pixels
    public float getFaceHeight() {
        return boundingBox[3] - boundingBox[1];
    }
}
