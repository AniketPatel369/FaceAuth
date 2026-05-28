package com.faceauth.service;

import com.faceauth.dto.DetectionResult;
import com.faceauth.util.AlignmentUtils;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

// Aligns a detected face to the ArcFace standard template (112x112)
// Uses 5-point landmarks from SCRFD to compute affine transform
@Service
public class FaceAlignmentService {

    private static final Logger log = LoggerFactory.getLogger(FaceAlignmentService.class);

    // Align face using landmarks from detection result
    // Returns a 112x112 BGR image with face in standard position
    public Mat alignFace(Mat originalImage, DetectionResult detection) {
        float[][] landmarks = detection.getLandmarks();

        // Validate landmarks exist
        if (landmarks == null || landmarks.length < 5) {
            throw new IllegalArgumentException(
                    "Insufficient landmarks for alignment. Expected 5, got: " +
                    (landmarks == null ? 0 : landmarks.length));
        }

        log.debug("Aligning face with landmarks: leftEye=[{},{}], rightEye=[{},{}]",
                landmarks[0][0], landmarks[0][1],
                landmarks[1][0], landmarks[1][1]);

        // Compute affine transform and warp face to 112x112
        Mat aligned = AlignmentUtils.alignFace(originalImage, landmarks);

        if (aligned.empty()) {
            aligned.close();
            throw new RuntimeException("Face alignment produced empty result");
        }

        log.info("Face aligned to {}x{}", aligned.cols(), aligned.rows());
        return aligned;
    }
}
