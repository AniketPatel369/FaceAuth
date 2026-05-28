package com.faceauth.onnx;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.faceauth.dto.DetectionResult;
import com.faceauth.util.ImageUtils;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.*;

// Runs SCRFD face detection model via ONNX Runtime
// Input: BGR image → Output: bounding boxes, confidence scores, 5-point landmarks
@Component
public class SCRFDDetector {

    private static final Logger log = LoggerFactory.getLogger(SCRFDDetector.class);

    // SCRFD expects 640x640 input
    private static final int INPUT_SIZE = 640;

    // Feature map strides used by SCRFD architecture
    private static final int[] FEAT_STRIDES = {8, 16, 32};

    private final OnnxModelManager modelManager;

    public SCRFDDetector(OnnxModelManager modelManager) {
        this.modelManager = modelManager;
    }

    // Detect faces in the given BGR image
    // Returns list of DetectionResult (bbox + confidence + landmarks)
    public List<DetectionResult> detect(Mat bgrImage, float confidenceThreshold) {
        List<DetectionResult> results = new ArrayList<>();

        // Scale factors to map detections back to original image size
        float scaleX = (float) bgrImage.cols() / INPUT_SIZE;
        float scaleY = (float) bgrImage.rows() / INPUT_SIZE;

        try {
            OrtEnvironment env = modelManager.getEnvironment();
            OrtSession session = modelManager.getDetectionSession();

            // Resize image to model input size
            Mat resized = ImageUtils.resize(bgrImage, INPUT_SIZE, INPUT_SIZE);

            // Create input tensor with SCRFD normalization
            OnnxTensor inputTensor = OnnxTensorUtils.matToDetectionTensor(
                    env, resized, INPUT_SIZE, INPUT_SIZE);

            // Get the input name from model metadata
            String inputName = session.getInputNames().iterator().next();

            // Run inference
            OrtSession.Result output = session.run(
                    Collections.singletonMap(inputName, inputTensor));

            // Log output tensor shapes to verify order
            output.forEach(entry -> log.info("Output key: {}, info: {}", entry.getKey(), entry.getValue().getInfo()));

            // Parse SCRFD output
            // SCRFD outputs 9 tensors: 3 strides × (scores, bboxes, landmarks)
            results = parseOutput(output, scaleX, scaleY, confidenceThreshold);

            // Cleanup
            inputTensor.close();
            output.close();
            resized.close();

        } catch (OrtException e) {
            log.error("SCRFD detection failed", e);
            throw new RuntimeException("Face detection failed", e);
        }

        return results;
    }

    // Parse SCRFD output tensors into DetectionResult objects
    private List<DetectionResult> parseOutput(OrtSession.Result output,
                                               float scaleX, float scaleY,
                                               float confidenceThreshold) throws OrtException {
        List<DetectionResult> detections = new ArrayList<>();

        // SCRFD outputs are organized by stride: 8, 16, 32
        // For each stride: score tensor, bbox tensor, landmark tensor
        int numOutputs = (int) output.size();

        // Determine output format based on number of outputs
        // Standard SCRFD: 9 outputs (3 strides × 3 outputs each)
        // Some variants: 6 outputs (3 strides × 2 outputs, landmarks in bbox)
        for (int strideIdx = 0; strideIdx < FEAT_STRIDES.length; strideIdx++) {
            int stride = FEAT_STRIDES[strideIdx];
            int gridH = INPUT_SIZE / stride;
            int gridW = INPUT_SIZE / stride;

            // Get score, bbox, and landmark tensors for this stride
            float[][] scores = (float[][]) output.get(strideIdx).getValue();
            float[][] bboxes = (float[][]) output.get(strideIdx + FEAT_STRIDES.length).getValue();
            float[][] landmarks = null;

            // Landmarks may be in a separate output group
            if (numOutputs >= 9) {
                landmarks = (float[][]) output.get(strideIdx + FEAT_STRIDES.length * 2).getValue();
            }

            int numAnchors = scores.length / (gridH * gridW);

            // Iterate over anchor grid
            for (int i = 0; i < scores.length; i++) {
                float score = scores[i][0];

                // Skip low-confidence detections
                if (score < confidenceThreshold) continue;

                // Compute anchor center
                int gridPos = i / numAnchors;
                int gridY = gridPos / gridW;
                int gridX = gridPos % gridW;
                float anchorCx = (gridX + 0.5f) * stride;
                float anchorCy = (gridY + 0.5f) * stride;

                // Decode bounding box (distance from anchor to edges)
                float x1 = (anchorCx - bboxes[i][0] * stride) * scaleX;
                float y1 = (anchorCy - bboxes[i][1] * stride) * scaleY;
                float x2 = (anchorCx + bboxes[i][2] * stride) * scaleX;
                float y2 = (anchorCy + bboxes[i][3] * stride) * scaleY;

                float[] bbox = {x1, y1, x2, y2};

                // Decode landmarks (5 points × 2 coordinates)
                float[][] lmks = new float[5][2];
                if (landmarks != null) {
                    for (int j = 0; j < 5; j++) {
                        lmks[j][0] = (anchorCx + landmarks[i][j * 2] * stride) * scaleX;
                        lmks[j][1] = (anchorCy + landmarks[i][j * 2 + 1] * stride) * scaleY;
                    }
                }

                detections.add(new DetectionResult(bbox, score, lmks));
            }
        }

        // Apply Non-Maximum Suppression to remove overlapping detections
        return nms(detections, 0.4f);
    }

    // Non-Maximum Suppression — keeps only the best non-overlapping detections
    private List<DetectionResult> nms(List<DetectionResult> detections, float nmsThreshold) {
        if (detections.isEmpty()) return detections;

        // Sort by confidence descending
        detections.sort((a, b) -> Float.compare(b.getConfidence(), a.getConfidence()));

        List<DetectionResult> kept = new ArrayList<>();
        boolean[] suppressed = new boolean[detections.size()];

        for (int i = 0; i < detections.size(); i++) {
            if (suppressed[i]) continue;
            kept.add(detections.get(i));

            for (int j = i + 1; j < detections.size(); j++) {
                if (suppressed[j]) continue;
                float iou = computeIoU(
                        detections.get(i).getBoundingBox(),
                        detections.get(j).getBoundingBox());
                if (iou > nmsThreshold) {
                    suppressed[j] = true;
                }
            }
        }

        return kept;
    }

    // Intersection over Union — measures overlap between two bounding boxes
    private float computeIoU(float[] boxA, float[] boxB) {
        float x1 = Math.max(boxA[0], boxB[0]);
        float y1 = Math.max(boxA[1], boxB[1]);
        float x2 = Math.min(boxA[2], boxB[2]);
        float y2 = Math.min(boxA[3], boxB[3]);

        float intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        float areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
        float areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

        float union = areaA + areaB - intersection;
        return union > 0 ? intersection / union : 0;
    }
}
