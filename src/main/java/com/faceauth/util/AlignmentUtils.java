package com.faceauth.util;

import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point2f;
import org.bytedeco.opencv.opencv_core.Size;

import static org.bytedeco.opencv.global.opencv_imgproc.*;

// Computes affine transform to align detected face to ArcFace standard template
public class AlignmentUtils {

    private AlignmentUtils() {}

    // ArcFace standard reference landmarks for 112x112 aligned face
    // Order: left eye, right eye, nose, left mouth, right mouth
    public static final float[][] ARCFACE_REFERENCE_POINTS = {
            {38.2946f, 51.6963f},   // left eye
            {73.5318f, 51.5014f},   // right eye
            {56.0252f, 71.7366f},   // nose tip
            {41.5493f, 92.3655f},   // left mouth corner
            {70.7299f, 92.2041f}    // right mouth corner
    };

    // Aligned face output size (ArcFace standard)
    public static final int ALIGNED_SIZE = 112;

    // Compute similarity transform and warp face to 112x112 aligned output
    // srcLandmarks: 5 detected landmark points from SCRFD [x,y] pairs
    // originalImage: the full original image (BGR)
    public static Mat alignFace(Mat originalImage, float[][] srcLandmarks) {
        // Use the two eye points for a simpler, robust affine transform
        // This avoids numerical instability from using all 5 points
        float[] leftEye = srcLandmarks[0];
        float[] rightEye = srcLandmarks[1];
        float[] refLeftEye = ARCFACE_REFERENCE_POINTS[0];
        float[] refRightEye = ARCFACE_REFERENCE_POINTS[1];

        // Compute angle between eyes
        double srcAngle = Math.atan2(rightEye[1] - leftEye[1], rightEye[0] - leftEye[0]);
        double dstAngle = Math.atan2(refRightEye[1] - refLeftEye[1], refRightEye[0] - refLeftEye[0]);
        double angle = Math.toDegrees(srcAngle - dstAngle);

        // Compute scale from eye distance ratio
        double srcDist = Math.sqrt(
                Math.pow(rightEye[0] - leftEye[0], 2) +
                Math.pow(rightEye[1] - leftEye[1], 2));
        double dstDist = Math.sqrt(
                Math.pow(refRightEye[0] - refLeftEye[0], 2) +
                Math.pow(refRightEye[1] - refLeftEye[1], 2));
        double scale = dstDist / srcDist;

        // Center point between eyes in source image
        float centerX = (leftEye[0] + rightEye[0]) / 2f;
        float centerY = (leftEye[1] + rightEye[1]) / 2f;

        // Get rotation matrix around eye center
        Point2f center = new Point2f(centerX, centerY);
        Mat rotMatrix = getRotationMatrix2D(center, angle, scale);

        // Adjust translation to move eyes to reference position
        double refCenterX = (refLeftEye[0] + refRightEye[0]) / 2.0;
        double refCenterY = (refLeftEye[1] + refRightEye[1]) / 2.0;

        // Adjust the translation component of the matrix
        double tx = rotMatrix.ptr(0, 2).getDouble();
        double ty = rotMatrix.ptr(1, 2).getDouble();
        rotMatrix.ptr(0, 2).putDouble(tx + refCenterX - centerX);
        rotMatrix.ptr(1, 2).putDouble(ty + refCenterY - centerY);

        // Warp the original image
        Mat aligned = new Mat();
        warpAffine(originalImage, aligned, rotMatrix,
                new Size(ALIGNED_SIZE, ALIGNED_SIZE));

        rotMatrix.close();
        center.close();

        return aligned;
    }
}
