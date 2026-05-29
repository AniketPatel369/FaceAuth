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

    // 5-Point Least Squares Similarity Transform (Umeyama Algorithm)
    // Uses all 5 landmarks for optimal alignment to ArcFace template.
    // srcLandmarks: 5 detected landmark points from SCRFD [x,y] pairs
    // originalImage: the full original image (BGR)
    public static Mat alignFace(Mat originalImage, float[][] srcLandmarks) {
        int n = 5;
        double meanXSrc = 0, meanYSrc = 0, meanXDst = 0, meanYDst = 0;
        for (int i = 0; i < n; i++) {
            meanXSrc += srcLandmarks[i][0];
            meanYSrc += srcLandmarks[i][1];
            meanXDst += ARCFACE_REFERENCE_POINTS[i][0];
            meanYDst += ARCFACE_REFERENCE_POINTS[i][1];
        }
        meanXSrc /= n; meanYSrc /= n; meanXDst /= n; meanYDst /= n;

        double A = 0, B = 0, varSrc = 0;
        for (int i = 0; i < n; i++) {
            double dxSrc = srcLandmarks[i][0] - meanXSrc;
            double dySrc = srcLandmarks[i][1] - meanYSrc;
            double dxDst = ARCFACE_REFERENCE_POINTS[i][0] - meanXDst;
            double dyDst = ARCFACE_REFERENCE_POINTS[i][1] - meanYDst;

            A += dxSrc * dxDst + dySrc * dyDst;
            B += dxSrc * dyDst - dySrc * dxDst;
            varSrc += dxSrc * dxSrc + dySrc * dySrc;
        }

        double a = A / varSrc;
        double b = B / varSrc;

        double tx = meanXDst - (a * meanXSrc - b * meanYSrc);
        double ty = meanYDst - (b * meanXSrc + a * meanYSrc);

        Mat rotMatrix = new Mat(2, 3, org.bytedeco.opencv.global.opencv_core.CV_64F);
        rotMatrix.ptr(0, 0).putDouble(a);
        rotMatrix.ptr(0, 1).putDouble(-b);
        rotMatrix.ptr(0, 2).putDouble(tx);
        rotMatrix.ptr(1, 0).putDouble(b);
        rotMatrix.ptr(1, 1).putDouble(a);
        rotMatrix.ptr(1, 2).putDouble(ty);

        Mat aligned = new Mat();
        warpAffine(originalImage, aligned, rotMatrix, new Size(ALIGNED_SIZE, ALIGNED_SIZE));

        rotMatrix.close();
        return aligned;
    }
}

