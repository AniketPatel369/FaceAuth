package com.faceauth.util;

import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import javax.imageio.ImageIO;

import static org.bytedeco.opencv.global.opencv_imgproc.*;

// Converts between Java image formats and OpenCV Mat
public class ImageUtils {

    private ImageUtils() {}

    // Convert raw bytes (from MultipartFile) to OpenCV Mat (BGR format)
    public static Mat bytesToMat(byte[] imageBytes) {
        Mat rawData = new Mat(1, imageBytes.length, org.bytedeco.opencv.global.opencv_core.CV_8UC1);
        rawData.data().put(imageBytes, 0, imageBytes.length);
        Mat decoded = opencv_imgcodecs.imdecode(rawData, opencv_imgcodecs.IMREAD_COLOR);
        rawData.close();

        if (decoded.empty()) {
            decoded.close();
            throw new IllegalArgumentException("Failed to decode image bytes into OpenCV Mat");
        }
        return decoded;
    }

    // Get image dimensions from raw bytes without full decode
    public static int[] getImageDimensions(byte[] imageBytes) throws Exception {
        BufferedImage img = ImageIO.read(new ByteArrayInputStream(imageBytes));
        if (img == null) {
            throw new IllegalArgumentException("Cannot read image dimensions");
        }
        return new int[]{img.getWidth(), img.getHeight()};
    }

    // Resize a Mat to target dimensions
    public static Mat resize(Mat src, int width, int height) {
        Mat dst = new Mat();
        opencv_imgproc.resize(src, dst, new Size(width, height));
        return dst;
    }

    // Convert BGR Mat to RGB Mat
    public static Mat bgrToRgb(Mat bgr) {
        Mat rgb = new Mat();
        cvtColor(bgr, rgb, COLOR_BGR2RGB);
        return rgb;
    }

    // Convert Mat to JPEG bytes
    public static byte[] matToBytes(Mat image) {
        org.bytedeco.javacpp.BytePointer bytePointer = new org.bytedeco.javacpp.BytePointer();
        org.bytedeco.opencv.global.opencv_imgcodecs.imencode(".jpg", image, bytePointer);
        byte[] bytes = new byte[(int) bytePointer.limit()];
        bytePointer.get(bytes);
        bytePointer.close();
        return bytes;
    }
}
