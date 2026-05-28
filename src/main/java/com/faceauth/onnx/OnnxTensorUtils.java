package com.faceauth.onnx;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import org.bytedeco.opencv.opencv_core.Mat;

import java.nio.FloatBuffer;

// Converts OpenCV Mat to ONNX tensor format and vice versa
public class OnnxTensorUtils {

    private OnnxTensorUtils() {}

    // Convert BGR Mat to NCHW float tensor with normalization
    // Output shape: [1, 3, height, width]
    // Normalization: (pixel - 127.5) / 127.5 → range [-1, 1]
    public static OnnxTensor matToTensor(OrtEnvironment env, Mat mat, int height, int width)
            throws OrtException {

        int channels = mat.channels();
        float[] data = new float[1 * channels * height * width];

        // Extract pixel bytes from Mat
        byte[] pixels = new byte[height * width * channels];
        mat.data().get(pixels);

        // Convert to NCHW format with normalization
        // OpenCV Mat is in HWC (height, width, channels) BGR order
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int pixelIndex = (h * width + w) * channels;

                // BGR → channels 0,1,2 in NCHW layout
                for (int c = 0; c < channels; c++) {
                    int nchwIndex = c * height * width + h * width + w;
                    // Convert unsigned byte to float, then normalize to [-1, 1]
                    float pixelValue = (pixels[pixelIndex + c] & 0xFF);
                    data[nchwIndex] = (pixelValue - 127.5f) / 127.5f;
                }
            }
        }

        long[] shape = {1, channels, height, width};
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape);
    }

    // Convert BGR Mat to NCHW float tensor with SCRFD normalization
    // SCRFD uses different normalization: (pixel - 127.5) / 128.0
    public static OnnxTensor matToDetectionTensor(OrtEnvironment env, Mat mat, int height, int width)
            throws OrtException {

        int channels = mat.channels();
        float[] data = new float[1 * channels * height * width];

        byte[] pixels = new byte[height * width * channels];
        mat.data().get(pixels);

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int pixelIndex = (h * width + w) * channels;

                for (int c = 0; c < channels; c++) {
                    int nchwIndex = c * height * width + h * width + w;
                    float pixelValue = (pixels[pixelIndex + c] & 0xFF);
                    // SCRFD normalization
                    data[nchwIndex] = (pixelValue - 127.5f) / 128.0f;
                }
            }
        }

        long[] shape = {1, channels, height, width};
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape);
    }
}
