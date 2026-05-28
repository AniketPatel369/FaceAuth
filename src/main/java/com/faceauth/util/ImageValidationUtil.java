package com.faceauth.util;

import com.faceauth.config.FaceAuthProperties;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;
import java.util.Set;

// Validates uploaded images before processing
@Component
public class ImageValidationUtil {

    private final FaceAuthProperties properties;

    // Allowed MIME types for upload
    private static final Set<String> ALLOWED_MIME_TYPES = Set.of(
            "image/jpeg",
            "image/png"
    );

    public ImageValidationUtil(FaceAuthProperties properties) {
        this.properties = properties;
    }

    // Validate the uploaded file (type, size, dimensions)
    public void validate(MultipartFile file, byte[] imageBytes) throws Exception {
        validateNotEmpty(file);
        validateMimeType(file);
        validateFileSize(file);
        validateDimensions(imageBytes);
    }

    private void validateNotEmpty(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new IllegalArgumentException("Image file is required");
        }
    }

    // Check MIME type against whitelist
    private void validateMimeType(MultipartFile file) {
        String contentType = file.getContentType();
        if (contentType == null || !ALLOWED_MIME_TYPES.contains(contentType.toLowerCase())) {
            throw new IllegalArgumentException(
                    "Invalid image type: " + contentType + ". Allowed: JPG, PNG");
        }
    }

    // Check file size (max 10MB configured in application.yml)
    private void validateFileSize(MultipartFile file) {
        long maxSize = 10 * 1024 * 1024; // 10MB
        if (file.getSize() > maxSize) {
            throw new IllegalArgumentException(
                    "File too large: " + file.getSize() + " bytes. Max: " + maxSize);
        }
    }

    // Check image dimensions against configured min/max
    private void validateDimensions(byte[] imageBytes) throws Exception {
        int[] dims = ImageUtils.getImageDimensions(imageBytes);
        int width = dims[0];
        int height = dims[1];

        FaceAuthProperties.Image img = properties.getImage();

        if (width < img.getMinWidth() || height < img.getMinHeight()) {
            throw new IllegalArgumentException(
                    "Image too small: " + width + "x" + height +
                    ". Minimum: " + img.getMinWidth() + "x" + img.getMinHeight());
        }

        if (width > img.getMaxWidth() || height > img.getMaxHeight()) {
            throw new IllegalArgumentException(
                    "Image too large: " + width + "x" + height +
                    ". Maximum: " + img.getMaxWidth() + "x" + img.getMaxHeight());
        }
    }
}
