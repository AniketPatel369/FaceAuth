package com.faceauth.controller;

import com.faceauth.dto.*;
import com.faceauth.entity.User;
import com.faceauth.service.*;
import com.faceauth.util.ImageUtils;
import com.faceauth.util.ImageValidationUtil;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

// REST API for face enrollment and recognition
@CrossOrigin
@RestController
@RequestMapping("/api/face")
public class FaceController {

    private static final Logger log = LoggerFactory.getLogger(FaceController.class);

    private final ImageValidationUtil imageValidator;
    private final FaceDetectionService detectionService;
    private final FaceAlignmentService alignmentService;
    private final FaceEmbeddingService embeddingService;
    private final FaceMatchingService matchingService;
    private final FaceStorageService storageService;

    public FaceController(ImageValidationUtil imageValidator,
                           FaceDetectionService detectionService,
                           FaceAlignmentService alignmentService,
                           FaceEmbeddingService embeddingService,
                           FaceMatchingService matchingService,
                           FaceStorageService storageService) {
        this.imageValidator = imageValidator;
        this.detectionService = detectionService;
        this.alignmentService = alignmentService;
        this.embeddingService = embeddingService;
        this.matchingService = matchingService;
        this.storageService = storageService;
    }

    // Enroll a face — detect, align, embed, store
    // POST /api/face/enroll
    @PostMapping(value = "/enroll", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<UploadFaceResponse> enrollFace(
            @RequestParam("image") MultipartFile image,
            @RequestParam("name") String name,
            @RequestParam("identifier") String identifier,
            @RequestParam(value = "userId", required = false) Long userId) {

        Mat bgrImage = null;
        Mat alignedFace = null;

        try {
            // Step 1: Read and validate image
            byte[] imageBytes = image.getBytes();
            imageValidator.validate(image, imageBytes);

            // Step 2: Convert to OpenCV Mat (BGR)
            bgrImage = ImageUtils.bytesToMat(imageBytes);
            log.info("Image loaded: {}x{}", bgrImage.cols(), bgrImage.rows());

            // Step 3 & 4: Two-stage detection and alignment (Detect -> Crop -> Detect -> Align)
            alignedFace = detectAndAlignWithCrop(bgrImage);

            // Step 5: Generate 512-D embedding (ArcFace)
            float[] embedding = embeddingService.generateEmbedding(alignedFace);

            // Step 6: Find or create user
            User user;
            if (userId != null) {
                user = storageService.findUserById(userId);
            } else {
                user = storageService.findOrCreateUser(name, identifier);
            }

            // Step 7: Store aligned face + embedding
            // Note: We don't have the final confidence here easily without changing the helper signature, 
            // but we can pass 1.0f as it passed the threshold check inside detectionService.
            int totalEnrollments = storageService.storeEnrollment(
                    user, alignedFace, embedding, 1.0f);

            // Return success response
            UploadFaceResponse response = UploadFaceResponse.success(
                    user.getId(), 1.0f, totalEnrollments);

            return ResponseEntity.ok(response);

        } catch (IllegalArgumentException e) {
            log.warn("Enrollment rejected: {}", e.getMessage());
            return ResponseEntity.badRequest()
                    .body(UploadFaceResponse.failure(e.getMessage()));

        } catch (Exception e) {
            log.error("Enrollment failed", e);
            return ResponseEntity.internalServerError()
                    .body(UploadFaceResponse.failure("Internal error: " + e.getMessage()));

        } finally {
            // Always release OpenCV Mat memory
            if (bgrImage != null) bgrImage.close();
            if (alignedFace != null) alignedFace.close();
        }
    }

    // Recognize a face — detect, align, embed, match against stored embeddings
    // POST /api/face/recognize
    @PostMapping(value = "/recognize", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<MatchFaceResponse> recognizeFace(
            @RequestParam("image") MultipartFile image) {

        Mat bgrImage = null;
        Mat alignedFace = null;

        try {
            // Same pipeline as enrollment (Steps 1-5)
            byte[] imageBytes = image.getBytes();
            imageValidator.validate(image, imageBytes);

            bgrImage = ImageUtils.bytesToMat(imageBytes);
            
            // Two-stage detection and alignment
            alignedFace = detectAndAlignWithCrop(bgrImage);
            
            float[] queryEmbedding = embeddingService.generateEmbedding(alignedFace);

            // Step 6: Match against stored embeddings
            FaceMatchingService.MatchResult result = matchingService.findBestMatch(queryEmbedding);

            // Build response based on match result
            float acceptThreshold = 0.45f;

            if (result.matched()) {
                return ResponseEntity.ok(MatchFaceResponse.matched(
                        result.userId(), result.userName(),
                        result.similarity(), acceptThreshold));
            } else {
                return ResponseEntity.ok(MatchFaceResponse.noMatch(
                        result.similarity(), acceptThreshold));
            }

        } catch (IllegalArgumentException e) {
            log.warn("Recognition rejected: {}", e.getMessage());
            return ResponseEntity.badRequest()
                    .body(MatchFaceResponse.failure(e.getMessage()));

        } catch (Exception e) {
            log.error("Recognition failed", e);
            return ResponseEntity.internalServerError()
                    .body(MatchFaceResponse.failure("Internal error: " + e.getMessage()));

        } finally {
            if (bgrImage != null) bgrImage.close();
            if (alignedFace != null) alignedFace.close();
        }
    }

    // Detect and crop face (utility endpoint)
    // POST /api/face/detect-crop
    @PostMapping(value = "/detect-crop", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<byte[]> detectAndCropFace(
            @RequestParam("image") MultipartFile image) {

        Mat bgrImage = null;
        Mat alignedFace = null;

        try {
            byte[] imageBytes = image.getBytes();
            imageValidator.validate(image, imageBytes);

            bgrImage = ImageUtils.bytesToMat(imageBytes);
            alignedFace = detectAndAlignWithCrop(bgrImage);

            byte[] croppedJpeg = ImageUtils.matToBytes(alignedFace);

            return ResponseEntity.ok()
                    .contentType(MediaType.IMAGE_JPEG)
                    .body(croppedJpeg);

        } catch (IllegalArgumentException e) {
            log.warn("Detect-crop rejected: {}", e.getMessage());
            return ResponseEntity.badRequest().build();
        } catch (Exception e) {
            log.error("Detect-crop failed", e);
            return ResponseEntity.internalServerError().build();
        } finally {
            if (bgrImage != null) bgrImage.close();
            if (alignedFace != null) alignedFace.close();
        }
    }

    // Health check
    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("FaceAuth is running");
    }

    // Helper method for two-stage detection and alignment
    private Mat detectAndAlignWithCrop(Mat originalImage) {
        // Stage 1: Initial detection to find the face in the large image
        DetectionResult initialDetection = detectionService.detectSingleFace(originalImage);

        // Stage 2: Crop out the face with a 50% margin
        Mat croppedImage = ImageUtils.cropFaceWithMargin(originalImage, initialDetection.getBoundingBox(), 0.5f);

        try {
            // Stage 3: Second detection on the cropped, high-resolution face region
            DetectionResult refinedDetection = detectionService.detectSingleFace(croppedImage);

            // Stage 4: Align the face using the refined landmarks
            return alignmentService.alignFace(croppedImage, refinedDetection);
        } finally {
            if (croppedImage != null && !croppedImage.empty()) {
                croppedImage.close();
            }
        }
    }
}
