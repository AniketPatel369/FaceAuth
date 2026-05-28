package com.faceauth.service;

import com.faceauth.config.FaceAuthProperties;
import com.faceauth.entity.FaceEmbedding;
import com.faceauth.entity.User;
import com.faceauth.repository.FaceEmbeddingRepository;
import com.faceauth.repository.UserRepository;
import com.faceauth.util.VectorNormalizationUtil;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

// Stores processed face images and embedding vectors
// Handles both filesystem (aligned face) and database (embedding vector) storage
@Service
public class FaceStorageService {

    private static final Logger log = LoggerFactory.getLogger(FaceStorageService.class);

    private final UserRepository userRepository;
    private final FaceEmbeddingRepository embeddingRepository;
    private final FaceAuthProperties properties;

    public FaceStorageService(UserRepository userRepository,
                               FaceEmbeddingRepository embeddingRepository,
                               FaceAuthProperties properties) {
        this.userRepository = userRepository;
        this.embeddingRepository = embeddingRepository;
        this.properties = properties;
    }

    // Save aligned face image to filesystem and embedding to database
    // Returns the total number of enrollments for this user
    public int storeEnrollment(User user, Mat alignedFace, float[] embedding, float qualityScore) {
        // Check enrollment limit (max 4 per user)
        int currentCount = embeddingRepository.countByUserId(user.getId());
        int maxImages = properties.getEnrollment().getMaxImagesPerUser();

        if (currentCount >= maxImages) {
            throw new IllegalArgumentException(
                    "User already has " + currentCount + " enrollments (max: " + maxImages + ")");
        }

        // Save aligned face image to filesystem
        String imagePath = saveAlignedFace(user.getId(), alignedFace, currentCount + 1);

        // Convert float[] to double[] for PostgreSQL float8[]
        double[] embeddingDoubles = VectorNormalizationUtil.toDoubleArray(embedding);

        // Save embedding to database
        FaceEmbedding faceEmbedding = new FaceEmbedding(user, embeddingDoubles, imagePath, qualityScore);
        embeddingRepository.save(faceEmbedding);

        int newCount = currentCount + 1;
        log.info("Stored enrollment {}/{} for userId={}", newCount, maxImages, user.getId());

        return newCount;
    }

    // Find or create a user by identifier
    public User findOrCreateUser(String name, String identifier) {
        return userRepository.findByIdentifier(identifier)
                .orElseGet(() -> {
                    User newUser = new User(name, identifier);
                    User saved = userRepository.save(newUser);
                    log.info("Created new user: id={}, identifier={}", saved.getId(), identifier);
                    return saved;
                });
    }

    // Find existing user by ID
    public User findUserById(Long userId) {
        return userRepository.findById(userId)
                .orElseThrow(() -> new IllegalArgumentException("User not found: " + userId));
    }

    // Save the aligned face image to: faces/user_{id}/aligned_{n}.png
    private String saveAlignedFace(Long userId, Mat alignedFace, int imageNumber) {
        String baseDir = properties.getStorage().getFaceDir();
        Path userDir = Paths.get(baseDir, "user_" + userId);

        // Create directory if not exists
        File dir = userDir.toFile();
        if (!dir.exists() && !dir.mkdirs()) {
            log.warn("Failed to create face storage directory: {}", userDir);
        }

        String filename = "aligned_" + imageNumber + ".png";
        Path filePath = userDir.resolve(filename);

        // Save using OpenCV (lossless PNG)
        opencv_imgcodecs.imwrite(filePath.toString(), alignedFace);
        log.debug("Saved aligned face to: {}", filePath);

        return filePath.toString();
    }
}
