package com.faceauth.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import java.time.LocalDateTime;

// Stores a single 512-D face embedding vector linked to a user
// Each user can have 1-4 embeddings for improved matching accuracy
@Entity
@Table(name = "face_embeddings")
@Getter
@Setter
@NoArgsConstructor
public class FaceEmbedding {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // The user this embedding belongs to
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    // 512-element float array — the actual identity vector
    // Stored as PostgreSQL float8[] for efficient read/write
    @Column(name = "embedding", nullable = false, columnDefinition = "float8[]")
    private double[] embedding;

    // Path to the aligned face image used to generate this embedding
    @Column(name = "image_path")
    private String imagePath;

    // SCRFD detection confidence when this face was captured
    @Column(name = "quality_score")
    private Float qualityScore;

    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @PrePersist
    protected void onCreate() {
        this.createdAt = LocalDateTime.now();
    }

    public FaceEmbedding(User user, double[] embedding, String imagePath, Float qualityScore) {
        this.user = user;
        this.embedding = embedding;
        this.imagePath = imagePath;
        this.qualityScore = qualityScore;
    }
}
