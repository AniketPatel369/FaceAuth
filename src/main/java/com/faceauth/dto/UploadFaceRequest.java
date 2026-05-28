package com.faceauth.dto;

import lombok.Getter;
import lombok.Setter;

// Request DTO for face enrollment — user info sent alongside the image upload
@Getter
@Setter
public class UploadFaceRequest {

    // User's display name (required for new users)
    private String name;

    // Unique identifier — employee ID, email, etc.
    private String identifier;

    // Optional: existing user ID (if adding more images to existing user)
    private Long userId;
}
