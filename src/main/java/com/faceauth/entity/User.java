package com.faceauth.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import java.time.LocalDateTime;

// Represents a registered user in the system
@Entity
@Table(name = "users")
@Getter
@Setter
@NoArgsConstructor
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // Display name of the user
    @Column(nullable = false)
    private String name;

    // Unique identifier (employee ID, email, etc.)
    @Column(nullable = false, unique = true)
    private String identifier;

    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    // Active flag — soft delete support
    @Column(nullable = false)
    private boolean active = true;

    @PrePersist
    protected void onCreate() {
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() {
        this.updatedAt = LocalDateTime.now();
    }

    public User(String name, String identifier) {
        this.name = name;
        this.identifier = identifier;
    }
}
