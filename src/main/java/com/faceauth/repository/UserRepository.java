package com.faceauth.repository;

import com.faceauth.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.Optional;

// Data access for User entity
@Repository
public interface UserRepository extends JpaRepository<User, Long> {

    // Find user by unique identifier (employee ID, email, etc.)
    Optional<User> findByIdentifier(String identifier);

    // Check if identifier already exists
    boolean existsByIdentifier(String identifier);
}
