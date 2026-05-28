package com.faceauth.config;

import jakarta.annotation.PostConstruct;
import org.bytedeco.opencv.global.opencv_core;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Configuration;

// Loads OpenCV native libraries at application startup
@Configuration
public class OpenCVConfig {

    private static final Logger log = LoggerFactory.getLogger(OpenCVConfig.class);

    // Called once when Spring context initializes
    @PostConstruct
    public void init() {
        try {
            // bytedeco auto-loads the correct native binary for the current OS
            int cores = opencv_core.getNumThreads();
            log.info("OpenCV loaded successfully. Available threads: {}", cores);
        } catch (Exception e) {
            log.error("Failed to load OpenCV native libraries", e);
            throw new RuntimeException("OpenCV initialization failed", e);
        }
    }
}
