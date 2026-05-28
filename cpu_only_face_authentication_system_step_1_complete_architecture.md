# CPU-Only Face Authentication System
## Complete Step 1 Architecture Documentation

---

# 1. Project Objective

Build a fully local CPU-only face authentication system for:
- employee punch systems
- office attendance
- kiosk login
- secure internal authentication

The system will:
- upload user face image
- detect face
- normalize face
- generate identity vector
- store vector for future matching

The system will NOT:
- use GPU
- use cloud AI APIs
- perform realtime CCTV analytics
- process video streams
- perform public surveillance

This architecture is designed specifically for:
- VPS deployment
- CPU-only execution
- controlled capture environments
- moderate user scale

---

# 2. System Design Philosophy

This system depends on:
- controlled image capture
- deterministic preprocessing
- stable normalization
- consistent vector generation

Instead of solving every real-world condition with huge AI models.

Controlled assumptions:
- user cooperates
- front-facing image
- decent lighting
- decent camera
- single face visible
- no sunglasses
- limited extreme angles

This simplifies the recognition problem enormously.

---

# 3. Recommended Technology Stack

## Backend

### Java Version
Java 21 LTS

Reason:
- stable long-term support
- modern concurrency
- virtual threads
- production-ready
- stable OpenCV compatibility

---

## Framework

Spring Boot 3.3+

Purpose:
- REST APIs
- dependency injection
- upload handling
- service architecture
- future scalability

---

## Build Tool

Maven

Reason:
- dependency stability
- OpenCV compatibility
- standard ecosystem support

---

## Computer Vision Library

OpenCV

Dependency:

```xml
<dependency>
    <groupId>org.bytedeco</groupId>
    <artifactId>opencv-platform</artifactId>
    <version>4.10.0-1.5.11</version>
</dependency>
```

Reason:
- includes native binaries
- CPU optimized
- easy deployment
- stable Java bindings

---

## Database

PostgreSQL

Purpose:
- user storage
- vector storage
- metadata storage
- authentication logs

---

# 4. Initial Project Structure

```text
src/main/java/com/company/faceauth/

├── controller/
│   └── FaceController.java
│
├── service/
│   ├── FaceDetectionService.java
│   ├── FaceNormalizationService.java
│   ├── FaceEmbeddingService.java
│   ├── FaceMatchingService.java
│   └── FaceStorageService.java
│
├── vision/
│   ├── OpenCVConfig.java
│   ├── HaarCascadeLoader.java
│   ├── LandmarkUtils.java
│   └── ImageUtils.java
│
├── repository/
│   ├── UserRepository.java
│   └── FaceEmbeddingRepository.java
│
├── entity/
│   ├── User.java
│   └── FaceEmbedding.java
│
├── dto/
│   ├── UploadFaceRequest.java
│   ├── UploadFaceResponse.java
│   └── MatchFaceResponse.java
│
└── util/
    ├── CosineSimilarityUtil.java
    └── ImageValidationUtil.java
```

---

# 5. Full Face Processing Pipeline

```text
Upload Image
    ↓
Validate File
    ↓
Convert To OpenCV Mat
    ↓
Convert To Grayscale
    ↓
Detect Face
    ↓
Validate Single Face
    ↓
Crop Face Region
    ↓
Align Face
    ↓
Normalize Face
    ↓
Histogram Equalization
    ↓
Resize Face
    ↓
Generate Feature Vector
    ↓
Store Processed Face
    ↓
Store Vector Embedding
```

This pipeline MUST remain deterministic.

Every image must go through the EXACT same processing sequence.

---

# 6. Step-by-Step Detailed Flow

# STEP 1 — Image Upload

## API Endpoint

```http
POST /api/face/upload
```

Multipart form upload:

```java
MultipartFile image
```

---

## File Validation Rules

Allowed:
- JPG
- JPEG
- PNG
- WEBP

Rejected:
- GIF
- HEIC
- SVG
- corrupted files

---

## Size Rules

Minimum:
```text
300x300
```

Maximum:
```text
5000x5000
```

Reason:
- avoid tiny unusable faces
- avoid massive memory usage

---

## File Size Limit

```text
10MB
```

---

# STEP 2 — Convert Image To OpenCV Mat

Spring Boot upload:

```java
BufferedImage
```

Convert into:

```java
Mat
```

Reason:
- OpenCV operates on Mat structures

---

## Conversion Flow

```text
MultipartFile
    ↓
BufferedImage
    ↓
Byte Array
    ↓
OpenCV Mat
```

---

# STEP 3 — Convert To Grayscale

Purpose:
- reduce noise
- reduce color dependency
- reduce memory
- improve classical CV stability

OpenCV:

```java
Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
```

Result:

```text
RGB Face → Grayscale Face
```

---

# STEP 4 — Face Detection

## Initial Detector

Haar Cascade Classifier

Model:

```text
haarcascade_frontalface_alt.xml
```

Reason:
- CPU friendly
- lightweight
- deterministic
- enough for MVP

---

## Detection Logic

```java
CascadeClassifier.detectMultiScale()
```

Returns:

```java
Rect[] faces
```

---

# Face Validation Rules

Reject if:
- no face found
- multiple faces found
- face too small
- face too tilted
- blurry image

Reason:
- prevent low-quality enrollment

---

## Blur Detection

Future implementation:

Laplacian variance.

Reason:
- blurry faces produce unstable vectors

---

# STEP 5 — Crop Face

Largest detected face selected.

Crop:

```java
Mat croppedFace = new Mat(grayImage, faceRect);
```

Result:

```text
background removed
only face retained
```

---

# STEP 6 — Face Alignment

Purpose:
- stabilize eye position
- reduce tilt variation
- improve matching consistency

---

## Landmark Detection

Detect:
- left eye
- right eye
- nose
- mouth corners

---

## Alignment Logic

Rotate image until:

```text
eyes become horizontally aligned
```

Reason:
- same person should produce similar geometry

---

# STEP 7 — Face Normalization

Purpose:
- create consistent face geometry
- stabilize recognition

Normalization includes:
- centered face
- fixed dimensions
- stable eye position
- consistent crop area

---

# STEP 8 — Histogram Equalization

Purpose:
- normalize brightness
- normalize contrast
- reduce minor lighting variation

OpenCV:

```java
Imgproc.equalizeHist(grayFace, normalizedFace);
```

---

# STEP 9 — Resize Face

Target size:

```text
160x160
```

Reason:
- lightweight
- CPU efficient
- sufficient detail

Resize:

```java
Imgproc.resize(face, output, new Size(160, 160));
```

Result:

```text
all faces become identical dimensions
```

---

# STEP 10 — Generate Vector Embedding

This converts face image into mathematical identity representation.

---

# Initial Embedding Strategy

Use:

```text
LBPH
```

Meaning:

```text
Local Binary Pattern Histogram
```

---

# Why LBPH

Advantages:
- CPU-only
- lightweight
- deterministic
- no GPU
- no cloud AI
- trainable locally
- works in controlled environments

Disadvantages:
- weaker under major appearance changes
- less robust than deep embeddings

Acceptable for controlled authentication systems.

---

# LBPH Processing Flow

```text
normalized face
    ↓
texture extraction
    ↓
local binary pattern generation
    ↓
histogram generation
    ↓
feature vector
```

Result:

```text
identity vector representation
```

---

# Generated Vector Example

```text
[0.12, -0.44, 0.81, ...]
```

This becomes the stored biometric identity.

---

# STEP 11 — Store Processed Face

Store processed normalized face.

DO NOT store raw uploaded image as primary matching source.

Folder structure:

```text
faces/
   user_101/
      normalized.png
```

Purpose:
- future debugging
- reprocessing
- retraining

---

# STEP 12 — Store Vector Embedding

Database table:

```sql
CREATE TABLE face_embeddings (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    embedding JSONB NOT NULL,
    image_path TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

# Stored Data

Per user:
- user_id
- embedding vector
- processed face path
- metadata

---

# 7. Future Recognition Flow

Recognition pipeline:

```text
new image
    ↓
upload
    ↓
same preprocessing
    ↓
same normalization
    ↓
same vector generation
    ↓
compare against DB vectors
    ↓
return best match
```

VERY IMPORTANT:

Recognition preprocessing MUST remain IDENTICAL to enrollment preprocessing.

Otherwise vector consistency breaks.

---

# 8. Similarity Matching

Comparison method:

```text
cosine similarity
```

Purpose:
- determine closeness between vectors

---

# Matching Logic

```java
similarity = cosine(vectorA, vectorB)
```

Result:

```text
0.0 → completely different
1.0 → identical
```

---

# Threshold Examples

```text
> 0.92 → accepted
0.85–0.92 → uncertain
< 0.85 → rejected
```

Threshold tuning comes later.

---

# 9. Performance Design

# VPS Requirements

Recommended:

```text
4–8 vCPU
8GB RAM
```

---

# Why CPU-Only Works

Because system is:
- single-face
- single-image
- controlled capture
- no realtime video
- no surveillance

The system processes:

```text
one face at a time
```

which is lightweight.

---

# Expected Performance

Approximate:

```text
image preprocessing → 50–150ms
vector generation → 20–50ms
vector comparison → <10ms
```

Total:

```text
under 300ms realistic
```

---

# 10. Memory Strategy

Keep vectors in memory cache later.

Reason:

```text
10k users × vectors
```

is very small memory usage.

---

# Why Vector DB Not Needed Initially

10k comparisons are trivial.

Simple loop comparison is enough:

```java
for each user:
   similarity = cosine(query, stored)
```

Only migrate to vector DB later if scale grows massively.

---

# 11. Security Rules

# Initial Security

Required:
- MIME validation
- file size validation
- single-face validation
- rate limiting
- upload sanitization

---

# Future Security

Later:
- liveness detection
- blink detection
- replay attack prevention
- challenge-response

Not required for MVP.

---

# 12. JVM Configuration

Recommended:

```bash
-Xms512m
-Xmx2g
```

Avoid oversized heap allocation.

OpenCV native memory handles most heavy image processing.

---

# 13. API Response Examples

## Success

```json
{
  "success": true,
  "userId": 101,
  "faceDetected": true,
  "vectorGenerated": true,
  "stored": true
}
```

---

## Failure

```json
{
  "success": false,
  "error": "Multiple faces detected"
}
```

---

# 14. Important Engineering Rules

# Rule 1

Always use identical preprocessing pipeline.

---

# Rule 2

Never compare raw uploaded images.

Always compare normalized vectors.

---

# Rule 3

Reject bad enrollment images aggressively.

Bad training data destroys recognition quality.

---

# Rule 4

Do not optimize early.

First make preprocessing stable.

---

# Rule 5

Normalization quality matters more than model complexity in controlled systems.

---

# 15. Next Step Dependencies

# Step 2 — Recognition Engine

Depends entirely on:
- stable face detection
- stable normalization
- consistent vector generation

Recognition flow:

```text
new image
    ↓
preprocess
    ↓
generate vector
    ↓
find closest vector
```

---

# Step 3 — Threshold Optimization

Future:
- reduce false positives
- improve rejection logic
- confidence scoring

---

# Step 4 — Authentication Integration

Future:
- attendance APIs
- employee punch systems
- secure login flows
- session management

---

# Step 5 — Anti-Spoofing

Future:
- blink detection
- head movement
- replay attack prevention

---

# Step 6 — Scaling

Future optional upgrades:
- pgvector
- Redis cache
- ONNX embeddings
- distributed matching

Still CPU-only.

---

# Final MVP Goal

The MVP objective is NOT perfect AI.

The MVP objective is:

```text
stable deterministic face processing
+
consistent vector generation
+
reliable controlled-environment authentication
```

That foundation determines all future recognition quality.

