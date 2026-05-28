# 👤 FaceAuth: CPU-Only Face Authentication System

FaceAuth is a high-performance, lightweight face enrollment and recognition system built using **Spring Boot 3.3**, **OpenCV**, and **ONNX Runtime**. It runs state-of-the-art deep learning models (SCRFD and ArcFace) directly on the **CPU** without requiring external GPU dependencies.

Designed for local deployments, microservices, and educational systems, FaceAuth provides a full pipeline from face detection to alignment, feature extraction (embeddings), and database-backed matching.

---

## 🚀 Key Features

*   **⚡ CPU-Optimized Inference**: Uses ONNX Runtime with configurable thread pooling for rapid CPU-based model execution.
*   **🎯 High-Accuracy Face Detection**: Leverages **SCRFD** (Sample-Efficient Face Detector) to detect bounding boxes and 5-point facial landmarks.
*   **📐 Affine Face Alignment**: Warps detected faces to a standard $112 \times 112$ resolution using eyes landmarks before embedding generation to ensure maximum similarity accuracy.
*   **🧬 ArcFace Embeddings**: Generates $512$-dimensional L2-normalized embedding vectors using the **ArcFace (ResNet-50)** model.
*   **💾 Database Integration**: Stores normalized embeddings in a **PostgreSQL** database and performs matching using fast cosine similarity (dot product).
*   **🖥️ Tester Client**: Includes an interactive HTML5 client (`test_client.html`) for webcam capture, face detection, face cropping, enrollment, and recognition testing.

---

## 🛠️ Tech Stack

*   **Backend Framework**: Spring Boot 3.3 / Java 21 LTS
*   **Inference Engine**: ONNX Runtime Java API
*   **Computer Vision**: OpenCV (via Bytedeco Java wrappers)
*   **Database**: PostgreSQL (JPA/Hibernate)
*   **Frontend**: HTML5, Vanilla CSS, JavaScript (Webcam/Fetch API)

---

## 📐 System Pipeline

```mermaid
graph TD
    A[User Image] --> B[SCRFD Face Detection]
    B -->|BBox + 5 Landmarks| C[Face Alignment & Warping]
    C -->|112x112 Aligned BGR| D[BGR-to-RGB & Norm]
    D -->|Float Tensor [-1, 1]| E[ArcFace Feature Extractor]
    E -->|512-D L2 Embedding| F{Operation}
    F -->|Enrollment| G[Save to PostgreSQL]
    F -->|Recognition| H[Cosine Similarity Search]
    H -->|Best Match| I[MatchResult & Score]
```

---

## 📂 Project Structure

```text
├── models/                     # Store your downloaded ONNX models here (Ignored by Git)
├── faces/                      # Storage folder for cropped/enrolled faces (Ignored by Git)
├── src/
│   ├── main/
│   │   ├── java/com/faceauth/
│   │   │   ├── config/         # App properties and OpenCV configuration
│   │   │   ├── controller/     # Rest Controllers for endpoints
│   │   │   ├── dto/            # Data Transfer Objects
│   │   │   ├── entity/         # Database Entities (User, FaceEmbedding)
│   │   │   ├── onnx/           # ONNX models, detector, embedder, and tensor utilities
│   │   │   ├── repository/     # JPA repositories
│   │   │   ├── service/        # Alignment, detection, embedding, matching services
│   │   │   └── util/           # Normalization, math, and OpenCV helper utils
│   │   └── resources/
│   │       ├── application.yml # Main configuration settings
│   │       └── db/migration/   # Flyway or SQL schema migrations
├── test_client.html            # Web-based testing dashboard
├── restart.bat                 # Script to easily kill and restart the local server
└── pom.xml                     # Maven dependencies
```

---

## 📥 Required Models Setup

You must download the pre-trained ONNX models and place them inside the `models/` directory in the root of the project:

| Model Filename | Model Architecture | Purpose | Size |
| :--- | :--- | :--- | :--- |
| **`det_2.5g.onnx`** | SCRFD-2.5G | Light & Fast Face Detection | ~9.0 MB |
| **`w600k_r50.onnx`** | ArcFace ResNet-50 | High-Accuracy Face Embedding | ~166 MB |

> [!TIP]
> You can download these models from the official InsightFace model zoo or public model sharing repositories. Ensure the filenames match the ones specified in `application.yml` exactly.

---

## ⚙️ Configuration & Database Setup

1. **Create PostgreSQL Database**:
   ```sql
   CREATE DATABASE faceauth;
   ```
2. **Update Database Credentials**:
   Open [src/main/resources/application.yml](file:///g:/Aniket/Projects/FaceAuth/src/main/resources/application.yml) and configure your database settings:
   ```yaml
   spring:
     datasource:
       url: jdbc:postgresql://localhost:5432/faceauth
       username: postgres
       password: yourpassword
   ```
3. **Threshold Tuning**:
   You can adjust matching thresholds inside `application.yml`:
   * `faceauth.matching.accept-threshold`: Default is `0.45`. Any score above this counts as a match.
   * `faceauth.matching.reject-threshold`: Default is `0.30`. Any score below this is auto-rejected.

---

## 🚀 Running the Application

### Running the Spring Boot Server
You can launch the server using Maven:
```bash
mvn spring-boot:run
```
Alternatively, on Windows, double-click **`restart.bat`** to automatically kill any process holding port `8089` and restart the application.

### Opening the Tester Interface
Simply double-click or open **`test_client.html`** in any modern web browser. It connects directly to the backend APIs running on `http://localhost:8089`.

---

## 📡 API Reference

### 1. Enroll a Face
* **Endpoint**: `POST /api/face/enroll`
* **Content-Type**: `multipart/form-data`
* **Parameters**:
  * `image`: Image file containing one face (JPEG/PNG)
  * `name`: User's full name
  * `identifier`: Unique user ID or email
* **Response**:
  ```json
  {
    "success": true,
    "userId": 1,
    "detectionConfidence": 0.985,
    "totalEnrollments": 1,
    "error": null
  }
  ```

### 2. Recognize / Match Face
* **Endpoint**: `POST /api/face/recognize`
* **Content-Type**: `multipart/form-data`
* **Parameters**:
  * `image`: Query image file (JPEG/PNG)
* **Response**:
  ```json
  {
    "matched": true,
    "userId": 1,
    "userName": "John Doe",
    "similarity": 0.8872,
    "threshold": 0.45
  }
  ```

### 3. Detect and Crop Face (Utility Endpoint)
* **Endpoint**: `POST /api/face/detect-crop`
* **Content-Type**: `multipart/form-data`
* **Parameters**:
  * `image`: Image file (JPEG/PNG)
* **Response**: Binary JPEG stream of the $112 \times 112$ aligned, cropped face.

---

## 📝 Math & Normalization Details

* **Input Normalization**:
  For SCRFD detection:
  $$x_{\text{norm}} = \frac{x - 127.5}{128.0}$$
  For ArcFace embedding generation:
  $$x_{\text{norm}} = \frac{x - 127.5}{128.0} \text{ or } \frac{x - 127.5}{127.5}$$
* **Similarity Computation**:
  Since face embeddings are **L2-normalized** ($\|v\|_2 = 1$) upon generation, the Cosine Similarity reduces to a simple dot product:
  $$\text{Cosine Similarity}(A, B) = A \cdot B = \sum_{i=1}^{512} A_i B_i$$
  This allows database matching to run in sub-millisecond speeds using standard floating-point arithmetic.
