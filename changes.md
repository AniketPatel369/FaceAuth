# Face Authentication — Score Improvement Changelog

> **Current Score: ~80%** (up from 50-60%)
> **Target: 90%+**

---

## ✅ Completed Changes

### 1. BGR-to-RGB Color Space Fix — `Priority: P0 (Critical)` ✅
*   **The Issue**: OpenCV reads images in BGR format, but ArcFace is trained on RGB images. The model was seeing faces with reversed color channels (e.g. blue skin), drastically degrading feature extraction and similarity scores.
*   **The Fix**: Convert the cropped, aligned face image from BGR to RGB right before passing it into the ArcFace embedding generation model.
*   **Impact**: ~15-20% score improvement.

### 2. Low Similarity Score Investigation — `Priority: P0 (Critical)` ✅
*   Scores were coming out around 50-60%. Root cause identified as BGR-to-RGB bug combined with alignment issues. Applied BGR-to-RGB and Black & White fixes together.

### 3. Two-Stage Face Alignment Pipeline — `Priority: P0 (Critical)` ✅
*   **The Issue**: Large/full-body photos cause the face to shrink to ~32x32px when resized to SCRFD's 640x640 input, destroying landmark accuracy.
*   **The Fix**: A two-stage detection pipeline:
    1.  Detect the face roughly in the original image.
    2.  Crop the face out with a 50% margin padding.
    3.  Run detection a *second time* on this high-resolution cropped version.
    4.  Align and generate embeddings. Guarantees ultra-precise 5-point landmarks.
*   **Impact**: ~10-15% score improvement on large/full-body photos.

---

## 🔥 High Priority (Biggest Remaining Gains)

### 4. Input Pixel Normalization — `Priority: P1` | Expected Gain: **+5-10%**
*   **The Issue**: ArcFace models are trained with specific pixel preprocessing — typically `(pixel - 127.5) / 128.0`, mapping values to [-1, 1]. If raw 0-255 pixels or a different normalization scheme is being fed, every single pixel the model processes is slightly "off", degrading all embeddings.
*   **The Fix**: Verify and apply the exact normalization the ArcFace ONNX model was trained with before inference. This is the single most likely source of the remaining ~20% gap.

### 5. ArcFace Alignment Template Verification — `Priority: P1` | Expected Gain: **+3-7%**
*   **The Issue**: The 5-point landmark target coordinates used for the similarity transform to produce the 112x112 aligned face must match *exactly* what ArcFace was trained on. Even a few pixels off will subtly distort every embedding.
*   **The Fix**: Cross-reference current alignment target coordinates against the official InsightFace/ArcFace canonical template and correct any discrepancies.

### 6. Multiple Enrollment Photos (Centroid Embedding) — `Priority: P1` | Expected Gain: **+3-5%**
*   **The Idea**: Instead of registering with a single photo, capture 3-5 photos during enrollment and **average the L2-normalized embeddings** into a single "centroid" embedding.
*   **The Benefit**: Smooths out per-photo noise (lighting, expression, slight angle differences), creating a much more robust reference embedding. Easy win with no model changes.

---

## ⚡ Medium Priority

### 7. Flip-Ensemble Verification — `Priority: P2` | Expected Gain: **+2-4%**
*   **The Idea**: Generate embeddings for both the original cropped face and a horizontally flipped (mirror) version of it during authentication.
*   **The Benefit**: Averaging these two embeddings together before comparison filters out asymmetric lighting and shadow artifacts, offering a solid boost in recognition stability.

### 8. Embedding L2 Normalization Check — `Priority: P2` | Expected Gain: **+2-3%**
*   **The Issue**: Cosine similarity assumes unit-length vectors. If embeddings are not L2-normalized before comparison, magnitude differences pollute the angle-based similarity calculation.
*   **The Fix**: Ensure all embeddings are explicitly L2-normalized (`embedding / ||embedding||`) immediately after generation and before any storage or comparison.

### 9. Dynamic Contrast Enhancement (CLAHE) — `Priority: P2` | Expected Gain: **+1-3%**
*   **The Idea**: Apply CLAHE to the **luminance channel only** (convert to LAB color space, apply CLAHE on L channel, convert back to RGB) of the cropped face before generating the embedding.
*   **The Benefit**: Fixes washed-out webcam images and standardizes lighting across different environments. Applying to luminance only avoids color distortion.

---

## 🎯 Refinement (Edge Cases & Robustness)

### 10. Image Quality Gating — `Priority: P3` | Expected Gain: **+1-2%** (prevents bad matches)
*   **The Idea**: Reject low-quality inputs before they produce bad embeddings:
    *   **Blur detection**: Use Laplacian variance — blurry faces produce unreliable embeddings.
    *   **Brightness check**: Reject too-dark or overexposed images where facial features are lost.
    *   **Minimum face size**: If detected face is below ~80x80 pixels, ask user for a closer photo.
*   **The Benefit**: Prevents garbage-in-garbage-out scenarios. Instead of returning a misleading low score, provide actionable feedback to the user.

### 11. Face Pose Filtering — `Priority: P3` | Expected Gain: **+1-2%** (prevents bad matches)
*   **The Idea**: Estimate rough face pose from the 5 detected landmarks. If the nose point is too far from center relative to the eyes (indicating extreme yaw/pitch), reject or warn.
*   **The Benefit**: Extreme side profiles or downward-looking faces produce poor embeddings. Better to reject and ask for a frontal photo than return a low score.

### 12. Consistent Preprocessing Pipeline — `Priority: P3` | Expected Gain: **prevents score drops**
*   **The Issue**: If registration and authentication use even slightly different preprocessing steps (e.g., CLAHE during auth but not registration, or different resize interpolation methods), scores will be systematically lower.
*   **The Fix**: Ensure the *exact same* pipeline (resize → align → color convert → normalize → CLAHE → embed) is used for both registration and authentication. Audit both code paths side by side.
