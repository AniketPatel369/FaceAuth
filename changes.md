# Planned Changes

1. **BGR-to-RGB Color Space Fix**: 
    *   **The Issue**: OpenCV reads images in BGR format, but ArcFace is trained on RGB images. Currently, the model sees faces with reversed color channels (e.g. blue skin), which drastically degrades feature extraction and similarity scores.
    *   **The Fix**: Convert the cropped, aligned face image from BGR to RGB right before passing it into the ArcFace embedding generation model. This will ensure the AI analyzes natural skin tones, significantly boosting similarity scores.

2. **Investigate Low Similarity Scores**: The face similarity/matching threshold scores are currently coming out around 50-60%. We will investigate why ArcFace is producing these scores. (Note: As discussed, 100% occurs on identical images, but the BGR-to-RGB bug may still be lowering scores for slightly different photos. We will apply the BGR-to-RGB and Black & White fixes together).

3. **Two-Stage Face Alignment Pipeline**:
    *   **The Issue**: If the user uploads a large, full-body photo where the face is very small, resizing the entire image down to the SCRFD model's 640x640 input tensor causes the face to shrink drastically (e.g., 32x32 pixels). This makes landmark detection highly inaccurate, completely destroying similarity scores.
    *   **The Fix**: A two-stage detection pipeline:
        1.  Detect the face roughly in the original image.
        2.  Crop the face out with a 50% margin padding.
        3.  Run detection a *second time* on this high-resolution cropped version.
        4.  Align and generate embeddings. This guarantees ultra-precise 5-point landmarks and maximizes authentication scores.
