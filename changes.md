# Planned Changes

1. **BGR-to-RGB Color Space Fix**: 
    *   **The Issue**: OpenCV reads images in BGR format, but ArcFace is trained on RGB images. Currently, the model sees faces with reversed color channels (e.g. blue skin), which drastically degrades feature extraction and similarity scores.
    *   **The Fix**: Convert the cropped, aligned face image from BGR to RGB right before passing it into the ArcFace embedding generation model. This will ensure the AI analyzes natural skin tones, significantly boosting similarity scores.

2. **Investigate Low Similarity Scores**: The face similarity/matching threshold scores are currently coming out around 50-60%. We will investigate why ArcFace is producing these scores. (Note: As discussed, 100% occurs on identical images, but the BGR-to-RGB bug may still be lowering scores for slightly different photos. We will apply the BGR-to-RGB and Black & White fixes together).
