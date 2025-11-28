# DeepDream MLX: Agents

## 1. The Mission
To resurrect the 2015 DeepDream aesthetic using modern 2025 Apple Silicon hardware, bypassing the need for archaic frameworks like Caffe or Torch7 by porting everything to native MLX.

## 2. Training & Fine-Tuning Plan (The "Punch-Card" Revival)
In the "classic" days (Intel Caffe era), training a custom DeepDream model meant fine-tuning a GoogLeNet on a dataset of specific objects (e.g., slugs, eyes, cars) so the network would hallucinate *those specific things* when dreaming.

**The Roadmap for MLX Training:**

### Phase 1: Dataset Prep
The `dream-creator` logic (from ProGamerGov) is still sound. We need:
1.  **Structure:** `dataset/class_name/*.jpg` (Standard PyTorch ImageFolder format).
2.  **Cleaning:** Remove corrupt images, deduplicate.
3.  **Resizing:** Resize to ~224x224 or 256x256.
4.  **Stats:** Calculate Mean/StdDev.

### Phase 2: The Trainer (`train_dream.py`)
We need to write a native MLX training loop.
*   **Base Model:** Load `googlenet_mlx.npz`.
*   **Architecture:** InceptionV1 (GoogLeNet).
*   **Layer Freezing:** 
    -   **Critical:** Freeze early layers (`conv1`, `conv2`, `inception3a/b`) to preserve the "visual vocabulary" (edges, textures).
    -   **Train:** Retrain only the higher layers (`inception4c`, `inception5b`, `fc`) and the Auxiliary Classifiers.
*   **Auxiliary Classifiers:** Inception has two side-branches (`aux1`, `aux2`) used for training stability. We must support training these or stripping them.
*   **Loss:** Cross-Entropy.
*   **Optimizer:** SGD with Momentum (classic) or Adam.

### Phase 3: "Decorrelation" (The Secret Sauce)
`dream-creator` confirms that "Color Decorrelation" is key.
*   **Matrix:** A 3x3 matrix calculated from the training set covariance.
*   **Effect:** "Whitens" the input image gradients during dreaming, preventing the image from converging to a mono-color blob.
*   **Implementation:** Port `data_tools/calc_cm.py` to MLX.

## 3. Animation & Video Strategy
The "Zoom" video effect is the second pillar of DeepDream.
*   **Logic:** Feedback Loop.
    1.  Dream on Frame N.
    2.  Zoom (Scale + Crop center) Frame N to create Frame N+1.
    3.  Repeat.
*   **Implementation:** A dedicated `dream_video.py` script.
*   **Tech:** Use `scipy.ndimage.zoom` (same as original 2015 code) for the scaling, as MLX's `resize` might differ slightly in sub-pixel interpolation.

## 4. Available Models & Wishlist
**Current:**
*   `alexnet`: The raw, chaotic ancestor.
*   `googlenet` (InceptionV1): The classic "slugs and dogs".
*   `vgg16/19`: The "painterly" style transfer beast.
*   `resnet50`: Modern, sharp, geometric.

**Wishlist (To Convert):**
*   `inception_v3`: More refined hallucinations.
*   `googlenet_places365`: Hallucinates landscapes/interiors. (Verified working via `convert.py --download googlenet` when URL is fixed/found).

## 5. Hugging Face Hygiene
*   **Repo:** `NickMystic/DeepDream-MLX`
*   **LFS:** Track `*.npz`.
*   **Cleanup:** Ensure `toConvert/` is empty of large raw files.
*   **Banner:** `assets/deepdream_header.jpg`.

---
*Docs derived from deep analysis of `dream-creator` and classic Caffe workflows.*
