# similarity_measures.py
from __future__ import annotations
import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
from numba import njit
from typing import List 
from helper import shut_gpu
# Disable GPU if needed
shut_gpu()

# Global variable to hold the pre-trained model
# This ensures the model is loaded only once, for efficiency.
DEEP_MODEL: Model | None = None


# This function loads the pre-trained ResNet50 model for deep feature extraction.
# It initializes the model only once, returning the already loaded model on subsequent calls.
def load_model():
    global DEEP_MODEL
    if DEEP_MODEL is None:
        base = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        DEEP_MODEL = Model(inputs=base.input, outputs=base.output)
    return DEEP_MODEL

# This function computes the deep embedding for a given image file path.
def compute_embedding(filepath, model=None, target_size=(224, 224)):
    # m is the pre-trained model, loaded if not provided
    m = model or load_model()
    # Load and preprocess the image
    img = keras_image.load_img(filepath, target_size=target_size)
    if img is None:
        raise FileNotFoundError(filepath)
    # Convert the image to an array and add a batch dimension
    x = keras_image.img_to_array(img)[None, ...]
    # Preprocess the image for the model
    x = preprocess_input(x)
    # Predict the embedding and normalize it
    vec = m.predict(x, verbose=0)[0].astype(np.float32)
    n = np.linalg.norm(vec) + 1e-12
    return (vec / n).astype(np.float32)

# This function computes deep embeddings for a *batch* of image file paths using ResNet50 avg-pool.
def compute_embeddings_batch(filepaths, model=None, target_size=(224, 224), batch_size=64):
    # Load or reuse the pre-trained model only once to avoid repeated initialization cost
    m = model or load_model()

    # Prepare containers for decoded images and a success mask
    arrays = []     # holds raw image tensors (H,W,3) for successfully loaded files
    ok_mask = []    # True/False per filepath → whether decoding/preprocess succeeded

    # Decode & resize each image upfront (I/O bound); keep failures as None to preserve order
    for fp in filepaths:
        try:
            # Load and resize to the network's expected input size
            img = keras_image.load_img(fp, target_size=target_size)
            # Convert PIL image to float32 numpy array (H,W,3)
            x = keras_image.img_to_array(img)
            arrays.append(x)
            ok_mask.append(True)
        except Exception:
            arrays.append(None)
            ok_mask.append(False)

    # Collect indices of successfully decoded images
    good_idx = [i for i, ok in enumerate(ok_mask) if ok]

    # If *none* succeeded, return an empty (0,2048) matrix and the mask
    if not good_idx:
        return np.zeros((0, 2048), dtype=np.float32), np.array(ok_mask, dtype=bool)

    # Stack only valid tensors into a batch and apply the model's preprocessing
    X = np.stack([arrays[i] for i in good_idx]).astype(np.float32)   # shape: (G, H, W, 3)
    X = preprocess_input(X)  # channel-wise normalization as expected by ResNet50

    # Run inference in mini-batches to fit memory and utilize vectorized compute
    outs = []
    for s in range(0, X.shape[0], batch_size):
        e = s + batch_size
        # Forward pass: outputs avg-pool features of shape (b, 2048)
        pred = m.predict(X[s:e], verbose=0).astype(np.float32)
        outs.append(pred)

    # Concatenate all mini-batch outputs → (G, 2048)
    Z = np.vstack(outs)

    # L2-normalize each row so cosine similarity ≡ dot product and matches ANN(L2) assumptions
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    # Return embeddings for the successful files and the original success mask (aligned with inputs)
    return Z.astype(np.float32), np.array(ok_mask, dtype=bool)


# This function computes a normalized color histogram for an image.
@njit(fastmath=True)
def _hist3d_bincount(lab_u8, bins):
    # Get the dimensions of the image
    h, w, _ = lab_u8.shape
    # Calculate the step size for each channel based on the number of bins
    step = 256 // bins 

    # Compute the 3D histogram using binning
    idx = (lab_u8[:, :, 0] // step) * (bins * bins) + (lab_u8[:, :, 1] // step) * bins + (lab_u8[:, :, 2] // step)
    # Flatten the index array to count occurrences
    hist = np.bincount(idx.ravel(), minlength=bins * bins * bins)
    return hist.astype(np.float32)

def compute_color_feature(filepath, bins=8):
    # Read the image
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(filepath)

    # Convert the image to the LAB color space
    space = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    

    # Compute the 3D histogram with fast Numba function
    hist = _hist3d_bincount(space.astype(np.uint8), bins=bins)

    # Normalize the histogram
    n = float(np.linalg.norm(hist) + 1e-12)
    hist = (hist / n).astype(np.float32)
    return hist