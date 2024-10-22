import os
import gradio as gr
import numpy as np
import tensorflow as tf
from skimage.io import imsave
from skimage.transform import resize
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Standardize function
def standardize(img):
    # Standardization using adjusted standard deviation
    N = np.shape(img)[0] * np.shape(img)[1]
    s = np.maximum(np.std(img), 1.0 / np.sqrt(N))
    m = np.mean(img)
    img = (img - m) / s
    if np.ndim(img) == 2:
        img = np.dstack((img, img, img))
    return img

# Load model
filepath = './saved_model'
model = tf.keras.layers.TFSMLayer(filepath, call_endpoint='serving_default')

# Segmentation function
def FRFsegment(input_img):
    dims = (512, 512)
    w, h = input_img.shape[:2]

    # Standardize and resize the input image
    img = standardize(input_img)
    img = resize(img, dims, preserve_range=True, clip=True)
    img = np.expand_dims(img, axis=0)

    # Model prediction
    est_label_dict = model(img)

    # Print available keys to understand what we are dealing with
    print("Available keys in the output dictionary:", est_label_dict.keys())

    # Extract the actual tensor from the dictionary using a dynamic key lookup
    key = list(est_label_dict.keys())[0]  # Use the first available key
    est_label = est_label_dict[key]

    # Check the shape of the predicted output
    if len(est_label.shape) == 4 and est_label.shape[-1] > 1:
        # Multi-class segmentation: apply argmax
        mask = np.argmax(np.squeeze(est_label, axis=0), -1)
    else:
        # Binary segmentation or unexpected shape
        mask = np.squeeze(est_label, axis=0)

    # Resize the mask back to original input dimensions
    pred = resize(mask, (w, h), preserve_range=True, clip=True)

    # Convert prediction to uint8 format
    pred_uint8 = (pred / np.max(pred) * 255).astype(np.uint8)

    # Save predicted mask
    imsave("label.png", pred_uint8)

    # Overlay the segmentation on the original input image
    plt.clf()
    plt.imshow(input_img, cmap='gray')
    plt.imshow(pred, cmap='jet', alpha=0.4)  # Use 'jet' colormap to enhance visibility
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.savefig("overlay.png", dpi=300, bbox_inches="tight")

    # Read the overlay image to return it as an output
    overlay_img = plt.imread("overlay.png")

    return overlay_img, "label.png", "overlay.png"

# Prepare absolute paths for example images
example_dir = os.path.join(os.getcwd(), "examples")
example_images = [
    os.path.join(example_dir, "FRF_c1_snap_20191112160000.jpg"),
    os.path.join(example_dir, "FRF_c1_snap_20170101.jpg")
]

# Gradio Interface
title = "Segment beach imagery taken from a tower in Duck, NC, USA"
description = "This model segments beach imagery into 4 classes: vegetation, sand, coarse sand, and background (water + sky + buildings + people)"

FRFSegapp = gr.Interface(
    fn=FRFsegment,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(label="Overlay Image"), gr.File(label="Segmentation Mask Download"), gr.File(label="Overlay Image Download")],
    examples=example_images,
    title=title,
    description=description
)

FRFSegapp.launch()
