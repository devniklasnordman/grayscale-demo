import numpy as np
from PIL import Image
import gradio as gr

def to_grayscale(image):
    img = np.array(image)
    weights = np.array([0.2126, 0.7152, 0.0722])
    gray = np.dot(img[..., :3], weights).astype(np.uint8)
    return Image.fromarray(gray)

demo = gr.Interface(fn=to_grayscale, inputs="image", outputs="image", title="Image to Grayscale")
demo.launch()