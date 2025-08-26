# app.py
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import numpy as np
import os

# -------------------------------
# Load BLIP Large Model Once
# -------------------------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# -------------------------------
# Caption Generator Function
# -------------------------------
def generate_caption(image_input):
    """
    Generate a single caption for an image (supports file path or numpy array).
    """
    if isinstance(image_input, str) and os.path.exists(image_input):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input.astype("uint8")).convert("RGB")
    else:
        raise ValueError("Unsupported image input type")

    inputs = processor(image, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=40,
        num_beams=5,
        repetition_penalty=1.2,
        early_stopping=True
    )

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


# -------------------------------
# Gradio UI
# -------------------------------
def caption_image(img):
    try:
        return generate_caption(img)
    except Exception as e:
        return f"❌ Error: {str(e)}"


with gr.Blocks() as demo:
    gr.Markdown("## 🖼️ Universal Image Identifier (BLIP Large)")
    
    img_input = gr.Image(type="numpy", label="Upload Image")
    caption_output = gr.Textbox(label="Predicted Caption / Object", lines=2)
    
    btn = gr.Button("Identify Image")
    btn.click(fn=caption_image, inputs=img_input, outputs=caption_output)

# Run app
demo.launch()
