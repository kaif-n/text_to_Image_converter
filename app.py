from diffusers import StableDiffusionPipeline
import torch
from IPython.display import display, Image, HTML
from google.colab import drive
import os
from PIL import Image as PILImage, ImageDraw, ImageFont
import getpass
import logging
from contextlib import redirect_stdout, redirect_stderr

# Suppress standard output using contextlib
def suppress_output(func):
    def wrapper(*args, **kwargs):
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                return func(*args, **kwargs)
    return wrapper

# Mount Google Drive (suppress output)
@suppress_output
def mount_drive():
    drive.mount('/content/drive', force_remount=True)

# Get Hugging Face Token from environment (Assumed to be set in the environment or by some external means)
def login_silently():
    token = os.getenv("HUGGINGFACE_TOKEN")
    # No output if token is present (to suppress message)
    if token:
        pass  # Do nothing on success to suppress the message
    else:
        print("Hugging Face token not found. Please ensure the token is set in the environment.")

# Perform silent login by retrieving the token from the environment
login_silently()

# Load the Stable Diffusion model from Hugging Face (suppress output)
@suppress_output
def load_model(model_id):
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")  # Use GPU for faster inference if available
        return pipe
    except Exception as e:
        print(f"Failed to load the model: {e}")
        return None

# Mount Google Drive
mount_drive()

# Load the Stable Diffusion model from Hugging Face
model_id = "runwayml/stable-diffusion-v1-5"  # Change the model if needed
pipe = load_model(model_id)

def add_text_overlay(image, text):
    # Convert image to PIL format for text overlay
    pil_image = image.convert("RGBA")
    draw = ImageDraw.Draw(pil_image)

    # Define font size and style
    try:
        # Try loading the default font
        font = ImageFont.load_default()
    except IOError:
        # Fallback font if default loading fails
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)  # Standard DejaVu font
        except IOError:
            print("Error: Unable to load font, using basic font.")
            font = ImageFont.load_default()  # Default to a very basic font if all else fails

    # Calculate text width and height to center it using textbbox()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]  # width is the difference between the right and left coordinates
    text_height = bbox[3] - bbox[1]  # height is the difference between the bottom and top coordinates
    
    image_width, image_height = pil_image.size

    # Position the text at the bottom of the image
    x = (image_width - text_width) / 2
    y = image_height - text_height - 10  # 10px padding from bottom

    # Add a semi-transparent background behind the text for better visibility
    background = (0, 0, 0, 128)  # Black with transparency (RGBA)
    draw.rectangle([(x-10, y-10), (x + text_width + 10, y + text_height + 10)], fill=background)

    # Add text on top of the background
    draw.text((x, y), text, font=font, fill="white")

    return pil_image

def generate_and_save_image(description, file_name):
    if not description:
        print("Description is required!")
        return
    
    if not file_name.endswith('.png'):
        file_name += '.png'  # Ensure the file name has a .png extension
    
    try:
        # Generate an image using Stable Diffusion (suppress output)
        print("Generating image, please wait...")
        image = pipe(description).images[0]
        
        # Add text overlay to the image
        image_with_text = add_text_overlay(image, description)
        
        # Save the image to Google Drive
        drive_path = os.path.join("/content/drive/My Drive", file_name)
        image_with_text.save(drive_path)
        print(f"Image saved to Google Drive as {drive_path}")
        
        # Display the image in Colab
        display(Image(drive_path))  # Display the image with overlay
        
        # Optional: Display the description as well
        print(f"Description: {description}")
    
    except Exception as e:
        print(f"Error during image generation: {e}")

# HTML & CSS for Frontend Input
html_code = """
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f7f7f7;
                color: #333;
            }
            h1 {
                color: #4CAF50;
            }
            .container {
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                width: 300px;
                margin: auto;
            }
            label {
                font-size: 14px;
                margin-bottom: 8px;
                display: block;
            }
            input[type="text"], input[type="submit"] {
                width: 100%;
                padding: 10px;
                margin: 10px 0;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
            }
            input[type="submit"] {
                background-color: #4CAF50;
                color: white;
                border: none;
                cursor: pointer;
            }
            input[type="submit"]:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Generate Image</h1>
            <form action="#" method="post">
                <label for="description">Image Description:</label>
                <input type="text" id="description" name="description" placeholder="Enter description">
                <label for="filename">File Name:</label>
                <input type="text" id="filename" name="filename" placeholder="Enter file name">
                <input type="submit" value="Generate Image" onclick="generateImage(); return false;">
            </form>
        </div>

        <script>
            function generateImage() {
                var description = document.getElementById("description").value;
                var filename = document.getElementById("filename").value;

                // Check if both inputs are filled
                if (!description || !filename) {
                    alert("Please fill in both fields.");
                    return;
                }

                google.colab.kernel.invokeFunction('notebook.generate_and_save_image', [description, filename], {});
            }
        </script>
    </body>
    </html>
"""

# Display HTML in the notebook
display(HTML(html_code))

# Make the Python function callable from the JavaScript frontend
from google.colab import output
output.register_callback('notebook.generate_and_save_image', generate_and_save_image)