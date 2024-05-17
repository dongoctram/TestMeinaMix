import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Define the path to the pre-trained model
model_path = "D:\\SELF STUDY\\Art Generation\\Pre-trained"

# Load the pre-trained model
pipe = StableDiffusionPipeline.from_pretrained(model_path)
pipe = pipe.to("cuda")  # Use "cpu" if you don't have a GPU

# Generate an image with a prompt
prompt = "A beautiful painting of a futuristic city"
with torch.no_grad():
    result = pipe(prompt)

# Save the generated image
image = result.images[0]
image.save("generated_image.png")

print("Image generated and saved as generated_image.png")
