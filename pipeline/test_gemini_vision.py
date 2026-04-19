"""
Quick test to debug Gemini vision scoring response format.
"""
import os
import sys
import base64
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBRAXQvldywp5YeOwRjSCeDUG56WhT7w9o")

# Find a test image
test_images = list(Path("/home/ubuntu/veritas_image_pipeline/output/images").rglob("*.jpg"))
if not test_images:
    # Try temp dir
    test_images = list(Path("/tmp/pipeline_test").rglob("*.jpg"))

if not test_images:
    print("No test images found. Run the worker first to download some candidates.")
    sys.exit(1)

test_image = str(test_images[0])
print(f"Testing with: {test_image}")

from google import genai
from google.genai import types as genai_types

client = genai.Client(api_key=GEMINI_API_KEY)

with open(test_image, "rb") as f:
    img_data = f.read()

prompt = """You are a food image quality assessor. Evaluate this image for use in a food database.

Food item: Soy Milk with Grass Jelly
Slot type: hero
Slot description: A clear, well-lit photo of Soy Milk with Grass Jelly in a glass, cup, or bottle on a clean background. Single serving, no clutter.

Respond with EXACTLY this format (no extra text):
REAL_PHOTO: YES or NO
FOOD_MATCH: YES or NO
CONFIDENCE: HIGH or MEDIUM or LOW
WATERMARK: YES or NO
FM_SCORE: integer 1-10
SF_SCORE: integer 1-10"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        genai_types.Part.from_bytes(
            data=img_data,
            mime_type="image/jpeg",
        ),
        prompt,
    ],
    config=genai_types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=200,
    ),
)

print("=== RAW RESPONSE ===")
print(repr(response.text))
print("=== FORMATTED ===")
print(response.text)
