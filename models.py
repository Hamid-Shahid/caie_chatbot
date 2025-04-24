import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()  # Add this line
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# List available models
for model in genai.list_models():
    print(f"Name: {model.name}")
    print(f"Supported methods: {model.supported_generation_methods}\n")