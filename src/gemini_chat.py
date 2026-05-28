import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(
    api_key = os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

def ask_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        print(f"Error: {e}, The API key is not valid")
        return None