import google.generativeai as genai

genai.configure(
    api_key = "AIzaSyBPGTCxMewXoMBZbKCPhPzq_FbMZLOjh7Q"
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