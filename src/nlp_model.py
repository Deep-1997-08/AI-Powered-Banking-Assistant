import openai
from src.config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def generate_response(query):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=query,
        max_tokens=150
    )
    return response.choices[0].text.strip()
