from dotenv import load_dotenv
import os


load_dotenv()

CHATGPT_API_KEY = os.getenv('CHATGPT_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

if not CHATGPT_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Please set the CHATGPT_API_KEY and PINECONE_API_KEY environment variables.")
