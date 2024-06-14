# AI-Powered Banking Assistant

This project aims to develop an AI-powered banking assistant that leverages various advanced AI and machine learning techniques to enhance customer interaction and streamline banking operations.

## Project Description

The AI-Powered Banking Assistant integrates multiple state-of-the-art technologies to deliver a seamless and intelligent banking experience. This project involves:

- **Natural Language Understanding and Response Generation**: Utilizes OpenAI's GPT-3.5 for interpreting and responding to customer queries, providing accurate and context-aware answers.
- **Speech-to-Text Functionality**: Implements Whisper for converting spoken language into text, ensuring accessibility and convenience for users.
- **Agentic Workflows**: Leverages Langchain and Haystack frameworks to manage and automate complex banking processes, reducing response times and improving efficiency.
- **Semantic Search**: Uses a vector database (Pinecone) for efficient and precise information retrieval, allowing quick access to relevant banking documents and customer records.
- **Embedding Creation**: Employs MPNet and Ada models to create high-quality embeddings, enhancing the accuracy of query matching and document retrieval tasks.
- **Advanced Techniques**: Incorporates techniques such as HyDE, MMR, and LLM reranking to optimize semantic search, significantly improving query resolution accuracy.

## Features
- Natural language understanding and response generation using OpenAI's GPT-3.5.
- Speech-to-text functionality using Whisper.
- Agentic workflows for handling complex banking operations.
- Semantic search using Langchain and Haystack.
- Vector database for efficient information retrieval using Pinecone.
- Embedding creation using MPNet and Ada models.
- Advanced techniques like HyDE, MMR, and LLM reranking for enhanced semantic search.

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/AI-Powered-Banking-Assistant.git
    cd AI-Powered-Banking-Assistant
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up environment variables:
    ```sh
    export OPENAI_API_KEY='your-openai-api-key'
    export PINECONE_API_KEY='your-pinecone-api-key'
    ```

## Usage
1. Place the customer query and banking document datasets in the `data/` directory.
2. Run the main script:
    ```sh
    python src/main.py
    ```

## Project Structure
- `data/`: Contains the datasets.
- `src/`: Contains the main scripts and modules.
- `requirements.txt`: Lists the dependencies.
- `README.md`: Project documentation.

## Contributing
Contributions are welcome. Please create a pull request to add new features or fix bugs.
