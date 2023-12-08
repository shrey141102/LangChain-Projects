# ChatBot Project 

## Introduction

Welcome to the ChatBot project! This project allows you to interact with an AI-powered chatbot through a Streamlit web application. You can find the deployed application [here](https://chatbot-langchain1.streamlit.app/).

## Project Structure

- **app.py:** The main Python script containing the Streamlit application code.
- **utils.py:** A utility module providing functions used in the main application.
- **requirements.txt:** List of Python dependencies required to run the project.

## Getting Started

### Prerequisites

Before running the project, make sure you have the following installed:

- [Python](https://www.python.org/) (version 3.6 or higher)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/chatbot-project.git
    ```

2. Navigate to the project directory:

    ```bash
    cd chatbot-project
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1. Run the Streamlit application using the following command:

    ```bash
    streamlit run app.py
    ```

2. Access the application by opening the provided URL in your web browser.

## Configuration

The application requires several configuration parameters to function properly. These parameters can be set in the Streamlit sidebar:

- **Choose your LLM:** Select the language model (LLM) from options such as "ChatGooglePalm," "ChatOpenAI," or "HuggingFaceHub."
- **Choose your Embeddings:** Select the embeddings from options like "HuggingFaceEmbeddings," "OpenAIEmbeddings," or "SentenceTransformerEmbeddings."
- **API Keys:** Enter API keys for OpenAI, Huggingface, and Google Palm API, if applicable.
- **Website URL:** Enter the target website URL for scraping.

## Usage

1. Fill in the configuration parameters in the Streamlit sidebar.
2. Click the "All details filled! ✅" button to initiate scraping if a website URL is provided.
3. Enter a message in the chat input box to interact with the chatbot.
4. The assistant's responses will be displayed in the chat.

## Troubleshooting

If you encounter any issues during installation or execution, please check the following:

- Ensure that Python and pip are properly installed.
- Verify that the required dependencies are installed by checking the `requirements.txt` file.

If problems persist, feel free to open an issue on the project's GitHub repository.

## Acknowledgments

This project utilizes Streamlit for the web interface and various language models for chatbot responses.

Happy chatting! 🤖🚀