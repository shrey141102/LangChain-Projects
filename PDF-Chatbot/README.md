# PDF ChatBot

## Introduction

Welcome to the PDF ChatBot project! This application allows you to interact with a language model-powered chatbot specifically designed for processing PDF files. The deployed application can be found [here](https://pdf-chatbot-langchain1.streamlit.app/).

## Project Structure

- **app.py:** The main Python script containing the Streamlit application code.
- **langchain:** A Python package providing text processing and language model functionalities.
- **requirements.txt:** List of Python dependencies required to run the project.

## Getting Started

### Prerequisites

Before running the project, make sure you have the following installed:

- [Python](https://www.python.org/) (version 3.6 or higher)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/pdf-chatbot.git
    ```

2. Navigate to the project directory:

    ```bash
    cd pdf-chatbot
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

The application requires you to enter your OpenAI API key. Follow the instructions in the application to input the key:

1. Enter your OpenAI API key in the provided text input.
2. Click the "Enter" button to confirm.

## Usage

1. Upload a PDF file using the file uploader in the application.
2. The application will extract text from the uploaded PDF and process it into chunks.
3. If embeddings for the PDF are not already stored, they will be generated and saved.
4. Enter questions about the PDF in the text input field and click on "Ask questions about your PDF file."
5. The chatbot will use language models and embeddings to provide answers based on the content of the PDF.

## Troubleshooting

If you encounter any issues during installation or execution, please check the following:

- Ensure that Python and pip are properly installed.
- Verify that the required dependencies are installed by checking the `requirements.txt` file.
- Make sure to enter a valid OpenAI API key.

If problems persist, feel free to open an issue on the project's GitHub repository.

## Acknowledgments

This project utilizes Streamlit for the web interface, LangChain for text processing, and OpenAI for language models.

Enjoy chatting with your PDF ChatBot! 🤖💬
