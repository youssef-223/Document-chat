# Document Assistant

## Overview
The Document Assistant is a Streamlit-based application that allows users to upload, process, and query documents using AI-powered natural language processing. It supports multiple document formats, including PDF, DOCX, and TXT, and provides concise answers to user queries based on the content of the uploaded documents.

## Features
- Upload and process multiple documents.
- Supported formats: PDF, DOCX, TXT.
- AI-powered question answering with context retrieval.
- Document chunking and embedding storage using ChromaDB.
- Easy-to-use web interface built with Streamlit.

## Requirements
- streamlit
- torch
- transformers
- chromadb
- pypdf2
- python-docx
- langchain
- pyngrok

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Document-chat
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have a CUDA-enabled GPU and the necessary drivers installed for PyTorch.

## Usage
1. Start the application:
   ```bash
   python run_app.py
   ```

2. Open the application in your browser. By default, it runs on `http://localhost:8501`.

3. Upload documents using the sidebar and start querying them.

## File Structure
- `app.py`: Main Streamlit application file.
- `generation_pipeline.py`: Contains the `DocumentAssistant` class for document processing and querying.
- `run_app.py`: Script to start the Streamlit app and set up ngrok for public access.
- `ngrok_config.py`: Configuration for setting up ngrok tunnels.
- `requirements.txt`: List of Python dependencies.

## Supported Document Formats
- **PDF**: Extracts text from all pages.
- **DOCX**: Extracts text from paragraphs.
- **TXT**: Reads plain text files.

## Troubleshooting
- **Error: Unsupported file format**: Ensure the uploaded file is in PDF, DOCX, or TXT format.
- **Streamlit process terminated unexpectedly**: Check the logs for errors and ensure all dependencies are installed.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- [Streamlit](https://streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [ChromaDB](https://www.trychroma.com/)

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.