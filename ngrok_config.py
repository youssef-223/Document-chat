from pyngrok import ngrok, conf
import os
import streamlit as st
import logging
from dotenv import load_dotenv


def setup_ngrok(port=8501):
    try:
        # Load environment variables from .env file
        load_dotenv()

        # Get ngrok auth token from environment variable
        ngrok_auth_token = os.getenv("NGROK_AUTH_TOKEN")
        if not ngrok_auth_token:
            raise ValueError("NGROK_AUTH_TOKEN not found in environment variables.")
        
        # Configure ngrok
        ngrok_auth_token = ""  # Replace with your token
        conf.get_default().auth_token = ngrok_auth_token
        
        # Create tunnel
        public_url = ngrok.connect(port).public_url
        
        # Update Streamlit configuration
        os.environ['STREAMLIT_SERVER_PORT'] = str(port)
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
        os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
        
        return public_url
    except Exception as e:
        logging.error(f"Ngrok setup failed: {e}")
        return None