import subprocess
import os
import time
from ngrok_config import setup_ngrok

def run_streamlit():
    try:
        # Set environment variables
        os.environ['STREAMLIT_SERVER_PORT'] = '8501'
        os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
        
        # Start Streamlit first
        process = subprocess.Popen(
            ["streamlit", "run", "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait for Streamlit to start up
        print("Starting Streamlit server...")
        time.sleep(5)
        
        # Setup ngrok only after Streamlit is running
        public_url = setup_ngrok()
        if public_url:
            print(f"Ngrok tunnel established at: {public_url}")
        else:
            print("Failed to establish ngrok tunnel")
        
        # Monitor the process
        while True:
            if process.poll() is not None:
                print("Streamlit process terminated unexpectedly")
                break
                
            output = process.stdout.readline()
            if output:
                print(output.strip())
            
            error = process.stderr.readline()
            if error:
                print(f"Error: {error.strip()}")
                
    except KeyboardInterrupt:
        print("Shutting down...")
        if 'process' in locals():
            process.terminate()
        
    except Exception as e:
        print(f"Error: {e}")
        if 'process' in locals():
            process.terminate()

if __name__ == "__main__":
    run_streamlit()