import subprocess
import sys
import os

def main():
    """
    Runs the Streamlit application.
    """
    app_path = os.path.join(os.path.dirname(__file__), 'ui.py')
    
    # Use subprocess to run the streamlit command
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', app_path])

if __name__ == "__main__":
    main()