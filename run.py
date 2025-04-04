#!/usr/bin/env python
"""
Main run script for the AgentNet Customer Support System.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def ensure_data_directories():
    """Ensure all necessary directories exist."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create directories if they don't exist
    directories = [
        os.path.join(script_dir, "data"),
        os.path.join(script_dir, "data", "raw"),
        os.path.join(script_dir, "data", "processed"),
        os.path.join(script_dir, "logs"),
        os.path.join(script_dir, "results")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    return script_dir

def run_streamlit():
    """Run Streamlit application."""
    script_dir = ensure_data_directories()
    app_path = os.path.join(script_dir, "app.py")
    
    print(f"Starting Streamlit application from {app_path}")
    subprocess.run(["streamlit", "run", app_path])

def run_server(host, port):
    """Run the API server."""
    # Import here to avoid loading everything when not needed
    from src.main import app
    import uvicorn
    
    print(f"Starting API server on {host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=True
    )

def run_cli(cli_args):
    """Run CLI tools."""
    # Import here to avoid loading everything when not needed
    from src.cli import main
    
    # Replace sys.argv with the CLI args
    sys.argv = [sys.argv[0]] + cli_args
    
    print(f"Running CLI command: {' '.join(cli_args)}")
    main()

def check_ollama():
    """Check if Ollama is running and available."""
    import requests
    
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            print("✅ Ollama is running and available")
            models = response.json().get("models", [])
            if models:
                print(f"Available models: {', '.join([m.get('name', 'unknown') for m in models])}")
            return True
        else:
            print(f"❌ Ollama returned unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Could not connect to Ollama: {str(e)}")
        print("Please make sure Ollama is installed and running.")
        print("You can download Ollama from https://ollama.ai/")
        return False

def check_requirements():
    """Check if all required packages are installed."""
    try:
        import streamlit
        import fastapi
        import uvicorn
        import numpy
        import pandas
        import loguru
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {str(e)}")
        print("Please install all requirements: pip install -r requirements.txt")
        return False

def main():
    """Main entry point for the script."""
    # Ensure necessary directories exist
    script_dir = ensure_data_directories()
    
    # Create command line parser
    parser = argparse.ArgumentParser(description="AgentNet Customer Support System")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # API server command
    server_parser = subparsers.add_parser("server", help="Run API server")
    server_parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind server to"
    )
    server_parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to bind server to"
    )
    
    # Streamlit command
    streamlit_parser = subparsers.add_parser("streamlit", help="Run Streamlit application")
    
    # CLI commands
    cli_parser = subparsers.add_parser("cli", help="Run CLI tools")
    cli_parser.add_argument(
        "cli_args", nargs="*",
        help="Arguments to pass to the CLI"
    )
    
    # Check environment command
    check_parser = subparsers.add_parser("check", help="Check environment and dependencies")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "server":
        # Check environment before running server
        deps_ok = check_requirements()
        if not deps_ok:
            sys.exit(1)
            
        run_server(args.host, args.port)
        
    elif args.command == "streamlit":
        # Check environment before running Streamlit
        deps_ok = check_requirements()
        ollama_ok = check_ollama()
        
        if not deps_ok:
            print("Attempting to continue despite missing dependencies...")
        
        run_streamlit()
        
    elif args.command == "cli":
        # Check environment before running CLI
        deps_ok = check_requirements()
        if not deps_ok:
            sys.exit(1)
            
        run_cli(args.cli_args)
        
    elif args.command == "check":
        # Check if all requirements are met and Ollama is running
        deps_ok = check_requirements()
        ollama_ok = check_ollama()
        
        if deps_ok and ollama_ok:
            print("\n✅ Environment is ready to run the AgentNet Customer Support System")
        else:
            print("\n❌ There are issues with your environment. Please fix them before running the system.")
            sys.exit(1)
    else:
        # If no command is provided, default to running Streamlit
        print("No command specified, starting Streamlit application by default...\n")
        # Check environment
        check_requirements()
        check_ollama()
        time.sleep(1)  # Give time to read the checks
        run_streamlit()

if __name__ == "__main__":
    main() 