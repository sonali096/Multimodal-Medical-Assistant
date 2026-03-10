#!/usr/bin/env python3
"""
Quick Setup Script for Multimodal Medical Assistant
Automates installation and verification of dependencies
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def run_command(cmd, description):
    """Run shell command and handle errors"""
    print(f"\n▶️  {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9 or higher required")
        return False
    
    print("✅ Python version OK")
    return True

def check_ollama():
    """Check if Ollama is installed"""
    print_header("Checking Ollama Installation")
    
    try:
        result = subprocess.run(
            "ollama --version",
            shell=True,
            capture_output=True,
            text=True
        )
        print(f"✅ Ollama installed: {result.stdout.strip()}")
        
        # Check if llama2 model is available
        result = subprocess.run(
            "ollama list",
            shell=True,
            capture_output=True,
            text=True
        )
        
        if "llama2" in result.stdout or "llama3" in result.stdout:
            print("✅ Llama model found")
        else:
            print("⚠️  Llama model not found")
            print("   Run: ollama pull llama2")
        
        return True
    except:
        print("⚠️  Ollama not found")
        print("   Install from: https://ollama.ai")
        print("   After installation, run: ollama pull llama2")
        return False

def print_next_steps():
    """Print next steps for user"""
    print_header("Quick Start Guide 🎉")
    
    activate_cmd = (
        ".venv\\Scripts\\activate" if sys.platform == "win32"
        else "source .venv/bin/activate"
    )
    
    print(f"""
Next Steps:

1. Activate virtual environment:
   {activate_cmd}

2. Install dependencies:
   pip install -r requirements.txt

3. Download spaCy model:
   python -m spacy download en_core_web_sm

4. (Optional) Start Ollama:
   ollama serve
   ollama pull llama2

5. Run interactive demo:
   python main.py --mode interactive

6. Or start API server:
   python main.py --mode api --port 8000

For detailed help, see README.md
""")

def main():
    """Main setup function"""
    print_header("Multimodal Medical Assistant - Environment Check")
    
    # Check prerequisites
    check_python_version()
    check_ollama()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
