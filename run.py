# Create this file at /mnt/data/amit/projects/ai/DeepSeek/unsloth/DI/DI/Qwen/TranformerRL/rl-transformer/run.py
import sys
import os

# Add the project directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import main function
from src.main import main

if __name__ == "__main__":
    main()