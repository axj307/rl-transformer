# filepath: /mnt/data/amit/projects/ai/DeepSeek/unsloth/DI/DI/Qwen/TranformerRL/rl-transformer/setup.py
from setuptools import setup, find_packages

setup(
    name="transformer_rl",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0", 
        "matplotlib>=3.3.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Transformer-based RL for control problems",
)
