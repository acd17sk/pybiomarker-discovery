from setuptools import setup, find_packages

setup(
    name="biomarker-discovery",
    version="0.1.0",
    author="Your Name",
    description="Multi-modal biomarker discovery platform using PyTorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/biomarker-discovery",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "wandb>=0.12.0",
        "transformers>=4.20.0",
        "librosa>=0.9.0",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
        "pyyaml>=5.4.0",
        "tensorboard>=2.8.0",
        "pytest>=7.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
    ],
    extras_require={
        "dev": [
            "jupyterlab>=3.0.0",
            "ipywidgets>=7.6.0",
            "pytest-cov>=3.0.0",
            "mypy>=0.910",
        ],
        "privacy": [
            "opacus>=1.0.0",  # Differential privacy
            "syft>=0.5.0",    # Federated learning
        ],
    },
)