from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="video-swin-transformer",
    version="0.1.0",
    author="Parker Martin",
    author_email="parker.martin@osumc.edu",
    description="A clean PyTorch implementation of Video Swin Transformer without mmlab dependencies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/video-swin-transformer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.8.0",
        "numpy>=1.19.0",
        "pillow>=7.0.0",
        "matplotlib>=3.3.0",
        "einops>=0.3.0",
    ],
    extras_require={
        "test": ["opencv-python", "matplotlib", "memory-profiler"],
        "video": ["opencv-python>=4.5.0", "av>=8.0.0"],
        "dev": ["tqdm>=4.50.0", "tensorboard>=2.4.0"],
        "export": ["onnx>=1.8.0", "onnxruntime>=1.7.0"],
    },
)