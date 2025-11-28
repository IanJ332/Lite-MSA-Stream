from setuptools import setup, find_packages

setup(
    name="Yayo-MSA-Stream",
    version="0.1.0",
    description="Real-time Multimodal Sentiment Analysis Stream",
    author="IanJ332",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "websockets",
        "numpy",
        "torch",
        "torchaudio",
        "onnxruntime",
        "transformers",
        "optimum[onnxruntime]",
    ],
)
