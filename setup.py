from setuptools import find_packages, setup

setup(
    name="minigpt",
    version="1.0.14",
    description="Experiments using NanoGPT",
    author="Christy George",
    author_email="christy.george@gmail.com",
    url="https://github.com/christyxgeorge/minigpt",
    packages=find_packages(),
    install_requires=[
        "semver==3.0.1",
        "kaggle==1.5.16",
        "numpy",  # ==1.25.2
        "pandas",  # ==2.1.0
        "psutil==5.9.5",
        "Requests==2.31.0",
        "sentencepiece==0.1.99",
        "tiktoken==0.4.0",
        "torch==2.0.1",
        "tqdm==4.66.1",
        "wandb==0.15.10",
        "python-dotenv==1.0.0",
    ],
)
