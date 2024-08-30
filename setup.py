from setuptools import setup, find_packages

setup(
    name="gcgnn",
    version="0.0.1",
    author="Shengli (Bruce) Jiang",
    author_email="sj0161@princeton.com",
    url="https://github.com/webbtheosim/gcgnn",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy==1.24.4",
        "matplotlib==3.4.3",
        "torch==1.13.1",
        "proplot==0.9.7",
        "networkx==2.8.4",
        "torch-geometric==2.3.0",
        "scikit-learn==1.2.2",
    ],
)
