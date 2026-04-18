"""
Setup script for Dog Breed Identification System
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dog-breed-identification",
    version="2.0.0",
    author="Đỗ Văn Minh",
    author_email="dovanminh100104@gmail.com",
    description="Deep Learning system for classifying 120 dog breeds with ensemble learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dovanminh100104-coder/dog-breed-identification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "mobile": [
            "tensorflow-model-optimization>=0.7.0",
            "coremltools>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dog-breed-train=src.final_dog_breed_classifier:main",
            "dog-breed-test=src.test_model:main",
            "dog-breed-api=src.api_server:main",
            "dog-breed-optimize=src.model_optimizer:main",
            "dog-breed-validate=src.data_validator:main",
            "dog-breed-mobile=src.mobile_deployment:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": [
            "config.py",
            "requirements.txt",
            "README.md",
            "LICENSE",
        ],
    },
)
