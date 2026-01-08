from setuptools import setup, find_packages

setup(
    name="chronos_forecaster",
    version="0.2.2",
    description="Unified interface for Amazon Chronos and Chronos-2 time series forecasting engines.",
    author="Sebastião Santos Lessa",
    author_email="sebastiao2002@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "chronos-forecasting",
        "setuptools",
        "pandas",
        "torch",
        "boto3",
    ], 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
