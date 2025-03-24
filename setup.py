from setuptools import setup, find_packages

setup(
    name="chronos_forecaster",
    version="0.1.1",
    description="Making time series forecasting with Amazon's Foundation Model Chronos simple and accessible.",
    author="Sebastião Santos Lessa",
    author_email="sebastiao.lessa@inesctec.pt",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "autogluon.timeseries",
        "setuptools",
        "pandas",
        "torch",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)
