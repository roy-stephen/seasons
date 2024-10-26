import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seasons",
    version="0.2.1",  # Update the version number as needed
    author="Stephen Ezin",
    author_email="iamstephenezin@gmail.com",
    description="A Python package for detecting seasonality in time series data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/your-username/seasons",
    license="MIT", 
    python_requires=">=3.8",  # Specify the required Python version
    install_requires=[
        "numpy", 
        "scipy",
        "matplotlib",
        "statsmodels",  
        "joblib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
    ],
    keywords="seasonality detection time series analysis",
)