from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="recommend",
    version="0.5",
    description="Implicit Recommender",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
    keywords=["recommend"],
    url="https://github.com/maxhumber/recommend",
    author="Max Humber",
    author_email="max.humber@gmail.com",
    license="MIT",
    packages=find_packages(),
    # pair with import pkg_resources
    package_data={"recommend": ["data/candy.csv"]},
    python_requires=">=3.6",
    install_requires=["scipy", "numpy"],
    setup_requires=["setuptools>=38.6.0"],
)
