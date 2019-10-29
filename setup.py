from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="recommend",
    version="0",
    description="squat",
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
    py_modules=["recommend"],
    python_requires=">=3.6",
    setup_requires=["setuptools>=38.6.0"],
)
