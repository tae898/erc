import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="erc", 
    version="0.0.0",
    author="Taewoon Kim",
    author_email="t.kim@vu.nl",
    description="erc pytorch package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tae898/erc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
