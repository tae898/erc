import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cltl-face-all",  # Replace with your own username
    version="0.0.0",
    author="Taewoon Kim",
    author_email="t.kim@vu.nl",
    description="face bounding box detection, 68 landmarks, age, gender, 512-D embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leolani/cltl-face-all",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
