import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="psycho_AI-pkg-sciosci", # Replace with your own username
    version="0.0.2",
    author="Lizhen Liang and Daniel Acuna",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sciosci/psycho_ai",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=['scipy', 'pandas', 'numpy', 'matplotlib', 'wget', 'tqdm']
)
