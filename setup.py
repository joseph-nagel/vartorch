from setuptools import setup, find_packages


NAME = 'vartorch'
VERSION = '0.0.1'
AUTHOR = 'Joseph Nagel'
EMAIL = 'JosephBNagel@gmail.com'
URL = 'https://github.com/joseph-nagel/vartorch'
LICENSE = 'MIT'
DESCRIPTION = 'Variational inference for Bayesian neural nets with PyTorch'


try:
    with open('README.md', 'r') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


setup(
    name = NAME,
    version = VERSION,
    author = AUTHOR,
    author_email = EMAIL,
    url = URL,
    license = LICENSE,
    description = DESCRIPTION,
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    packages = find_packages(),
    install_requires = [
        'numpy',
        'torch'
    ],
    python_requires = '>=3.6'
)

