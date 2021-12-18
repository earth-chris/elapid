from setuptools import setup

version = open("elapid/__version__.py").read().strip('"\n')
long_description = open("README.md", "r", encoding="utf-8").read()
requirements = open("requirements.txt", "r", encoding="utf-8").read().strip().split()

setup_args = {
    "name": "elapid",
    "version": version,
    "url": "https://github.com/earth-chris/elapid",
    "license": "MIT",
    "author": "Christopher Anderson",
    "author_email": "cbanders@stanford.edu",
    "description": "Species distribution modeling support tools",
    "long_description": long_description,
    "long_description_content_type": "text/markdown",
    "keywords": [
        "biogeography",
        "ecology",
        "conservation",
        "SDM",
        "species distribution modeling",
        "maxent",
    ],
    "packages": ["elapid"],
    "include_package_data": True,
    "install_requires": requirements,
    "python_requires": "<3.9.0",
    "platforms": "any",
    "classifiers": [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
}

setup(**setup_args)
