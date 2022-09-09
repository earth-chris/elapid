import os
import platform

from setuptools import setup

this_dir, this_path = os.path.split(os.path.abspath(__file__))
version = open(os.path.join(this_dir, "elapid", "__version__.py")).read().strip('"\n')
long_description = open(os.path.join(this_dir, "README.md"), "r", encoding="utf-8").read()
requirements = open(os.path.join(this_dir, "requirements.txt"), "r", encoding="utf-8").read().strip().split()

# remove glmnet requirement for windows installs
if platform.system() != "Linux":
    [requirements.pop(idx) for idx, pkg in enumerate(requirements) if "glmnet" in pkg]

setup_args = {
    "name": "elapid",
    "version": version,
    "url": "https://elapid.org",
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
    "python_requires": ">=3.7.0",
    "platforms": "any",
    "classifiers": [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
}

setup(**setup_args)
