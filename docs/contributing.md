# Contributing to elapid

Contributions to `elapid` are welcome, particularly from biogeographers, regular geographers, and machine teachers. You're welcome to fix bugs, add new features, expand or clarify documentation, or posit outlandish new modeling approaches.

All contributions should go through the GitHub repository. Bug reports, ideas, and questions should be raised by opening an issue on the GitHub tracker. Suggestions for changes in code or documentation should be submitted as a pull request. All discussion should take place on GitHub to keep the development of `elapid` transparent.

If you decide to contribute, ensure that you're using an up-to-date `main` branch. The latest development version will always be there, including the documentation.

---

## Steps for contributing

1. Fork the git repository
1. Create a development environment & install dependencies
1. Create a new branch, make changes to code & add tests
1. Update the docs
1. Submit a pull request

---

## Fork the git repository

You will need your own fork to work on the code. Go to the `elapid` project page and hit the Fork button. You will want to clone your fork to your machine:

```bash
git clone git@github.com:YOUR-USER-NAME/elapid.git elapid-YOURNAME
cd elapid-YOURNAME
git remote add upstream git://github.com/elapid/elapid.git
```

This creates the directory `elapid-YOURNAME` and connects your repository to the upstream (main project) elapid repository.

---

## Create a development environment & install dependencies

A development environment is a virtual space where you can keep an independent `elapid` install. This makes it easy to keep both a stable version in one place you use for work, and a development version (which you may break while playing with code) in another. First, you should:

- Install [miniconda](http://conda.pydata.org/miniconda.html) or [anaconda](http://docs.continuum.io/anaconda/)
- `cd` to the `elapid` source directory

Linux and Mac users can then create the development environment with:

```bash
make init
```

This will create a conda environment named `elapid` then install the package, it's dependencies, `pre-commit` & `pytest`.

Windows users need to do things a bit differently, as is often the case:

```bash
conda create -n elapid -python=3.8 -y
activate elapid
conda install geopandas rasterio scikit-learn tqdm pre-commit pytest pytest-cov pytest-xdist
pre-commit install
pip install -e .
```

This library uses `black`, `flake8` and `isort` pre-commit hooks. You should be familiar with [pre-commit](https://pre-commit.com/) before contributing.

---

## Create a new branch, make changes to code & add tests

Make changes to the code on a separate branch to keep you main branch clean:

```bash
git checkout -b shiny-new-feature
```

Make changes to your code and write tests as you go. Write clear, [self-documenting code](https://realpython.com/documenting-python-code/) to spend more time developing and less time describing how the code works.

If your branch is no longer up-to-date with `main`, run the following code to update it:

```bash
git fetch upstream
git rebase upstream/main
```

Testing is done with `pytest`, which you can run with either:

```bash
# for linux/mac
make test

# for windows
pytest -x -n auto --cov --no-cov-on-fail --cov-report=term-missing:skip-covered
```

---

## Update the docs

There are two places to update docs. One is required (docstrings), the other optional (`mkdocs` web documentation).

Adding docstrings to each new function/class is required. `elapid` uses [Google-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) and, when you contribute to it, you should too. `mkdocs` automatically renders the API docs for all functions written with this style, so you don't need to re-document each function outside of the code.

If your code contributes important new features, or introduces novel/interesting concepts, write new documentation in the `docs/` directory. The docs system is managed by `mkdocs`, which renders from Markdown.

You can install `mkdocs` and the associated plugins with:

```bash
pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-jupyter livereload
```

Then you can render the docs locally with:

```bash
mkdocs serve
```

---

## Submit a pull request

Once you’ve made changes and pushed them to your forked repository, you then submit a pull request to have them integrated into the `elapid` code base.

For more information, you can find a PR tutorial in [GitHub’s Help Docs](https://help.github.com/articles/using-pull-requests/).

---

Thanks, and happy mapping!
