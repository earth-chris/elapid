# Installation

## pip

```bash
pip install elapid
```

You can also clone the source repository and install it locally.

```bash
git clone https://github.com/earth-chris/elapid.git
cd elapid
pip install -e .
```

## conda

You can use `conda` to ensure you have all the required dependencies (`glmnet` and `rasterio` have some library dependencies). From the repository's base directory:

```bash
conda env update
conda activate elapid
pip install -e .
```

This will create an environment named `elapid` and install an editable version of the package that you can use.
