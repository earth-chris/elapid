# Installation guide

`elapid` is accessible from [pypi](https://pypi.org/project/elapid/):

```bash
pip install elapid
```

It's also accessible from [conda](https://anaconda.org/conda-forge/elapid):

```bash
conda install -c conda-forge elapid
```

This should suffice for most linux/mac users, as there are builds available for most of the dependencies (`numpy`, `sklearn`, `glmnet`, `geopandas`, `rasterio`).

Windows installs can be more difficult. There are two primary challenges you may face: installing the key [geospatial dependencies](#resolving-geospatial-dependencies), and installing [glmnet](#installing-glmnet).

You can avoid both of them by using [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about), which creates a linux environment on your Windows machine. If you're using an older version of Windows, see the options below.

---

## Resolving geospatial dependencies

### With conda

If you have `conda` installed, use that to install elapid's dependencies:

```bash
conda create -n elapid python=3.8 -y
conda activate elapid # or just `activate elapid` on windows
conda install -y -c conda-forge geopandas rasterio rtree scikit-learn tqdm
pip install elapid
```

### Installing on Windows without conda

You can get Windows builds of several key geospatial packages using `pipwin`, which installs wheels from an unofficial source:

```bash
pip install wheel pipwin
pipwin install numpy
pipwin install pandas
pipwin install shapely
pipwin install gdal
pipwin install fiona
pipwin install pyproj
pipwin install six
pipwin install rtree
pip install geopandas
pip install elapid
```

---

## Installing glmnet

`glmnet` is optional. When it's importable, `elapid` uses it to fit the Maxent
logistic regression (matching the R [maxnet][r-maxnet] reference); when it
isn't, `elapid` falls back to `sklearn`.

**Caveat — installing `glmnet` via pip is currently broken.** The [python
wrapper][glmnet-py] hasn't been released since 2020. Its `setup.py` predates
PEP 517 and fails to build under modern installers (`pip`, `uv`, etc.), and no
wheels are published on PyPI. The only working install paths are:

1. **conda-forge (recommended).** The conda-forge `glmnet` build is current on
   Python 3.8–3.11 (no 3.12+ build at time of writing):

   ```bash
   conda install -c conda-forge glmnet
   ```

2. **The elapid `dev-glmnet` pixi environment.** Equivalent to (1) but managed
   for you alongside the rest of the dev tooling:

   ```bash
   pixi run -e dev-glmnet test
   ```

3. **Manual Fortran build.** Install `gfortran` + `setuptools` and run
   `pip install --no-build-isolation glmnet`. Brittle; not recommended.

If `glmnet` isn't installable on your target Python, `elapid` still works —
the `sklearn` solver path covers the same modeling surface with slightly
different regularization handling (see [An important consideration](#an-important-consideration) below).

### An important consideration

To simplify installing and working with `elapid`, you can install the package and fit Maxent models with or without `glmnet`. This is handled by fitting the maximum likelihood model with either `glmnet` or with `sklearn`.

The results are similar, but not the same. This is largely because of differences in how regularization is handled.

`glmnet` can handle arrays of regularization terms, which allows fine-scale control over the potential importance of different features. There are a series of calibrated defaults used by `elapid`, which were originally defined in maxnet but have been updated in [opinionated ways](../sdm/maxent#differences-between-elapid-and-maxnet).

You can typically just assign a single value for regularization scores with `sklearn`.

The differences are relatively small. When comparing models fit in `maxnet` to `elapid` models, the level of agreement was `r2 = 0.91` for glmnet models, `r2 = 0.85` for sklearn models.

Still, I recommend users try their best to install `glmnet` if you're interested in maintaining fidelity to the other family of Maxent tools.


[glmnet-py]: https://github.com/civisanalytics/python-glmnet/
[r-maxnet]: https://github.com/mrmaxent/maxnet
