# Installation guide

`elapid` is accessible from [pypi](https://pypi.org/project/elapid/):

```bash
pip install elapid
```

This should suffice for most linux/mac users, as there are builds available for most of the dependencies (`numpy`, `sklearn`, `glmnet`, `geopandas`, `rasterio`).

Windows installs can be more difficult. There are two primary challenges you may face: installing the key [geospatial dependencies](#resolving-geospatial-dependencies), and installing [glmnet](#installing-glmnet).

You can avoid both of them by using [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about), which creates a linux environment on your Windows machine. If you're using an older version of Windows, see the options below.

---

## Resolving geospatial dependencies

### With conda

If you have `conda` installed, use that to install elapid's dependencies:

```bash
conda create -n elapid -python=3.8 -y
activate elapid
conda install -y geopandas rasterio rtree scikit-learn tqdm
```

Then you should be able to run `pip install elapid`.

### Without conda

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

`glmnet` needs to be manually installed on Windows. But, technically, it's not required.

`elapid` was written to try and match the modeling framework of the R version of Maxent, [maxnet][r-maxnet]. `maxnet` uses an inhomogeneous Poisson process model, which fits penalized maximum likelihood models, and is handled by the package [glmnet][glmnet-fortran].

There is a python wrapper for [glmnet][glmnet-py], which is used by `elapid`. But it has no Windows build, so it has to compile some fortran code on install. This means you need to have a fortran compiler running (like [MinGW-w64][mingw] or [Cygwin](https://www.cygwin.com/)) if you want to install it (`pip install glmnet`).

You can also checkout [this GitHub issue][fortran-issue] to read about other people's solutions or contribute a better solution.

### An important consideration

To simplify installing and working with `elapid`, you can install the package and fit Maxent models with or without `glmnet`. This is handled by fitting the maximum likelihood model with either `glmnet` or with `sklearn`.

The results are similar, but not the same. This is largely because of differences in how regularization is handled.

`glmnet` can handle arrays of regularization terms, which allows fine-scale control over the potential importance of different features. There are a series of calibrated defaults used by `elapid`, which were originally defined in maxnet but have been updated in [opinionated ways](../sdm/maxent#differences-between-elapid-and-maxnet).

You can typically just assign a single value for regularization scores with `sklearn`.

The differences are relatively small. When comparing models fit in `maxnet` to `elapid` models, the level of agreement was `r2 = 0.91` for glmnet models, `r2 = 0.85` for sklearn models.

Still, I recommend users try their best to install `glmnet` if you're interested in maintaining fidelity to the other family of Maxent tools.


[glmnet-fortran]: https://glmnet.stanford.edu/articles/glmnet.html
[glmnet-py]: https://github.com/civisanalytics/python-glmnet/
[fortran-issue]: https://github.com/earth-chris/elapid/issues/9
[mingw]: https://www.mingw-w64.org/
[r-maxnet]: https://github.com/mrmaxent/maxnet
