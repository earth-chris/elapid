# elapid

<img src="https://earth-chris.github.io/elapid/img/amazon.jpg" alt="the amazon"/>

<p align="center">
  <em>Contemporary species distribution modeling tools for python.</em>
</p>

![GitHub](https://img.shields.io/github/license/earth-chris/elapid)
![PyPI](https://img.shields.io/pypi/v/elapid)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/earth-chris/elapid/docs)
![PyPI - Downloads](https://img.shields.io/pypi/dm/elapid)
![GitHub last commit](https://img.shields.io/github/last-commit/earth-chris/elapid)
![Lines of code](https://img.shields.io/tokei/lines/github/earth-chris/elapid)

---

**Documentation**: [earth-chris.github.io/elapid](https://earth-chris.github.io/elapid)

**Source code**: [earth-chris/elapid](https://github.com/earth-chris/elapid)

---

## Introduction

`elapid` provides python support for species distribution modeling. This includes a custom implementation of [Maxent][home-maxent] and a suite of tools to simplify working with biogeography data.

The name is an homage to *A Biogeographic Analysis of Australian Elapid Snakes* (H.A. Nix, 1986), the paper widely credited with defining the essential bioclimatic variables to use in species distribution modeling. It's also a snake pun (a python wrapper for mapping snake biogeography).

---

## Installation

```bash
pip install elapid
```

This should suffice for most linux/mac users, as there are available unix builds of the underlying python dependencies (`numpy`, `sklearn`, `glmnet`, `rasterio`, etc.).

Windows installs are more challenging. [glmnet][glmnet] has to compile some fortran code on install, meaning you need to have a fortran compiler running (like [MinGW-w64][mingw] or [Cygwin](https://www.cygwin.com/)).

You can review Windows install instructions with slightly more detail, or contribute a better solution, at [this issue][fortran-issue].

---

## Package design

The amount and quality of bioegeographic data has increased dramatically over the past decade, as have cloud-based tools for working with it. `elapid` was designed to provide a set of modern, python-based tools for working with species occurrence records and environmental covariates to map different dimensions of a species' niche.

`elapid` supports working with modern geospatial data formats and uses contemporary approaches to training statistical models. It uses `sklearn` conventions to fit and apply models, `rasterio` to handle raster operations, `geopandas` for vector operations, and processes data under the hood with `numpy`.

It does the following things reasonably well:

**Point sampling**

Drawing random geographic point samples (or `background` points) within polygons or rasters, handling `nodata` locations, as well as sampling from bias maps (using `elapid.sample_geoseries()`, `elapid.sample_raster()`, or `elapid.sample_bias_file()`).

**Vector annotation**

Annotating point data with coincident raster data, creating `GeoDataFrames` with sample locations and co-aligned covariate values (using `elapid.annotate()`).

**Zonal statistics**

Calculating zonal statistics from multi-band, multi-raster data into a single `GeoDataFrame` from one command (using `elapid.zonal_stats()`).

**Feature transformations**

Transforming covariate data into derivative `features` to expand data dimensionality (primarily the `elapid.MaxentFeatureTransformer()`, but see others under `elapid.features`)

**Species distribution modeling**

Training and applying generic species distribution models (like `elapid.MaxentModel()` and `elapid.NicheEnvelopeModel()`).

**Applying models to rasters**

Applying pixel-based models with a `.predict()` method to rasters (like training a `RandomForestClassifier()` and applying with `elapid.apply_model_to_rasters()`).

**Cloud-native geo support**

Working with cloud- or web-hosted raster and vector data (data hosted on `s3://`, `gs://`, or `https://`).

---

`elapid` requires some effort on the user's part to draw samples and extract covariate data. This is by design. Selecting background samples, splitting train/test data, and specifying model parameters are all critical modeling choices that have profound effects on model prediction and interpretation. This extra flexibility provides more control over the seemingly black-box approach of Maxent's java implementation, and enabling users to better tune and evaluate their models.

---

## Contact

<a href="https://twitter.com/earth_chris">![Twitter Follow](https://img.shields.io/twitter/follow/earth_chris)</a>
<a href="https://twitter.com/forestobs">![Twitter Follow](https://img.shields.io/twitter/follow/forestobs)</a>


[home-maxent]: https://biodiversityinformatics.amnh.org/open_source/maxent/
[r-maxnet]: https://github.com/mrmaxent/maxnet
[glmnet]: https://github.com/civisanalytics/python-glmnet/
[fortran-issue]: https://github.com/earth-chris/elapid/issues/9
[mingw]: https://www.mingw-w64.org/
