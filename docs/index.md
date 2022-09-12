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

## :snake: Introduction

`elapid` is a series of species distribution modeling tools for python. This includes a custom implementation of [Maxent][home-maxent] and a suite of methods to simplify working with biogeography data.

The name is an homage to *A Biogeographic Analysis of Australian Elapid Snakes* (H.A. Nix, 1986), the paper widely credited with defining the essential bioclimatic variables to use in species distribution modeling. It's also a snake pun (a python wrapper for mapping snake biogeography).

---

## :seedling: Installation

```bash
pip install elapid
```

This should suffice for most linux/mac users, as there are builds available for most of the dependencies (`numpy`, `sklearn`, `glmnet`, `geopandas`, `rasterio`).

While there is a pip distribution for Windows, you may experience some challenges during install. The easiest way to overcome these challenges is to use [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/about). Otherwise see [this page](/install) for install support.

---

## :deciduous_tree: Why use elapid?

The amount and quality of bioegeographic data has increased dramatically over the past decade, as have cloud-based tools for working with it. `elapid` was designed to provide a set of modern, python-based tools for working with species occurrence records and environmental covariates to map different dimensions of a species' niche.

`elapid` supports working with modern geospatial data formats and uses contemporary approaches to training statistical models. It uses `sklearn` conventions to fit and apply models, `rasterio` to handle raster operations, `geopandas` for vector operations, and processes data under the hood with `numpy`.

This makes it easier to do things like fit/apply models to multi-temporal and multi-scale data, fit geographically-weighted models, create ensembles, precisely define background point distributions, and summarize model predictions.

It does the following things reasonably well:

:globe_with_meridians: **Point sampling**

Select random geographic point samples (aka background or pseudoabsence points) within polygons or rasters, handling `nodata` locations, as well as sampling from bias maps (using `elapid.sample_raster()`, `elapid.sample_vector()`, or `elapid.sample_bias_file()`).

:chart_with_upwards_trend: **Vector annotation**

Extract and annotate point data from rasters, creating `GeoDataFrames` with sample locations and their matching covariate values (using `elapid.annotate()`). On-the-fly reprojection, dropping nodata, multi-band inputs and multi-file inputs are all supported.

:bar_chart: **Zonal statistics**

Calculate zonal statistics from multi-band, multi-raster data into a single `GeoDataFrame` from one command (using `elapid.zonal_stats()`).

:bug: **Feature transformations**

Transform covariate data into derivative `features` to expand data dimensionality and improve prediction accuracy (like `elapid.features.ProductTransformer()` or the all-in-one `elapid.MaxentFeatureTransformer()`).

:bird: **Species distribution modeling**

Train and apply species distribution models based on annotated point data, configured with sensible defaults (like `elapid.MaxentModel()` and `elapid.NicheEnvelopeModel()`).

:earth_asia: **Applying models to rasters**

Apply any pixel-based model with a `.predict()` method to raster data to easily create prediction probability maps (like training a `RandomForestClassifier()` and applying with `elapid.apply_model_to_rasters()`).

:cloud: **Cloud-native geo support**

Work with cloud- or web-hosted raster/vector data (on `https://`, `gs://`, `s3://`, etc.) to keep your disk free of temporary files.

Check out some example code snippets and workflows on the [examples](examples/geo) page.

---

:snake: `elapid` requires some effort on the user's part to draw samples and extract covariate data. This is by design.

Selecting background samples, computing sample weights, splitting train/test data, and specifying training parameters are all critical modeling choices that have profound effects on inference and interpretation.

The extra flexibility provided by `elapid` enables more control over the seemingly black-box approach of Maxent, enabling users to better tune and evaluate their models.

---

## Developed by

[Christopher Anderson](https://cbanderson.info)

<a href="https://twitter.com/earth_chris">![Twitter Follow](https://img.shields.io/twitter/follow/earth_chris)</a>
<a href="https://github.com/earth-chris">![GitHub Stars](https://img.shields.io/github/stars/earth-chris?affiliations=OWNER%2CCOLLABORATOR&style=social)</a>

[home-maxent]: https://biodiversityinformatics.amnh.org/open_source/maxent/
[r-maxnet]: https://github.com/mrmaxent/maxnet
