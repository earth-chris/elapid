---
title: 'elapid: Species distribution modeling tools for Python'
tags:
  - biogeography
  - species distribution modeling
  - gespatial analysis
  - machine learning
  - Python
authors:
  - name: Christopher B. Anderson
    orcid: 0000-0001-7392-4368
    affiliation: "1, 2"
affiliations:
 - name: Salo Sciences, San Francisco, CA, USA
   index: 1
 - name: Center for Conservation Biology, Stanford University, Stanford, CA, USA
   index: 2
date: 18 September 2022
bibliography: paper.bib
---

# Summary

Species distribution modeling (SDM) is based on the Grinellean niche concept: the environmental conditions that allow individuals of a species to survive and reproduce will constrain the distributions of those species over space and time [@Grinnell:1917; @Wiens:2009]. The inputs to these models are typically spatially-explicit species occurrence records and a series of environmental covariates, which might include information on climate, topography, land cover or hydrology [@Booth:2014]. While many modeling methods have been developed to quantify and map these species-environment interactions, few software systems include both a) the appropriate statistical modeling routines and b) support for handling the full suite of geospatial analysis required to fit, apply, and summarize these models.

`elapid` is both a geospatial analysis and a species distribution modeling package. It provides an interface between vector and raster data for selecting random point samples, annotating point locations with coincident raster data, and summarizing raster values inside a polygon with zonal statistics. It provides a series of covariate transformation routines for increasing feature dimensionality, quantifying interaction terms and normalizing unit scales. It provides a Python implementation of the popular Maxent SDM [@Phillips:2008] and a standard Niche Envelope Model [@Nix:1986], written to match the software design patterns of modern machine learning packages like `sklearn` [@scikit-learn]. It also allows users to add spatial context to any model by providing methods for spatially splitting train/test data and computing geographically-explicit sample weights.

Biogeographers use species occurrence records and environmental data to understand the current, past, and future spatial distributions of biodiversity. `elapid` is a contemporary SDM package built on best practices from the past, and aspires to help build the next generation of biodiversity modeling tools.

# Statement of need

Species occurrence data—georeferenced point locations where a species has been observed and identified—are an important resource for understanding the environmental conditions that predict habitat suitability for that species. These data are now abundant thanks to the proliferation of open data policies, large-scale institutional collaboration across research groups and museums, advances in the quality of citizen science-driven applications, and increases in participation by citizen scientists. Tools for working with these data are less abundant, however, especially ones that support modern geospatial data formats and machine learning workflows.

`elapid` builds on a suite of well-known statistical modeling tools commonly used by biogeographers, extending them to add novel features, to work with cloud-hosted data, and to save and share models. It provides methods for managing the full lifecyle of modeling data: generating background point data, extracting raster values for each point (i.e. point annotation), splitting train/test data, fitting models, and applying predictions to rasters. It provides a very high degree of control for model design, which is important for several reasons.

First is to provide simple and flexible methods for working with spatial data. Point data are managed as `GeoSeries` and `GeoDataFrame` objects [@geopandas], which can be easily merged and split using traditional indexing method as well as with geograhic methods. They can also be reprojected on-the-fly. `elapid` reads and writes raster data with `rasterio`, which provides a similarly convenient set of methods for geographically-indexed pixel locations and handling on-the-fly reprojection [@rasterio]. These features are wrapped to handle many of the routine tasks and gotchas of working with geospatial data. It doesn't require data to be rigorously pre-processed so that all rasters are perfectly aligned, nor does it require that all datasets are in matching projections. `elapid` can extract pixel-level raster data from datasets at different resolutions, from multi-band files, and harmonize projections on-the-fly, for both model fitting and for inference.

Another advantage of `elapid`'s flexible design is that it can be used to extend traditional species distribution models in ways that are difficult to implement in other software systems. Working with multi-temporal data, for example—fitting SDMs to occurrence records and environmental data from multiple time periods—is also supported. Each time period's occurrence data can be annotated using the coincident environmental data, then concatenated into a single `GeoDataFrame`. Random background samples can likewise be generated for each time period, which ensures the background represents a broad distribution of conditions across the full temporal extent. Fitted models can be applied to multi-temporal environmental data to map changes in habitat suitability over time, and can be saved and restored later for future inference.

The main scientific contribution of `elapid` is in extending and modifying the Maxent SDM. First published in 2006, Maxent remains a valuable approach because it is a presence-only model designed to work with the kinds of species occurrence data data that have proliferated lately. Presence-only models formulate binary classification models as presence/background (1/0) instead of presence/absence, which changes how models are fit and interpreted. Background points are a spatially-random sample of the extent where a species might be found, which should be sampled with the same level of effort and bias as the species occurrence data. The conceptual basis for presence/background models is: our null expectation is a species is equally likely to be found anywhere within it's range, and differences in environmental conditions between where a species occurs and in the conditions across its full range should indicate niche preferences. Presence-only models reduce the burden of finding absence data, which are problematic to boot, but they increase the burden of precisely selecting background points.

`elapid` includes several methods for sampling the background. Points can be sampled uniformly within a polygon, like a range map or an ecoregion extent. Sampling points from rasters can be done uniformly across the full extent or only from pixels with valid, unmasked data. Working with bias rasters is also supported. Any raster with monotonically increasing values can be used as a sample probability map, increasing the probability a that a sample is drawn in locations with higher pixel values. One important role for the niche envelope model is to create bias maps to ensure background points are only sampled within the broad climatic envelope where a species occurs.

Creating spatially-explicit train/test splits and cross-validation folds is another key feature of `elapid`. Spatial models should include methods for handling spatially-specific modeling paradigms, like the lack of independence of nearby samples or spatial biases in sample density. Quantifying and understanding model skill requires accounting for these spatial autocorrelations. Checkerboard cross-validation can mitigate bias introduced by spatially clustered points. Creating spatially-explicit $k$-fold splits—independent clusters based on x/y locations—can quantify how well model predictions generalize to new areas. And tuning sample weights based on the density of nearby points decreases the risk of overfitting to autocorellated environmental features from areas with high sample density.

These methods are not solely restricted to the SDMs implemented in `elapid`, but can add spatial context to other machine learning models. Geographic sample weights can be used to fit random forests, boosted regression trees, generalized linear models, and other approaches commonly used to predict spatial distributions. `elapid` also includes a series of feature transformers, including the transformations used in Maxent, which can extend covariate feature space to improve model skill.

`elapid` was designed to provide a series of modern tools for quantifying biodiversity change. The target audience for the package includes ecologists, biodiversity scientists, spatial analysts and machine learning scientists. Working with software to understand the rapid changes reshaping our biosphere should be easy and enjoyable. Because thinking about the ongoing annihilation of nature that's driving our current extinction crisis very much is not.

# Acknowledgments

Many thanks to Jeffrey R. Smith for many long and thought-provoking discussions on species distribution modeling. Thanks also to David C. Marvin for helping me think creatively about novel applications for Maxent. And many thanks to Gretchen C. Daily for promoting and supporting access to open source software for biodiversity and ecosystem services modeling.

# References
