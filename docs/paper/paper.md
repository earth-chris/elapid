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

Species distribution modeling (SDM) is based on the Grinellean niche concept: the environmental conditions that allow individuals of a species to survive and reproduce will constrain the distributions of those species over space and time [@Grinnell:1917; @Wiens:2009]. The inputs to these models are typically spatially-explicit species occurrence records and gridded environmental covariates, which might include information on climate, topography, land cover or hydrology [@Booth:2014]. While many modeling methods have been developed to quantify and map these species-environment interactions, few software systems include support for both a) the required statistical modeling routines and b) handling the full suite of geospatial analysis tools required to fit, apply, and summarize these models.

`elapid` is both a geospatial analysis and a species distribution modeling package. It provides an interface between vector and raster data for selecting random point samples, annotating point locations with coincident raster values, and summarizing raster values inside a polygon with zonal statistics. It provides a series of covariate transformation routines for increasing feature dimensionality, quantifying interaction terms and normalizing unit scales. It provides a Python implementation of the popular Maxent SDM [@Phillips:2008] and a standard Niche Envelope Model [@Nix:1986], written to match the software design patterns of modern machine learning packages like `sklearn` [@scikit-learn]. It also allows users to add spatial context to any model by providing methods for computing geographically-explicit sample weights.

Biogeographers use species occurrence records and environmental data to understand the current, past, and future spatial distributions of biodiversity. `elapid` is a contemporary SDM package built on best practices from the past and aspires to support the next generation of biodiversity models.

# Statement of need

Species occurrence data—georeferenced point locations where a species has been observed and identified—are an important resource for understanding the environmental conditions that predict habitat suitability for that species. These data are now abundant thanks to the proliferation of open data policies, large-scale institutional collaboration across research groups and museums, advances in the quality of citizen science-driven applications, and increases in participation by citizen scientists. Tools for working with these data are less abundant, especially tools that support modern geospatial data formats and machine learning workflows.

`elapid` builds on a suite of well-known statistical modeling tools commonly used by biogeographers, extending them to add novel features, read and write modern spatial data formats, including cloud-hosted data, and save and share models. It provides methods for managing the full lifecyle of modeling data, including background point generation, extracting raster values for each point (i.e. point annotation), splitting train/test data, fitting models, and applying predictions to raster data. This provides users a very high degree of control over their modeling workflows, which is important for several reasons.

Point data are managed as `GeoSeries` and `GeoDataFrame` objects [@geopandas], which can be easily merged and split using traditional indexing method as well as with geograhic methods, and can be reprojected on the fly. It reads and writes data with `rasterio`, which provides a similarly convenient set of methods for geographically-indexed pixel sampling and handling on-the-fly reprojection [@rasterio]. `elapid` uses these features to silently handle many of the routine gotchas of working with geospatial data. It doesn't require data to be rigorously pre-processed so all rasters are perfectly aligned or all datasets are in matching projections. `elapid` can extract pixel-level raster data from datasets at different resolutions, from multi-band files, and harmonize projections as necessary, even when reading multi-scaler or mutli-source raster data during model inference.

Working with multi-temporal data—fitting SDMs to occurrence records and environmental data from multiple time periods—is also supported. Each time period's occurrence data can be annotated using the coincident environmental data, then concatenated into a single `GeoDataFrame`. Random background samples can likewise be generated for each time period, which ensures the background represents a broad distribution of conditions across the full temporal extent. Fitted models can be applied to multi-temporal environmental data to map changes in habitat suitability over time, and can be saved and restored later for future inference.

The main scientific contribution of `elapid` is in extending and modifying the Maxent SDM. Maxent important because presence-only. But its hard to control things like background point sampling, bias adjustments, train/test splits. Its also not spatial. Elapid addresses each of those things, in that order.

- sampling from raster or vector extents (like a range map or an ecoregion)
- bias adjustment in background sampling
- splitting train/test data yourself, including checkerboard split or geographic k-fold
- can add sample weights based on point density, removing typical assumptions about independence between samples, decreasing weights for nearby/clustered points.

The target audience for the package includes ecologists, biodiversity scientists, spatial analysts and machine learning scientists.

# Acknowledgments

Many thanks to Jeffrey R. Smith for many long and thought-provoking discussions on species distribution modeling. Thanks also to David C. Marvin for helping me think creatively about counter-intuitive applications for Maxent. And many thanks to Gretchen C. Daily for promoting and supporting open source software in biodiversity and ecosystem services modeling.

# References
