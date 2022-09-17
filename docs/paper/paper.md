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

Species distribution modeling (SDM) is based on the Grinellean niche concept: the environmental conditions that allow individuals of a species to survive and reproduce will constrain the distributions of those species over space and time [@Grinnell:1917; @Wiens:2009]. The inputs to these models are typically spatially-explicit species occurrence records and gridded environmental covariates, which might include information on climate, topography, land cover or hydrology [@Booth:2014]. While many modeling methods have been developed to quantify and map these species-environment interactions, few software systems include support for both the required statistical modeling routines and for handling the full suite of geospatial analysis tools required to fit, apply, and summarize these models.

`elapid` is both a geospatial analysis and a species distribution modeling package. It provides an interface between vector and raster data for selecting random point samples, annotating point locations with coincident raster values, and summarizing raster values inside a polygon with zonal statistics. It provides a series of covariate transformation routines for increasing feature dimensionality, quantifying interaction terms, and normalizing unit scales. It provides a Python implementation of the popular Maxent SDM [@Phillips:2008], of a standard Niche Envelope Model [@Nix:1986], and support for working with other machine learning packages like `sklearn` [@scikit-learn]. And it allows users to add spatial context to any model by providing methods for computing geographically-explicit sample weights.

Biogeographers use species occurrence records and environmental data to understand the current, past, and future spatial distributions of biodiversity. `elapid` is a contemporary SDM package built on best practices from the past and aspires to support the next generation of biodiversity models.

# Statement of need

- presence-only data more abundant via GBIF, iNaturalist. Absence data are rare; presence-only models required.
- maxent commonly used, but it has limits - hard to fit multi-temporal, not a spatial model per se, weird raster formats, all data needs to be perfectly co-aligned (how to deal with multiple scales?)
- this is a python port of `maxnet` [@Phillips:2017], statistically, but includes some differences (regularization defaults, feature defaults). one complaint is that it's not spatial (assumed independence of samples). can now use geographic weights with same model form.
- includes other features, like suite of geospatial analysis tools. built on top of modern packages like geopandas and rasterio, supports working with data hosted on web and cloud storage.

Even fewer support working with modern, cloud-native geospatial data, making it hard to work with the large volume and of high resolution

# Acknowledgments

Many thanks to Jeffrey R. Smith for many long and thought-provoking discussions on species distribution modeling. Many thanks to David C. Marvin for helping me think creatively about counter-intuitive applications for Maxent. Many thanks to Gretchen C. Daily for promoting and supporting open source software in biodiversity and ecosystem services modeling.

# References
