# Maxent

Maxent is a species distribution modeling (SDM) system, which uses species observations and environmental data to predict where a species might be found under past, present or future environmental conditions.

Its a presence/background model, meaning it uses data on where a species is present and, instead of data on where it is absent, a random sample of the region where you might expect to find that species. This is convenient because it reduces data collection burdens (absence data isn't required) but it also introduces some unique challenges for fitting and interpreting models.

Formally, Maxent estimates habitat suitability (i.e. the fundamental niche) using species occurrence records (`y = 1`), randomly sampled "background" location records (`y = 0`), and environmental covariate data (`x`) for each location of `y`.

Maxent doesn't directly estimate relationships between presence/background data and environmental covariates (so, not just `y ~ x`). Instead, it fits a series of feature transformatons (`z`) to the covariate data (e.g. computing pairwise products between covariates, setting random covariate thresholds). Maxent then estimates the conditional probability of finding a species given a set of environmental conditions as:

```
Pr(y = 1 | f(z)) = (f1(z) * Pr(y = 1)) / f(z)
```

`elapid` provides python tools for fitting maxent models, computing features, and working with geospatial data. It's goal is to help us better understand where the species are, why they are there, and where they will go in the future.
