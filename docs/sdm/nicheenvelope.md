# Niche Envelope Models

Niche envelope models use the range of environmental conditions where a species occurs to constrain predictions of where they might occur.

`elapid` implements a simple but straightforward approach to estimating niche envelopes.

It uses the distribution of covariate values where a species is observed to assign binary (0/1) suitability scores on a covariate-by-covariate basis if the conditions at any location are within the range of observed covariate values.

```python
import elapid as ela

x, y = ela.load_sample_data()
envelope = ela.NicheEnvelopeModel(percentile_range=[2.5, 97.5])
envelope.fit(x, y)
```

The `percentile_range` parameter controls how inclusive the envelope is. Setting it to [0, 100] includes all observed covariate values within the range of suitability. Modifying this range may be useful to clip spurious or extreme values. The default is to use the 95th percentile range.

Binary suitability is assessed on a per-covariate basis, so the envelope has as many dimensions as it does covariates. `elapid` includes three methods for estimating the envelope of suitability across these dimensions.

```python
# overlay options include ['average', 'intersection', 'union']
envelope = ela.NicheEnvelopeModel(overlay='intersection')
```

`average` computes the pixel-wise mean of the the binary suitability scores across each covariate to produce a 0.0-1.0 score. `intersection` only includes areas with suitable conditions for all covariates within the envelope. Conversely, `union` includes an area within the envelop if any covariate is within the suitable range.

---

## Chaining with another SDM

Niche envelope models are also a useful way to select background points for other species distribution models.

Since many methods expect that background points represent the landscape of potentially suitable locations for a species, a NicheEnvelopeModel can filter out points where a species is never expected to be found.

```python
# estimate the envelope of suitability
x, y = ela.load_sample_data()
envelope = ela.NicheEnvelopeModel(overlay="average")
envelope.fit(x, y)

# only select points with >50% envelope suitability
in_range = envelope.predict(x) > 0.5
xsub, ysub = x[in_range], y[in_range]

# fit a maxent model
maxent = ela.MaxentModel()
maxent.fit(xsub, ysub)
```

This approach is useful for reducing overfitting by excluding point locations that don't represent the range of conditions across the landscape where a species won't occur.
