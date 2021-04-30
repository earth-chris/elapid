In addition to species occurrence records, maxent requires a set of pseudo-absence (i.e. background) points. These are a random geographic samping of where you might expect to find a species.

## From a raster's extent

You can use `elapid` to create a random geographic sampling of points from unmasked locations within a raster's extent:

```python
count = 10000 # the number of points to generate
pseudoabsence_points = elapid.pseudoabsence_from_raster(raster_path, count)
```

## From a vector polygon

Species occurrence records are often biased in their collection (collected near roads, in protected areas, etc.), so we typically need to be more precise in where we select pseudo-absence points. You could use a vector with a species range map to select records:

```python
range_path = "/home/slug/ariolimax-californicus-range.shp"
pseudoabsence_points = elapid.pseudoabsence_from_vector(range_path, count)
```

If you've already computed a polygon using geopandas, you can pass it instead to `elapid.pseudoabsence_from_geoseries()`, which is what `pseudoabsence_from_vector()` does under the hood.

## From a bias raster

You could also pass a raster bias file, where the raster grid cells contain information on the probability of sampling an area:

```python
# assume covariance between vertebrate and invertebrate banana slugs
bias_path = "/home/slug/proximity-to-ucsc.tif"
pseudoabsence_points = pseudoabsence_from_bias_file(bias_path)
```

The grid cells can be an arbitrary range of values. What's important is that the values encode a linear range of numbers that are higher where you're more likely to draw a sample. The probability of drawing a sample is dependent on two factors: the range of values provided and the frequency of values across the dataset.

So, for a raster with values of `1` and `2`, you're sampling probability for raster locations of `2` is twice that as `1` locations. If these occur in equal frequency (i.e. half the data are `1` valuess, half are `2` values), then you'll likely sample twice as many areas with `2` values. But if the frequency of `1` values is much greater than `2` values, you'll shift the distribution. But you're still more likely, on a per-draw basis, to draw samples from `2` locations.

The above example prioritizes sampling frequency in the areas around UC Santa Cruz, home to all types of slug, based on the distance to the campus.
