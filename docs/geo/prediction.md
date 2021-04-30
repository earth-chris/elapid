Once you've fit a model, you can apply it to a set of raster covariates to produce gridded suitability maps.

```python
elapid.apply_model_to_rasters(
    model,
    raster_paths,
    output_path,
    template_idx = 0,
    transform = "cloglog",
    nodata=-9999,
)
```

The list and band order of the rasters must match the order of the covariates used to train the model. It reads each dataset in a block-wise basis, applies the model, and writes gridded predictions.

If the raster datasets are not consistent (different extents, resolutions, etc.), it wll re-project the data on the fly, with the grid size, extent and projection based on a 'template' raster. Use the `template_idx` keyword to specify the index of which raster file to use as the template (`template_idx=0` sets the first raster as the template).

In the example above, it's important to set the template to the `california-leaf-area-index.tif` file. This is because this is the smallest extent with data, and it'll only read and apply the model to the `usa` and `global` datasets in the area covered by `california`. If you were to set the extent to `usa-mean-temperature.tif`, it would still technically function, but there would be a large region of `nodata` values where there's insufficient covariate coverage.
