In addition to the maxent modeling support tools, `elapid` includes a series of geospatial data processing routines. These should make it easy to work with species occurrence records and raster covariates in multiple formats.

It uses `geopandas` for vector processing operations and `rasterio` for raster processing operatons.

Here's an example end-to-end workflow (using dummy paths to demonstrate functionality).

```python
import elapid

vector_path = "/home/slug/ariolimax-californicus.shp"
raster_path = "/home/slug/california-climate-veg.tif"
output_path = "/home/slug/ariolimax-californicus-habitat.tif"
model_path = "/home/slug/ariolimax-claifornicus-model.ela"

# sample the raster values at point locations
presence = elapid.raster_values_from_vector(vector_path, raster_path)
pseudoabsence_points = elapid.pseudoabsence_from_raster(raster_path)
pseudoabsence = elapid.raster_values_from_geoseries(pseudoabsence_points, raster_path)

# merge the datasets into one dataframe
pseudoabsence['presence'] = 0
presence['presence'] = 1
y = presence['presence'].append(pseudoabsence['presence']).reset_index(drop=True)
x = presence.drop(['presence'], axis=1).append(pseudoabsence.drop(['presence'], axis=1)).reset_index(drop=True)

# train the model
model = elapid.MaxentModel()
model.fit(x, y)

# apply it to the full extent and save the model for later
elapid.apply_model_to_rasters(model, raster_path, output_path, transform="logistic")
elapid.save_object(model, model_path)
```

To work with this saved model later, you can run:

```python
model = elapid.load_object(model_path)
```
