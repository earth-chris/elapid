Here's an example end-to-end example for training a Maxent model (using dummy paths to demonstrate functionality).

This is to demonstrate the simplest pattern of model training. Full model training and evaluation should include creating train/test splits, cross-validation, feature selection, etc. These are not covered here.

```python
import elapid as ela

vector_path = "/home/slug/ariolimax-californicus.gpkg"
raster_path = "/home/slug/california-climate.tif"
output_raster_path = "/home/slug/ariolimax-californicus-habitat.tif"
output_model_path = "/home/slug/ariolimax-claifornicus-model.ela"

# sample the raster values for background point locations
pseudoabsence_points = ela.saple_raster(raster_path, count=5000)

# read the raster covariates at each point location
presence = ela.annotate(vector_path, raster_path)
pseudoabsence = ela.annotate(pseudoabsence_points, raster_path)

# merge the datasets into one dataframe
pseudoabsence['presence'] = 0
presence['presence'] = 1
merged = presence.append(pseudoabsence).reset_index(drop=True)
x = merged.drop(columns=['presence'])
y = merged['presence']

# train the model
model = ela.MaxentModel()
model.fit(x, y)

# apply it to the full extent and save the model for later
ela.apply_model_to_rasters(model, raster_path, output_raster_path, transform="cloglog")
ela.save_object(model, output_model_path)
```

To work with this saved model later, you can run:

```python
model = ela.load_object(model_path)
```
