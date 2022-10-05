Here's an example end-to-end example for training a Maxent model (using dummy paths to demonstrate functionality).

This is to demonstrate the simplest pattern of model training. Full model training and evaluation should include creating train/test splits, cross-validation, feature selection, etc. These are not covered here.

```python
import geopandas as gpd
import elapid as ela

vector_path = "/home/slug/ariolimax-californicus.gpkg"
raster_path = "/home/slug/california-climate.tif"
output_raster_path = "/home/slug/ariolimax-californicus-habitat.tif"
output_model_path = "/home/slug/ariolimax-claifornicus-model.ela"

# sample the raster values for background point locations
presence = gpd.read_file(vector_path)
pseudoabsence = ela.saple_raster(raster_path, count=5000)

# read the raster covariates at each point location
merged = ela.stack_geometries(presence, pseudoabsence, add_class_label=True)
annotated = ela.anotate(merged, raster_path)

# split the x/y data
x = annotated.drop(columns=['class'])
y = annotated['class']

# train the model
model = ela.MaxentModel(transform="cloglog")
model.fit(x, y)

# apply it to the full extent and save the model for later
ela.apply_model_to_rasters(model, raster_path, output_raster_path)
ela.save_object(model, output_model_path)
```

To work with this saved model later, you can run:

```python
model = ela.load_object(model_path)
```

The `save_object` and `load_object` method just pickles a python object, meaning most variables and classs can be easily stored and accessed for later use.
