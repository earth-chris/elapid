Almost all of the data sampling and indexing uses `geopandas.GeoSeries` objects. These are the format of the `geometry` column for a `GeoDataFrame`.

```python
import geopandas as gpd

gdf = gpd.read_file(vector_path)
print(type(gdf.geometry))

> <class 'geopandas.geoseries.GeoSeries'>
```

## From CSVs

Sometimes you don't have a vector of point-format location data. The `java` implementation of maxent uses csv files, for example. You can work with those using the `xy_to_geoseries` function:

```python
import pandas as pd

csv_path = "/home/cba/ariolimax-californicus.csv"
df = pd.read_csv(csv_path)
presence = elapid.xy_to_geoseries(df.x, df.y, crs="EPSG:32610")
```

Make sure you specify the projection of your x/y data. The default assumption is lat/lon, which in many cases is not correct.

## From arrays or lists

You can also convert arbitrary arrays of x/y data:

```python
lons = [-122.49, 151.0]
lats = [37.79, -33.87]
locations = elapid.xy_to_geoseries(lons, lats)
print(locations)

> 0    POINT (-122.49000 37.79000)
> 1    POINT (151.00000 -33.87000)
> dtype: geometry
```
