import numpy as np
import rasterio

path = "data/raw/FloodCastBench/High-fidelity flood forecasting/30m/Australia/0.tif"

# Open the GeoTIFF. rasterio reads the georeferencing along with the pixels.
with rasterio.open(path) as src:
    print("shape (rows, cols):", src.height, src.width)
    print("bands:", src.count, " dtype:", src.dtypes[0])
    print("pixel size (x, y):", src.res)
    print("CRS:", src.crs)
    print("nodata value:", src.nodata)
    depth = src.read(1)

# Basic statistics of the water depth array.
print("\nmin:", float(np.nanmin(depth)), " max:", float(np.nanmax(depth)))
print("mean:", float(np.nanmean(depth)))
print("fraction of cells above 0.01 m:", float((depth > 0.01).mean()))
