import re
import xarray as xr

# drop nas in xarray and export to zarr
def output_clean_chips(dar, tiles, fname):
    xd = dar.sel({'tile_input': tiles}).stack(
        batch=('input_batch', 'tile_input'))
    xd = xd.dropna('batch', how='any')
    xd = xd.chunk({'batch': 1, 'x_input': -1, 'y_input': -1, 'band_input': -1})
    xd = xd.drop('batch')
    xd.to_zarr(fname)
    return

# convert to binary string
def to_bin_string(x):
    x = f'{int(x):08b}'
    return x[::-1]

# check bit at position
def check_bit(x, pos):
    return int(x[pos]) == 1

# open raster tif and add metadata
def open_ras(x):
    x = xr.open_dataset(x, engine='rasterio')
    string = x.encoding["source"]
    match = re.search(r'sen2cor_(.*?)_clouds', string)
    x = x.assign_coords({'tile': match.group(1)})
    x = x.chunk({'band': 1, 'x': 5000, 'y': 5000})
    return x.drop(('x', 'y'))

# open cloud tif and add metadata
def open_cloud(x):
    x = xr.open_dataset(x, engine='rasterio')
    string = x.encoding["source"]
    match = re.search(r'mask_(.*?)_clouds', string)
    x = x.assign_coords({'tile': match.group(1)})
    x = x.chunk({'x': 5000, 'y': 5000})
    return x.drop(('x', 'y', 'band')).squeeze()