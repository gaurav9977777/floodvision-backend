#!/bin/bash
apt-get install -y libgdal-dev libgeos-dev
pip install -r requirements.txt
```
Then set **Build Command** in Render to: `bash build.sh`

Alternatively, pin to pre-built wheels by replacing in `requirements.txt`:
```
geopandas==0.14.4
rasterio==1.3.10
```
with:
```
geopandas
rasterio
