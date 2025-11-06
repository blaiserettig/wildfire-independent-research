import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

occ = gpd.read_file("data/mtbs_fod_pts_data/mtbs_FODpoints_DD.shp")
bnd = gpd.read_file("data/mtbs_perimeter_data/mtbs_perims_DD.shp")

# These files do not have a state or geographic field explicitly, but the beginning two characters from EVENT_ID are the state they belong to ex: "WA123..."
occ["STATE"] = occ["Event_ID"].str[:2]
bnd["STATE"] = bnd["Event_ID"].str[:2]

pnw = ["WA", "OR", "ID"]
occ_pnw = occ[occ["STATE"].isin(pnw)]
bnd_pnw = bnd[bnd["STATE"].isin(pnw)]

occ_pnw.to_file("data/mtbs_fod_pts_data/mtbs_FODpoints_DD_pnw.shp")
bnd_pnw.to_file("data/mtbs_perimeter_data/mtbs_perims_DD_pnw.shp")

print(occ_pnw["STATE"].value_counts())
print(bnd_pnw["STATE"].value_counts())

print(occ_pnw.columns)
print(occ_pnw.head())

print(bnd_pnw.columns)
print(bnd_pnw.head())

bnd_pnw['Ig_Date'] = pd.to_datetime(bnd_pnw['Ig_Date'], errors='coerce')
bnd_pnw['YEAR'] = bnd_pnw['Ig_Date'].fillna(
    bnd_pnw['Event_ID'].str.extract(r'(\d{4})$')[0].astype(float)
)

bnd_pnw = bnd_pnw.to_crs(epsg=5070)
bnd_pnw['area_ha'] = bnd_pnw.geometry.area / 10000.0

annual = bnd_pnw.groupby('YEAR').agg(
    fire_count = ('Event_ID','count'),
    total_area_ha = ('area_ha','sum'),
    median_area_ha = ('area_ha','median'),
    mean_area_ha = ('area_ha','mean')
).reset_index()

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.lineplot(data=annual, x="YEAR", y="fire_count", ax=ax[0])
sns.lineplot(data=annual, x="YEAR", y="total_area_ha", ax=ax[1])
ax[0].set_title("Fire Count per Year")
ax[1].set_title("Total Burned Area (ha)")
plt.tight_layout()
plt.show()

