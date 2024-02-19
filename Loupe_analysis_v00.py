# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:38:35 2023

@author: noo029
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
# import contextily as cx
import xyzservices.providers as xyz
from datetime import datetime, timedelta
from dateutil import parser
#%% Projection
import os
os.environ['PROJ_LIB'] = 'C:\\Users\\noo029\\Anaconda3\\envs\\py310\\Library\\share\\proj'
os.environ['GDAL_DATA'] = 'C:\\Users\\noo029\\Anaconda3\\envs\\py310\\Library\\share'

#%% Defs

def get_loupe_header(loupe_fn, header_line_num):
    with open(loupe_fn, 'r') as f:
        lines = f.readlines()
        loupe_header = lines[header_line_num].split() 
    return loupe_header

#%%  Get the Loupe data first
loupe_fn = 'Kartoo_2024_02_13.dat'
header_line_num = 6

with open(loupe_fn, 'r') as f:
    lines = f.readlines()
    full_loupe_header = lines[0:header_line_num] 

loupe_full_df = (pd.read_csv(loupe_fn, skiprows=7, delim_whitespace=True, header=None, 
                        names=get_loupe_header(loupe_fn, header_line_num))
            .dropna(how='all')
            .dropna())

loupe_df = (pd.read_csv(loupe_fn, skiprows=7, delim_whitespace=True, header=None, 
                       names=get_loupe_header(loupe_fn, header_line_num))
            .dropna(subset=['TIME', 'EAST', 'NORTH', 'HEIGHT'], how='any')
            [['TIME', 'EAST', 'NORTH', 'HEIGHT', 'C']]
            .assign(TIME=lambda df: pd.to_datetime(df['TIME']),
                    EAST=lambda df: pd.to_numeric(df['EAST'], errors='coerce'),
                    NORTH=lambda df: pd.to_numeric(df['NORTH'], errors='coerce'),
                    HEIGHT=lambda df: pd.to_numeric(df['HEIGHT'], errors='coerce'))
            .dropna(how='all')
            .rename(columns={"TIME":'time',
                             'EAST':'x',
                             'NORTH':'y',
                             'HEIGHT':'z'
                            })
            .dropna()
            )

loupeX_df = loupe_df[loupe_df.C == 'X']

loupe_gdf = gpd.GeoDataFrame(
    loupeX_df, 
    geometry=gpd.points_from_xy(loupeX_df['x'], loupeX_df['y']), 
    crs="EPSG:32754")

loupe_gdf ['time'] = loupe_gdf.time.astype(str)

loupe_gdf.to_file('2023_09_14_avon_NZ.shp')

loupeX_df = loupeX_df.set_index('time')

#%% Get the Catalyst data next
# **Note column names and date format may differ**
catalyst_fn = '2024-02-13-Kartoo.csv'
time_coln, y_coln, x_coln, z_coln = "Gnss DateTime (Local Time)"," Northing(m)"," Easting(m)"," Height MSL(m)"

catalyst_df = pd.read_csv(catalyst_fn)[[time_coln, y_coln, x_coln, z_coln]]       
t = catalyst_df[time_coln].tolist()[0]         
# the colons in the "14/09/2023 12:57:27 PM (UTC+12:00)" needs to be removed to allow datetime.strptime to recognise the UTC timezone: https://protect-au.mimecast.com/s/jggLCL7EAOCwrkXAhmfJvX?domain=docs.python.org
dt_format = '%d/%m/%Y %I:%M:%S %p (%Z%z)'  # 13/02/2024 2:28:03 PM (UTC+10.5:30) -> "14/09/2023 12:57:27 PM (UTC+1200)"   
catalyst_df[time_coln] = [datetime.strptime(t[:-6]+t[-3:], dt_format) for t in catalyst_df[time_coln]]   

catalyst_df = (catalyst_df.assign(time_utc=lambda df: 
                        df[time_coln].dt.tz_convert('UTC'))
                .rename(columns={'time_utc':'time',
                                x_coln:'x', 
                                y_coln:'y', 
                                z_coln:'z'}
                )[['time', 'x', 'y', 'z']]
)

#%% Timestamp analysis
# catalyst_gdf = catalyst_gdf.assign(sec_prev=lambda df: (df['time'].diff().seconds))
nidx = pd.date_range(catalyst_df.time.min(), catalyst_df.time.max(), freq='1s')
catalyst_df = catalyst_df.set_index('time').sort_index()
catalyst_df['raw'] = 'Y' 

catalyst_df_new = catalyst_df.reindex(nidx)
catalyst_df_new[['x', 'y', 'z']] = catalyst_df_new[['x', 'y', 'z']].interpolate()
catalyst_df_new['raw'] = catalyst_df_new['raw'].fillna('N')

catalyst_gdf_new = gpd.GeoDataFrame(
    catalyst_df_new, 
    geometry=gpd.points_from_xy(catalyst_df_new['x'], catalyst_df_new['y']), 
    crs="EPSG:7854")
catalyst_gdf_new = catalyst_gdf_new.to_crs("32754")

catalyst_df_new['xx'] = catalyst_gdf_new.get_coordinates()['x'].values
catalyst_df_new['yy'] = catalyst_gdf_new.get_coordinates()['y'].values


catalyst_gdf_new['datetime'] = catalyst_gdf_new.index.astype(str)
catalyst_gdf_new = catalyst_gdf_new.reset_index(drop=True)
catalyst_gdf_new.to_file('2024-02-13-Catalyst_Kartoo.shp')

#%% Figure
fig, ax = plt.subplots(1,1)
loupe_gdf.plot(ax=ax, marker='*')
catalyst_gdf_new.plot(ax=ax, column='raw', 
                      marker='o', 
                      edgecolor='none',
                      legend=True)
# cx.add_basemap(ax, crs=catalyst_gdf_new.crs.to_string(), 
#                 source=cx.providers.OpenStreetMap.Mapnik)
plt.show()

#%%

fig, ax = plt.subplots(1,1)
ax.plot(loupe_df.time, loupe_df.z,'k*', label="Loupe")
for idx, grp in catalyst_df_new.groupby('raw'):
    ax.plot(grp.time, grp.z,'+', label=f"raw data = {idx}")
ax.set_xlabel('Time')
ax.set_ylabel('Elevation')
ax.legend()
plt.show()

#%% Concat data sources
df = pd.concat([catalyst_df_new.add_prefix('cata_'), 
                loupeX_df.add_prefix('loupe_')], 
               axis=1)
df['offset_xy'] = ((df['cata_xx'] - df['loupe_x']) ** 2 + (df['cata_yy'] - df['loupe_y']) ** 2) ** 0.5
df.to_csv('loupe_catalyst_combined.dat')

catalyst_df_new = (catalyst_df_new.reset_index(drop=False).rename(columns={'index':'time'}))

#%%
def replace_values(loupe_df, catalyst_df):
    # create a copy of the loupe_df to avoid modifying the original DataFrame
    result_df = loupe_df.copy()
    
    result_df['source'] = 'Loupe'
    # create a lookup table for the catalyst_df and rten_df time columns
    catalyst_lookup = {t: (x, y, z) for t, x, y, z in zip(catalyst_df['time'], 
                                                          catalyst_df['xx'], 
                                                          catalyst_df['yy'], 
                                                          catalyst_df['z'])}
    
    # iterate over the rows in the loupe_df
    for index, row in loupe_df.iterrows():
        # look up the x, y, and z values for the current time in the catalyst_df and rten_df
        x, y, z = catalyst_lookup.get(row['time'], None)
        source = 'Catalyst'
                
        # replace the x, y, and z values in the result_df for the current row
        result_df.at[index, 'x'] = x
        result_df.at[index, 'y'] = y
        result_df.at[index, 'z'] = z
        result_df.at[index, 'source'] = source
       
    return result_df
    
loupe_catalyst_df = replace_values(loupe_df, catalyst_df_new)
loupe_catalyst_df.to_csv('loupe_catalyst_replaced.dat')

loupe_full_df [['EAST', 'NORTH', 'HEIGHT']] = loupe_catalyst_df[['x', 'y', 'z']]
loupe_full_df [['EAST', 'NORTH', 'HEIGHT']] = loupe_full_df [['EAST', 'NORTH', 'HEIGHT']].round(1)

with open(loupe_fn[:-4] + "_catalyst_fix.dat", 'w') as f:
    f.write("".join(full_loupe_header))
    f.write(loupe_full_df.to_string(index=False))
