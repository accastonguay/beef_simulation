from rasterstats import zonal_stats
from rasterstats import point_query
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import time
from glob import glob
from math import isnan
import multiprocessing
import logging
from affine import Affine
from rasterio import features
import rasterio
from functools import wraps
# from memory_profiler import profile
import rasterstats
import os.path

# print("Rasterstats version: {}".format(rasterstats.__version__))
LOG_FORMAT = "%(asctime)s - %(message)s"
try:
    logging.basicConfig(
    filename="/home/uqachare/model_file/current_state_pq.log",
    level=logging.INFO,
    format=LOG_FORMAT,
    filemode='w')
except:
    logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    filemode='w')
logger = logging.getLogger()
truck_emission_factor = 2.6712 # Emissions factor for heavy trucks (CO2/l)
fuel_efficiency = 0.4 # in l/km
regions = pd.read_csv("tables/glps_regions.csv")
fuel_cost = pd.read_csv("tables/fuel_costs.csv")

def zstats_partial(feats):
    """
    Imports raster values into a dataframe partition and returns the partition

    Arguments:
    feats (array)-> partion of dataframe

    Output: returns a gridded dataframe
    """
    logger.info("Start parralel")

    # Get all tif rasters in folder 'rasters'
    folder = 'rasters/'
    rasters = glob(folder + '*.tif')

    dict_data_name = {}
    feats['centroid_column'] = feats.centroid

    # Loop over rasters to create a dictionary of raster path and raster name for column names and add values to grid
    for i in rasters:
        dict_data_name[i] = os.path.basename(i).split('.')[0]
        start = time.time()
        colname = dict_data_name[i]
        logger.info(colname)

        if colname == 'glps':
            stats = zonal_stats(feats.set_geometry('geometry'), i, stats='majority', nodata = 255)
            feats[colname] = pd.Series(np.asarray([0 if d['majority'] is None else d['majority'] for d in stats]), index=feats.index, dtype='int8')

        elif colname in ['harvested_biomass', 'grazed_biomass']:
            stats = point_query(feats.set_geometry('centroid_column'), i, interpolate = 'nearest')
            feats[colname] = pd.Series([0 if d is None else 0 if d < 0 else d for d in stats], index=feats.index, dtype='float32')

        elif colname == 'accessibility':
            stats = zonal_stats(feats.set_geometry('geometry'), i, stats='mean', nodata=-9999)
            # feats[colname] = pd.Series([0 if d['mean'] is None | d['mean'] < 0 else d['mean'] for d in stats], index=feats.index)
            feats[colname] = pd.Series([d['mean'] for d in stats], index=feats.index, dtype='float32')

        else:
            # stats = zonal_stats(feats.set_geometry('geometry'), i, stats='mean')
            # feats[colname] = pd.Series(np.asarray([0 if d['mean'] is None or d['mean'] < 0 else d['mean'] for d in stats]), index=feats.index)
            stats = point_query(feats.set_geometry('centroid_column'), i, interpolate = 'nearest')
            feats[colname] = pd.Series([0 if d is None else 0 if d < 0 else d for d in stats], index=feats.index, dtype='float32')

        print('      Done with {} in {} seconds.'.format(colname, time.time()-start))
        logger.info("   Done with "+colname)

    feats["opp_cost"] = feats["agri_opp_cost"] + feats['ls_opp_cost']*0.01
    feats = feats.merge(fuel_cost[['ADM0_A3', 'Diesel']], how = 'left', left_on = 'ADM0_A3', right_on ='ADM0_A3')
    feats["transport_cost"] = feats["accessibility"] * feats['Diesel'] * fuel_efficiency
    logger.info("Done with transport_cost")

    feats["transport_emissions"] = feats["accessibility"] * fuel_efficiency * truck_emission_factor
    logger.info("Done with transport_emissions")

    return feats

def parallelize(df, func, ncores):
    """
    Splits the dataframe into a number of partitions corresponding to the number of cores,
    applies a function to each partition and returns the dataframe.

    Arguments:
    df (pandas dataframe)-> Dataframe on which to apply function
    func (function)-> function to apply to partitions
    ncores (int)-> number of cores to use

    Output: returns a gridded dataframe
    """
    num_cores = int(ncores)
    # number of partitions to split dataframe based on the number of cores
    num_partitions = num_cores
    # logger.info("Type num partitions: {}".format(type(num_partitions)))
    df_split = np.array_split(df, num_partitions)
    pool = multiprocessing.Pool(num_cores)
    df = pd.concat(pool.imap(func, df_split))
    pool.close()
    pool.join()
    return df

def create_grid():
    """
    Create grid for a defined location and resolution

    Arguments:
    location (str)-> extent of simulation at country level using country code or 'Global'
    resolution (float)-> resolution of cells in degrees

    Output: returns an empty grid
    """

    # Load countries file
    extent = gpd.read_file('map/world.gpkg')
    extent.crs = {'init': 'epsg:4326'}
    # Only keep three columns
    extent = extent[['geometry', 'ADM0_A3']]
    extent = extent[extent.ADM0_A3.notnull() & (extent.ADM0_A3 != "ATA")]

    # Filter countries on location argument
    # if "Global" in location:
    #     extent = extent[extent.ADM0_A3.notnull() & (extent.ADM0_A3 != "ATA")]
    # elif location in beef_table.index:
    #     extent = extent[extent.ADM0_A3 == location]
    # else:
    #     print("Location not in choices")

    # Create grid based on extent bounds
    start_init = time.time()
    xmin, ymin, xmax, ymax = -179.1412506102777797,-55.9794960022222057,180.0254160563888490,83.1038373311111087


    resolution = (xmax - xmin) / 4310.
    rows = abs(int(np.ceil((ymax - ymin) / resolution)))
    cols = abs(int(np.ceil((xmax - xmin) / resolution)))
    logger.info(rows)
    logger.info(cols)
    logger.info(resolution)

    x1 = np.cumsum(np.full((rows, cols), resolution), axis=1) + xmin - resolution
    x2 = np.cumsum(np.full((rows, cols), resolution), axis=1) + xmin
    y1 = np.cumsum(np.full((rows, cols), resolution), axis=0) + ymin - resolution
    y2 = np.cumsum(np.full((rows, cols), resolution), axis=0) + ymin
    polys = [Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]) for x1, x2, y1, y2 in
             zip(x1.flatten(), x2.flatten(), y1.flatten(), y2.flatten())]
    grid = gpd.GeoDataFrame({'geometry': polys})

    logger.info(grid.shape)

    grid.crs = {'init': 'epsg:4326'}

    # Find which grid fall within a country or on land
    grid['centroid_column'] = grid.centroid

    grid = grid.set_geometry('centroid_column')

    start = time.time()

    grid = gpd.sjoin(grid, extent, how='left', op='within')
    grid.drop(['index_right'], axis=1, inplace=True)
    grid = grid[(grid.ADM0_A3.notnull())]
    print('Joined grid to country in {} seconds'.format(time.time() - start))

    # Filter cells to keep those on land
    grid = grid.merge(regions, how='left')
    grid = grid.set_geometry('geometry')

    print('### Done Creating grid in {} seconds. '.format(time.time() - start))
    return grid

def main(export_folder ='.', ncores= 16):
    """
    Main function that optimises beef production for a given location and resolution, using a given number of cores.
    
    Arguments:
    location (str)-> extent of simulation at country level using country code or 'Global'
    resolution (float)-> resolution of cells in degrees
    ncores (int)-> number of cores to use
    export_folder (str)-> folder where the output file is exported
    constraint (str)-> Whether the production should be constrained to equal actual country-specific production, or unconstrained 
    
    Output: Writes the grid as GPKG file
    """

    # grid = gpd.read_file("grid.gpkg")
    # grid = grid[['geometry', 'region', 'group', 'ADM0_A3', 'area',
    #              # 'accessibility',
    #              'opp_cost', 'distance_port', 'Diesel',
    #              'transport_cost', 'transport_emissions']]

    grid = create_grid()
    # Measure area of cells in hectare

    logger.info("Number of cores {}, type: {}".format(ncores, type(ncores)))
    logger.info("Type of grid: {}".format(type(grid)))

    grid = grid.to_crs({'init': 'epsg:3857'})
    grid["area"] = grid['geometry'].area / 10. ** 4
    grid = grid.to_crs({'init': 'epsg:4326'})


    # Parallelise the input data collection
    start = time.time()
    grid = parallelize(grid, zstats_partial, ncores)

    print('### Done Collecting inputs in {} seconds'.format(time.time() - start))
    logger.info("Done Collecting inputs")
    logger.info(grid.dtypes)

    # Fill missing grass
    grass = grid[['ADM0_A3', 'harvested_biomass']].groupby('ADM0_A3').mean()
    grid['mean_grass_yield'] = grid[['ADM0_A3']].merge(grass, how = 'left', left_on = 'ADM0_A3', right_index = True)['harvested_biomass'].values
    grid['mean_grass_yield'] = grid['mean_grass_yield'].astype('float32')
    # grid['harvested_biomass'] = np.where(grid['harvested_biomass'].values == 0, grid['mean_grass_yield'].values, grid['harvested_biomass'].values)

    ######### Export #########
    grid = grid.loc[grid.bvmeat > 0]
    start = time.time()
    grid.drop('centroid_column', axis = 1).set_geometry('geometry').to_file(export_folder+"/grid.gpkg", driver="GPKG")
    print('### Exporting results finished in {} seconds. ###'.format(time.time()-start))

    logger.info("Exporting results finished")

if __name__ == '__main__':

    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('export_folder', help='Name of exported file')
    argparser.add_argument('ncores', type = int,  help='Number of cores for multiprocessing')

    args = argparser.parse_args()
    export_folder = args.export_folder
    ncores = args.ncores

    main(export_folder, ncores)