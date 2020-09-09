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
logging.basicConfig(
    filename="/home/uqachare/model_file/gridinit.log",
    level=logging.INFO,
    format=LOG_FORMAT,
    filemode='w')
logger = logging.getLogger()

######################### Load tables #########################

# Fuel cost by country
fuel_cost = pd.read_csv("tables/fuel_costs.csv")

# Load ratio actual-potential yield
ratios = pd.read_csv("tables/ratios.csv")

# Load coefficients to get meat production and GHG emissions as a function of biomass consumed
coefficients = pd.read_csv("tables/coefficients.csv")

# Load feed costs per country
feed_costs = pd.read_csv("tables/feed_costs.csv")

# Load transition costs for grass/grain
transition = pd.read_csv("tables/transitioning_costs.csv", index_col="current")

# Load PPP conversion factors per country
# ppp = pd.read_csv("tables/ppp_conv.csv", index_col="SOVEREIGNT")
# ppp = pd.read_csv("tables/ppp_conv.csv")

# Load GLPS regions
regions = pd.read_csv("tables/glps_regions.csv")

# Groups
grouped_ME = pd.read_csv("tables/nnls_group_ME.csv")

grass_energy = pd.read_csv("tables/grass_energy.csv")

# Load landuse coding to get land use names
landuse_code = pd.read_csv("tables/landuse_coding.csv", index_col="code")
cmap = landuse_code.to_dict()['landuse']

# Beef production for different locations
beef_table = pd.read_csv("tables/beef_production.csv", index_col="Code")

fertiliser_prices = pd.read_csv("tables/fertiliser_prices.csv")

nutrient_req_grass = pd.read_csv("tables/nutrient_req_grass.csv")

nutrient_req_alfa = pd.read_csv("tables/nutrient_req_alfa.csv")

# bovine_supply = pd.read_csv("tables/bovine_supply.csv")

######################### Set parameters #########################

# N20 emission_factors from N application
emission_factors = {"maize": 0.0091, "soybean": 0.0066, "wheat": 0.008, "grass": 0.007}

# List of land covers for which livestock production is allowed
suitable_landcovers = ["area_tree", "area_sparse", "area_shrub", "area_mosaic", "area_grass", "area_crop",
                       "area_protected", "area_intact", "area_barren"]
list_suitable_cover = [i for i in cmap if 'area_'+cmap[i] in suitable_landcovers]

# k landuses to include in the simulation
landuses = ['grass_low', 'grass_high', 'alfalfa_high', 'maize', 'soybean', 'wheat']

fuel_efficiency = 0.4 # in l/km
pasture_utilisation = 0.3 # Proportion of grazing biomass consumed
truck_emission_factor = 2.6712 # Emissions factor for heavy trucks (CO2/l)

no_transition_lu = [i for i in cmap if cmap[i] in ['crop', 'grass']]
grass = [i for i in cmap if cmap[i] == 'grass']
crop = [i for i in cmap if cmap[i] == 'crop']
forest = [i for i in cmap if cmap[i] =='tree']

# def lc_summary(landcovers, cell_area):
#     est_cost = 0
#     suitable_area = 0
#     for lc_code in landcovers:
#         area = float(landcovers[lc_code])/sum(landcovers.values())* cell_area
#         if lc_code in list_suitable_cover:
#             suitable_area += area
#             if lc_code not in no_transition_lu:
#                 est_cost += area
#
#     d = (est_cost,  suitable_area)
#     return d

def lc_summary(landcovers, cell_area):
    est_cost,area_pasture,area_crop,area_forest = 0,0,0,0
    suitable_area = 0
    for lc_code in landcovers:
        area = float(landcovers[lc_code])/sum(landcovers.values())* cell_area
        if lc_code in list_suitable_cover:
            suitable_area += area
            if lc_code not in no_transition_lu:
                est_cost += area
            if lc_code in grass:
                area_pasture += area
            if lc_code in crop:
                area_crop += area
            if lc_code in forest:
                area_forest += area
    d = (est_cost,  suitable_area, area_pasture, area_crop, area_forest)
    return d

def zstats_partial(feats):
    """
    Imports raster values into a dataframe partition and returns the partition

    Arguments:
    feats (array)-> partion of dataframe

    Output: returns a gridded dataframe
    """

    # Get all tif rasters in folder 'rasters'
    folder = 'rasters/'
    rasters = glob(folder + '*.tif')

    # Loop over rasters to create a dictionary of raster path and raster name for column names and add values to grid
    for i in rasters:

        with rasterio.open(i) as dataset:
            col_dtype = dataset.meta['dtype']

        colname= os.path.basename(i).split('.')[0]
        logger.info("   Col name {} dtype {}".format(colname, col_dtype))

        start = time.time()
        feats['centroid_column'] = feats.centroid

        if 'efftemp' in colname:
            stats = point_query(feats.set_geometry('centroid_column'), i, interpolate='nearest')
            feats[colname] = pd.Series([d for d in stats], index=feats.index, dtype = col_dtype)

        elif colname =='landcover':
            # For land cover raster, count the number of covers in each cell
            stats =  zonal_stats(feats.set_geometry('geometry'), i, categorical=True)
            result = list(map(lc_summary, stats, feats['area'].values))
            for cols, pos in zip(['est_area', 'suitable_area', "pasture_area", "crop_area", 'tree_area'], range(5)):
                feats[cols] = [i[pos] for i in result]

        elif colname =='accessibility':
            stats =  zonal_stats(feats.set_geometry('geometry'), i, stats = 'mean', nodata=-9999)
            # feats[colname] = pd.Series([0 if d['mean'] is None | d['mean'] < 0 else d['mean'] for d in stats], index=feats.index)
            feats[colname] = pd.Series([d['mean'] for d in stats], index=feats.index)

        else:
            # For all other rasters do a point query instead of zonal statistics and replace negative values by NaN
            stats = point_query(feats.set_geometry('centroid_column'), i, interpolate = 'nearest')
            feats[colname] = pd.Series([0 if d is None else 0 if d < 0 else d for d in stats], index=feats.index, dtype = col_dtype)
        print('      Done with {} in {} seconds.'.format(colname, time.time()-start))
        logger.info("   Done with "+colname)
    # print(feats.columns)

    # Establishment cost of 8 '000$ per ha where land cover requires a transition (not grass or crop) from
    # (Dietrich et al 2019 Geosci. Model Dev.
    feats['est_cost'] = feats['est_area'] * 8
    print(feats[['area', 'suitable_area', 'est_area', 'est_cost']].head())

    # feats["opp_cost"] = (feats['suitable_area'] * feats["agri_opp_cost"]) + (feats['suitable_area'] * feats['ls_opp_cost']*0.01)
    feats["opp_cost"] = feats["agri_opp_cost"] + feats['ls_opp_cost']*0.01

    logger.info("Done with opp_cost")
    # feats = feats.merge(fuel_cost[['ADM0_A3', 'Diesel']], how = 'left', left_on = 'ADM0_A3', right_on ='ADM0_A3')
    # feats["transport_cost"] = feats["accessibility"] * feats['Diesel'] * fuel_efficiency
    # logger.info("Done with transport_cost")
    #
    # feats["transport_emissions"] = feats["accessibility"] * fuel_efficiency * truck_emission_factor
    # logger.info("Done with transport_emissions")

    feats["nutrient_availability"] = feats['nutrient_availability'].replace(0, 2)

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
    df_split = np.array_split(df, num_partitions)
    pool = multiprocessing.Pool(num_cores)
    df = pd.concat(pool.imap(func, df_split))
    pool.close()
    pool.join()
    return df

def create_grid(location, resolution):
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
    # extent.crs = "EPSG:4326"

    # Only keep three columns
    extent = extent[['geometry', 'ADM0_A3']]

    # Filter countries on location argument
    if "Global" in location:
        extent = extent[extent.ADM0_A3.notnull() & (extent.ADM0_A3 != "ATA")]
    elif location in beef_table.index:
        extent = extent[extent.ADM0_A3 == location]
    else:
        print("Location not in choices")

    # Create grid based on extent bounds
    start_init = time.time()
    xmin, ymin, xmax, ymax = extent.total_bounds
    # if "Global" in location:
    #     xmin, xmax = -180, 180
    ################ Vectorized technique ################
    # resolution = float(resolution)

    resolution = 360 / 4321.
    rows = abs(int(np.ceil((ymax - ymin) / resolution)))
    cols = abs(int(np.ceil((xmax - xmin) / resolution)))
    x1 = np.cumsum(np.full((rows, cols), resolution), axis=1) + xmin - resolution
    x2 = np.cumsum(np.full((rows, cols), resolution), axis=1) + xmin
    y1 = np.cumsum(np.full((rows, cols), resolution), axis=0) + ymin - resolution
    y2 = np.cumsum(np.full((rows, cols), resolution), axis=0) + ymin
    polys = [Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]) for x1, x2, y1, y2 in
             zip(x1.flatten(), x2.flatten(), y1.flatten(), y2.flatten())]
    grid = gpd.GeoDataFrame({'geometry': polys})
    grid.crs = {'init': 'epsg:4326'}
    # extent.crs = "EPSG:4326"

    # Find which grid fall within a country or on land
    grid['centroid_column'] = grid.centroid

    grid = grid.set_geometry('centroid_column')

    start = time.time()

    grid = gpd.sjoin(grid, extent, how='left', op='within')
    grid.drop(['index_right'], axis=1, inplace=True)

    print('Joined grid to country in {} seconds'.format(time.time()-start))

    # Filter cells to keep those on land
    grid = grid.merge(regions, how='left')

    if "Global" in location:
        grid = grid[(grid.ADM0_A3.notnull())]
    elif location in beef_table.index:
        grid = grid[(grid.ADM0_A3 == location) & (grid.ADM0_A3.notnull())]
    else:
        print("no beef demand for location")

    grid = grid.set_geometry('geometry')

    print('### Done Creating grid in {} seconds. '.format(time.time()-start))
    # # grid.drop('centroid_column', axis = 1).set_geometry('geometry').to_file(export_folder+"/grid1_"+location+".geojson", driver="GeoJSON")

    return grid

def main(location = 'TLS', resolution = 0.1, ncores =1, export_folder ='.'):
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

    grid = create_grid(location, resolution)
    logger.info('Shape of grid: {}'.format(grid.shape[0]))

    # Measure area of cells in hectare
    start_area = time.time()
    grid = grid.to_crs({'init': 'epsg:3857'})
    # grid = grid.to_crs("EPSG:3857")

    grid["area"] = grid['geometry'].area / 10. ** 4
    # grid = grid.to_crs("EPSG:4326")
    grid = grid.to_crs({'init': 'epsg:4326'})

    grid = grid.loc[grid.area < 1000000]
    logger.info(grid.shape[0])
    logger.info(grid.loc[grid.area < 1000000].shape[0])

    print('### Done calculating area in {} seconds'.format(time.time() - start_area))
    logger.info("Done calculating area")

    # Parallelise the input data collection
    start = time.time()
    grid = parallelize(grid, zstats_partial, ncores)
    grid['climate_bin'] = np.select([grid['soilmoisture'].values < np.ceil(
        grid['soilmoisture'].min() + 1 * ((grid['soilmoisture'].max() - grid['soilmoisture'].min()) / 10.)),
                                     grid['soilmoisture'].values < np.ceil(grid['soilmoisture'].min() + 2 * (
                                                 (grid['soilmoisture'].max() - grid['soilmoisture'].min()) / 10.)),
                                     grid['soilmoisture'].values < np.ceil(grid['soilmoisture'].min() + 3 * (
                                                 (grid['soilmoisture'].max() - grid['soilmoisture'].min()) / 10.)),
                                     grid['soilmoisture'].values < np.ceil(grid['soilmoisture'].min() + 4 * (
                                                 (grid['soilmoisture'].max() - grid['soilmoisture'].min()) / 10.)),
                                     grid['soilmoisture'].values < np.ceil(grid['soilmoisture'].min() + 5 * (
                                                 (grid['soilmoisture'].max() - grid['soilmoisture'].min()) / 10.)),
                                     grid['soilmoisture'].values < np.ceil(grid['soilmoisture'].min() + 6 * (
                                                 (grid['soilmoisture'].max() - grid['soilmoisture'].min()) / 10.)),
                                     grid['soilmoisture'].values < np.ceil(grid['soilmoisture'].min() + 7 * (
                                                 (grid['soilmoisture'].max() - grid['soilmoisture'].min()) / 10.)),
                                     grid['soilmoisture'].values < np.ceil(grid['soilmoisture'].min() + 8 * (
                                                 (grid['soilmoisture'].max() - grid['soilmoisture'].min()) / 10.)),
                                     grid['soilmoisture'].values < np.ceil(grid['soilmoisture'].min() + 9 * (
                                                 (grid['soilmoisture'].max() - grid['soilmoisture'].min()) / 10.)),
                                     grid['soilmoisture'].values < np.ceil(grid['soilmoisture'].min() + 10 * (
                                                 (grid['soilmoisture'].max() - grid['soilmoisture'].min()) / 10.))],
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], default=np.nan) + np.select(
        [grid['gdd'].values < np.ceil(grid['gdd'].min() + 1 * ((grid['gdd'].max() - grid['gdd'].min()) / 10.)),
         grid['gdd'].values < np.ceil(grid['gdd'].min() + 2 * ((grid['gdd'].max() - grid['gdd'].min()) / 10.)),
         grid['gdd'].values < np.ceil(grid['gdd'].min() + 3 * ((grid['gdd'].max() - grid['gdd'].min()) / 10.)),
         grid['gdd'].values < np.ceil(grid['gdd'].min() + 4 * ((grid['gdd'].max() - grid['gdd'].min()) / 10.)),
         grid['gdd'].values < np.ceil(grid['gdd'].min() + 5 * ((grid['gdd'].max() - grid['gdd'].min()) / 10.)),
         grid['gdd'].values < np.ceil(grid['gdd'].min() + 6 * ((grid['gdd'].max() - grid['gdd'].min()) / 10.)),
         grid['gdd'].values < np.ceil(grid['gdd'].min() + 7 * ((grid['gdd'].max() - grid['gdd'].min()) / 10.)),
         grid['gdd'].values < np.ceil(grid['gdd'].min() + 8 * ((grid['gdd'].max() - grid['gdd'].min()) / 10.)),
         grid['gdd'].values < np.ceil(grid['gdd'].min() + 9 * ((grid['gdd'].max() - grid['gdd'].min()) / 10.)),
         grid['gdd'].values < np.ceil(grid['gdd'].min() + 10 * ((grid['gdd'].max() - grid['gdd'].min()) / 10.))],
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], default=np.nan)

    # Add soil organic carbon change according to Guo and Gifford 2002 Glob Chang Biol, f to p + c to p & p to c + p to f
    # grid['crop_soc'] = -0.59 * grid['pasture_area'] * grid['soil_carbon10km'] + -0.42 * grid['tree_area'] * grid[
    #     'soil_carbon10km'] + grid['suitable_area'] * grid['soil_carbon10km']
    # grid['pasture_soc'] = 0.19 * grid['crop_area'] * grid['soil_carbon10km'] + 0.08 * grid['tree_area'] * grid[
    #     'soil_carbon10km'] + grid['suitable_area'] * grid['soil_carbon10km']
    # grid['soil_carbon'] = grid['suitable_area'] * grid['soil_carbon10km']

    print('### Done Collecting inputs in {} seconds'.format(time.time() - start))
    logger.info("Done Collecting inputs")

    ######### Export #########

    start = time.time()
    grid.drop(['centroid_column', 'soilmoisture', "gdd", 'ls_opp_cost', 'agri_opp_cost','est_area'],
              axis = 1).set_geometry('geometry').to_file(export_folder+"/grid.gpkg", driver="GPKG")
    # grid.set_geometry('geometry').to_file(export_folder+"/grid.gpkg", driver="GPKG")
    print('### Exporting results finished in {} seconds. ###'.format(time.time()-start))

    logger.info("Exporting results finished")

if __name__ == '__main__':

    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('location', help='Spatial extent of simulation')
    argparser.add_argument('resolution', help='Resolution of pixels (in degrees)')
    argparser.add_argument('ncores', help='Number of cores for multiprocessing')
    argparser.add_argument('export_folder', help='Name of exported file')

    args = argparser.parse_args()
    location = args.location
    resolution = args.resolution
    ncores = args.ncores
    export_folder = args.export_folder

    main(location, resolution, ncores, export_folder)