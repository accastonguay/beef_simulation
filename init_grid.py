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
    # filename="/home/uqachare/model_file/gridinit.log",
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
ppp = pd.read_csv("tables/ppp_conv.csv")

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

bovine_supply = pd.read_csv("tables/bovine_supply.csv")

######################### Set parameters #########################

# N20 emission_factors from N application
emission_factors = {"maize": 0.0091, "soybean": 0.0066, "wheat": 0.008, "grass": 0.007}

# List of land covers for which livestock production is allowed
suitable_landcovers = ["area_tree", "area_sparse", "area_shrub", "area_mosaic", "area_grass", "area_crop",
                       "area_barren"]
list_suitable_cover = [i for i in cmap if 'area_'+cmap[i] in suitable_landcovers]

# k landuses to include in the simulation
landuses = ['grass_low', 'grass_high', 'alfalfa_high', 'maize', 'soybean', 'wheat']

fuel_efficiency = 0.4 # in l/km
pasture_utilisation = 0.3 # Proportion of grazing biomass consumed
truck_emission_factor = 2.6712 # Emissions factor for heavy trucks (CO2/l)

def lc_summary(landcovers, cell_area):
    grassl = 0
    grainl = 0
    suitable_area = 0
    for lc_code in landcovers:
        area = float(landcovers[lc_code])/sum(landcovers.values())* cell_area
        if lc_code in list_suitable_cover:
            suitable_area += area
            grassl += area * transition.at[cmap[lc_code], 'pasture']
            grainl += area * transition.at[cmap[lc_code], 'crop']
    d = (grassl,  grainl,  suitable_area)
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

    dict_data_name = {}

    # Loop over rasters to create a dictionary of raster path and raster name for column names and add values to grid
    for i in rasters:
        # dict_data_name[i] = i.split("\\")[-1].split('.')[0]
        dict_data_name[i] = os.path.basename(i).split('.')[0]
        start = time.time()
        colname = dict_data_name[i]
        feats['centroid_column'] = feats.centroid
        # print(feats.head())
        # print(i, colname)
        if 'efftemp' in colname:
            stats = point_query(feats.set_geometry('centroid_column'), i, interpolate='nearest')
            feats[colname] = pd.Series([d for d in stats], index=feats.index)
        elif colname =='landcover':
            # For land cover raster, count the number of covers in each cell
            stats =  zonal_stats(feats.set_geometry('geometry'), i, categorical=True)
            result = list(map(lc_summary, stats, feats['area'].values))
            for cols, pos in zip(['grass_transition', 'grain_transition', 'suitable_area'], range(3)):
                feats[cols] = [i[pos] for i in result]

        elif colname =='accessibility':
            stats =  zonal_stats(feats.set_geometry('geometry'), i, stats = 'mean', nodata=-9999)
            # feats[colname] = pd.Series([0 if d['mean'] is None | d['mean'] < 0 else d['mean'] for d in stats], index=feats.index)
            feats[colname] = pd.Series([d['mean'] for d in stats], index=feats.index)



        # else:
        #     # For all other rasters do a point query instead of zonal statistics and replace negative values by NaN
        #     stats = point_query(feats.set_geometry('centroid_column'), i, interpolate = 'nearest')
        #     feats[colname] = pd.Series([0 if d is None else 0 if d < 0 else d for d in stats], index=feats.index)
        print('      Done with {} in {} seconds.'.format(colname, time.time()-start))
        logger.info("   Done with "+colname)

    # feats["opp_cost"] = feats["opp_cost"] * feats['suitable_area']
    feats["agri_opp_cost"] = feats["opp_cost"]
    feats["opp_cost"] = (feats['suitable_area'] * feats["agri_opp_cost"]) + (feats['suitable_area'] * feats['ls_opp_cost'])

    # print type(feats), type(feats["accessibility"].values), type(feats[['ADM0_A3']].merge(fuel_cost, how='left')['Diesel'].values), type(fuel_efficiency)

    # print feats["accessibility"].values, feats[['ADM0_A3']].merge(fuel_cost, how='left')['Diesel'].values, fuel_efficiency
    logger.info("Done with opp_cost")
    feats = feats.merge(fuel_cost[['ADM0_A3', 'Diesel']], how = 'left', left_on = 'ADM0_A3', right_on ='ADM0_A3')
    feats["transport_cost"] = feats["accessibility"] * feats['Diesel'] * fuel_efficiency
    logger.info("Done with transport_cost")

    feats["transport_emissions"] = feats["accessibility"] * fuel_efficiency * truck_emission_factor
    logger.info("Done with transport_emissions")

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
    # Only keep three columns
    extent = extent[['geometry', 'SOVEREIGNT', 'ADM0_A3']]

    # Filter countries on location argument
    if "Global" in location:
        extent = extent[extent.SOVEREIGNT.notnull() & (extent.SOVEREIGNT != "Antarctica")]
    elif location in beef_table.index:
        extent = extent[extent.ADM0_A3 == location]
    else:
        print("Location not in choices")

    # Create grid based on extent bounds
    start_init = time.time()
    xmin, ymin, xmax, ymax = extent.total_bounds

    ################ Numpy technique ################
    # res = float(resolution)
    # x = np.arange(xmin, xmax, res)
    # y = np.arange(ymin, ymax, res)
    # ssize = res
    # xy_ll = np.vstack(np.dstack(np.meshgrid(x, y)))
    # xy_ur = xy_ll + ssize
    # xy_lr = np.column_stack([xy_ll[:, 0] + ssize, xy_ll[:, 1]])
    # xy_ul = np.column_stack([xy_ll[:, 0], xy_ll[:, 1] + ssize])
    # cs = np.column_stack([xy_ur, xy_ul, xy_ll, xy_lr, xy_ur])
    # cs_shape = cs.shape
    # cs = cs.reshape(cs_shape[0], int(cs_shape[1] / 2), 2)
    # polygons = gpd.GeoSeries(pd.Series(cs.tolist()).apply(lambda x: Polygon(x)))
    # grid = gpd.GeoDataFrame({'geometry': polygons})

    ################ Looping technique ################

    width = float(resolution)
    height = float(resolution)
    rows = abs(int(np.ceil((ymax - ymin) / height)))
    cols = abs(int(np.ceil((xmax - xmin) / width)))
    Xleftorigin = xmin
    Xrightorigin = xmin + width
    Ytoprigin = ymax
    Ybottomrigin = ymax - height
    polygons = []
    for i in range(cols):
        Ytop = Ytoprigin
        Ybottom = Ybottomrigin

        for j in range(rows):
            polygons.append(
                Polygon([(Xleftorigin, Ytop), (Xrightorigin, Ytop), (Xrightorigin, Ybottom), (Xleftorigin, Ybottom)]))
            Ytop = Ytop - height
            Ybottom = Ybottom - height
        Xleftorigin = Xleftorigin + width

        Xrightorigin = Xrightorigin + width

        grid = gpd.GeoDataFrame({'geometry': polygons})

    print('Created whole grid in {} seconds'.format(time.time() - start_init))

    # grid.to_file(export_folder+"/grid1_"+location+".gpkg", driver="GPKG")

    grid.crs = {'init': 'epsg:4326'}


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
    # grid.drop('centroid_column', axis = 1).set_geometry('geometry').to_file(export_folder+"/grid1_"+location+".geojson", driver="GeoJSON")

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
    start_module = time.time()

    grid = create_grid(location, resolution)
    # Measure area of cells in hectare
    start_area = time.time()
    grid = grid.to_crs({'init': 'epsg:3857'})
    grid["area"] = grid['geometry'].area / 10. ** 4
    grid = grid.to_crs({'init': 'epsg:4326'})
    print('### Done calculating area in {} seconds'.format(time.time() - start_area))
    # print('### {} cells with area greater than 100k ha'.format(grid.loc[grid["area"] > 100000].head()))
    # grid = grid.loc[grid["area"] < 100000]
    logger.info("Done calculating area")

    # Parallelise the input data collection
    start = time.time()
	
    #grid = parallelize(grid, zstats_partial, ncores)
    grid = zstats_partial(grid)
    print('### Done Collecting inputs in {} seconds'.format(time.time() - start))
    logger.info("Done Collecting inputs")

    ######### Export #########

    start = time.time()
    grid.drop('centroid_column', axis = 1).set_geometry('geometry').to_file(export_folder+"/grid.gpkg", driver="GPKG")
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