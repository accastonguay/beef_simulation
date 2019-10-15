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
from multiprocessing import Process

grid = gpd.read_file("grid.gpkg")

# import os
# os.environ["PROJ_LIB"] = "C:/Program Files/Anaconda3/envs/myenvironment/Library/share"
# from functools import wraps
# from memory_profiler import profile

LOG_FORMAT = "%(asctime)s - %(message)s"
try:
    logging.basicConfig(
    filename="/home/uqachare/model_file/new_opt.log",
    level=logging.INFO,
    format=LOG_FORMAT,
    filemode='w')
except:
    logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    filemode='w')
logger = logging.getLogger()


######################### Load tables #########################

fuel_cost = pd.read_csv("tables/fuel_costs.csv") # Fuel cost by country
cscc = pd.read_csv("tables/cscc.csv") # Carbon cost
ratios = pd.read_csv("tables/ratios.csv") # Load ratio actual-potential yield

# Load coefficients to get meat production and GHG emissions as a function of biomass consumed
coefficients = pd.read_csv("tables/coefficients.csv")
feed_costs = pd.read_csv("tables/feed_costs.csv") # Load feed costs per country
transition = pd.read_csv("tables/transitioning_costs.csv", index_col="current") # Load transition costs for grass/grain
ppp = pd.read_csv("tables/ppp_conv.csv") # Load PPP conversion factors per country
regions = pd.read_csv("tables/glps_regions.csv") # Load GLPS regions
grouped_ME = pd.read_csv("tables/nnls_group_ME.csv") # Load country groups
grass_energy = pd.read_csv("tables/grass_energy.csv") # Load energy in grasses
landuse_code = pd.read_csv("tables/landuse_coding.csv", index_col="code") # Load landuse coding to get land use names
cmap = landuse_code.to_dict()['landuse'] # Change dataframe to dictionary
beef_production = pd.read_csv("tables/beef_production.csv", index_col="Code") # Load country-level beef supply
fertiliser_prices = pd.read_csv("tables/fertiliser_prices.csv") # Load fertiliser prices
nutrient_req_grass = pd.read_csv("tables/nutrient_req_grass.csv") # Load nutrient requirement for grasses
nutrient_req_alfa = pd.read_csv("tables/nutrient_req_alfa.csv") # Load nutrient requirment for alfalfa
beef_demand = pd.read_csv("tables/beef_demand.csv", index_col="ADM0_A3") # Load country-level beef demand
sea_distances = pd.read_csv("tables/sea_distances.csv") # Load averaged distances between countries
trans_margins = pd.read_csv("tables/trans_margins.csv") # Load country-level transport margins
sea_t_costs = pd.read_csv("tables/sea_t_costs.csv") # Load transport costs

######################### Set parameters #########################

# N20 emission_factors from N application from Gerber et al 2016
emission_factors = {"maize": 0.0091, "soybean": 0.0066, "wheat": 0.008, "grass": 0.007}
suitable_landcovers = ["area_tree", "area_sparse", "area_shrub", "area_mosaic", "area_grass", "area_crop",
                       "area_barren"] # List of land covers for which livestock production is allowed
list_suitable_cover = [i for i in cmap if 'area_'+cmap[i] in suitable_landcovers]
landuses = ['grass_low', 'grass_high', 'alfalfa_high', 'maize', 'soybean', 'wheat'] # k landuses to include in the simulation

# Regression coefficients to get fertiliser needed for a given yield
grain_fertiliser = {'intercept' : {'maize' :3.831, 'soybean':2.287, 'wheat': 6.18018},
                    'coefficent': {'maize' :0.02416, 'soybean':0.01642, 'wheat': 0.03829}}

me_forrage = {'maize': 0.096, 'wheat': 0.096,  'soybean': 0.092, 'alfalfa': 0.094} # Energy in DM from feedipedia.com (MJ/t)
fuel_efficiency = 0.4 # in l/km
pasture_utilisation = 0.3 # Proportion of grazing biomass consumed
truck_emission_factor = 2.6712 # Emissions factor for heavy trucks (CO2/l)
sea_emissions =  0.048  # Emissions factor for heavy trucks (kg CO2/ton-km)

# bestlu_dict = {'None': 0, 'grass_low': 1, 'grass_high': 2, 'alfalfa_high': 3, 'maize': 4, 'soybean': 5, 'wheat': 6}

# Create list on column names for monthly effective temperature
months = ["efftemp0" + str(i) for i in range(1, 10)] + ["efftemp10", "efftemp11", "efftemp12"]

# Function used to determine suitable area based on the land cover raster
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

#@profile
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
        dict_data_name[i] = i.split("/")[-1].split('.')[0]
        start = time.time()
        colname = dict_data_name[i]
        feats['centroid_column'] = feats.centroid

        if colname =='landcover':
            # For land cover raster, count the number of covers in each cell
            stats =  zonal_stats(feats.set_geometry('geometry'), i, categorical=True)
            result = map(lc_summary, stats, feats['area'].values)
            for cols, pos in zip(['grass_transition', 'grain_transition', 'suitable_area'], range(3)):
                feats[cols] = [i[pos] for i in result]

        elif colname =='accessibility':
            stats =  zonal_stats(feats.set_geometry('geometry'), i, stats = 'mean')
            feats[colname] = pd.Series(np.asarray([0 if d['mean'] < 0 else d['mean'] for d in stats]), index=feats.index)

        else:
            # For all other rasters do a point query instead of zonal statistics and replace negative values by NaN
            stats = point_query(feats.set_geometry('centroid_column'), i, interpolate = 'nearest')
            feats[colname] = pd.Series(np.asarray([0 if d < 0 else d for d in stats]), index=feats.index)

        print('      Done with {} in {} seconds.'.format(colname, time.time()-start))
        logger.info("   Done with "+colname)

    # Calculate opportunity cost based on suitable area (ha) and crop value ($/ha)
    feats["opp_cost"] = (feats['suitable_area'] * feats["agri_opp_cost"])
    # Calculate transport emissions based on distance to market, fuel efficiency and road emission factor
    feats["transport_emissions"] = feats["accessibility"].values * fuel_efficiency * truck_emission_factor

    return feats

#@profile
def scoring(feats, scenario, carbon_price, gap_reduce):
    """
    Finds the best landuse for each cell in the partition and returns the partition

    Arguments:
    feats (array)-> partion of dataframe

    Output: returns a gridded dataframe
    """
    for i in ["maize_ratio", "soybean_ratio", "wheat_ratio"]:
        ratios[i] = np.where(ratios[i] + int(gap_reduce) > 100, 100, ratios[i] + int(gap_reduce))

    # Initialise columns
    cols = ['_meat','_meth','_cost','_trans_cost', '_trans_emiss', '_cstock','_n2o','_tot_cost', '_ghg', '_rel_ghg', '_rel_cost']
    for l in landuses:
        for c in cols:
          feats[l+c]=0.
    feats['destination'] = 0

    for l in ['grass_low', 'grass_high', 'alfalfa_high']:

        # GLPS coding: 1 = Temperate, 2 = Arid, 3 = Humid

        # Grass productivity in t/ha is multiplied by the suitable area (ha)
        # Assume pasture utilistion of 30% to get grazing biomass consumed
        biomass = feats[l].values * 3.1 * feats['suitable_area'].values * pasture_utilisation

        # Calculate meat produced and emissions from coefficients
        if l == 'grass_low':
            # Calculate energy consumed (MJ): biomass consumed (t) x energy in biomass (MJ/kg)
            energy = biomass * feats.merge(grass_energy, how='left', left_on=['region', 'glps'], right_on=['region', 'glps'])[
                'ME'].values/100.
            # Calculate meat produced from energy with coefficients
            meat = energy * feats[['group']].merge(grouped_ME, how='left')['ME'].values

            # Estimate meat production as a function of temperature when effective temp < -1:
            feats[l + '_meat'] = np.sum(np.where(feats[months] < -1,
                                                 (meat[:, None] - (
                                                             meat[:, None] * (-0.0182 * feats[months] - 0.0182))) / 12.,
                                                 meat[:, None] / 12.), axis=1)

            # Calculate emissions generated from energy with coefficients
            feats[l+'_meth'] = energy * feats[['group']].merge(grouped_ME, how='left')['emissions'].values

            # Calculate fertiliser application in tons (0 for rangeland, assuming no input)
            n_applied = 0

            # Load fertiliser prices (US$/ton)
            # fert_costs = 0
            # Multiple fertiliser application x fertiliser price for all fertilisers
            feats[l+'_cost'] = 0
            # Calculate N20 emissions based on N application
            feats[l+'_n2o'] = (n_applied * emission_factors["grass"]) * 298

        elif l == 'grass_high':
            # Calculate energy consumed (MJ): biomass consumed (t) x energy in biomass (MJ/kg)
            energy = biomass * feats.merge(grass_energy, how='left', left_on=['region', 'glps'], right_on=['region', 'glps'])[
                'ME'].values/100.
            # Calculate meat produced from energy with coefficients
            meat = energy * feats[['group']].merge(grouped_ME, how='left')['ME'].values

            feats[l + '_meat'] = np.sum(np.where(feats[months] < -1,
                                                 (meat[:, None] - (
                                                         meat[:, None] * (-0.0182 * feats[months] - 0.0182))) / 12.,
                                                 meat[:, None] / 12.), axis=1)
            # Calculate emissions generated from energy with coefficients
            feats[l+'_meth'] = energy * feats[['group']].merge(grouped_ME, how='left')['emissions'].values

            # Calculate fertiliser application in tons
            # N is measure from biomass. Urea contains 46-0-0, so we need to multiply N required by 2.2 to get Urea
            n_applied = biomass * feats[['nutrient_availability']].merge(nutrient_req_grass, how = 'left',
                                                                         left_on = 'nutrient_availability',
                                                                         right_on = 'nutrient_availability')['N'].values*2.2/1000.

            k_applied = feats['suitable_area'] * feats[['nutrient_availability']].merge(nutrient_req_grass, how = 'left',
                                                                         left_on = 'nutrient_availability',
                                                                         right_on = 'nutrient_availability')['K'].values*2.2/1000.

            p_applied = feats['suitable_area'] * feats[['nutrient_availability']].merge(nutrient_req_grass, how = 'left',
                                                                         left_on = 'nutrient_availability',
                                                                         right_on = 'nutrient_availability')['P'].values*1.67/1000.

            # Load fertiliser prices (US$/ton)
            fert_costs = feats[['ADM0_A3']].merge(fertiliser_prices, how = 'left')

            # Multiple fertiliser application x fertiliser price for all fertilisers
            feats[l+'_cost'] = n_applied * fert_costs['n'].values + k_applied * fert_costs['k'].values + p_applied * fert_costs['p'].values

            # Calculate N20 emissions based on N application
            feats[l+'_n2o'] = (n_applied * emission_factors["grass"]) * 298

        else:
            energy = biomass * me_forrage['alfalfa']
            meat = energy * feats[['group']].merge(grouped_ME, how='left')['ME'].values

            feats[l + '_meat'] = np.sum(np.where(feats[months] < -1,
                                                 (meat[:, None] - (
                                                             meat[:, None] * (-0.0182 * feats[months] - 0.0182))) / 12.,
                                                 meat[:, None] / 12.), axis=1)

            feats[l+'_meth'] = energy * feats[['group']].merge(grouped_ME, how='left')['emissions'].values
            n_applied = 0
            k_applied = feats['suitable_area'] * feats[['nutrient_availability']].merge(nutrient_req_alfa, how = 'left',
                                                                         left_on = 'nutrient_availability',
                                                                         right_on = 'nutrient_availability')['K'].values*2.2/1000.
            p_applied = feats['suitable_area'] * feats[['nutrient_availability']].merge(nutrient_req_alfa, how = 'left',
                                                                         left_on = 'nutrient_availability',
                                                                         right_on = 'nutrient_availability')['P'].values*2.2/1000.
            fert_costs = feats[['ADM0_A3']].merge(fertiliser_prices, how = 'left')
            feats[l+'_cost'] = 0 * fert_costs['n'].values + k_applied * fert_costs['k'].values + p_applied * fert_costs['p'].values
            # Estimate n2o from fertiliser application and convert to tons of CO2 eq (* 298):
            feats[l+'_n2o'] = (n_applied * emission_factors["grass"]) * 298

        # Number of trips to market; assuming 15 tons per trip, return
        ntrips = (feats[l+'_meat']/int(15)+1)*2
        # Transport cost to market: number of trips * transport cost ('000 US$)
        feats[l+'_trans_cost'] = ntrips * feats['transport_cost']/1000.
        # Transport emissions: number of trips * emissions per trip (tons CO2 eq)
        feats[l+'_trans_emiss'] = ntrips * feats['transport_emissions']/1000.
        # Estimate carbon content as 47.5% of remaining grass biomass. Then convert to CO2 eq (*3.67)
        feats[l+'_cstock'] = 0.475 * biomass * (1-pasture_utilisation) * 3.67

    for l in ['maize', 'soybean', 'wheat']:

        # Estimate feed yield base on potential yield, suitable area and ratio actual-potential yield
        # Assume all biomass is consumed
        biomass = feats[l+'_high'] * feats['suitable_area'] * feats.merge(ratios, how='left')[l+'_ratio'].values/100.

        # Calculate meat produced and emissions from coefficients
        energy = biomass * me_forrage[l]
        meat = energy * feats[['group']].merge(grouped_ME, how='left')['ME'].values

        feats[l + '_meat'] = np.sum(np.where(feats[months] < -1,
                                             (meat[:, None] - (meat[:, None] * (-0.0182*feats[months] - 0.0182)))/12.,
                                             meat[:, None]/12.), axis = 1)

        feats[l + '_meth'] = energy * feats[['group']].merge(grouped_ME, how='left')['emissions'].values

        # Calculate production cost based on producer price ('000 USD/tons)
        # feats[l+'_cost'] = np.sum(np.where(feats[months] < -1,
        #                                    (0.982 * biomass[:, None] - 0.018182*feats[months] * biomass[:, None])/12.,
        #                                    biomass[:, None]/12.), axis = 1) *feats[['ADM0_A3']].merge(feed_costs, how='left')[l].values
        feats[l+'_cost'] = biomass * feats[['ADM0_A3']].merge(feed_costs, how='left')[l].values
        # Number of trips to market; assuming 15 tons per trip, return
        ntrips = (feats[l+'_meat']/int(15)+1)*2
        # Transport cost to market: number of trips * transport cost ('000 US$)
        feats[l+'_trans_cost'] = ntrips *feats['transport_cost']/1000.
        # Transport emissions: number of trips * emissions per trip (tons CO2 eq)
        feats[l+'_trans_emiss'] = ntrips *feats['transport_emissions']/1000.
        # Estimate 0 carbon stock because of 0 standing biomass
        feats[l+'_cstock'] = 0
        # Estimate nitrogen fertiliser application using regression results:
        n_applied = grain_fertiliser['intercept'][l] + grain_fertiliser['coefficent'][l]*biomass
        # Estimate n2o from fertiliser application, crop-specific emission factors and convert to tons of CO2 eq:
        feats[l+'_n2o'] = n_applied * emission_factors[l] * 298
    feats['opp_cost'] = feats['opp_cost'].astype(float)
    feats = feats.fillna(0)

    # Only keep cells where at least 1 feed option produces meat
    feats = feats.loc[feats[[l + '_meat' for l in landuses]].sum(axis=1) > 0]


    ### SCENARIOS
    if scenario == 'weighted_sum':

        for l in landuses:
            # For all landuse, calculate total costs over 20 years
            feats[l + '_tot_cost'] = feats['grass_transition'] + \
                                     (feats[l + '_cost'] + feats[l + '_trans_cost'] + feats['opp_cost']) * 20
            # For all landuse, calculate emissions over 20 years
            flow = (feats[l + '_n2o'] + feats[l + '_meth'] + feats[l + '_trans_emiss']) * 20
            # For all landuse, calculate change in carbon stock
            stock_change = feats['carbon_stock'] * feats['suitable_area'] * 3.67 - feats[l + '_cstock']
            # Calculate total loss of carbon
            feats[l + '_stock_schange'] = stock_change
            feats[l + '_ghg'] = stock_change + flow

            # Calculate costs and emissions per unit of meat produced for each land use. Convert 0 to NaN to avoid error
            feats[l + '_exp_emiss'] = 0
            feats[l + '_exp_costs'] = 0
            # Calculate relative GHG and costs
            if l + '_meat' in feats:
                feats[l+'_rel_ghg'] = np.where(feats[l+'_meat'] == 0, np.NaN, feats[l+'_ghg']/(feats[l+'_meat']*20))
                feats[l+'_rel_cost'] = np.where(feats[l+'_meat'] == 0, np.NaN, feats[l+'_tot_cost']/(feats[l+'_meat']*20))

        # Set of weights
        costw = np.array([0, 0.25, 0.5, 0.75, 1])

        # for each landuse, append to a list an array with the lowest weighted sum of rel ghg and rel cost per cell
        # list_scores = [np.min(np.dot(feats[i + '_rel_ghg'].values[:, None], (1 - costw)[None, :]) +
        #                       np.dot(feats[i + '_rel_cost'].values[:, None], costw[None, :]), axis=1) for i in landuses]
        list_scores = [np.min((feats[i + '_rel_ghg'].values[:, None] * (1 - costw)[None, :]) +
                              (feats[i + '_rel_cost'].values[:, None] * costw[None, :]), axis=1) for i in landuses]
        # Stack arrays
        allArrays = np.stack(list_scores, axis=-1)

        feats['best_score'] = np.nanmin(allArrays, axis=1)
        feats['bestlu'] = np.nanargmin(allArrays, axis=1)

        # column names for optimal costs/emissions sources
        optimal = ['production', 'opt_emissions', 'opt_exp_emiss', 'opt_exp_costs', 'opt_trans_emiss', 'opt_trans_cost',
                   'opt_tot_cost', 'opt_ghg', 'opt_n2o', 'opt_prod_cost', 'opt_stock_change']

        # Column suffixes for landuse specific costs/emissions sources
        old = ['_meat', '_meth', '_exp_emiss', '_exp_costs', '_trans_emiss', '_trans_cost', '_tot_cost', '_ghg', '_n2o', '_cost','_stock_schange']

        # Get costs/emissions sources for optimal land use
        for new_name, old_name in zip(optimal, old):
            feats[new_name] = np.take_along_axis(feats[[l + old_name for l in landuses]].values,
                                                 feats['bestlu'].values[:, None], axis=1)

        del allArrays
        return feats

    # Minimise total costs given country-level social cost of carbon
    elif scenario == 'cscc':
        for l in landuses:
            feats[l + '_tot_cost'] = np.where(feats[l + '_meat'] > 0, feats['grass_transition'] + \
                                              (feats[l + '_cost'] + feats[l + '_trans_cost'] + feats['opp_cost']) * 20,
                                              0)
            flow = (feats[l + '_n2o'] + feats[l + '_meth'] + feats[l + '_trans_emiss']) * 20
            stock_change = np.where(feats[l + '_meat'] > 0,
                                    feats['carbon_stock'] * feats['suitable_area'] * 3.67 - feats[l + '_cstock'], 0)
            feats[l + '_ghg'] = np.where(feats[l + '_meat'] > 0,
                                         (stock_change + flow) * feats[['ADM0_A3']].merge(cscc, how='left')[
                                             'cscc'].values, 0)
            feats[l + '_all_costs'] = feats[l + '_ghg'] + feats[l + '_tot_cost']
            feats[l + '_relative_costs'] = np.where(feats[l + '_meat'] > 0,
                                                    feats[l + '_all_costs'] / feats[l + '_meat'], 0)

    elif scenario == 'carbon_price':
        for l in landuses:
            # For all landuse, calculate total costs over 20 years
            feats[l + '_tot_cost'] = feats['grass_transition'] + \
                                     (feats[l + '_cost'] + feats[l + '_trans_cost'] + feats['opp_cost']) * 20
            # For all landuse, calculate emissions over 20 years
            flow = (feats[l + '_n2o'] + feats[l + '_meth'] + feats[l + '_trans_emiss']) * 20
            # For all landuse, calculate change in carbon stock

            stock_change = feats['carbon_stock'] * feats['suitable_area'] * 3.67 - feats[l + '_cstock']
            # Calculate total loss of carbon
            feats[l + '_ghg'] = stock_change + flow
            # Calculate costs and emissions per unit of meat produced for each land use. Convert 0 to NaN to avoid error
            feats[l + '_exp_emiss'] = 0
            feats[l + '_exp_costs'] = 0
            # Calculate relative GHG and costs
            if l + '_meat' in feats:
                feats[l+'_rel_ghg'] = np.where(feats[l+'_meat'] == 0, np.NaN, feats[l+'_ghg']/(feats[l+'_meat']*20))
                feats[l+'_rel_cost'] = np.where(feats[l+'_meat'] == 0, np.NaN, feats[l+'_tot_cost']/(feats[l+'_meat']*20))

        list_scores = [feats[i + '_rel_cost'] + feats[i + '_rel_ghg'] * carbon_price for i in landuses]
        # Stack arrays
        allArrays = np.stack(list_scores, axis=-1)

        feats['best_score'] = np.nanmin(allArrays, axis=1)
        feats['bestlu'] = np.nanargmin(allArrays, axis=1)

        # column names for optimal costs/emissions sources
        optimal = ['production', 'opt_emissions', 'opt_exp_emiss', 'opt_exp_costs', 'opt_trans_emiss', 'opt_trans_cost',
                   'opt_tot_cost', 'opt_ghg', 'opt_n2o', 'opt_prod_cost']

        # Column suffixes for landuse specific costs/emissions sources
        old = ['_meat', '_meth', '_exp_emiss', '_exp_costs', '_trans_emiss', '_trans_cost', '_tot_cost', '_ghg', '_n2o', '_cost']

        # Get costs/emissions sources for optimal land use
        for new_name, old_name in zip(optimal, old):
            feats[new_name] = np.take_along_axis(feats[[l + old_name for l in landuses]].values,
                                                 feats['bestlu'].values[:, None], axis=1)

        del allArrays
        return feats

    # Minimise total costs of production
    elif scenario == "costs":

        for l in landuses:
            feats[l + '_tot_cost'] = np.where(feats[l + '_meat'] > 0, feats['grass_transition'] + \
                                              (feats[l + '_cost']+feats[l + '_trans_cost'] + feats['opp_cost']) * 20, 0)
            feats[l + '_relative_costs'] = np.where(feats[l + '_meat'] > 0,
                                                    feats[l + '_tot_cost'] / feats[l + '_meat'], 0)
    # Minimise total GHG emissions
    elif scenario == 'ghg':

        for l in landuses:
            flow = (feats[l + '_n2o'] + feats[l + '_meth'] + feats[l + '_trans_emiss']) * 20
            stock_change = np.where(feats[l + '_meat'] > 0,
                                    feats['carbon_stock'] * feats['suitable_area'] * 3.67 - feats[l + '_cstock'], 0)
            feats[l + '_ghg'] = np.where(feats[l + '_meat'] > 0, (stock_change + flow),0)
            feats[l + '_rel_cost'] = np.where(feats[l + '_meat'] > 0, feats[l + '_meth'] / feats[l + '_meat'], np.nan)
            feats[l + '_exp_emiss'] = 0
            feats[l + '_exp_costs'] = 0
    else:
        print('Scenarios not in choice')
    # Get meat production and GHG emissions of best land use per cell
    if scenario in ['cscc', 'costs', 'ghg']:
        columns = [l + '_rel_cost' for l in landuses]

        feats['best_score'] = np.nanmin(feats[columns].values, axis=1)
        feats['bestlu'] = np.nanargmin(feats[columns].values, axis=1)

        # column names for optimal costs/emissions sources
        optimal = ['production', 'opt_emissions', 'opt_exp_emiss', 'opt_exp_costs', 'opt_trans_emiss', 'opt_trans_cost',
                   'opt_tot_cost', 'opt_ghg', 'opt_n2o']

        # Column suffixes for landuse specific costs/emissions sources
        old = ['_meat', '_meth', '_exp_emiss', '_exp_costs', '_trans_emiss', '_trans_cost', '_tot_cost', '_ghg', '_n2o']

        # Get costs/emissions sources for optimal land use
        for new_name, old_name in zip(optimal, old):
            feats[new_name] = np.take_along_axis(feats[[l + old_name for l in landuses]].values,
                                                 feats['bestlu'].values[:, None], axis=1)

        del allArrays
        return feats

def trade(feats, scenario, carbon_price):

    if scenario == 'weighted_sum':

        for l in landuses:
            # For all landuse, calculate total costs over 20 years

            ntrips = (feats[l + '_meat'] / int(15) + 1) * 2

            # Calculate transport cost to nearest port
            feats[l + '_trans_cost'] = ntrips * feats["distance_port"] * feats['Diesel'] * fuel_efficiency/ 1000.

            # Calculate international transport costs based on average sea distance (km), transport cost to port, used for FOB ('000$) and transport cost percentage ($/(FOB*km))
            # feats[l + '_exp_costs'] =  feats[['ADM0_A3']].merge(sea_distances[['ADM0_A3', 'ave_distance']], how='left')['ave_distance'].values * feats[l + '_trans_cost'] * feats[['ADM0_A3']].merge(sea_t_costs[['ADM0_A3', 'tcost']], how='left')['tcost'].values

            # Calculate transport costs as a function of quantity traded
            feats[l + '_exp_costs'] =  feats[l + '_meat'] * feats[['ADM0_A3']].merge(sea_t_costs[['ADM0_A3', 'tcost']], how='left')['tcost'].values

            # Transport emissions to port
            feats[l + '_trans_emiss'] = ntrips * feats["distance_port"] * fuel_efficiency * truck_emission_factor / 1000.
            # Transport emissions by sea
            feats[l + '_exp_emiss'] =  feats[['ADM0_A3']].merge(sea_distances[['ADM0_A3', 'ave_distance']], how='left')['ave_distance'].values * feats[l + '_meat'] * sea_emissions / 1000.

            feats[l + '_tot_cost'] = feats['grass_transition'] + (feats[l + '_cost'] + feats[l + '_trans_cost'] + feats['opp_cost'] + feats[l + '_exp_costs']) * 20
            # feats[l + '_tot_cost'] = (feats['grass_transition'] + \
            #                          (feats[l + '_cost'] + feats[l + '_trans_cost'] + feats['opp_cost']) * 20) / (1 - feats[['ADM0_A3']].merge(trans_margins, how='left')[
            #         'mean'].values)
            # Calculate emissions over 20 years (t CO2 eq)
            flow = (feats[l + '_n2o'] + feats[l + '_meth'] + feats[l + '_trans_emiss'] + feats[l + '_exp_emiss'] ) * 20
            # Calculate change in carbon stock (t CO2 eq)
            stock_change = feats['carbon_stock'] * feats['suitable_area'] * 3.67 - feats[l + '_cstock']
            feats[l + '_stock_schange'] = stock_change

            # Calculate total loss of carbon (t CO2 eq)
            feats[l + '_ghg'] = stock_change + flow
            # Calculate costs and emissions per unit of meat produced for each land use. Convert 0 to NaN to avoid error

            # Calculate relative GHG and costs
            if l + '_meat' in feats:
                feats[l+'_rel_ghg'] = np.where(feats[l+'_meat'] == 0, np.NaN, feats[l+'_ghg']/(feats[l+'_meat']*20))
                feats[l+'_rel_cost'] = np.where(feats[l+'_meat'] == 0, np.NaN, feats[l+'_tot_cost']/(feats[l+'_meat']*20))

        # List weights
        costw = np.array([0, 0.25, 0.5, 0.75, 1])

        # Create list of arrays with minimum weighted sum for each feed option
        list_scores = [np.min((feats[i + '_rel_ghg'].values[:, None] * (1 - costw)[None, :]) +
                              (feats[i + '_rel_cost'].values[:, None] * costw[None, :]), axis=1) for i in landuses]

        # Stack arrays horizontally
        allArrays = np.stack(list_scores, axis=-1)

        # Take lowest score across feed options
        feats['best_score'] = np.nanmin(allArrays, axis=1)
        # Get best feed option based on position of best score
        feats['bestlu'] = np.nanargmin(allArrays, axis=1)

        # column names for optimal costs/emissions sources
        optimal = ['production', 'opt_emissions', 'opt_exp_emiss', 'opt_exp_costs', 'opt_trans_emiss', 'opt_trans_cost',
                   'opt_tot_cost', 'opt_ghg', 'opt_n2o', 'opt_prod_cost', 'opt_stock_change']

        # Column suffixes for landuse specific costs/emissions sources
        old = ['_meat', '_meth', '_exp_emiss', '_exp_costs', '_trans_emiss', '_trans_cost', '_tot_cost', '_ghg', '_n2o',
               '_cost', '_stock_schange']

        # Get costs/emissions sources for optimal land use
        for new_name, old_name in zip(optimal, old):
            feats[new_name] = np.take_along_axis(feats[[l + old_name for l in landuses]].values,
                                                 feats['bestlu'].values[:, None], axis=1)

        del allArrays
        return feats

    if scenario == 'carbon_price':

        for l in landuses:
            # For all landuse, calculate total costs over 20 years

            ntrips = (feats[l + '_meat'] / int(15) + 1) * 2
            # Transport cost to port
            feats[l + '_trans_cost'] = ntrips * feats["distance_port"] * feats['Diesel'] * fuel_efficiency/ 1000.

            # fob = feats['grass_transition'] + (feats[l + '_cost'] + feats[l + '_trans_cost'] + feats['opp_cost']) * 20
            feats[l + '_exp_costs'] =  feats[['ADM0_A3']].merge(sea_distances[['ADM0_A3', 'ave_distance']], how='left')['ave_distance'].values * feats[l + '_trans_cost'] * feats[['ADM0_A3']].merge(sea_t_costs[['ADM0_A3', 'tcost']], how='left')['tcost'].values

            # Transport emissions to port + emissions by sea
            feats[l + '_trans_emiss'] = ntrips * feats["distance_port"] * fuel_efficiency * truck_emission_factor / 1000.

            feats[l + '_exp_emiss'] =  feats[['ADM0_A3']].merge(sea_distances[['ADM0_A3', 'ave_distance']], how='left')['ave_distance'].values * feats[l + '_meat'] * sea_emissions / 1000.

            feats[l + '_tot_cost'] = feats['grass_transition'] + (feats[l + '_cost'] + feats[l + '_trans_cost'] + feats['opp_cost'] + feats[l + '_exp_costs']) * 20
            # feats[l + '_tot_cost'] = (feats['grass_transition'] + \
            #                          (feats[l + '_cost'] + feats[l + '_trans_cost'] + feats['opp_cost']) * 20) / (1 - feats[['ADM0_A3']].merge(trans_margins, how='left')[
            #         'mean'].values)
            # For all landuse, calculate emissions over 20 years
            flow = (feats[l + '_n2o'] + feats[l + '_meth'] + feats[l + '_trans_emiss'] + feats[l + '_exp_emiss'] ) * 20
            # For all landuse, calculate change in carbon stock
            stock_change = feats['carbon_stock'] * feats['suitable_area'] * 3.67 - feats[l + '_cstock']
            # Calculate total loss of carbon
            feats[l + '_ghg'] = stock_change + flow
            # Calculate costs and emissions per unit of meat produced for each land use. Convert 0 to NaN to avoid error

            # Calculate relative GHG and costs
            if l + '_meat' in feats:
                feats[l+'_rel_ghg'] = np.where(feats[l+'_meat'] == 0, np.NaN, feats[l+'_ghg']/(feats[l+'_meat']*20))
                feats[l+'_rel_cost'] = np.where(feats[l+'_meat'] == 0, np.NaN, feats[l+'_tot_cost']/(feats[l+'_meat']*20))

        list_scores = [feats[i + '_rel_cost'] + feats[i + '_rel_ghg'] * carbon_price for i in landuses]

        allArrays = np.stack(list_scores, axis=-1)

        feats['best_score'] = np.nanmin(allArrays, axis=1)
        feats['bestlu'] = np.nanargmin(allArrays, axis=1)

        # column names for optimal costs/emissions sources
        optimal = ['production', 'opt_emissions', 'opt_exp_emiss', 'opt_exp_costs', 'opt_trans_emiss', 'opt_trans_cost',
                   'opt_tot_cost', 'opt_ghg', 'opt_n2o', 'opt_prod_cost']

        # Column suffixes for landuse specific costs/emissions sources
        old = ['_meat', '_meth', '_exp_emiss', '_exp_costs', '_trans_emiss', '_trans_cost', '_tot_cost', '_ghg', '_n2o', '_cost']

        # Get costs/emissions sources for optimal land use
        for new_name, old_name in zip(optimal, old):
            feats[new_name] = np.take_along_axis(feats[[l + old_name for l in landuses]].values,
                                                     feats['bestlu'].values[:, None], axis=1)

        del allArrays
        return feats

    elif scenario == 'ghg':
        for l in landuses:
            # For all landuse, calculate total costs over 20 years

            ntrips = (feats[l + '_meat'] / int(15) + 1) * 2
            # Transport cost to port
            feats[l + '_trans_cost'] = ntrips * feats["distance_port"] * feats['Diesel'] * fuel_efficiency/ 1000.

            # fob = feats['grass_transition'] + (feats[l + '_cost'] + feats[l + '_trans_cost'] + feats['opp_cost']) * 20
            feats[l + '_exp_costs'] =  feats[['ADM0_A3']].merge(sea_distances[['ADM0_A3', 'ave_distance']], how='left')['ave_distance'].values * feats[l + '_trans_cost'] * feats[['ADM0_A3']].merge(sea_t_costs[['ADM0_A3', 'tcost']], how='left')['tcost'].values

            # Transport emissions to port + emissions by sea
            feats[l + '_trans_emiss'] = ntrips * feats["distance_port"] * fuel_efficiency * truck_emission_factor / 1000.

            feats[l + '_exp_emiss'] =  feats[['ADM0_A3']].merge(sea_distances[['ADM0_A3', 'ave_distance']], how='left')['ave_distance'].values * feats[l + '_meat'] * sea_emissions / 1000.

            feats[l + '_tot_cost'] = feats['grass_transition'] + (feats[l + '_cost'] + feats[l + '_trans_cost'] + feats['opp_cost'] + feats[l + '_exp_costs']) * 20
            # feats[l + '_tot_cost'] = (feats['grass_transition'] + \
            #                          (feats[l + '_cost'] + feats[l + '_trans_cost'] + feats['opp_cost']) * 20) / (1 - feats[['ADM0_A3']].merge(trans_margins, how='left')[
            #         'mean'].values)
            # For all landuse, calculate emissions over 20 years
            flow = (feats[l + '_n2o'] + feats[l + '_meth'] + feats[l + '_trans_emiss'] + feats[l + '_exp_emiss'] ) * 20
            # For all landuse, calculate change in carbon stock
            stock_change = feats['carbon_stock'] * feats['suitable_area'] * 3.67 - feats[l + '_cstock']
            # Calculate total loss of carbon
            feats[l + '_ghg'] = stock_change + flow

            feats[l + '_rel_cost'] = np.where(feats[l + '_meat'] > 0,
                                                    feats[l + '_meth'] / feats[l + '_meat'], np.nan)

        if scenario in ['cscc', 'costs', 'ghg']:
            columns = [l + '_rel_cost' for l in landuses]
            # print feats[columns]

            feats['best_score'] = np.nanmin(feats[columns].values, axis=1)
            feats['bestlu'] = np.nanargmin(feats[columns].values, axis=1)

            # column names for optimal costs/emissions sources
            optimal = ['production', 'opt_emissions', 'opt_exp_emiss', 'opt_exp_costs', 'opt_trans_emiss',
                       'opt_trans_cost', 'opt_tot_cost', 'opt_ghg', 'opt_n2o']

            # Column suffixes for landuse specific costs/emissions sources
            old = ['_meat', '_meth', '_exp_emiss', '_exp_costs', '_trans_emiss', '_trans_cost', '_tot_cost', '_ghg',
                   '_n2o']

            # Get costs/emissions sources for optimal land use
            for new_name, old_name in zip(optimal, old):
                feats[new_name] = np.take_along_axis(feats[[l + old_name for l in landuses]].values,
                                                     feats['bestlu'].values[:, None], axis=1)

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
#@profile

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
    elif location in beef_production.index:
        extent = extent[extent.ADM0_A3 == location]
    else:
        print("Location not in choices")

    # Create grid based on extent bounds
    start_init = time.time()
    xmin, ymin, xmax, ymax = extent.total_bounds

    res = float(resolution)
    x = np.arange(xmin, xmax, res)
    y = np.arange(ymin, ymax, res)

    ssize = res
    xy_ll = np.vstack(np.dstack(np.meshgrid(x, y)))
    xy_ur = xy_ll + ssize
    xy_lr = np.column_stack([xy_ll[:, 0] + ssize, xy_ll[:, 1]])
    xy_ul = np.column_stack([xy_ll[:, 0], xy_ll[:, 1] + ssize])
    cs = np.column_stack([xy_ur, xy_ul, xy_ll, xy_lr, xy_ur])
    cs_shape = cs.shape
    cs = cs.reshape(cs_shape[0], int(cs_shape[1] / 2), 2)

    polygons = gpd.GeoSeries(pd.Series(cs.tolist()).apply(lambda x: Polygon(x)))
    grid = gpd.GeoDataFrame({'geometry': polygons})
    print('Created whole grid in {} seconds'.format(time.time() - start_init))

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
    elif location in beef_production.index:
        grid = grid[(grid.ADM0_A3 == location) & (grid.ADM0_A3.notnull())]
    else:
        print("no beef demand for location")

    grid = grid.set_geometry('geometry')

    print('### Done Creating grid in {} seconds. '.format(time.time()-start))
    # grid.drop('centroid_column', axis = 1).set_geometry('geometry').to_file(export_folder+"/grid1_"+location+".geojson", driver="GeoJSON")

    return grid

def export_raster(grid, b, resolution, export_column, scenario, export_folder, scale):
    resolution = float(resolution)
    width = abs(int((b[2] - b[0]) / resolution))
    heigth = abs(int((b[3] - b[1]) / resolution))
    out_shape = (heigth, width)
    grid['bestlu'] = np.array(grid['bestlu'], dtype='uint8')
    # print(grid['bestlu'].dtype)
    for i in export_column:
    #     if i == 'bestlu':
    #         dt = 'uint8'
    #     else:
    #         dt = grid[i].dtype
        dt = grid[i].dtype

        print("Type of array: {}, type of file: {}".format(grid[i].dtype, dt))
        meta = {'driver': 'GTiff',
            'dtype': dt,
            'nodata': 0,
            'width': width,
            'height': heigth,
            'count': 1,
            'crs': {'init': 'epsg:4326'},
            'transform': Affine(resolution, 0.0, b[0],
                                0.0, -resolution, b[3]),
            'compress': 'lzw',
            }
        # for m in meta: print(m, meta[m])
        out_fn = export_folder + '/' + scenario + "_" + i + '_' + str(scale) + ".tif"

        with rasterio.open(out_fn, 'w', **meta) as out:
            # Create a generator for geom and value pairs
            grid_cell = ((geom, value) for geom, value in zip(grid.geometry, grid[i]))

            burned = features.rasterize(shapes=grid_cell, fill=0, out_shape=out_shape, dtype = dt,
                                        transform=Affine(resolution, 0.0, b[0],
                                                         0.0, -resolution, b[3]))
            print("Burned value dtype: {}".format(burned.dtype))
            out.write_band(1, burned)

def export_grid(resolution):
    """
    Export an empty grid of a defined resolution
    Arguments:
    resolution (float)-> resolution of cells in degrees

    Output: Writes the grid as GPKG file
    """
    
    grid = create_grid(resolution)
    grid.to_file("init_grid"+str(float(resolution)*100)+"km.gpkg", driver = 'GPKG')

def main(location = 'TLS', export_folder ='.', scenario = 'weighted_sum', trade_scenario = 'trade', gap_reduce = 0, cprice = 10, resolution = 0.1 , constraint = 'global',
         exp_global_cols = ['best_score', 'bestlu'], exp_changed_cols = ['best_score', 'bestlu', 'production'],
        grid = grid):
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
    if cprice:
        carbon_price = int(cprice)
    else:
        carbon_price = 0

    logger.info("Simulation start")
    # Calculate opportunity cost from suitable area and crop value (in '000 $)
    grid["opp_cost"] = grid['suitable_area'] * grid["agri_opp_cost"]/1000.

    # Set amount of beef to be produced based on the chosen location
    # demand = beef_production.at[location, 'Value']
    demand = beef_demand.at[location, 'Demand']

    # demand = beef_demand.loc[beef_demand.ADM0_A3 == location, 'Demand'].iloc[0]

    start_module = time.time()

    # Parallelise the scoring
    start = time.time()
    grid = scoring(grid, scenario, carbon_price, gap_reduce)

    print('### Done scoring in {} seconds'.format(time.time()-start))
    logger.info("Done scoring")

    start = time.time()

    # Keep on ly rows where at least one feed option has a score
    grid = grid.loc[grid.best_score > 0]
    grid = grid.reset_index(drop=True)
    total_production = 0
    grid['changed'] = 0
    grid['destination'] = 0
    grid['exporting'] = 0
    # Get country-level domestic demand
    grid['dom_demand'] = grid.merge(beef_demand, how='left', left_on='ADM0_A3', right_index=True)['Demand']
    # Sort rows by increasing 'best score'
    grid = grid.sort_values('best_score')
    # Get cumulative country level production in order of increasing best score
    grid.loc[grid.changed == 0, 'cumdomprod'] = grid.groupby('ADM0_A3')['production'].transform(pd.Series.cumsum)

    if trade_scenario == 'trade':
        # Create original best score to compare scores for domestic vs international destination
        grid['orig_bestscore'] = grid['best_score']
        # Set new production > 0 to compare old and new production to avoid new == old and infinite while loop
        new_production = 1
        while total_production < demand and grid.loc[(grid.changed == 0)].shape[0] > 0 and new_production != 0:

            # Calculate old production to compare with new production
            old_production = grid.loc[grid.changed == 1, 'production'].sum()

            # Sort by increasing best score
            grid = grid.sort_values('best_score')

            # Recalculate cumulative production based on total production and according to sorted values
            grid.loc[grid.changed == 0, 'cumprod'] = grid.loc[grid.changed == 0, 'production'].cumsum() + total_production

            # Convert cells to production if (1) cells have not been changed yet, (2) cumulative domestic production is lower than domestic demand OR the country of the cell is already exporting,
            # (3) Cumulative production is lower than global demand and (4) best score is lower than the highest score meeting these conditions

            grid.loc[(grid['changed'] == 0) &
                     ((grid['cumdomprod'] < grid['dom_demand']) | (grid['exporting'] == 1)) &
                     ((demand + grid['production'] - grid['cumprod']) > 0) &
                     (grid['best_score'] <= grid.loc[(grid['changed'] == 0) &
                                                     ((demand + grid['production'] - grid['cumprod']) > 0) &
                                                     ((grid['cumdomprod'] < grid['dom_demand']) | (grid[
                                                                                                       'exporting'] == 1)), 'best_score'].max()), 'changed'] = 1

            # Select all countries that have been converted and that are not yet exporting for which we recalculate costs
            ADM0_A3 = grid.loc[(grid.best_score <= grid.loc[grid.changed == 1].best_score.max()) &
                               (grid.exporting == 0) &
                               (grid.destination == 0) &
                               (grid.cumdomprod > grid.dom_demand), 'ADM0_A3']

            # Set these countries to exporting (0 = not exporting; 1 = exporting)
            grid.loc[(grid['changed'] == 0) & (grid['ADM0_A3'].isin(ADM0_A3)), 'exporting'] = 1

            start = time.time()
            # Recalculate costs and emissions of cells from listed countries
            if grid.loc[(grid['ADM0_A3'].isin(ADM0_A3)) & (grid['changed'] == 0)].shape[0] > 0:
                grid.loc[(grid['ADM0_A3'].isin(ADM0_A3)) & (grid['changed'] == 0)] = trade(
                    grid.loc[(grid['ADM0_A3'].isin(ADM0_A3)) & (grid['changed'] == 0)], scenario, carbon_price)
            logger.info("Trade done in {}".format(time.time() - start))
            # Set destination of production depending on whether domestic demand is met (1 = local; 2 = international)
            grid.loc[(grid['destination'] == 0) &
                     (grid['changed'] == 1), 'destination'] = np.where(grid.loc[(grid['destination'] == 0) &
                                                                                (grid['changed'] == 1), 'cumdomprod'] <
                                                                       grid.loc[(grid['destination'] == 0) & (
                                                                                   grid['changed'] == 1), 'dom_demand'], 1,
                                                                       2)
            # Recalculate total production
            total_production = grid.loc[grid.changed == 1, 'production'].sum()
            #
            new_production = total_production - old_production
            logger.info("Total production: {}".format(total_production))
    else:
        # Calculate cumulative beef production by increasing score
        grid['cumprod'] = grid['production'].cumsum()
        # Convert cells as long as the targeted demand has not been achieved
        grid.loc[(demand + grid['production'] - grid['cumprod']) > 0, 'changed'] = 1
    # # Reset index which had duplicates somehow
    # grid = grid.reset_index(drop=True)
    #
    # if constraint == 'global':
    #     # Aggregate beef production by increasing order of score
    #     grid['cumprod'] = grid['production'].cumsum()
    #     # Convert cells as long as the targeted demand has not been achieved
    #     grid.loc[(demand + grid['production'] - grid['cumprod']) > 0, 'changed'] = 1
    #
    # elif constraint == 'country':
    #     total_production = 0
    #     # Keep track of country-level production
    #     country_prod_dict = {}
    #     # Loop through cells
    #     for index, cell in grid.iterrows():
    #         # Record country
    #         country = cell["ADM0_A3"]
    #         # Check if the demand has been achieved and if there is demand in the country:
    #         if total_production < demand and country in beef_production.index:
    #             # Get beef demand in the country of the cell
    #             country_demand = beef_production.at[country, 'Value']
    #             # Get beef production from optimal landuse for this cell
    #             cell_production = grid.at[index, "production"]
    #             # Check if beef has previously been produced in this country
    #             if country not in country_prod_dict:
    #                 # If not, convert cell
    #                 country_prod_dict[country] = cell_production
    #                 grid.at[index, 'changed'] = 1
    #                 total_production += cell_production
    #                 grid.at[index, 'cumprod'] = total_production
    #
    #             # If previous cells have been converted, check if the country demand has been met
    #             elif country_prod_dict[country] < country_demand:
    #                 # If not, convert cell
    #                 country_prod_dict[country] += cell_production
    #                 grid.at[index, 'changed'] = 1
    #                 total_production += cell_production
    #                 grid.at[index, 'cumprod'] = total_production

    print('### Done sorting and selecting cells in  {} seconds. ###'.format(time.time() - start))
    #
    print('### Main simulation finished in {} seconds. ###'.format(time.time()-start_module))
    logger.info("Main simulation finished")

    ######### Export #########

    # start = time.time()
    grid.to_file(export_folder + '/' + scenario + "_" + str(gap_reduce) + ".gpkg", driver="GPKG")
    # b = list(grid.total_bounds)
    # export_raster(grid.loc[grid['changed'] == 1], b, 0.0833, ['production', 'opt_emissions'], scenario, export_folder, gap_reduce)

    start = time.time()

    print('### Exporting GPKG finished in {} seconds. ###'.format(time.time()-start))
    logger.info("Exporting results finished")

def parallelise(location, export_folder, scenario, trade_scenario):
    for i in range(0,100,10):
    # for i in ["weighted_sum"]:
        Process(target=main, args = (location, export_folder, scenario, trade_scenario, i,)).start()

if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('location', help='Spatial extent of simulation')
    # argparser.add_argument('resolution', help='Resolution of pixels (in degrees)')
    # argparser.add_argument('ncores', help='Number of cores for multiprocessing')
    argparser.add_argument('export_folder', help='Name of exported file')
    # argparser.add_argument('constraint', help='Whether to achieve optimisation globally or by keeping country-specific prodution')
    # argparser.add_argument('--exp_global_cols', nargs='+', help='Which column to export as global rasters')
    # argparser.add_argument('--exp_changed_cols', nargs='+', help='Which column to export as changed rasters')
    argparser.add_argument('scenario', help='Which scenario of optimisation to run ("weighted_sum", "carbon_price", "cscc", "costs", "ghg")')
    argparser.add_argument('trade_scenario', help="Trade scenario('trade' or 'notrade') ")
    argparser.add_argument('--cprice', help="Optional argument: Price of carbon (US$/t CO eq) to use if scenario is 'carbon_price'")
    argparser.add_argument('--ratio', help="Optional argument: Percentage reduction in yield gap")

    args = argparser.parse_args()
    location = args.location
    # resolution = args.resolution
    # ncores = args.ncores
    export_folder = args.export_folder
    # constraint = args.constraint
    # exp_global_cols = args.exp_global_cols
    # exp_changed_cols = args.exp_changed_cols
    scenario = args.scenario
    trade_scenario = args.trade_scenario
    cprice = args.cprice
    ratio = args.ratio

    # scenario = args.scenario
    # parallelise(location, export_folder, scenario, trade_scenario)
    main(location, export_folder, scenario, trade_scenario)