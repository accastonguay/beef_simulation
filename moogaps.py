#    Copyright (C) 2010 Adam Charette-Castonguay
#    the University of Queensland
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
__author__ = "Adam Charette-Castonguay"

import rasterio
from rasterio import features
from glob import glob
import geopandas as gpd
import pandas as pd
import numpy as np
import multiprocessing
import logging
from numpy import ones, vstack
from numpy.linalg import lstsq
import random

pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
### Read csv tables
regions = pd.read_csv("tables/glps_regions.csv")
grass_energy = pd.read_csv("tables/grass_energy.csv")  # Load energy in grasses
beef_production = pd.read_csv("tables/emissions_increase.csv")  # Load country-level beef supply
nutrient_req_grass = pd.read_csv("tables/nutrient_req_grass.csv")  # Load nutrient requirement for grasses
beef_demand = pd.read_csv("tables/beef_demand.csv")  # Load country-level beef demand
sea_distances = pd.read_csv("tables/sea_distances.csv")  # Load averaged distances between countries
sea_t_costs = pd.read_csv("tables/sea_t_costs.csv")  # Load sea transport costs
# energy_conversion = pd.read_csv("tables/energy_conversion.csv")  # Load energy conversion table
fuel_cost = pd.read_csv("tables/fuel_costs.csv")  # fuel cost
crop_area = pd.read_csv("tables/crop_area.csv")  # proportion of crop areas by country
feed_energy = pd.read_csv("tables/feed_energy.csv")  # ME in different feeds
partner_me = pd.read_csv("tables/partner_me.csv")  # Weighted average of ME to meat conversion factor in export partner countries
potential_yields = pd.read_csv("tables/potential_yields.csv")  # Potential yields by climate bins
yield_fraction = pd.read_csv("tables/yield_fraction.csv")  # Fraction yield gap
percent_exported = pd.read_csv("tables/percent_exported.csv")  # Fraction of exported feed
feedprices = pd.read_csv("tables/feedprices.csv")  # Crop prices
crop_emissions_factors = pd.read_csv("tables/emissions_factors.csv")  # N2O emission factors from N for crops
feedpartners = pd.read_csv("tables/feedpartners.csv")  # Trade partners for each feed
expcosts = pd.read_csv("tables/expcosts.csv")  # Export cost of feeds
sea_dist = pd.read_csv("tables/sea_dist.csv")  # Sea distances matrix
exp_access = pd.read_csv("tables/partner_access.csv")  # Access to market in importing country
fuel_partner = pd.read_csv("tables/fuel_partner.csv")  # Fuel cost in partner countries
fertiliser_requirement = pd.read_csv("tables/fertiliser_requirement.csv")  # fertiliser requirement per crop production
energy_efficiency = pd.read_csv("tables/energy_efficiency.csv")  # Energy efficiency
# crop_residues= pd.read_csv("tables/crop_residues.csv")  # Residue to product ratio
residue_energy= pd.read_csv("tables/residue_energy.csv")  # Energy in crop residues
stover_frac = pd.read_csv("tables/stover_frac.csv")  # Fraction of stover feed for beef cattle vs all livestock
# sc_change = pd.read_csv("tables/sc_change.csv")  # Fraction of stover feed for beef cattle vs all livestock
beefprices = pd.read_csv("tables/beef_price.csv", usecols = ['ADM0_A3', 'price'])
grain_stover_compo = pd.read_csv("tables/grain_stover_compo.csv")
feed_composition = pd.read_csv("tables/PNAS_feed_composition.csv")
beef_exports = pd.read_csv("tables/beef_exports.csv")
beef_export_partners = pd.read_csv("tables/beef_export_partners.csv")

aff_costs = pd.read_csv("tables/aff_costs.csv")
crop_emissions = pd.read_csv("tables/crop_emissions.csv")
dressing_table = pd.read_csv("tables/dressing_pct.csv")
wages = pd.read_csv("tables/wages.csv")
beef_increase = pd.read_csv("tables/beef_increase.csv")
foddercrop_area = pd.read_csv("tables/foddercrop_area.csv")
fertiliser_application = pd.read_csv("tables/fertiliser_application.csv")
beef_export_costs = pd.read_csv("tables/export_costs.csv")
distances = pd.read_csv("tables/distances.csv")
stemwood_c_parameters = pd.read_csv("tables/stemwood_c_parameters.csv")
herd_param = pd.read_csv("tables/herd_param.csv")
animal_weights = pd.read_csv("tables/animal_weights.csv")
lending_rates = pd.read_csv("tables/lending_rates_iso3.csv")

# stover_removal = 0.4  # Availability of crop residues
# Grass N20 emission_factors from N application from Gerber et al 2016
grass_n2o_factor = 0.007
speed = 1
fuel_efficiency = 0.4  # fuel efficiency in l/km
truck_emission_factor = 2.6712  # Emissions factor for heavy trucks (kg CO2/l)
sea_emissions = 0.048  # Emissions factor for heavy trucks (kg CO2/ton-km)
past_est_cost = 2.250
crop_est_cost = 4.500
process_pack = 1.45
GWP_N2O = 298



# column names for optimal costs/emissions sources
new_colnames = {'production': '_meat',
                'enteric': '_meth',
                'manure': '_manure',
                # 'export_emissions': '_exp_emiss',
                # 'export_cost': '_exp_costs',
                # 'transp_emission': '_trans_emiss',
                # 'transp_cost': '_trans_cost',
                'total_cost': '_tot_cost',
                'total_emission': '_ghg',
                'n2o_emissions': '_n2o',
                'production_cost': '_cost',
                'agb_change': '_agb_change',
                'opportunity_cost': '_opp_cost',
                'bgb_change': '_bgb_change',
                # 'processing_energy': '_process_energy',
                'postfarm_cost': '_postfarm_cost',
                'postfarm_emi': '_postfarm_emi',

                # 'compensation': '_compensation',
                'beef_area': '_area',
                'establish_cost': '_est_cost',
                'biomass': '_BM'
                }

# List of all 16 crops, used to determine available area on cell to grow feed
crop_list = ['barley', 'cassava', 'groundnut', 'maize', 'millet', 'oilpalm', 'potato', 'rapeseed', 'rice', 'rye',
             'sorghum', 'soybean', 'sugarbeet', 'sugarcane', 'sunflower', 'wheat']

def grain_composition_clean(foddercrop_list):
    grain_compo = pd.read_csv("tables/FAOSTAT_data_5-26-2021_feeds.csv",
                              usecols=['Area Code', 'Item', 'Value', 'Unit'])

    # Clean grain_compo df
    grain_compo['Value'] = grain_compo['Value'] * 1e-3
    grain_compo['crop'] = grain_compo['Item'].str.lower().str.replace(r' and products', '').replace({
        'rape and mustardseed': 'rapeseed',
        'soyabeans': 'soybean'})
    grain_compo = grain_compo.drop(['Unit', 'Item'], axis=1).rename(columns={'Area Code': 'ADM0_A3'})
    grain_compo = pd.pivot_table(grain_compo, values='Value', index='ADM0_A3', columns='crop').reset_index()
    grain_compo = grain_compo[['ADM0_A3'] + [f for f in foddercrop_list]]
    grain_compo[[f for f in foddercrop_list]] = grain_compo[[f for f in foddercrop_list]].div(
        grain_compo.drop('ADM0_A3', axis=1).sum(axis=1), axis="index")

    return grain_compo

def domestic_feed_clean(foddercrop_list):
    domestic_feed = pd.read_csv("tables/FAOSTAT_data_5-26-2021_domestic_grain_prod.csv",
                                usecols=['Area Code', 'Element', 'Item', 'Value', 'Unit'])

    # Clean domestic_feed df
    domestic_feed['Value'] = domestic_feed['Value'] * 1e-3
    domestic_feed['crop'] = domestic_feed['Item'].str.lower().str.replace(r' and products', '').replace({
        'rape and mustardseed': 'rapeseed',
        'soyabeans': 'soybean'})
    domestic_feed.rename(columns={'Area Code': 'ADM0_A3'}, inplace=True)

    print(domestic_feed)

    domestic_feed = domestic_feed.loc[domestic_feed.crop.isin(foddercrop_list)].drop(['Unit', 'Item'], axis=1).pivot_table(
        values='Value', index=['ADM0_A3', 'crop'], columns='Element').reset_index()

    domestic_feed['proportion_domestic'] = domestic_feed['Production'] / (domestic_feed['Import Quantity'] + domestic_feed['Production'])
    domestic_feed = domestic_feed.drop(['Import Quantity',  'Production'], axis = 1).pivot_table(
        values='proportion_domestic', index='ADM0_A3', columns='crop').reset_index()
    return domestic_feed

def producer_prices_clean(foddercrop_list):
    # Clean producer price df
    FAO_producer_prices = pd.read_csv("tables/FAOSTAT_data_5-26-2021_prod_prices.csv",
                                      usecols=['Area Code', 'Item', 'Value', 'Unit']
                                      )
    FAO_producer_prices['Value'] = FAO_producer_prices['Value'] * 1e-3
    FAO_producer_prices['crop'] = FAO_producer_prices['Item'].str.lower().str.replace(r' and products', '').replace({
        'rape and mustardseed': 'rapeseed',
        'rice, paddy': 'rice',
        'soybeans': 'soybean'})
    FAO_producer_prices.rename(columns={'Area Code': 'ADM0_A3'}, inplace=True)
    FAO_producer_prices = FAO_producer_prices.loc[FAO_producer_prices.crop.isin(foddercrop_list)].drop(['Unit', 'Item'], axis=1)
    return FAO_producer_prices

def fertiliser_sampling(simulation):
    init_fertiliser_prices = pd.read_csv("tables/fertiliser_prices.csv")  # Load fertiliser prices

    f_df = init_fertiliser_prices.merge(regions, how='left')

    fertiliser_prices = pd.DataFrame({'ADM0_A3': f_df.ADM0_A3,
                          'group': f_df.group})
    for i in ['n', 'p', 'k']:
        if simulation == 'uncertainty':
            fert_series = f_df.groupby('group')[i].apply(
                lambda x: np.random.uniform(low=x.min(), high=x.max())).reset_index(name=i)
            fertiliser_prices = fertiliser_prices.merge(fert_series, how='left', left_on='group', right_on='group')

        else:
            fertiliser_prices = f_df.copy()
    return fertiliser_prices

def energy_conversion_sampling(simulation):
    energy_conversion = pd.read_csv("tables/energy_conversion.csv")  # Load energy conversion table
    region = pd.read_csv("tables/glps_regions.csv", usecols=['region', 'group']).drop_duplicates(subset='group')

    energy_conversion = energy_conversion.merge(region, how='left')
    new_conv_df = energy_conversion.copy().drop(['curr', 'curr_methane', 'curr_manure', 'max', 'min_emissions', 'system'], axis=1)

    if simulation == 'uncertainty':
        for i in ['curr', 'curr_methane', 'curr_manure']:
            regional_conversion = energy_conversion.groupby(['region', 'feed'])[i].apply(
                lambda x: np.random.uniform(low=x.min(), high=x.max())).reset_index(name=i)
            new_conv_df = new_conv_df.merge(regional_conversion, how='left', left_on=['region', 'feed'], right_on=['region', 'feed'])
    else:
        new_conv_df = energy_conversion.copy()
    return new_conv_df

def parameter_sampling(grid, sampling_data, crop_list, logger, simulation):
    world = grid[['ADM0_A3', 'region', 'group']].drop_duplicates(subset=['ADM0_A3'])

    sampling_data = sampling_data.loc[sampling_data.crop.isin(crop_list)]
    sampling_data = pd.pivot_table(sampling_data, values='Value', index='ADM0_A3',
                                   columns='crop').reset_index()

    sampling_data = world.loc[~world.group.isna()].merge(sampling_data, how='left')

    new_prod_prices = pd.DataFrame({'ADM0_A3': sampling_data.ADM0_A3, 'group': sampling_data.group,
                                    'region': sampling_data.region})

    for c in crop_list:
        if simulation == 'uncertainty':
            fert_series_group = sampling_data.loc[sampling_data[c] > 0].groupby('group')[c].apply(
                lambda x: np.random.uniform(low=x.min(), high=x.max())).reset_index()
            fert_series_continent = sampling_data.loc[sampling_data[c] > 0].groupby('region')[c].apply(
                lambda x: np.random.uniform(low=x.min(), high=x.max())).reset_index()

            print(fert_series_continent.head())

            fert_series_group = new_prod_prices.merge(fert_series_group, how='left', left_on='group', right_on='group')[
                c].values
            fert_series_continent = \
                new_prod_prices.merge(fert_series_continent, how='left', left_on='region', right_on='region')[c].values
            global_average = np.nanmean(sampling_data[c].values)
            new_prod_prices[c] = np.select([fert_series_group > 0, fert_series_continent > 0],
                                           [fert_series_group, fert_series_continent],
                                           default=global_average)
        else:
            fert_series_group = sampling_data.loc[sampling_data[c] > 0].groupby('group')[c].mean().reset_index()
            fert_series_continent = sampling_data.loc[sampling_data[c] > 0].groupby('region')[c].mean().reset_index()

            print(fert_series_continent.head())

            fert_series_group = new_prod_prices.merge(fert_series_group, how='left', left_on='group', right_on='group')[
                c].values
            fert_series_continent = \
                new_prod_prices.merge(fert_series_continent, how='left', left_on='region', right_on='region')[c].values
            global_average = np.nanmean(sampling_data[c].values)
            new_prod_prices[c] = np.select([fert_series_group > 0, fert_series_continent > 0],
                                           [fert_series_group, fert_series_continent],
                                           default=global_average)

    return new_prod_prices

def opp_soc_change_sampling(change_type, aff, simulation, logger = 'logger'):
    sc_change = pd.read_csv("tables/sc_change.csv")  # Fraction of stover feed for beef cattle vs all livestock
    new_sc_change = sc_change.copy()
    if change_type == 'opp' and aff == 'regrowth':
        logger.info('type1: {}'.format(type(new_sc_change)))
        new_sc_change = new_sc_change.loc[
            (new_sc_change.code > 0)].drop(['grassland', 'cropland', 'tree', 'min_tree', 'max_tree'], axis=1)
        logger.info('type2: {}'.format(type(new_sc_change)))

        for i in ['grassland', 'cropland']:
            if simulation == 'uncertainty':
                new_sc_change[i] = [random.uniform(x, y) for x, y in zip(new_sc_change['min_' + i].values,
                                                                                new_sc_change['max_' + i].values)]
            else:
                new_sc_change[i] = new_sc_change[['min_' + i, 'max_' + i]].mean(axis=1)

            logger.info('type of {}: {}'.format(i, type(new_sc_change)))

    elif change_type == 'expansion':
        new_sc_change = new_sc_change.loc[(new_sc_change.code < 0)].drop(['grassland', 'cropland', 'tree', 'code'], axis=1)
        for i in ['grassland', 'cropland', 'tree']:
            if simulation == 'uncertainty':
                new_sc_change[i] = [random.uniform(x, y) for x, y in zip(new_sc_change['min_' + i].values,
                                                                                new_sc_change['max_' + i].values)]
            else:
                new_sc_change[i] = new_sc_change[['min_' + i, 'max_' + i]].mean(axis=1)
    logger.info('Final type {}'.format(type(new_sc_change)))

    return new_sc_change

def opportunity_cost_carbon(feats, sc_change, aff_scenario, logger, horizon, lam):
    ################## SOC and AGB change from removing current beef ##################
    for c in ['current_grazing', 'current_cropping']:
        feats[c] = np.where(feats.newarea == 1, 0, feats[c].values)

    feats['regrowth_area'] = np.nan_to_num(np.nansum(feats[['current_grazing', 'current_cropping']].values, axis=1))

    if aff_scenario == 'noaff':
        for c in ['opp_aff', 'aff_cost', 'opp_soc', 'best_regrowth']:
            feats[c] = np.zeros_like(feats.ADM0_A3, dtype='int8')

    elif aff_scenario == 'regrowth':

        max_cstock = feats['regrowth_area'].values * feats['potential_carbon'].values * -1
        # max_cstock = np.where(feats['ecoregions'].values  , max_cstock, 0)
        # tempdf = pd.DataFrame()
        tempdf = feats[['ADM0_A3']].copy()

        aff_opp_cost = feats['opp_cost'].values.astype(float) * 1e-3 * feats['c_area'].values

        print('type aff_opp_cost: ', type(aff_opp_cost))
        for aff in ['nat', 'man']:

            # Get percentage change of grassland and cropland to 'original' ecoregion
            soc_change = feats[['ecoregions']].merge(
                sc_change.loc[(sc_change.regrowth_type == aff) & (sc_change.code > 0)][['code', 'grassland', 'cropland']], how='left',
                left_on='ecoregions', right_on='code')

            # 'Opportunity cost' of soil carbon = sum over land uses of current area of land use * current soil carbon * percentage change * negative emission (-1)
            tempdf['opp_soc_'+ aff] = (feats['current_grazing'].fillna(0).values * feats['bgb_spawn'].values * soc_change[
                'grassland'].values + feats['current_cropping'].fillna(0).values * feats['bgb_spawn'].values * soc_change[
                           'cropland'].values) * -1 * 3.67

            # Make sure that opportunity cost of soil carbon is only where there currently is beef production
            tempdf['opp_soc_'+ aff] = np.where(feats.newarea.values == 0, tempdf['opp_soc_'+ aff], 0) * 3.67
            tempdf['opp_soc_'+ aff] = eac(tempdf['opp_soc_'+ aff], rate=0, type = 'ghg', lifespan=horizon)

            # Annualize opportunity cost of soil carbon

            t = float(horizon)
            subset_stemwood = stemwood_c_parameters.loc[stemwood_c_parameters.regrowth_type == aff]
            k = np.nan_to_num(feats[['ecoregions']].merge(subset_stemwood, how = 'left')['k'].values)
            p = np.nan_to_num(feats[['ecoregions']].merge(subset_stemwood, how = 'left')['p'].values)

            tempdf['opp_aff_' + aff] = (max_cstock * (1 - np.exp(-k * t)) ** p) * 3.67
            tempdf['opp_aff_' + aff] = np.where((soc_change.ecoregions <= 6) | (soc_change.ecoregions == 12),
                eac(tempdf['opp_aff_' + aff].values, rate=0, type = 'ghg', lifespan=horizon), 0)

            if aff == 'nat':
                # tempdf['aff_cost_' + aff] = np.zeros_like(feats.ADM0_A3, dtype='int8')
                tempdf['aff_cost_' + aff] = aff_opp_cost

            else:
                # Initial cost and long term rotation annual cost (in '000 USD)
                initial_aff_cost = feats[['ADM0_A3']].merge(aff_costs, how='left')['initial'].values / 1000. * feats[
                    'regrowth_area'].values
                annual_aff_cost = feats[['ADM0_A3']].merge(aff_costs, how='left')['annual'].values / 1000. * feats[
                    'regrowth_area'].values
                # Annualise initial cost
                r = tempdf[['ADM0_A3']].merge(lending_rates, how = 'left', on = 'ADM0_A3')['lending_rate'].values
                tempdf['aff_cost_' + aff] = eac(initial_aff_cost, rate = r, type = 'cost', lifespan=horizon) + annual_aff_cost + aff_opp_cost

            carbon_regrowth = np.nansum(tempdf[['opp_aff_' + aff, 'opp_soc_'+ aff]].values, axis = 1)
            tempdf['regrowth_score_' + aff] = (carbon_regrowth * (1 - lam)) + (tempdf['aff_cost_' + aff] * lam)

        for c in ['opp_aff', 'aff_cost', 'opp_soc', 'regrowth_score']:
            tempdf[c + '_noaff'] = np.zeros_like(feats.ADM0_A3, dtype='int8')

        feats['best_regrowth'] = np.nanargmin(tempdf[['regrowth_score' + man for man in ['_noaff', '_nat', '_man']]].values, axis=1)

        # logger.info('tempdf')
        # logger.info(tempdf[['regrowth_score' + man for man in ['_noaff', '_nat', '_man']]])
        #
        # logger.info('best_regrowth')
        # logger.info(feats[['best_regrowth']])

        for cname in ['opp_aff', 'aff_cost', 'opp_soc']:
            feats[cname] = np.take_along_axis(tempdf[[cname + man for man in ['_noaff', '_nat', '_man']]].values,
                                                         feats['best_regrowth'].values[:, None], axis=1).flatten()
        del tempdf

    else:
        logger.info("Afforestation scenario {} not in choices".format(aff_scenario))

    return feats

def current_state(grid, grain_compo, domestic_feed, producer_prices, fertiliser_prices, foddercrop_list,
                  profit_margin_method, graz_cap, feed_area,
                  subset_country = None, logger = None, scenario_id = 1):

    logger.info("Started simulation")
    logger.info("subset_country: {}".format(subset_country))

    percdict = {1: 90,
                # 2:92.5,
                2:95,
                # 4:99,
                3:100}
    # perc = percdict[perc_scn]

    ### Convert kg/km2 to kg

    # grid['feed'] = grid['feed'] * 0.01 * grid['cell_area'] * 1e-3
    for f in ['c_grain', 'c_graz', 'c_stover', 'c_occa']:
        grid[f] = grid[f] * 0.01 * grid['cell_area'] * 1e-3

    grid['total_nitrogen'] = grid['total_nitrogen'].values * 0.01 * grid['cell_area'].values * 1e-3
    grid['c_meat'] = grid['c_meat'].values * 0.01 * grid['cell_area'].values * 1e-3
    grid['c_meth'] = grid['c_meth'].values * 0.01 * grid['cell_area'].values* 1e-3
    grid['c_manure'] = grid['c_manure'].values * 0.01 * grid['cell_area'].values* 1e-3

    # logger.info('total_nitrogen 2')
    # logger.info('min: {}, max: {}'.format(grid['total_nitrogen'].min(), grid['total_nitrogen'].max()))

    country_beef = grid.groupby('ADM0_A3', as_index=False)[['c_meat', 'c_meth', 'c_manure']].sum()
    beef_increase2 = country_beef.merge(beef_production, how='left')

    for d, cname in zip(['c_meth', 'c_manure', 'c_meat'], ['curr_methane', 'curr_manure', 'curr_beef_meat']):

        beef_increase2['change'] = (beef_increase2[cname] - beef_increase2[d]) / (
                    beef_increase2[d])

        increase = 1 + grid[['ADM0_A3']].merge(beef_increase2[['ADM0_A3', 'change']], how='left', left_on='ADM0_A3',
                                               right_on='ADM0_A3')['change'].values
        grid[d] = grid[d] * increase

    # Adjust feed and total nitrogen for 2018 based on increase in beef production

    beef_increase2['change'] = (beef_increase2['curr_beef_meat'] - beef_increase2['c_meat']) / (
        beef_increase2['c_meat'])

    beef_increase = 1 + grid[['ADM0_A3']].merge(beef_increase2[['ADM0_A3', 'change']], how='left', left_on='ADM0_A3',
                                           right_on='ADM0_A3')['change'].values
    beef_increase = np.nan_to_num(beef_increase, nan=0.0, posinf=0, neginf=0)

    for d in ['total_nitrogen', 'c_grain', 'c_graz', 'c_stover', 'c_occa']:
        grid[d] = grid[d].values * beef_increase

    ### Calculate feed composition (tons)

    # for i in ['grass', 'grain', 'stover', 'occasional']:
    #     grid['c_' + i] = grid['feed'] * grid[['region', 'glps']].merge(feed_composition, how = 'left', left_on= ["region", "glps"],
    #                                         right_on = ["region", "glps"])[i].values/100.

    ### n2o (ton CO2) = N (ton) x N2O factor (%) * GWP_N2O
    curr_grass_n2o = np.nan_to_num(grid['total_nitrogen'] * grass_n2o_factor * GWP_N2O)

    # Cost from grassland fertiliser ('000$) = Total nitrogen application (ton) x Nitrogen prices ('000$/ton)
    grass_cost = np.nan_to_num(np.where(grid['c_meat'].values > 0,
                                        grid['total_nitrogen'].values, 0) * grid[['ADM0_A3']].merge(fertiliser_prices, how = 'left', left_on = 'ADM0_A3',
                                                                                 right_on = 'ADM0_A3')['n'].values)
    # Transport wage ($)
    transport_wage_mkt = (grid['c_meat'].values / 15.) * 2 * (grid["accessibility"].values / 60.) * grid[['ADM0_A3']].merge(wages, how = 'left')['wage'].values
    transport_wage_prt = (grid['c_meat'].values / 15.) * 2 * (grid["distance_port"].values / 60.) * grid[['ADM0_A3']].merge(wages, how = 'left')['wage'].values

    # For each feed
    # for f in foddercrop_list:
    #     grain_qty = grid[['ADM0_A3']].merge(grain_compo[['ADM0_A3', f]], how = 'left')[f].values * grid['grain'].values
    #     prop_dom = grid[['ADM0_A3']].merge(domestic_feed.loc[domestic_feed.feed == f],
    #                                                            how='left')['proportion_domestic'].values
    #     domestic_price = grid[['ADM0_A3']].merge(producer_prices.loc[producer_prices.feed == f], how='left')['Value'].values
    #
    #     # trade_matrix = grid[['ADM0_A3']].merge(feeds.loc[feeds.feed == f].drop('feed', axis=1), how='left').drop('ADM0_A3',
    #     #                                                                                                     axis=1).values
    #     #
    #     # feed_prices = grid[['ADM0_A3']].merge(prices.loc[prices.feed == f].drop('feed', axis=1), how='left').drop('ADM0_A3',
    #     #                                                                                                       axis=1).values
    #
    #     curr_grain_cost += np.nan_to_num((grain_qty * prop_dom * domestic_price)
    #                                      # + np.nansum(grain_qty[:, None] * trade_matrix * feed_prices,axis=1)
    #                                      )
    #
    #     # Quantity of feed traded (tons)
    #     allqtysum += grain_qty * (1 - prop_dom)

    # Grain production (ton) = grain (ton) * grain porportions (%)
    grain_prod = grid['c_grain'].values[:, None] * grid[['ADM0_A3']].merge(grain_compo, how = 'left').drop('ADM0_A3', axis = 1).values

    # Fraction of feed produced within country (%)
    prop_dom = grid[['ADM0_A3']].merge(domestic_feed, how='left').drop('ADM0_A3', axis = 1).values

    # Domestic grain prices
    domestic_price = grid[['ADM0_A3']].merge(producer_prices, how='left').drop(['ADM0_A3','group', 'region'], axis = 1).values

    # grid['curr_grain_cost'] = np.nansum(grain_prod * prop_dom * domestic_price, axis = 1)

    # Cost of producing grain ('000 USD) = grain production (ton) * domestic prices ('000 USD/ton)
    curr_grain_cost = np.nan_to_num(np.nansum(grain_prod * domestic_price, axis = 1))

    # Quantity of import feed (ton) = total grain consumed (ton) * fraction of grain imported (%)
    qty_feed_imp = np.nansum(grain_prod * (1 - prop_dom), axis = 1)

    ymax = fertiliser_application.loc[fertiliser_application['crop'].isin(
        foddercrop_list), 'max_yield'].values

    k = fertiliser_application.loc[fertiliser_application['crop'].isin(
        foddercrop_list), 'n05'].values

    curr_grain_n2o = np.nan_to_num(np.nansum(N_app(grain_prod, k[None, :], ymax[None, :]) * (
                crop_emissions_factors.loc[crop_emissions_factors['crop'].isin(foddercrop_list), 'factor'].values[
                None, :] / 100), axis=1))

    ntrips_port = qty_feed_imp / 15. * 2
    ntrips_port = np.where(ntrips_port < 0, 0, ntrips_port)
    # Calculate transport cost to bring imported feed from nearest port ($)
    # trans_feed_cost.append(ntrips_port * grid["distance_port"] * grid['Diesel'] * fuel_efficiency)

    # 1. Get proportion of beef exported
    # 2. Meat * prop domestic -> ntrip -> local emissions, local costs
    # 3. Proportion importing countries -> distance importing countries
    # 4. Meat * prop imported * emissions/kg-distance
    # 5. Meat * prop imported * cost/kg-distance

    ### Transport cost to bring grain from port ('000 USD) = trips x distance to port (min) x speed (km/min) x fuel efficiency (l/km) x fuel_cost ($/l) + wage ($) x $/'000$
    trans_feed_cost = np.nan_to_num(ntrips_port * grid["distance_port"] * speed * fuel_efficiency *  grid[['ADM0_A3']].merge(
        fuel_cost[['ADM0_A3', 'Diesel']], how = 'left', left_on = 'ADM0_A3', right_on ='ADM0_A3')['Diesel'].values + transport_wage_prt) * 1e-3

    ### Transport GHG to bring grain from port (t CO2) = trips x distance to port (min) x speed (km/min) x fuel efficiency (l/km) x diesel emission factor (kg CO2/l) x ton/kg
    trans_feed_emissions = np.nan_to_num(ntrips_port * grid["distance_port"] * speed * fuel_efficiency * truck_emission_factor) * 1e-3

    ### Fraction of beef exported (%)
    exported_beef = np.nan_to_num(grid[['ADM0_A3']].merge(beef_exports, how = 'left')['prop_exp'].values)

    ### Trips to market/abattoir based on live animal weight
    ntrips_beef_local = grid['c_meat'].values / grid.merge(dressing_table, how = 'left', left_on = 'region',
                                                           right_on = 'region')['dressing'].values * (1-exported_beef) / 15.0 * 2

    ntrips_beef_local = np.where(ntrips_beef_local < 0, 0, ntrips_beef_local)

    ### Transport cost to bring cattle to market ('000 USD) = trips x distance to port (min) x speed (km/min) x fuel efficiency (l/km) x fuel_cost ($/l) + wage ($) x $/'000$
    loc_trans_cost = np.nan_to_num(ntrips_beef_local * grid["accessibility"].values * speed * fuel_efficiency *  grid[['ADM0_A3']].merge(
        fuel_cost[['ADM0_A3', 'Diesel']], how = 'left', left_on = 'ADM0_A3',  right_on ='ADM0_A3')['Diesel'].values + transport_wage_mkt)* 1e-3

    ### Transport GHG to bring cattle to market (ton CO2) = trips x distance to port (min) x speed (km/min) x fuel efficiency (l/km) x diesel emission factor (kg CO2/l) x ton/kg
    loc_trans_emissions =  np.nan_to_num(ntrips_beef_local * grid["accessibility"] * speed * fuel_efficiency * truck_emission_factor * 1e-3)

    ### Trips to port for exports
    ntrips_beef_exp = grid['c_meat'] / grid.merge(dressing_table, how = 'left', left_on = 'region',
                                                  right_on = 'region')['dressing'].values * exported_beef / 15. * 2

    ntrips_beef_exp = np.where(ntrips_beef_exp < 0, 0, ntrips_beef_exp)

    ### Transport cost to bring cattle to port ('000 USD) = trips x distance to port (min) x speed (km/min) x fuel efficiency (l/km) x fuel_cost ($/l) + wage ($) x $/'000$
    meat_export_cost = np.nan_to_num(ntrips_beef_exp * grid["distance_port"] * speed * fuel_efficiency * grid[['ADM0_A3']].merge(
        fuel_cost[['ADM0_A3', 'Diesel']], how = 'left', left_on = 'ADM0_A3',  right_on ='ADM0_A3')['Diesel'].values  + transport_wage_prt) * 1e-3

    ### Sea shipping cost ('000 USD) = meat (ton) x fraction of beef export to partner countries (%) x export cost to partner country (USD/ton) x $/'000$
    meat_exp_sea_cost =  np.nan_to_num(np.nansum(grid['c_meat'].values[:, None] * grid[['ADM0_A3']].merge(beef_export_partners, how = 'left').drop('ADM0_A3', axis=1).values * \
                                      grid[['ADM0_A3']].merge(beef_export_costs, how = 'left').drop('ADM0_A3', axis=1).values, axis = 1)) * 1e-3

    ### Transport emissions of beef to trade partners
    ### Transport GHG to bring cattle to market ('000 USD) = trips x distance to port (min) x speed (km/min) x fuel efficiency (l/km) x diesel emission factor (kg CO2/l) x ton/kg
    meat_exp_emissions = np.nan_to_num(ntrips_beef_exp * grid["distance_port"].values * speed * fuel_efficiency * truck_emission_factor) * 1e-3

    # Emissions from sea transport (ton C02) = meat shipped (ton) x fraction of exports to partner countries (%) x distances between exporting and importing countries (km) sea emissionsa (kg CO2/ton-km) * ton/kg
    meat_exp_sea_emissions = np.nan_to_num(np.nansum(grid['c_meat'].values[:, None] * grid[['ADM0_A3']].merge(beef_export_partners, how = 'left').drop('ADM0_A3', axis=1).values * \
                                      grid[['ADM0_A3']].merge(distances, how = 'left').drop('ADM0_A3', axis=1).values, axis = 1) * sea_emissions) * 1e-3

    ### Emissions from processing and packaging (ton CO2) = beef (ton) x emission factor ('000 MJ/ton) x ton CO2/'000 MJ
    process_emissions  = np.nan_to_num(grid['c_meat'].values * process_pack * \
                                   grid[['ADM0_A3']].merge(energy_efficiency, how='left')['energy'].fillna(0).values)

    # Fraction of grazed area vs harvested area on cell (%)
    grazed_fraction = grid['grazed_area'].values/(grid['grazed_area'].values + grid['harvested_area'].values)
    grazed_fraction = np.nan_to_num(grazed_fraction, nan=1)

    harvested_fraction = 1 - grazed_fraction

    grid['grazed_biomass'] = np.where(grid['grazed_biomass'].values == 0, np.nan, grid['grazed_biomass'].values)
    grid['harvested_biomass'] = np.where(grid['harvested_biomass'].values == 0, np.nan, grid['harvested_biomass'].values)

    grid['current_grazing'] = np.nan_to_num(grid['c_graz'].values * grazed_fraction / grid['grazed_biomass'].values) + np.nan_to_num(((grid['c_graz'].values * harvested_fraction)) / grid['harvested_biomass'].values)
    grid['current_grazing'] = np.where(grid.current_grazing > grid.cell_area, grid.cell_area, grid.current_grazing)

    if graz_cap ==2:
        grid['current_grazing'] = np.minimum(np.nan_to_num(grid['current_grazing'].values),
                                             np.nan_to_num(grid['grass'].values * grid['cell_area'].values))

    grid = grid.drop(['grazed_biomass', 'harvested_biomass', 'grazed_area', 'harvested_area'], axis = 1)

    # Calculate local opp_cost
    # grid['grain_opp_cost'] = 0
    # grid['dom_grain_opp'] = 0
    imp_grain_opp = np.zeros_like(grid['current_grazing'].values, dtype = 'int8')
    # grid['dom_cropping_area'] = 0
    imp_cropping_area = np.zeros_like(grid['current_grazing'].values, dtype = 'int8')

    # for f in partner_feed_yields.feeds.unique():
    #     print("start " + f)
    #     domyields = grid[['ADM0_A3']].merge(feed_yields[["ADM0_A3", f]], how='left')[f].values
    #     domyields[domyields == 0] = np.nan
    #
    #     # Calculate cropping area (ha) = grain production (t) / grain yield (t/ha)
    #     grid['dom_cropping_area'] = grid['dom_cropping_area'].fillna(0) + grid[f].values / 1000. * \
    #                             grid[['ADM0_A3']].merge(domestic_feed.loc[domestic_feed.feed == f], how='left')[
    #                                 'proportion_domestic'].values / domyields
    #
    #     feedprod = grid[f].values / 1000.
    #     proptraded = (1 - grid[['ADM0_A3']].merge(domestic_feed.loc[domestic_feed.feed == f], how='left')[
    #         'proportion_domestic'].values)
    #     feedtraded = feedprod * proptraded
    #     feedtradedpartners = feedtraded[:, None] * grid[['ADM0_A3']].merge(
    #         feeds.loc[feeds.feed == f].drop('feed', axis=1), how='left').drop('ADM0_A3', axis=1).values
    #
    #     yields = partner_feed_yields.loc[partner_feed_yields.feeds == f].drop('feeds', axis=1).iloc[0].values[None, :]
    #     yields[yields == 0] = np.nan
    #     areatradedp = feedtradedpartners / yields
    #
    #     opp = areatradedp * opp_cost_range.iloc[opp_cost_range.index == 'opp_cost_median'].values
    #
    #     grid['imp_cropping_area'] = grid['imp_cropping_area'].fillna(0) + np.nansum(areatradedp, axis = 1)
    #     grid['imp_grain_opp'] = grid['imp_grain_opp'].values + np.nansum(opp, axis=1)

    # grain area (ha) = grain production (t)/yields (t/ha)
    if feed_area == 2:
        dom_cropping_area = np.nansum(grain_prod / np.ma.masked_values(grid[[f+'current' for f in foddercrop_list]].values, 0), axis = 1)
    else:
        fodder_yield_fraction = yield_fraction[['ADM0_A3'] + foddercrop_list]
        dom_cropping_area = np.nansum(grain_prod / grid[foddercrop_list].values * grid[['ADM0_A3']].merge(
                                    fodder_yield_fraction, how="left").drop('ADM0_A3', axis=1).values, axis=1)

    # logger.info("min dom_cropping_area: {}".format(np.min(dom_cropping_area)))
    # logger.info("max dom_cropping_area: {}".format(np.max(dom_cropping_area)))
    #
    # grid['dom_cropping_area'] = dom_cropping_area
    # logger.info("rows with inf {}".format(grid.loc[grid.dom_cropping_area == np.inf][['dom_cropping_area'] + [f+'current' for f in foddercrop_list]]))

    # full_grain_prod = np.nansum(feats['available_area'].values[:, None] * \
    #                             feats[['ADM0_A3']].merge(foddercrop_area, how="left").drop('ADM0_A3',
    #                                                                                        axis=1).values *
    #                             feats[foddercrop_list].values *
    #                             feats[['ADM0_A3']].merge(
    #                                 fodder_yield_fraction, how="left").drop('ADM0_A3', axis=1).values, axis=1)

    grid["pasture_opp_cost"] = np.nan_to_num(grid['opp_cost'].values * grid['current_grazing'].values)
    dom_grain_opp_cost =  np.nan_to_num(grid['opp_cost'].values * dom_cropping_area)
    grid['grain_opp_cost'] =  dom_grain_opp_cost + imp_grain_opp

    # threshold = oppcost_thresholds.loc[oppcost_thresholds.percentile == perc, 'threshold'].iloc[0]
    # grid["pasture_" + i + a] = np.where(grid["pasture_" + i + a]/grid['bvmeat'] > threshold,
    #                                     threshold * grid['bvmeat'].values,
    #                                     grid["pasture_" + i + a])

    logger.info("Finished calculating opportunity cost")

    # Check that cropping area is not greater than cell area
    grid['current_cropping'] = dom_cropping_area + imp_cropping_area

    # for c in grid.columns:
    #     try:
    #         logger.info('{}: {}'.format(c, np.nanmin(grid[c])))
    #     except:
    #         logger.info("Couldn't print min of column {} because of datatype {}".format(c, grid[c].dtype))
    grid["c_opp_cost"] = np.nansum(grid[["pasture_opp_cost", 'grain_opp_cost']].values,
                                   axis = 1) * 1e-3

    # cost_meat = np.nan_to_num(grid["c_opp_cost"].values[grid["c_meat"].values>0])/grid["c_meat"].values[grid["c_meat"].values>0]
    # threshold = np.percentile(cost_meat, perc)
    # logger.info("Perc: {}, Threshold: {}".format(perc, threshold))

    # grid["c_opp_cost"] = np.where(grid["c_opp_cost"].values/grid["c_meat"].values > threshold,
    #                               threshold * grid["c_meat"].values,
    #                               grid["c_opp_cost"].values)
    from scipy import stats
    threshold = 6

    perc = stats.percentileofscore(grid["c_opp_cost"].values[grid["c_meat"].values>0]/grid["c_meat"].values[grid["c_meat"].values>0], threshold)
    logger.info("Percentile of 6 USD/kg: {}".format(perc))

    grid["c_opp_cost"] = np.where(grid["c_opp_cost"].values/grid["c_meat"].values > threshold,
                                  threshold * grid["c_meat"].values,
                                  grid["c_opp_cost"].values)

    # logger.info("Max opp per meat {}".format(np.nanmax(grid["c_opp_cost"].values/grid["c_meat"].values)))

    export_costs = meat_export_cost + meat_exp_sea_cost
    export_emissions = meat_exp_emissions + meat_exp_sea_emissions

    # grid['export_costs'] = export_costs
    # grid['loc_trans_cost'] = loc_trans_cost

    # grid['grass_cost'] = grass_cost
    # grid['trans_feed_cost'] = trans_feed_cost
    # grid['curr_grain_cost'] = curr_grain_cost

    grid['c_postfarm_cost'] = export_costs + loc_trans_cost
    grid['c_cost'] = curr_grain_cost + trans_feed_cost + grass_cost
    grid['c_postfarm_emi'] = export_emissions + loc_trans_emissions + trans_feed_emissions + process_emissions
    grid['c_n2o'] = curr_grass_n2o + curr_grain_n2o
    grid['c_tot_cost'] = np.nansum(grid[['c_postfarm_cost', 'c_cost', "c_opp_cost"]].values, axis = 1)
    grid['c_ghg'] = np.nansum(grid[['c_postfarm_emi', 'c_n2o', 'c_meth', 'c_manure']].values, axis = 1)

    grid['c_agb_change'] = np.zeros_like(grid['c_meat'].values, dtype = 'int8')
    grid['c_bgb_change'] = np.zeros_like(grid['c_meat'].values, dtype = 'int8')
    grid['c_est_cost'] = np.zeros_like(grid['c_meat'].values, dtype = 'int8')

    grid['c_area'] = np.nansum(grid[['current_grazing', 'current_cropping']].values, axis = 1)

    # grid.rename(columns={'feed': 'c_BM'}, inplace=True)
    grid['c_BM'] = np.nansum(grid[['c_graz', 'c_grain', 'c_stover', 'c_occa']].values, axis = 1)

    newdf = pd.DataFrame({"total_emissions": grid.c_ghg.sum(),
                          "total_costs": grid.c_tot_cost.sum(),
                          "total_production": grid.c_meat.sum(),
                          "export_costs": np.nansum(export_costs),
                          "meat_exp_emissions": np.nansum(meat_exp_emissions),
                          'meat_export_cost': np.nansum(meat_export_cost),
                          "meat_exp_sea_cost": np.nansum(meat_exp_sea_cost),
                          "meat_exp_sea_emissions": np.nansum(meat_exp_sea_emissions),
                          "loc_trans_cost": np.nansum(loc_trans_cost),
                          "curr_grain_cost": np.nansum(curr_grain_cost),
                          "grass_cost": np.nansum(grass_cost),
                          "trans_feed_cost": np.nansum(trans_feed_cost),
                          "export_emissions": np.nansum(export_emissions),
                          "loc_trans_emissions": np.nansum(loc_trans_emissions),
                          "trans_feed_emissions": np.nansum(trans_feed_emissions),
                          "process_emissions": np.nansum(process_emissions),
                          "curr_grass_n2o": np.nansum(curr_grass_n2o),
                          "curr_grain_n2o": np.nansum(curr_grain_n2o),
                          # 'imp_grain_opp': grid.imp_grain_opp.sum(),
                          # 'dom_grain_opp_cost': grid.dom_grain_opp_cost.sum(),
                          'grain_opp_cost': grid.grain_opp_cost.sum(),
                          'pasture_opp_cost': grid.pasture_opp_cost.sum(),
                          "c_meth": grid.c_meth.sum(),
                          "c_manure": grid.c_manure.sum(),
                          "c_opp_cost": grid.c_opp_cost.sum(),
                          "c_postfarm_cost": grid.c_postfarm_cost.sum(),
                          "c_cost": grid.c_cost.sum(),
                          "c_postfarm_emi": grid.c_postfarm_emi.sum(),
                          "c_n2o": grid.c_n2o.sum(),
                          'c_area':grid.c_area.sum(),
                          'current_grazing': grid.current_grazing.sum(),
                          'current_cropping': grid.current_cropping.sum(),
                          # 'dom_cropping_area': grid.dom_cropping_area.sum(),
                          # 'imp_cropping_area': grid.imp_cropping_area.sum(),
                          'c_graz': grid.c_graz.sum(),
                          'c_grain': grid.c_grain.sum(),
                          'c_stover': grid.c_stover.sum(),
                          'c_occa': grid.c_occa.sum(),
                          'profit_margin_method' : profit_margin_method,
                          'percentile': str(perc)},
                         index=[0])

    logger.info('Total costs: {}, total emissions: {}'.format(grid.c_tot_cost.sum(), grid.c_ghg.sum()))
    country_agg = grid[['ADM0_A3', 'c_ghg', 'c_tot_cost', 'c_graz', 'c_stover', 'c_grain',
                        'current_grazing', 'current_cropping', 'c_meat', 'c_postfarm_cost', 'c_opp_cost',
                        'c_postfarm_emi', 'c_n2o', 'c_cost', 'c_manure', 'c_meth']].groupby('ADM0_A3', as_index=False).sum()

    country_agg.to_csv("./country_current_{}.csv".format(scenario_id), index = False)

    newdf.to_csv("./current_{}.csv".format(scenario_id), index = False)
    return grid

def N_app(x, k, ymax):

    result = np.where(x >= ymax,
                     k* np.arctanh(0.999)/np.arctanh(0.5),
                     k* np.arctanh(x/ymax)/np.arctanh(0.5))
    return result

def findcprice(emissions, costs):
    files = glob("./total_*.csv")

    data = pd.DataFrame()
    for i in files:
        sub = pd.read_csv(i)
        data = pd.concat([data, sub])
    data = data.sort_values('weight')
    print(data[['total_costs', 'total_emissions', 'weight']])

    target = emissions * 0.55

    x2 = data.weight[data.total_emissions > target].values[0]
    x1 = data.weight[data.total_emissions < target].values[-1]
    y2 = data.total_emissions[data.total_emissions > target].values[0]
    y1 = data.total_emissions[data.total_emissions < target].values[-1]

    points = [(y1, x1), (y2, x2)]

    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]

    target_weight = m * target + c

    ### Current emissions
    y = emissions

    x2 = data.weight[data.total_emissions > y].values[0]
    x1 = data.weight[data.total_emissions < y].values[-1]
    y2 = data.total_emissions[data.total_emissions > y].values[0]
    y1 = data.total_emissions[data.total_emissions < y].values[-1]

    points = [(y1, x1), (y2, x2)]

    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]

    emi_weight = m * y + c

    ### Current emissions

    y = costs

    x1 = data.weight[data.total_costs > y].values[-1]
    x2 = data.weight[data.total_costs < y].values[0]
    y1 = data.total_costs[data.total_costs > y].values[-1]
    y2 = data.total_costs[data.total_costs < y].values[0]

    points = [(y1, x1), (y2, x2)]

    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]

    cost_weight = m * y + c

    weight_list = [emi_weight, cost_weight, target_weight]
    return weight_list

def calc_establishment_cost(feats, area_col, horizon, cost):

    # Area that requires transition (ha)
    # transition_area = np.where(
    #     area_col < (feats['current_grazing'].fillna(0).values + feats['current_cropping'].fillna(0).values),
    #     0,
    #     area_col - (feats['current_grazing'].fillna(0).values + feats['current_cropping'].fillna(0).values))

    transition_area = np.where(feats.newarea == 1, area_col, 0)

    # Fraction of suitable area that is not grass or cropland (%)
    transition_fraction = ((feats['suitable'] - (feats['crop'] + feats['grass']))/feats['suitable'])
    transition_fraction[transition_fraction < 0] = 0
    r = feats[['ADM0_A3']].merge(lending_rates, how='left', on='ADM0_A3')['lending_rate'].values

    return eac(transition_fraction * transition_area * cost, rate = r, type = 'cost', lifespan=horizon)

def create_grid(foddercrop_list, profit_margin_sampling, pnas_inputs, simulation, suit_area, feed_area):
    grid = pd.DataFrame({})

    pnas_rasters = glob('./rasters/pnas_inputs/{}/*.tif'.format(pnas_inputs))

    cnames = {'feed': 'feed',
              'meat': 'c_meat',
              'beef': 'c_meat',
              'manure': 'c_manure',
              'methane': 'c_meth',
              'grai': 'c_grain',
              'graz': 'c_graz',
              'stov': 'c_stover',
              'occa': 'c_occa',
              }

    for r in pnas_rasters:
        name = r.split('/')[-1].split('.')[0]
        for i in ['feed', 'manure', 'meat', 'methane', 'beef', 'grai', 'graz', 'occa', 'stov']:
            if i in name:
                colname = cnames[i]
        # if name != 'accessibility':
        with rasterio.open(r) as f:
            meta = f.meta
            grid[colname] = f.read(1).flatten()

    rasters = glob('./rasters/*.tif')

    for r in rasters:
        name = r.split('/')[-1].split('.')[0]
        # if name != 'accessibility':
        with rasterio.open(r) as f:
            meta = f.meta
            # if meta['dtype'] == 'float64' and name not in ['c_ghg_95', 'c_tot_cost_95', 'c_meat_95']:
            #     dt = 'float32'
            if name in ['ecoregions', 'climate']:
                dt = 'int8'
            else:
                dt = 'float64'
                # dt = meta['dtype']

            grid[name] = f.read(1).flatten()
            # if name in ['c_ghg', 'c_tot_cost', 'c_meat']:
            #     grid[name] = np.where(grid[name] == 0, np.nan, grid[name])
    if feed_area == 2:
        feed_area_rasters = glob('./rasters/current_yield/*.tif')
        for r in feed_area_rasters:
            name = r.split('/')[-1].split('.')[0]
            with rasterio.open(r) as f:
                grid[name] = f.read(1).flatten()

    grid['potential_carbon'] = np.where(grid['potential_carbon'].values < 0, 0, grid['potential_carbon'].values)

    # grid['potential_carbon'] = [random.uniform(x,y) for x,y in zip(grid.potential_carbon.values,grid.potential_carbon_west.values)]

    dict_suitable = {1:'suitable_',
                     2: 'suitable_protected',
                     3:'suitable_barren',
                     4:'suitable_barren_sparse',
                     5:'suitable_barren_sparse_shrub',
                     6: 'suitable_protected_mosaic_barren_sparse_shrub'}

    grid['suitable'] = grid[dict_suitable[suit_area]]

    for col in ['suitable', 'crop', 'grass']:
        grid[col] = np.where(grid[col].values == 255, 0, grid[col].values)

    dict_colname = {}
    threshold_scenario = '_99'
    for c in grid.columns:
        if threshold_scenario in c:
            dict_colname[c] = c.replace(threshold_scenario, '')
    grid.rename(columns = dict_colname, inplace = True)

    world = gpd.read_file('map/admin_boundaries.gpkg')
    world['id'] = world.index

    out_shape = (meta['height'], meta['width'])
    grid_cell = ((geom, value) for geom, value in zip(world.geometry, world.id))

    burned = features.rasterize(shapes=grid_cell, fill=-1, out_shape=out_shape, dtype='int16',
                                transform=meta['transform'])
    grid['cell_id'] = np.asarray(grid.index, dtype = 'int32')
    # Keep all cells for exporting rasters later on

    grid['id'] = np.asarray(burned.flatten(), dtype = 'int32')
    grid['ADM0_A3'] = grid[['id']].merge(world[['id', 'ADM0_A3']], how='left')['ADM0_A3'].values

    grid = grid.loc[grid.id > -1]
    # grid = grid.loc[grid.ADM0_A3.isin(['BEL', 'NLD', 'LUX'])]

    grid = grid.merge(regions, how='left')

    del burned, world

    if profit_margin_sampling == 0:
        profit_margin = (0.833*grid['field_size'].values  -8.33)/100.
        profit_margin = np.where(profit_margin < 0, 0, profit_margin)
    else:
        small = np.random.choice([0.05, 0.15, 0.25], grid.shape[0], p=[0.56, 0.12, 0.32])
        medium = np.random.choice([0.05, 0.15, 0.25], grid.shape[0], p=[0.42, 0.13, 0.45])
        large = np.random.choice([0.05, 0.15, 0.25], grid.shape[0], p=[0.26, 0.12, 0.62])

        profit_margin = np.select([grid.field_size >= 30, grid.field_size >= 20, grid.field_size >= 10],
                                   [large, medium, small],
                                   default=0)
        del small, medium, large

    print('redis_feed_GLW2_country' in grid.columns)

    grid.rename(columns={
        # "redis_beef_GLW2_country": "c_meat",
        # 'redis_manure_GLW2_country': 'c_manure',
        # "redis_methane_GLW2_country": "c_meth",
        # 'redis_feed_GLW2_country': 'feed',
        'nitrogen': 'total_nitrogen',
                         }, inplace=True)
    
    grid['total_nitrogen'] = np.nan_to_num(
        np.where(grid['total_nitrogen'].values < 0, 0, grid['total_nitrogen'].values))

    if simulation == 'uncertainty':
        crop_value = np.asarray([random.uniform(x,y) for x,y in zip(grid.opp_uminn.values,grid.opp_ifpri.values)])
        crop_value = np.where(crop_value < 0, 0, crop_value)

    else:
        crop_value = np.where(grid['opp_uminn'].values < 0, 0, grid['opp_uminn'].values)

    # if rumi == 2:
    grid['ruminant_value'] = grid['ruminnats_redistributed'].values

    grid['opp_cost'] = np.maximum(np.nan_to_num(crop_value), np.nan_to_num(grid['ruminant_value'].values)/100.) * profit_margin

    del profit_margin, crop_value
    grid["nutrient_availability"] = grid['nutrient_availability'].replace(0, 2)

    # Get net cropping area to 'protect'
    # grid['net_fodder_area'] = np.where(grid['sum_area'] - grid['current_cropping'] < 0,
    #                                    0, grid['sum_area'] - grid['current_cropping'])

    ### Only keep cells where there is feed ###
    feeds = [c for c in grid.columns if 'grass' in c or c in foddercrop_list]

    # Remove cells that have no potential feed and no space, or no current beef
    # grid = grid.loc[((grid[feeds].sum(axis=1) > 0) &
    #                 (grid['suitable'].values * grid['cell_area'].values - grid['net_fodder_area'].values > 0)) |
    #                 (grid['c_meat'].values > 0)]
    grid = grid.loc[(grid[feeds].sum(axis=1) > 0) |  (grid['c_meat'].values > 0)]
    # print('----> Beef in EGY 5: {}'.format(np.nansum(grid.loc[grid.ADM0_A3 == 'EGY', 'c_meat'])))

    dropcols = ['other_rum', 'agri_opp2', 'field_size', 'opp_uminn', 'opp_ifpri',
                'iiasa_ag_opp','potential_carbon_west','beef_gs']

    for d in dropcols:
        if d in grid.columns:
            grid = grid.drop(d, axis=1)

    return grid

def eac(cost, rate, type, lifespan=30.):
    """
    Function to annualize a cost based on a discount rate and a lifespan

    Arguments:
    cost (float) -> One-off cost to annualise
    rate (float)-> Discount rate, default = 7% (Wang et al 2016 Ecol Econ.)
    lifespan (float)-> Time horizon, default: 30 commonly practiced in agricultural investment (Wang et al 2016 Ecol Econ.)

    Output: returns the annualised cost as a float
    """

    if type == 'ghg':  # For emissions -> no discount rate
        return cost / lifespan
    else:
        return (cost * rate) / (1 - (1 + rate) ** -lifespan)

def weighted_score(feats, l, lam, horizon, logger, transport_wage_mkt, dressing):
    """
    Function to calculate score based on relative costs and emissions

    Arguments:
    feats (dataframe) -> Dataframe in which to look for land use to optimise
    l (str)-> land use for which to calculate score
    lam (float)-> Lambda weight for optimisation

    Output: returns the annualised cost as a float
    """

    # Annualise establishment cost, same for pasture or crops, adjusted by the ratio of area used:cell area

    # Percentage of area that needs transition * area other than current crop and current grass * 8 ('000$) adjusted based on suitable ratio

    # transition_area = np.where(
    #     feats[l + '_area'] < (feats['current_grazing'].fillna(0).values + feats['current_cropping'].fillna(0).values),
    #     0,
    #     feats[l + '_area'] - (feats['current_grazing'].fillna(0).values + feats['current_cropping'].fillna(0).values))
    #
    # transition_fraction = ((feats['suitable'] - (feats['crop'] + feats['grass']))/feats['suitable'])
    # transition_fraction[transition_fraction < 0] = 0
    # feats[l + '_est_cost'] = eac(transition_fraction * transition_area * 8, lifespan=horizon)

    ### Infrastructure cost

    # Calculate current heads in feedlots (assuming that if there is grain in diet on the cell, beef is produced in a feedlot)

    logger.info('land use: {}'.format(l))

    if l in ['grass_grain', 'stover_grain']:

        LW_head = feats[['region']].merge(herd_param, how='left')['liveweight'].values

        current_heads = np.where(feats.c_grain > 0,
                                 np.ceil((feats.c_meat * 1e3) / dressing / LW_head),
                                 0).astype(int)

        delta_heads = np.where(np.ceil((feats[l + '_meat'] * 1e3 / dressing / LW_head)) > current_heads,
                               np.ceil((feats[l + '_meat'] * 1e3 / dressing / LW_head)) - current_heads,
                               0).astype(int)

        # cost_per_heads =  2051.5 * (delta_heads ** -0.221) * 108/71.
        cost_per_heads =  250 * 108/71.

        # infcost ('000 USD) = change head (head) * cost per head (USD/head) * '000 USD/USD
        r = feats[['ADM0_A3']].merge(lending_rates, how='left', on='ADM0_A3')['lending_rate'].values

        feats[l + '_infcost'] = eac((delta_heads * cost_per_heads) * 1e-3, rate = r, type = 'cost', lifespan=horizon)
        feats[l + '_est_cost'] = np.nan_to_num(feats[l + '_est_cost']) + np.nan_to_num(feats[l + '_infcost'])

        temp = pd.DataFrame({'current_heads': current_heads,
                             'new_heads': np.ceil((feats[l + '_meat'] * 1e3 / dressing / LW_head)),
                             'delta_heads': delta_heads,
                            'infcost': feats[l + '_infcost'],
                             'est_cost': feats[l + '_est_cost']}
        )
        logger.info('Total establishment cost of {} is:'.format(l))
        logger.info(temp.loc[temp.new_heads > 0 ])
        del temp

    # feats['transition_fraction'] = transition_fraction

    LW_diff = (feats[l + '_meat'] -  feats['c_meat']) * 1e3
    LW_diff[LW_diff < 0] = 0
    animals = np.ceil(LW_diff / dressing / feats[['region']].merge(animal_weights, how='left')['adult'].values)

    LW_calfs = (animals * feats[['region']].merge(animal_weights, how='left')['calf'].values) * 1e-3

    ntrips = np.ceil(LW_calfs / int(15)) * 2

    # Transport cost to market: number of trips * transport cost ('000 US$)
    trans_cost_calf = (ntrips * feats[['ADM0_A3']].merge(
        fuel_cost[['ADM0_A3', 'Diesel']], how='left', left_on='ADM0_A3', right_on='ADM0_A3')['Diesel'].values * \
                  feats["accessibility"] * speed * fuel_efficiency + ntrips * transport_wage_mkt) / 1000.

    logger.info('total production cost: {}, total calf transport cost: {}'.format(np.round(np.nansum(feats[l + '_cost'])), np.round(np.nansum(trans_cost_calf))))
    logger.info('min production cost: {}, min calf transport cost: {}'.format(np.round(np.nanmin(feats[l + '_cost'])), np.round(np.nanmin(trans_cost_calf))))

    feats[l + '_cost'] = feats[l + '_cost'] + trans_cost_calf

    calf_trans_emiss = ntrips * feats["accessibility"] * speed * fuel_efficiency * truck_emission_factor / 1000.
    logger.info('total postfarm emissions: {}, total calf transport GHG: {}'.format(np.round(np.nansum(feats[l + '_postfarm_emi'])), np.round(np.nansum(calf_trans_emiss))))
    feats[l + '_postfarm_emi'] = feats[l + '_postfarm_emi'] + calf_trans_emiss

    # Calculate opportunity cot in '000 USD: ag value (USD/ha) x area used (ha)
    feats[l + '_opp_cost'] =  feats['opp_cost'].astype(float) / 1000. * feats[l + '_area']

    # Calculate current production: cell area (ha) x meat (kg/km2) x kg/km2-t/ha conversion
    # current_production = feats['cell_area'].values * feats['bvmeat'].values * 1e-5
    # Get price of beef in '000 USD
    # beef_price = feats[['ADM0_A3']].merge(beefprices, how='left')['price'].values / 1000.
    # Compensation for loss of revenues from beef ('000 USD) = max(0, beef prices ('000 USD) x (current production - new production))
    # feats[l + '_compensation'] = np.maximum(0, beef_price * (current_production - feats[l + '_meat'].values))
    # del beef_price, current_production

    # Estimate cost of importing animals

    # Emissions for processing and packaging energy = meat (t) * process energy (MJ/kg) * energy efficiency (kg CO2/kg)

    # Total costs ('000 USD) = Establishment cost (annualised '000 USD) + production cost + transport cost + opportunity cost + cost of afforestation
    cost_cols = ['_est_cost', '_cost', '_opp_cost', '_postfarm_cost']
    feats[l + '_tot_cost'] = np.nansum(
        feats[[l + c for c in cost_cols]].values, axis=1, dtype='float64') - np.nan_to_num(feats.aff_cost)
    # + feats[l + '_compensation']

    # Annualise the loss of carbon stock
    feats[l + '_agb_change'] = eac(feats[l + '_cstock'], rate=0, type = 'ghg', lifespan=horizon)

    # feats[l + '_exp_costs'] + feats[l + '_trans_cost']+ feats[l + '_compensation'])

    # Update annual emissions (t CO2 eq)
    emissions_cols = ['_n2o', '_meth', '_manure', '_postfarm_emi', '_agb_change', '_bgb_change']

    # feats[l + '_trans_emiss'] + feats[l + '_exp_emiss'] + feats[l + '_process_energy']
    feats[l + '_ghg'] = np.nansum(
        feats[[l + c for c in emissions_cols]].values, axis=1, dtype='float64') - np.nansum(
        feats[['opp_aff', 'opp_soc']].values, axis=1, dtype='float64')

    # Calculate relative GHG (GHG/meat)(t CO2 eq)/Meat (ton)
    rel_ghg = np.where(feats[l + '_meat'] < 1, np.NaN, feats[l + '_ghg'] / (feats[l + '_meat']))
    # Calculate relative Cost (Cost/meat) Cost ('000 USD)/Meat (ton)
    rel_cost = np.where(feats[l + '_meat'] < 1, np.NaN,
                        feats[l + '_tot_cost'] / (feats[l + '_meat']))

    feats[l + '_bgb_change'] = np.where(feats['suitable'].values > 0, feats[l + '_bgb_change'], 0)

    feats[l + '_score'] = np.where(feats['suitable'].values > 0,
                                   (rel_ghg * (1 - lam)) + (rel_cost * lam),
                                   np.nan)
    # logger.info("Done calculating rel cost & emissions for  {}".format(l))
    return feats

def scoring(feats, foddercrop_list, feedprices, fertiliser_prices, energy_conversion, sc_change_opp, sc_change_exp,
            grain_max, stover_removal, crop_yield, lam, beef_yield, aff_scenario, logger, feed_option, landuses,
            horizon):
    """
    Finds the best landuse for each cell in the partition and returns the partition

    Arguments:
    feats (pandas dataframe) -> Main dataframe
    crop_yield (int)-> Scenario of crop yield (0 = current, 1 = no yield gap)
    lam (float)-> Lambda weight ([0,1])
    beef_yield (str)-> Scenario of beef yield ('me_to_meat' = current, 'max_yield' = no yield gap)
    logger (RootLogger) -> logger defined in main function
    feed_option (str)-> folder where the output file is exported

    Output: returns a gridded dataframe with scores
    """

    # Adjust yield fraction based on yield gap reduction scenario
    yield_fraction[foddercrop_list] = yield_fraction[foddercrop_list] + crop_yield

    # Cap yield fraction to 1 (cannot be high than attainable yield)
    yield_fraction[foddercrop_list] = yield_fraction[foddercrop_list].where(~(yield_fraction[foddercrop_list] > 1), other=1)

    ################## Calculate available area ##################

    feats['available_area'] = np.where(feats['newarea'].values == 1,
                                       feats['suitable'].values * feats['cell_area'].values  - feats['net_fodder_area'].values,
                                       feats['current_grazing'].fillna(0).values + feats['current_cropping'].fillna(
                                           0).values)

    # logger.info('Grid shape after available_area: {}, current meat {}'.format(feats.shape[0], feats.c_meat.sum()))
    feats = feats.loc[(feats['available_area'] > 0) | (feats['c_meat'] > 0)]
    # logger.info('Grid shape after available_area: {}, current meat {}'.format(feats.shape[0], feats.c_meat.sum()))
    # logger.info('Grid shape after available_area: {}, current meat {}'.format(feats.shape[0], feats.c_meat.sum()))

    logger.info('min available area {}'.format(np.nanmin(feats.loc[feats.newarea == 1, 'available_area']) ))

    feats.loc[(feats.newarea == 1) &
              (feats['suitable'].values * feats['cell_area'].values  < feats['net_fodder_area'].values), 'available_area'] = 0

    # Set transport wage based on travel time and country wage
    transport_wage_mkt = (feats["accessibility"].values / 60.) * feats[['ADM0_A3']].merge(wages, how='left')[
        'wage'].values
    transport_wage_prt = (feats["distance_port"].values / 60.) * feats[['ADM0_A3']].merge(wages, how='left')[
        'wage'].values
    dressing = feats[['region']].merge(dressing_table, how='left')['dressing'].values

    if feed_option in ['v1', 'v2']:

        for l in grass_cols:

            # For grazing, convert all area
            feats[l + '_area'] = feats['available_area'].values
            feats[l + '_est_cost'] = calc_establishment_cost(feats, feats[l + '_area'].values, horizon, past_est_cost)

            # Calculate biomass consumed (ton) = (grazed biomass (t/ha) * area (ha))
            # biomass_consumed = feats[l].values * feats['suitable'].values
            feats[l + '_BM'] = feats[l].values * feats[l + '_area']

            # Subset energy conversion table to keep grazing systems and ME to meat conversion column.
            # Climate coding: 1 = Temperate, 2 = Arid, 3 = Humid
            subset_table = energy_conversion.loc[energy_conversion.feed == 'grazing'][['group', 'climate', beef_yield,
                                                                                       'curr_methane', 'curr_manure']]

            # Calculate energy consumed ('000 MJ) = biomass consumed (t) * energy in grass (MJ/kg)
            energy = feats[l + '_BM'] * feats.merge(
                grass_energy, how='left', left_on=['region', 'climate'], right_on=['region', 'climate'])['ME'].values

            # Meat production (t) = energy consumed ('000 MJ) * energy conversion (kg/MJ) * dressing (%)
            meat = energy * feats[['group', 'climate']].merge(subset_table, how='left', left_on=['group', 'climate'],
                                                           right_on=['group', 'climate'])[beef_yield].values * dressing

            # Adjust meat prodution based on effective temperature
            monthly_beef = meat / 12.
            feats[l + '_meat'] = (feats.count_neg * monthly_beef - (feats.count_neg * monthly_beef * (
                    -0.0182 * feats.mean_neg - 0.0182))) + (monthly_beef * (12 - feats.count_neg))

            # Calculate methane production (ton CO2eq) = biomass consumed (t) * conversion factor (ton CO2eq/ton biomass)
            feats[l + '_meth'] = feats[l + '_BM'] * feats[['group', 'climate']].merge(
                subset_table, how='left', left_on=['group', 'climate'], right_on=['group', 'climate'])['curr_methane'].values

            # Calculate N2O from manure from energy consumed with coefficients (ton CO2eq) = biomass consumed (ton) * conversion factor (ton CO2eq/tom DM)
            feats[l + '_manure'] = feats[l + '_BM'] * feats[['group', 'climate']].merge(
                subset_table, how='left', left_on=['group', 'climate'], right_on=['group', 'climate'])['curr_manure'].values

            # Calculate fertiliser application in tons (0 for rangeland, assuming no N, P, K inputs)
            # Extract N application from column name, convert to ton
            n = int(l.split("_N")[1]) / 1000.

            if n == 0:
                n_applied = 0
                k_applied = 0
                p_applied = 0
            else:
                n_applied = int(l.split("_N")[1]) / 1000. * feats[l + '_area']

                k_applied = feats[l + '_area'] * feats[['nutrient_availability']].merge(
                    nutrient_req_grass, how='left')['K'].values * 2.2 / 1000.

                p_applied = feats[l + '_area'] * feats[['nutrient_availability']].merge(
                    nutrient_req_grass, how='left')['P'].values * 1.67 / 1000.

            # Get cost of fertilisers per country (USD/ton)
            fert_costs = feats[['ADM0_A3']].merge(fertiliser_prices, how='left')

            # Get total cost of fertilisers (USD) (N content in nitrate = 80%)
            feats[l + '_cost'] = n_applied * 1.2 * fert_costs['n'].values + k_applied * fert_costs[
                'k'].values + p_applied * fert_costs['p'].values
            logger.info('Landuse: {} minimum cost: {}'.format(l, feats[l + '_cost'].min()))
            # Calculate N20 emissions based on N application = N application (ton) * grass emission factor (%) * CO2 equivalency
            feats[l + '_n2o'] = (n_applied * grass_n2o_factor) * GWP_N2O

            # Number of trips to market; assuming 15 tons per trip, return
            ntrips = np.ceil(feats[l + '_meat'] / dressing / int(15)) * 2

            # Transport cost to market: number of trips * transport cost ('000 US$)
            trans_cost = (ntrips * feats[['ADM0_A3']].merge(
                fuel_cost[['ADM0_A3', 'Diesel']], how='left', left_on='ADM0_A3', right_on='ADM0_A3')['Diesel'].values * \
                                        feats["accessibility"] * speed *  fuel_efficiency + ntrips  * transport_wage_mkt) / 1000.

            # Transport emissions: number of trips * emissions per trip (tons CO2 eq)
            trans_emiss = ntrips * feats[
                "accessibility"] * speed * fuel_efficiency * truck_emission_factor / 1000.

            process_energy = feats[l + '_meat'].values * process_pack * \
                                        feats[['ADM0_A3']].merge(energy_efficiency, how='left')['energy'].fillna(
                                            0).values

            feats[l + '_postfarm_emi'] = process_energy + trans_emiss
            feats[l + '_postfarm_cost'] = trans_cost

            # Estimate carbon content as 47.5% of remaining grass biomass. Then convert to CO2 eq (*3.67)
            grazing_intensity = (1 - (int(l.split("_")[1]) / 1000.))
            # feats[l + '_cstock'] = 0.475 * (feats[l + '_BM'] / grazing_intensity * (1 - grazing_intensity)) * 3.67

            feats[l + '_cstock'] = np.where(feats['newarea'] == 1,
                                            feats['agb_spawn'] * 3.67 * feats[l + '_area'].values - \
                                            0.475 * (
                                                    (feats[l].values * feats[
                                                        l + '_area'].values) / grazing_intensity * (
                                                                1 - grazing_intensity)
                                            ) * 3.67,
                                            - 0.475 * ((feats[l].values * (feats[l + '_area'].values - feats[
                                                'current_grazing'])) / grazing_intensity * (
                                                                   1 - grazing_intensity)) * 3.67)

            # Change in soil carbon (t CO2 eq) = change from land use to grassland (%) * current land use area * current soil carbon (t/ha) * C-CO2 conversion * emission (-1)
            crop_to_grass = sc_change_exp.loc[sc_change_exp.new_cover == 'grassland', 'cropland'].iloc[0]
            tree_to_grass = sc_change_exp.loc[sc_change_exp.new_cover == 'grassland', 'tree'].iloc[0]

            weighted_area = feats['cell_area'].values

            bgb_change_new = ((crop_to_grass * (feats['crop'] * feats['cell_area']) * feats['bgb_spawn']) + (tree_to_grass * (feats['tree'] * feats['cell_area']) * feats[
                        'bgb_spawn'])) * 3.67 * -1 * (feats[l + '_area'] / weighted_area)

            # only consider current grazing/cropping area if looking at beef in current areas
            bgb_change_curr = (crop_to_grass * (feats['current_cropping']) * feats['bgb_spawn']) * 3.67 * -1

            bgb_change = np.where(feats.newarea == 1, bgb_change_new, bgb_change_curr)
            # Annualise change in soil carbon

            feats[l + '_bgb_change'] = eac(bgb_change, rate=0, type = 'ghg', lifespan=horizon)

            del bgb_change, grazing_intensity, ntrips, fert_costs, n_applied, k_applied, p_applied, monthly_beef, n,
            meat, energy, subset_table
            feats = weighted_score(feats, l, lam, horizon, logger, transport_wage_mkt, dressing)

        logger.info("Done with grass columns")

        for l in ['grass_grain']:
            # Keep 9 main fodder crops
            # fodder_potential_yields = potential_yields[['climate_bin'] + [c for c in foddercrop_list]]
            fodder_yield_fraction = yield_fraction[['ADM0_A3'] + foddercrop_list]
            # Potential production if all available area is converted to grain
            full_grain_prod = np.nansum(feats['available_area'].values[:, None] * \
                                        feats[['ADM0_A3']].merge(foddercrop_area, how="left").drop('ADM0_A3',
                                                                                                   axis=1).values *
                                        feats[foddercrop_list].values *
                                        # feats[['climate_bin']].merge(fodder_potential_yields, how="left").drop('climate_bin', axis=1).values *
                                        feats[['ADM0_A3']].merge(
                                            fodder_yield_fraction, how="left").drop('ADM0_A3', axis=1).values, axis=1)

            # Start area division with 80% grain, 20% grass
            cropping_prop_area = np.full_like(feats['available_area'].values, grain_max)
            grazing_prop_area = 1 - cropping_prop_area

            # Set grain production = grain prop area * 100% grain production
            grain_production = cropping_prop_area * full_grain_prod

            grasslu = np.nanargmin(np.ma.masked_array(feats[[g + '_score' for g in grass_cols]].values,
                                                      np.isnan(feats[[g + '_score' for g in
                                                                      grass_cols]].values)), axis=1)

            grasscol = np.take_along_axis(feats[[lu for lu in grass_cols]].values,
                                          grasslu[:, None], axis=1).flatten()



            # Set grass production = grazing proportion area * available area * grazing (t/ha)
            # grazing_production = grazing_prop_area * feats['available_area'].values * feats['grass_0500_N000'].values
            grazing_production = grazing_prop_area * feats['available_area'].values * grasscol

            # Total_production = grain + grazing
            total_production = grain_production + grazing_production

            # Proportion of grain vs total production
            grain_qty_prop = grain_production / total_production
            print(grain_qty_prop[grain_qty_prop > grain_max].shape[0])

            # logger.info(feats[['grass_0500_N000', 'available_area']])
            logger.info('Start grain area adjustment')
            pd.set_option('display.max_columns', 8)

            shape = grain_qty_prop[grain_qty_prop > grain_max].shape[0]

            while shape > 0:
                # df = pd.DataFrame({'Crop_area_frac': cropping_prop_area,
                #                    'Grass_area_frac': grazing_prop_area,
                #                    'grain_qty': grain_production,
                #                    'grass_qty': grazing_production,
                #                    'total': grain_production + grazing_production,
                #                    'grain_bm_prop': grain_qty_prop,
                #                    'grass_bm_prop': 1 - grain_qty_prop})
                # logger.info(df)
                # print(df)
                # If grain BM > 80% of total BM, Cropping frac  = Cropping frac - 0.1, else Cropping frac
                cropping_prop_area = np.where(grain_qty_prop > grain_max, cropping_prop_area - 0.01, cropping_prop_area)
                cropping_prop_area = np.where(cropping_prop_area < 0, 0, cropping_prop_area)

                # Updtate grazing area fraction = 1-cropping area fraction
                grazing_prop_area = 1 - cropping_prop_area

                grain_production = cropping_prop_area * full_grain_prod
                grazing_production = grazing_prop_area * feats['available_area'].values * grasscol
                grain_qty_prop = grain_production / (grain_production + grazing_production)
                shape = grain_qty_prop[grain_qty_prop > grain_max].shape[0]
                # logger.info('Number of cells with grain BM fraction exceeding 80% to toal BM: {}'.format(shape))

                # df = pd.DataFrame({'Crop_area_frac': cropping_prop_area,
                #                    'Grass_area_frac': grazing_prop_area,
                #                    'grain_qty': grain_production,
                #                    'grass_qty': grazing_production,
                #                    'total': grain_production + grazing_production,
                #                    'grain_bm_prop': grain_qty_prop,
                #                    'grass_bm_prop': 1 - grain_qty_prop})
                # logger.info(df)

            logger.info('End grain area adjustment')

            grain_area = feats['available_area'].values * cropping_prop_area
            grass_area = feats['available_area'].values * grazing_prop_area
            feats[l + '_est_cost'] = calc_establishment_cost(feats, grain_area, horizon, crop_est_cost) + calc_establishment_cost(feats, grass_area, horizon, past_est_cost)

            feats[l + '_area'] = grain_area + grass_area

            feats['grain_grassBM'] = grass_area * grasscol

            grain_prod = grain_area[:, None] * feats[['ADM0_A3']].merge(foddercrop_area, how="left").drop(
                'ADM0_A3', axis=1).values * feats[foddercrop_list].values * \
                         feats[['ADM0_A3']].merge(fodder_yield_fraction, how="left").drop(
                             'ADM0_A3', axis=1).values

            #          feats[['climate_bin']].merge(fodder_potential_yields, how="left").drop(
            # 'climate_bin', axis=1).values * \

            feats['grain_grainBM'] = np.nansum(grain_prod, axis=1)
            feats[l + '_BM'] = feats['grain_grainBM'] + feats['grain_grassBM']

            # Biomass consumed for domestic production (t) = actual production (t) x (1 - fraction exported feed)
            biomass_dom = grain_prod * (1 - feats[['ADM0_A3']].merge(percent_exported[['ADM0_A3'] + foddercrop_list],
                                                                     how="left").drop('ADM0_A3', axis=1).values)

            # Biomass consumed for domestic production (t) = actual production (t) x fraction exported feed
            biomass_exported = grain_prod * feats[['ADM0_A3']].merge(percent_exported[['ADM0_A3'] + foddercrop_list],
                                                                     how="left").drop('ADM0_A3', axis=1).values

            # Subset ME in conversion per region and climate
            subset_table = energy_conversion.loc[energy_conversion.feed == 'mixed'][['group', 'climate', beef_yield,
                                                                                     'curr_methane', 'curr_manure']]

            # Meat production (t) = sum across feeds (Domestic biomass (t) x ME in feed (MJ/kd DM)) x ME to beef conversion ratio * dressing (%)
            grass_meat = feats['grain_grassBM'].values * \
                         feats.merge(grass_energy, how='left', left_on=['region', 'climate'], right_on=['region', 'climate'])[
                             'ME'].values * \
                         feats[['group', 'climate']].merge(subset_table, how='left', left_on=['group', 'climate'],
                                                        right_on=['group', 'climate'])[beef_yield].values * dressing

            local_grain_meat = (np.nansum(biomass_dom * feed_energy[foddercrop_list].iloc[0].values[None, :], axis=1)) * \
                               feats[['group', 'climate']].merge(subset_table, how='left', left_on=['group', 'climate'],
                                                              right_on=['group', 'climate'])[beef_yield].values * dressing

            # Update meat production after climate penalty
            monthly_beef = (grass_meat + local_grain_meat) / 12.

            local_meat = (feats.count_neg * monthly_beef - (feats.count_neg * monthly_beef * (
                    -0.0182 * feats.mean_neg - 0.0182))) + (monthly_beef * (12 - feats.count_neg))

            # Calculate methane produced from local beef production (ton) = biomass consumed (ton) x biomass-methane conversion (ton/ton)
            local_methane = (np.nansum(biomass_dom, axis=1) + feats['grain_grassBM'].values) * \
                            feats[['group', 'climate']].merge(subset_table, how='left', left_on=['group', 'climate'],
                                                           right_on=['group', 'climate'])['curr_methane'].values

            # Calculate N2O from manure from energy consumed with coefficients (ton CO2eq) = biomass consumed (ton) * conversion factor (ton CO2eq/tom DM)
            local_manure = (np.nansum(biomass_dom, axis=1) + feats['grain_grassBM'].values) * \
                           feats[['group', 'climate']].merge(subset_table, how='left', left_on=['group', 'climate'],
                                                          right_on=['group', 'climate'])['curr_manure'].values

            # Calculate nitrous N2O (ton) = Actual production (ton) x fertiliser requirement (kg) x crop_emission factors (% per thousand)
            ###
            ymax = fertiliser_application.loc[fertiliser_application['crop'].isin(
                foddercrop_list), 'max_yield'].values

            k = fertiliser_application.loc[fertiliser_application['crop'].isin(
                foddercrop_list), 'n05'].values

            grain_n2o = np.nansum(N_app(grain_prod, k[None, :], ymax[None, :]) * (crop_emissions_factors.loc[crop_emissions_factors['crop'].isin(foddercrop_list), 'factor'].values[
                                              None, :] / 100), axis=1)

            grassarea2 = np.take_along_axis(feats[[lu + '_area' for lu in grass_cols]].values, grasslu[:, None],
                                            axis=1).flatten()

            grass_n2o = np.take_along_axis(feats[[lu + '_n2o' for lu in grass_cols]].values, grasslu[:, None],
                                           axis=1).flatten()

            feats[l + '_n2o'] = grain_n2o + (grass_n2o * (grass_area / grassarea2))
            logger.info("Done with local meat production")

            ##### Exported feed #####
            # Create empty arrays to fill in
            meat_abroad = np.zeros_like(feats.ADM0_A3, dtype = 'float32')
            methane_abroad = np.zeros_like(feats.ADM0_A3, dtype = 'float32')
            manure_abroad = np.zeros_like(feats.ADM0_A3, dtype = 'float32')
            exp_costs = np.zeros_like(feats.ADM0_A3, dtype = 'float32')
            sea_emissions_ls = np.zeros_like(feats.ADM0_A3, dtype = 'float32')
            emissions_partner_ls = np.zeros_like(feats.ADM0_A3, dtype = 'float32')
            trancost_partner_ls = np.zeros_like(feats.ADM0_A3, dtype = 'float32')

            for f in foddercrop_list:  # Loop though feeds
                ### Meat produced abroad
                # Quantity of feed f exported
                if feed_option == "v1":
                    # Qty exported (t) = Suitable area (ha) * crop area fraction * crop yield (t/ha) * yield gap (%) * export fraction
                    qty_exported = (((feats['suitable'].values *feats['cell_area'].values) * feats[['ADM0_A3']].merge(
                        foddercrop_area[['ADM0_A3', f + '_area']], how="left")[f + '_area'].values * feats[f].values *
                                     feats[['ADM0_A3']].merge(yield_fraction, how="left")[f].values)) * \
                                   feats[['ADM0_A3']].merge(percent_exported[['ADM0_A3', f]], how="left")[f].values
                    # feats[['climate_bin']].merge(potential_yields[['climate_bin', f]],
                    #                              how="left")[f].values * \

                if feed_option == "v2":
                    # Qty exported (t) = (Suitable area (ha) * crop area fraction * crop yield (t/ha) * yield gap (%)) - production for other uses (t) * export fraction
                    qty_exported = ((grain_area *
                                     feats[['ADM0_A3']].merge(foddercrop_area[['ADM0_A3', f + '_area']], how="left")[
                                         f + '_area'].values * feats[f].values *
                                     feats[['ADM0_A3']].merge(yield_fraction, how="left")[f].values)
                                       # - feats['diff_' + f].values
                                       ) * feats[['ADM0_A3']].merge(percent_exported[['ADM0_A3', f]],
                                                                    how="left")[f].values

                qty_exported = np.where(qty_exported < 0, 0, qty_exported)

                # trade partners
                trade_partners = feats[['ADM0_A3']].merge(feedpartners.loc[feedpartners.crop == f], how='left').drop(
                    ['ADM0_A3', 'crop'], axis=1).values

                # Meat produced from exported feed (t) = Exported feed (t) * partner fraction (%) * energy in feed ('000 MJ/t) * energy conversion in partner country (t/'000 MJ) * dressing (%)
                meat_abroad += np.nansum(
                    qty_exported[:, None] * trade_partners * feed_energy[f].iloc[0] * partner_me['meat'].values[None,
                                                                                      :],
                    axis=1) * dressing

                ### Methane emitted abroad (t CO2 eq) = Exported feed (t) * partner fraction (%) * methane emissions per biomass consumed (t/t)
                methane_abroad += np.nansum(
                    qty_exported[:, None] * trade_partners * partner_me["methane"].values[None, :], axis=1)

                ### N2O from manure emitted abroad (t CO2 eq) = Exported feed (t) * partner fraction (%) * N2O emissions per biomass consumed (t/t)
                manure_abroad += np.nansum(
                    qty_exported[:, None] * trade_partners * partner_me["manure"].values[None, :], axis=1)

                ### Export cost ('000 USD) = Exported feed (t) * partner fraction (%) * value of exporting crop c to partner p ('000 USD/t)
                exp_costs += np.nansum(qty_exported[:, None] * trade_partners * feats[['ADM0_A3']].merge(
                    expcosts.loc[expcosts.crop == f], how='left').drop(['ADM0_A3', 'crop'], axis=1).values, axis=1)

                ### Sea emissions (t CO2 eq) = Exported feed (t) * partner fraction (%) * sea distance from partner p (km) * sea emissions (kg CO2 eq/t-km) * kg-t conversion
                sea_emissions_ls += np.nansum(qty_exported[:, None] * trade_partners * feats[['ADM0_A3']].merge(
                    sea_dist, how='left').drop(['ADM0_A3'], axis=1).values * sea_emissions, axis=1) / 1000.

                ### Number of local transport cost in importing country
                ntrips_local_transp = qty_exported[:, None] * trade_partners / int(15) * 2

                ### Transport cost in partner country ('000 USD) = trips * accessibility to market in partner country (km) * fuel cost in partner country * fuel efficiency * USD-'000 USD conversion
                trancost_partner_ls += np.nansum(
                    ntrips_local_transp * exp_access['access'].values[None, :] * fuel_partner[
                                                                                     'Diesel'].values[None,
                                                                                 :] * speed *  fuel_efficiency / 1000., axis=1)

                ### Transport emissions in partner country (t CO2 eq) = trips * accessibility to market in partner country (km) *
                # fuel efficiency (l/km) * truck emission factor (kg CO2 eq/l) * kg-ton conversion
                emissions_partner_ls += np.nansum(
                    ntrips_local_transp * exp_access['access'].values[None,
                                          :] * speed * fuel_efficiency * truck_emission_factor / 1000., axis=1)
                logger.info("   Done with {}".format(f))

                ### Local transport emissions in importing country
            logger.info("Done looping through feeds")

            local_cost_grain = np.nan_to_num(np.nansum(
                biomass_dom * feats[['ADM0_A3']].merge(feedprices[['ADM0_A3'] + foddercrop_list], how="left").drop(
                    "ADM0_A3", axis=1).values, axis=1))

            grass_cost = np.nan_to_num(np.take_along_axis(feats[[lu + '_cost' for lu in grass_cols]].values, grasslu[:, None],
                                            axis=1).flatten())

            local_cost = local_cost_grain + (grass_cost * (np.nan_to_num(grass_area) / np.nan_to_num(grassarea2)))

            # Get price from trade database
            # Cost of producing feed to be exported

            # Number of trips to bring feed to port
            ntrips_feed_exp = np.nansum(biomass_exported, axis=1) / dressing / int(15) * 2
            ntrips_feed_exp = np.where(ntrips_feed_exp < 0, 0, ntrips_feed_exp)
            # Cost of sending feed to port
            feed_to_port_cost = (ntrips_feed_exp * feats["distance_port"] * \
                                 feats[['ADM0_A3']].merge(fuel_cost[['ADM0_A3', 'Diesel']],
                                                          how='left')[
                                     'Diesel'].values * speed * fuel_efficiency + ntrips_feed_exp * transport_wage_prt) / 1000.

            # Total cost of exporting feed
            # Emissions from transporting feed to nearest port (tons)
            feed_to_port_emis = ntrips_feed_exp * feats[
                'distance_port'] * speed * fuel_efficiency * truck_emission_factor / 1000.

            feats[l + '_meat'] = meat_abroad + local_meat


            feats['grain_grain_meat'] = local_grain_meat + meat_abroad
            feats['grain_grass_meat'] = grass_meat

            # Number of trips to markets
            ntrips_beef_mkt = feats[l + '_meat'].values / dressing / int(15) * 2
            ntrips_beef_mkt = np.where(ntrips_beef_mkt < 0, 0, ntrips_beef_mkt)

            beef_trans_cost = (ntrips_beef_mkt * feats[['ADM0_A3']].merge(fuel_cost[[
                'ADM0_A3', 'Diesel']], how='left', left_on='ADM0_A3', right_on='ADM0_A3')['Diesel'].values * \
                               feats["accessibility"] * speed * fuel_efficiency + ntrips_beef_mkt * transport_wage_mkt) / 1000.

            # Transport emissions: number of trips * emissions per trip (tons CO2 eq)
            beef_trans_emiss = ntrips_beef_mkt * feats[
                "accessibility"] * speed * fuel_efficiency * truck_emission_factor / 1000.
            logger.info("Done calculating costs and emissions")

            feats[l + '_meth'] = methane_abroad + local_methane
            feats[l + '_manure'] = manure_abroad + local_manure

            process_energy = feats[l + '_meat'].values * process_pack * \
                                        feats[['ADM0_A3']].merge(energy_efficiency, how='left')['energy'].fillna(
                                            0).values
            feats[l + '_cost'] = local_cost + feed_to_port_cost + exp_costs + trancost_partner_ls

            feats[l + '_postfarm_emi'] = beef_trans_emiss + feed_to_port_emis + sea_emissions_ls + emissions_partner_ls + process_energy
            feats[l + '_postfarm_cost'] = beef_trans_cost

            cstock_grain = np.where(feats['newarea'].values == 1,
                                    feats['agb_spawn'].values * 3.67 * feats[l + '_area'].values,
                                    0)

            cstock_grass = np.take_along_axis(feats[[lu + '_cstock' for lu in grass_cols]].values, grasslu[:, None],
                                              axis=1).flatten()

            # 0 C stock for grain. For grass, take cstock from the best grass land use and multiply by the fraction of area
            feats[l + '_cstock'] = cstock_grass * (grass_area / grassarea2) + cstock_grain


            grass_to_grain = sc_change_exp.loc[sc_change_exp.new_cover == 'cropland', 'grassland'].iloc[0]
            tree_to_grain = sc_change_exp.loc[sc_change_exp.new_cover == 'cropland', 'tree'].iloc[0]
            crop_to_grass = sc_change_exp.loc[sc_change_exp.new_cover == 'grassland', 'cropland'].iloc[0]
            tree_to_grass = sc_change_exp.loc[sc_change_exp.new_cover == 'grassland', 'tree'].iloc[0]

            # Calculate change in BGB for expansion areas
            bgb_change_grain_new = (((grass_to_grain * (feats['grass'] * feats['cell_area']) * feats['bgb_spawn']) + (
                    tree_to_grain * (feats['tree'] * feats['cell_area']) * feats[
                'bgb_spawn'])) * 3.67 * -1) * (grain_area / feats['cell_area'].values)

            bgb_change_grass_new = ((crop_to_grass * (feats['crop'] * feats['cell_area']) * feats['bgb_spawn']) + (
                    tree_to_grass * feats['tree'] * feats['cell_area'] * feats[
                'bgb_spawn'])) * 3.67 * -1 * (grass_area / weighted_area)


            # only consider current grazing/cropping area if looking at beef in current areas
            bgb_change_grain_curr = (grass_to_grain * (feats['current_grazing']) * feats['bgb_spawn']) * 3.67 * -1
            bgb_change_grass_curr = (crop_to_grass * (feats['current_cropping']) * feats['bgb_spawn']) * 3.67 * -1

            bgb_change_grain = np.where(feats.newarea == 1, bgb_change_grain_new, bgb_change_grain_curr)
            bgb_change_grass = np.where(feats.newarea == 1, bgb_change_grass_new, bgb_change_grass_curr)

            feats[l + '_bgb_change'] = eac(np.nan_to_num(bgb_change_grain) + np.nan_to_num(bgb_change_grass),
                                                      rate=0, type = 'ghg', lifespan=horizon)

            logger.info("Done writing cropland columns")

            feats = weighted_score(feats, l, lam, horizon, logger, transport_wage_mkt, dressing)

            feats['grain_grassBM'] = feats['grain_grassBM'].fillna(0)
            feats['grain_grainBM'] = feats['grain_grainBM'].fillna(0)

            feats[l + '_score'] = np.where((feats['grain_grassBM'] == 0) | (feats['grain_grainBM'] == 0),
                                           np.nan, feats[l + '_score'].values)

            del beef_trans_emiss, feed_to_port_emis, sea_emissions_ls, emissions_partner_ls, beef_trans_cost, \
                feed_to_port_cost, exp_costs, trancost_partner_ls, local_cost, manure_abroad, local_manure, \
                methane_abroad, local_methane, meat_abroad, local_meat, ntrips_beef_mkt, ntrips_feed_exp, meat, \
                biomass_dom, bgb_change_grass, bgb_change_grain, cstock_grass, grass_area, grassarea2, cstock_grain,\
                grass_cost, fodder_yield_fraction, full_grain_prod, cropping_prop_area, grazing_prop_area, \
                grain_production, grasslu, grasscol, grazing_production, total_production, grain_qty_prop, shape, \
                grain_area, grain_prod, biomass_exported, subset_table, grass_meat, local_grain_meat, monthly_beef, \
                grain_n2o,  grass_n2o

        for l in ['stover_grass']:
            grasslu = np.nanargmin(np.ma.masked_array(feats[[g + '_score' for g in grass_cols]].values,
                                                      np.isnan(feats[[g + '_score' for g in
                                                                      grass_cols]].values)), axis=1)

            feats['stover_grass_grassBM'] = np.take_along_axis(feats[[lu + '_BM' for lu in grass_cols]].values,
                                                               grasslu[:, None], axis=1).flatten()

            stover_fraction = feats[['region']].merge(grain_stover_compo[['region', 'stover']], how='left')['stover'].values
            stover_max = feats['stover_grass_grassBM'] * (stover_fraction / (1 - stover_fraction))

            fraction_total_stover = np.where(feats.newarea == 0,
                                                 ((feats.current_grazing + feats.current_cropping)/feats.cell_area),
                                                 (1-(feats.current_grazing + feats.current_cropping))/feats.cell_area)

            fraction_total_stover = np.select([fraction_total_stover > 1, fraction_total_stover < 0],
                                                       [1, 0], default = fraction_total_stover)

            potential_stover = fraction_total_stover * feats['stover_bm'].values * stover_removal

            # Adjust stover production based on what is needed
            # If potential is greater than maximum, adjust stover production by a ratio of max/potential
            # stov_adjustment = np.where(np.nansum(potential_stover, axis = 1) > stover_max, stover_max/np.nansum(potential_stover, axis = 1), 1.)
            stov_adjustment = np.where(potential_stover > stover_max, stover_max / potential_stover, 1.)

            # stover_production = potential_stover * stov_adjustment[:,None]

            stover_production = potential_stover * stov_adjustment

            # Stover energy ('000 MJ) = sum across rows of Stover production (t) * stover energy (MJ/kg dm)

            # feats['stover_grass_stoverBM'] = np.nansum(stover_production, axis=1)
            feats['stover_grass_stoverBM'] = stover_production

            feats[l + '_BM'] = feats['stover_grass_stoverBM'] + feats['stover_grass_grassBM']
            # Subset meat table for mixed systems
            subset_table = energy_conversion.loc[energy_conversion.feed == 'mixed'][['group', 'climate', beef_yield,
                                                                                     'curr_methane', 'curr_manure']]
            # Meat production (t) = Stover energy ('000 MJ) * energy converion (kg/MJ) * dressing percentage
            meat_stover = feats['stover_energy'].values * stov_adjustment * fraction_total_stover * stover_removal *\
                          feats[['group', 'climate']].merge(subset_table, how='left', left_on=['group', 'climate'],
                                                         right_on=['group', 'climate'])[beef_yield].values * dressing

            # Update meat production after climate penalty
            monthly_beef = meat_stover / 12.

            meat_stover = (feats.count_neg * monthly_beef - (feats.count_neg * monthly_beef * (
                    -0.0182 * feats.mean_neg - 0.0182))) + (monthly_beef * (12 - feats.count_neg))

            grass_meat = np.take_along_axis(feats[[lu + '_meat' for lu in grass_cols]].values,
                                            grasslu[:, None], axis=1).flatten()

            feats[l + '_meat'] = meat_stover + grass_meat

            feats['stover_grass_grass_meat'] = grass_meat
            feats['stover_grass_stover_meat'] = meat_stover

            # Methane emissions (t CO2 eq) = Biomass consumed (t) * CH4 emissions (t CO2 eq/t)
            stover_methane = stover_production * \
                             feats[['group', 'climate']].merge(subset_table, how='left', left_on=['group', 'climate'],
                                                            right_on=['group', 'climate'])['curr_methane'].values
            feats[l + '_meth'] = stover_methane + np.take_along_axis(feats[[lu + '_meth' for lu in grass_cols]].values,
                                                                     grasslu[:, None], axis=1).flatten()

            # Manure N20 (t CO2 eq) = Biomass consumed (t) * N2O emissions (t CO2 eq/t)
            stover_manure = stover_production * \
                            feats[['group', 'climate']].merge(subset_table, how='left', left_on=['group', 'climate'],
                                                           right_on=['group', 'climate'])['curr_manure'].values
            feats[l + '_manure'] = stover_manure + np.take_along_axis(
                feats[[lu + '_manure' for lu in grass_cols]].values,
                grasslu[:, None], axis=1).flatten()

            # Trips to market
            ntrips_beef_mkt = feats[l + '_meat'].values / dressing / int(15) * 2
            ntrips_beef_mkt = np.where(ntrips_beef_mkt < 0, 0, ntrips_beef_mkt)

            # Transport emissions = number of trips to nearest market * distance to market (km) * fuel efficeincy (l/km) * Diesel cost (USD/l)
            trans_cost = (ntrips_beef_mkt * feats["accessibility"] * speed * fuel_efficiency * feats[
                ['ADM0_A3']].merge(fuel_cost[['ADM0_A3', 'Diesel']], how='left', left_on='ADM0_A3',
                                   right_on='ADM0_A3')['Diesel'].values + ntrips_beef_mkt * transport_wage_mkt) / 1000.

            # Transport emissions = number of trips to nearest market * distance to market (km) * fuel efficeincy (l/km) * emissions factor (kg CO2/l) * kg/t conversion
            trans_emiss = ntrips_beef_mkt * feats[
                "accessibility"] * speed * fuel_efficiency * truck_emission_factor / 1000.

            process_energy = feats[l + '_meat'].values * process_pack * \
                                        feats[['ADM0_A3']].merge(energy_efficiency, how='left')['energy'].fillna(
                                            0).values

            feats[l + '_postfarm_emi'] = trans_emiss + process_energy
            feats[l + '_postfarm_cost'] = trans_cost

            for col in ['_cost', '_n2o', '_cstock', '_bgb_change']:
                feats[l + col] = np.nan_to_num(np.take_along_axis(feats[[glu + col for glu in grass_cols]].values,
                                                    grasslu[:, None], axis=1).flatten())

            # For grazing, convert all area
            feats[l + '_area'] = feats['available_area'].values
            feats[l + '_est_cost']  = calc_establishment_cost(feats, feats[l + '_area'].values, horizon, past_est_cost)

            feats = weighted_score(feats, l, lam, horizon, logger, transport_wage_mkt, dressing)

            feats['stover_grass_grassBM'] = feats['stover_grass_grassBM'].fillna(0)
            feats['stover_grass_stoverBM'] = feats['stover_grass_stoverBM'].fillna(0)

            feats[l + '_score'] = np.where((feats['stover_grass_grassBM'] == 0) | (feats['stover_grass_stoverBM'] == 0),
                                           np.nan, feats[l + '_score'].values)

        for l in ['stover_grain']:
            # fodder_potential_yields = potential_yields[['climate_bin'] + [c for c in foddercrop_list]]
            fodder_yield_fraction = yield_fraction[['ADM0_A3'] + foddercrop_list]

            #### Local feed consumption ####
            grain_production = feats['available_area'].values[:, None] * \
                               feats[['ADM0_A3']].merge(foddercrop_area, how="left").drop('ADM0_A3', axis=1).values * \
                               feats[foddercrop_list].values * \
                               feats[['ADM0_A3']].merge(fodder_yield_fraction, how="left").drop('ADM0_A3',
                                                                                                axis=1).values
            fraction_total_stover = np.where(feats.newarea == 0,
                                             ((feats.current_grazing + feats.current_cropping) / feats.cell_area),
                                             (1 - (feats.current_grazing + feats.current_cropping)) / feats.cell_area)

            fraction_total_stover = np.select([fraction_total_stover > 1, fraction_total_stover < 0],
                                              [1, 0], default=fraction_total_stover)

            potential_stover = fraction_total_stover * feats['stover_bm'].values * stover_removal

            # Adjust stover production based on what is needed
            # If potential is greater than maximum, adjust stover production by a ratio of max/potential
            # stov_adjustment = np.where(np.nansum(potential_stover, axis = 1) > stover_max, stover_max/np.nansum(potential_stover, axis = 1), 1.)
            stov_adjustment = np.where(potential_stover > stover_max, stover_max / potential_stover, 1.)

            stover_fraction = feats[['region']].merge(grain_stover_compo[['region', 'stover']], how='left')['stover'].values
            stover_max = np.nansum(grain_production, axis=1) * (stover_fraction / (1 - stover_fraction))

            # Adjust stover production based on what is needed
            # If potential is greater than maximum, adjust stover production by a ratio of max/potential
            # stov_adjustment = np.where(np.nansum(potential_stover, axis = 1) > stover_max, stover_max/np.nansum(potential_stover, axis = 1), 1.)
            stov_adjustment = np.where(potential_stover > stover_max, stover_max / potential_stover, 1.)

            # stover_production = potential_stover * stov_adjustment[:,None]

            stover_production = potential_stover * stov_adjustment
            # feats['stover_grain_stoverBM'] = np.nansum(stover_production, axis = 1)

            grain_max = stover_production * (grain_max / (1 - grain_max))

            grain_adjustment = np.where(np.nansum(grain_production, axis=1) > grain_max,
                                        grain_max / np.nansum(grain_production, axis=1), 1.)
            grain_production = grain_production * grain_adjustment[:, None]

            feats['stover_grain_grainBM'] = np.nansum(grain_production, axis=1)
            feats['stover_grain_stoverBM'] = stover_production
            feats[l + '_BM'] = feats['stover_grain_grainBM'] + feats['stover_grain_stoverBM']
            feats[l + '_area'] = feats['available_area'].values
            feats[l + '_est_cost'] = calc_establishment_cost(feats, feats[l + '_area'].values, horizon, crop_est_cost)

            # Biomass consumed for domestic production (t) = actual production (t) x (1 - fraction exported feed)
            # biomass_dom = total_prod * grain_perc * (
            #         1 - feats[['ADM0_A3']].merge(percent_exported, how="left").drop('ADM0_A3', axis=1).values)

            # # Biomass consumed for domestic production (t) = actual production (t) x fraction exported feed
            biomass_exported = grain_production * feats[['ADM0_A3']].merge(
                percent_exported[['ADM0_A3'] + foddercrop_list],
                how="left").drop('ADM0_A3', axis=1).values
            biomass_dom = grain_production * (
                        1 - feats[['ADM0_A3']].merge(percent_exported[['ADM0_A3'] + foddercrop_list],
                                                     how="left").drop('ADM0_A3', axis=1).values)
            # Subset ME in conversion per region and climate
            subset_table = energy_conversion.loc[energy_conversion.feed == 'mixed'][['group', 'climate', beef_yield,
                                                                                     'curr_methane', 'curr_manure']]

            # stover_energy = feats['stover_energy'].values * stov_adjustment

            # Meat production (t) = sum across feeds (Domestic biomass (t) x ME in feed (MJ/kd DM)) x ME to beef conversion ratio * dressing (%)
            stover_meat = feats['stover_energy'].values * stov_adjustment * fraction_total_stover * stover_removal * \
                          feats[['group', 'climate']].merge(subset_table, how='left', left_on=['group', 'climate'],
                                                         right_on=['group', 'climate'])[beef_yield].values * dressing

            local_grain_meat = np.nansum(biomass_dom * feed_energy[foddercrop_list].iloc[0].values[None, :], axis=1) * \
                               feats[['group', 'climate']].merge(subset_table, how='left', left_on=['group', 'climate'],
                                                              right_on=['group', 'climate'])[beef_yield].values * dressing
            # meat = (np.nansum(biomass_dom * feed_energy[foddercrop_list].iloc[0].values[None, :], axis=1) + (
            #     np.nansum(stover_production * residue_energy[foddercrop_list].iloc[0].values[None, :], axis=1))) * \
            #        feats[['group', 'glps']].merge(subset_table, how='left', left_on=['group', 'glps'],
            #                                       right_on=['group', 'glps'])[beef_yield].values * dressing
            local_grain_meat = np.where(local_grain_meat < 0, 0, local_grain_meat)
            stover_meat = np.where(stover_meat < 0, 0, stover_meat)

            # Update meat production after climate penalty
            monthly_beef = (stover_meat + local_grain_meat) / 12.

            local_meat = (feats.count_neg * monthly_beef - (feats.count_neg * monthly_beef * (
                    -0.0182 * feats.mean_neg - 0.0182))) + (monthly_beef * (12 - feats.count_neg))


            # Calculate methane produced from local beef production (ton) = biomass consumed (ton) x biomass-methane conversion (ton/ton)
            local_methane = (np.nansum(grain_production, axis=1) + stover_production) * \
                            feats[['group', 'climate']].merge(subset_table, how='left', left_on=['group', 'climate'],
                                                           right_on=['group', 'climate'])['curr_methane'].values

            # Calculate N2O from manure from energy consumed with coefficients (ton CO2eq) = biomass consumed (ton) * conversion factor (ton CO2eq/tom DM)
            local_manure = (np.nansum(grain_production, axis=1) + stover_production) * \
                           feats[['group', 'climate']].merge(subset_table, how='left', left_on=['group', 'climate'],
                                                          right_on=['group', 'climate'])['curr_manure'].values

            # Calculate nitrous N2O (ton) = Actual production (ton) x fertiliser requirement (kg) x crop_emission factors (% per thousand)
            # feats[l + '_n2o'] = np.nansum(grain_production * fertiliser_requirement['fertiliser'].values[None, :] * (
            #         crop_emissions_factors['factor'].values[None, :] / 100), axis=1)

            ymax = fertiliser_application.loc[fertiliser_application['crop'].isin(
                foddercrop_list), 'max_yield'].values

            k = fertiliser_application.loc[fertiliser_application['crop'].isin(
                foddercrop_list), 'n05'].values

            feats[l + '_n2o'] = np.nansum(N_app(grain_production, k[None, :], ymax[None, :]) * (crop_emissions_factors.loc[
                                                                                      crop_emissions_factors[
                                                                                          'crop'].isin(
                                                                                          foddercrop_list), 'factor'].values[
                                                                                  None, :] / 100), axis=1)

            # feats[l + '_n2o'] = np.nansum(grain_production * fertiliser_requirement.loc[fertiliser_requirement['crop'].isin(
            #     foddercrop_list), 'fertiliser'].values[None, :] * (
            #                                       crop_emissions_factors.loc[crop_emissions_factors['crop'].isin(
            #                                           foddercrop_list), 'factor'].values[None, :] / 100.), axis=1)

            # feats[l + '_n2o'] = np.nansum(feats[l + '_area'].values[:, None] * feats[['ADM0_A3']].merge(foddercrop_area, how="left").drop(
            #     'ADM0_A3', axis=1).values * feats[['ADM0_A3']].merge(crop_emissions[['ADM0_A3'] + foddercrop_list], how="left").drop('ADM0_A3', axis=1).values,
            #           axis = 1)

            logger.info("Done with local meat production")

            ##### Exported feed #####
            # Create empty arrays to fill in
            meat_abroad = np.zeros_like(feats.ADM0_A3, dtype = 'float32')
            methane_abroad = np.zeros_like(feats.ADM0_A3, dtype = 'float32')
            manure_abroad = np.zeros_like(feats.ADM0_A3, dtype = 'float32')
            exp_costs = np.zeros_like(feats.ADM0_A3, dtype = 'float32')
            sea_emissions_ls = np.zeros_like(feats.ADM0_A3, dtype = 'float32')
            emissions_partner_ls = np.zeros_like(feats.ADM0_A3, dtype = 'float32')
            trancost_partner_ls = np.zeros_like(feats.ADM0_A3, dtype = 'float32')

            for f in foddercrop_list:  # Loop though feeds
                ### Meat produced abroad
                # Quantity of feed f exported
                if feed_option == "v1":
                    # Qty exported (t) = Suitable area (ha) * crop area fraction * crop yield (t/ha) * yield gap (%) * export fraction
                    qty_exported = ((feats[l + '_area'].values * feats[['ADM0_A3']].merge(
                        foddercrop_area[['ADM0_A3', f + '_area']], how="left")[f + '_area'].values * \
                                     # feats[['climate_bin']].merge(fodder_potential_yields[['climate_bin', f]],
                                     #                              how="left")[f + '_potential'].values * \
                                     feats[f].values * feats[['ADM0_A3']].merge(fodder_yield_fraction, how="left")[
                                         f].values)) * \
                                   feats[['ADM0_A3']].merge([['ADM0_A3', f]], how="left")[f].values * grain_adjustment

                if feed_option == "v2":
                    # Qty exported (t) = (Suitable area (ha) * crop area fraction * crop yield (t/ha) * yield gap (%)) - production for other uses (t) * export fraction
                    qty_exported = ((feats[l + '_area'].values * \
                                     feats[['ADM0_A3']].merge(foddercrop_area[['ADM0_A3', f + '_area']], how="left")[
                                         f + '_area'].values * feats[f].values *
                                     feats[['ADM0_A3']].merge(fodder_yield_fraction, how="left")[f].values)

                                       # feats[['climate_bin']].merge(fodder_potential_yields[['climate_bin', f]],
                                       #                              how="left")[f].values * \
                                       # - feats['diff_' + f].values
                                       ) * feats[['ADM0_A3']].merge(percent_exported[['ADM0_A3', f]], how="left")[
                                       f].values * grain_adjustment

                qty_exported = np.where(qty_exported < 0, 0, qty_exported)

                # trade partners
                trade_partners = feats[['ADM0_A3']].merge(feedpartners.loc[feedpartners.crop == f], how='left').drop(
                    ['ADM0_A3', 'crop'], axis=1).values

                # Meat produced from exported feed (t) = Exported feed (t) * partner fraction (%) * energy in feed ('000 MJ/t) * energy conversion in partner country (t/'000 MJ) * dressing (%)
                meat_abroad += np.nansum(
                    qty_exported[:, None] * trade_partners * feed_energy[f].iloc[0] * partner_me['meat'].values[None,
                                                                                      :], axis=1) * dressing

                ### Methane emitted abroad (t CO2 eq) = Exported feed (t) * partner fraction (%) * methane emissions per biomass consumed (t/t)
                methane_abroad += np.nansum(
                    qty_exported[:, None] * trade_partners * partner_me["methane"].values[None, :], axis=1)

                ### N2O from manure emitted abroad (t CO2 eq) = Exported feed (t) * partner fraction (%) * N2O emissions per biomass consumed (t/t)
                manure_abroad += np.nansum(
                    qty_exported[:, None] * trade_partners * partner_me["manure"].values[None, :], axis=1)

                ### Export cost ('000 USD) = Exported feed (t) * partner fraction (%) * value of exporting crop c to partner p ('000 USD/t)
                exp_costs += np.nansum(qty_exported[:, None] * trade_partners * feats[['ADM0_A3']].merge(
                    expcosts.loc[expcosts.crop == f], how='left').drop(['ADM0_A3', 'crop'], axis=1).values, axis=1)

                ### Sea emissions (t CO2 eq) = Exported feed (t) * partner fraction (%) * sea distance from partner p (km) * sea emissions (kg CO2 eq/t-km) * kg-t conversion
                sea_emissions_ls += np.nansum(qty_exported[:, None] * trade_partners * feats[['ADM0_A3']].merge(
                    sea_dist, how='left').drop(['ADM0_A3'], axis=1).values * sea_emissions, axis=1) / 1000.

                ### Number of local transport cost in importing country
                ntrips_local_transp = qty_exported[:, None] * trade_partners / int(15) * 2

                ### Transport cost in partner country ('000 USD) = trips * accessibility to market in partner country (km) * fuel cost in partner country * fuel efficiency * USD-'000 USD conversion
                trancost_partner_ls += np.nansum(
                    ntrips_local_transp * exp_access['access'].values[None, :] * fuel_partner[
                                                                                     'Diesel'].values[None,
                                                                                 :] * speed * fuel_efficiency / 1000., axis=1)

                ### Transport emissions in partner country (t CO2 eq) = trips * accessibility to market in partner country (km) *
                # fuel efficiency (l/km) * truck emission factor (kg CO2 eq/l) * kg-ton conversion
                emissions_partner_ls += np.nansum(
                    ntrips_local_transp * exp_access['access'].values[None,
                                          :] * speed * fuel_efficiency * truck_emission_factor / 1000., axis=1)
                logger.info("   Done with {}".format(f))

                ### Local transport emissions in importing country
            logger.info("Done looping through feeds")

            local_cost = np.nansum(
                grain_production * feats[['ADM0_A3']].merge(feedprices[['ADM0_A3'] + foddercrop_list], how="left").drop(
                    "ADM0_A3", axis=1).values, axis=1)

            # Number of trips to markets
            ntrips_beef_mkt = local_meat / dressing / int(15) * 2
            ntrips_beef_mkt = np.where(ntrips_beef_mkt < 0, 0, ntrips_beef_mkt)

            beef_trans_cost = (ntrips_beef_mkt * feats[['ADM0_A3']].merge(fuel_cost[['ADM0_A3', 'Diesel']],
                                                                          how='left')['Diesel'].values * \
                               feats["accessibility"] * speed * fuel_efficiency + ntrips_beef_mkt * transport_wage_mkt) / 1000.

            # Transport emissions: number of trips * emissions per trip (tons CO2 eq)
            beef_trans_emiss = ntrips_beef_mkt * feats[
                "accessibility"] * speed * fuel_efficiency * truck_emission_factor / 1000.
            logger.info("Done calculating costs and emissions")

            feats[l + '_meat'] = local_meat + meat_abroad

            feats['stover_grain_grain_meat'] = meat_abroad + local_grain_meat
            feats['stover_grain_stover_meat'] = stover_meat

            feats[l + '_meth'] = local_methane + methane_abroad
            feats[l + '_manure'] = local_manure + manure_abroad

            # Number of trips to bring feed to port
            ntrips_feed_exp = np.nansum(biomass_exported, axis=1) / dressing / int(15) * 2
            ntrips_feed_exp = np.where(ntrips_feed_exp < 0, 0, ntrips_feed_exp)
            # Cost of sending feed to port
            feed_to_port_cost = (ntrips_feed_exp * feats["distance_port"] * \
                                 feats[['ADM0_A3']].merge(fuel_cost[['ADM0_A3', 'Diesel']],
                                                          how='left')[
                                     'Diesel'].values * speed * fuel_efficiency + ntrips_feed_exp * transport_wage_prt) / 1000.

            # Total cost of exporting feed
            # Emissions from transporting feed to nearest port (tons)
            feed_to_port_emis = ntrips_feed_exp * feats[
                'distance_port'] * speed * fuel_efficiency * truck_emission_factor / 1000.

            # feats[l + '_trans_cost'] = beef_trans_cost + feed_to_port_cost + exp_costs + trancost_partner_ls
            # feats[l + '_trans_emiss'] = beef_trans_emiss + feed_to_port_emis + sea_emissions_ls + emissions_partner_ls

            process_energy = feats[l + '_meat'].values * process_pack * \
                                        feats[['ADM0_A3']].merge(energy_efficiency, how='left')['energy'].fillna(
                                            0).values
            feats[l + '_cost'] = local_cost + feed_to_port_cost + exp_costs + trancost_partner_ls
            feats[l + '_postfarm_cost'] = beef_trans_cost 
            feats[l + '_postfarm_emi'] = beef_trans_emiss + feed_to_port_emis + sea_emissions_ls + emissions_partner_ls + process_energy

            feats[l + '_cstock'] = np.where(feats['newarea'].values == 1,
                                            feats['agb_spawn'].values * 3.67 * feats[l + '_area'].values,
                                            0)

            grass_to_grain = sc_change_exp.loc[sc_change_exp.new_cover == 'cropland', 'grassland'].iloc[0]
            tree_to_grain = sc_change_exp.loc[sc_change_exp.new_cover == 'cropland', 'tree'].iloc[0]


            weighted_area = feats['cell_area'].values

            bgb_change_grain_new  = (((grass_to_grain * (feats['grass'] * feats['cell_area']) * feats['bgb_spawn']) + (
                    tree_to_grain * (feats['tree'] * feats['cell_area'])* feats['bgb_spawn'])) * 3.67 * -1) * (feats[l + '_area'] / \
                                                                                      feats['cell_area'].values)

            # only consider current grazing/cropping area if looking at beef in current areas
            bgb_change_grain_curr = (grass_to_grain * (feats['current_grazing']) * feats['bgb_spawn']) * 3.67 * -1

            bgb_change_grain = np.where(feats.newarea == 1, bgb_change_grain_new, bgb_change_grain_curr)

            # Annualise change in soil carbon
            feats[l + '_bgb_change'] = eac(bgb_change_grain, rate=0, type = 'ghg', lifespan=horizon)

            feats['stover_grain_grainBM'] = feats['stover_grain_grainBM'].fillna(0)
            feats['stover_grain_stoverBM'] = feats['stover_grain_stoverBM'].fillna(0)

            feats = weighted_score(feats, l, lam, horizon, logger, transport_wage_mkt, dressing)

            feats[l + '_score'] = np.where((feats['stover_grain_grainBM'] == 0) | (feats['stover_grain_stoverBM'] == 0),
                                           np.nan, feats[l + '_score'].values)
    else:
        logger.info('Feed option {} not in choices'.format(feed_option))
    # Drop monthly temperature and other crop uses columns

    # Do not consider production lower than 1 ton
    # feats[[l + '_meat' for l in landuses]] = np.where(feats[[l + '_meat' for l in landuses]] < 0.1, 0, feats[[l + '_meat' for l in landuses]])

    # Only keep cells where at least 1 feed option produces meat
    # logger.info('Grid shape before sum: {}, current meat {}'.format(feats.shape[0], feats.c_meat.sum()))
    # feats = feats.loc[feats[[l + '_meat' for l in landuses]].sum(axis=1) > 0]
    # logger.info('Grid shape after sum: {}, current meat {}'.format(feats.shape[0], feats.c_meat.sum()))

    # Drop rows where no land use has a score (all NAs)

    feats['c_ghg'] = np.nan_to_num(feats.c_ghg) - np.nan_to_num(feats.opp_aff) - np.nan_to_num(feats.opp_soc)
    feats['c_tot_cost'] = np.nan_to_num(feats.c_tot_cost) - np.nan_to_num(feats.aff_cost)

    rel_c_ghg = feats['c_ghg'].values / feats['c_meat'].values
    rel_c_cost = feats['c_tot_cost'].values / feats['c_meat'].values

    feats['c_score'] = np.where(feats.newarea == 0, (rel_c_ghg * (1 - lam)) + (rel_c_cost * lam), np.nan)
    # feats['c_score'] = np.where(feats.newarea == 0,
    #                             ((feats['c_ghg'].values / feats['c_meat'].values) * (1 - lam)) + ((feats['c_tot_cost'].values/feats['c_meat'].values) * lam),
    #                             np.nan)

    logger.info('Grid shape before dropna: {}, current meat {}'.format(feats.shape[0], feats.c_meat.sum()))

    feats = feats.dropna(how='all', subset=[l + '_score' for l in landuses])

    logger.info('Grid shape after dropna: {}, current meat {}'.format(feats.shape[0], feats.c_meat.sum()))

    # Select lowest score
    feats['best_score'] = np.nanmin(feats[[l + '_score' for l in landuses]].values, axis=1)

    try:
        # Select position (land use) of lowest score
        feats['bestlu'] = np.nanargmin(feats[[l + '_score' for l in landuses]].values, axis=1)
    except:
        # If there is no best land use, export dataframe
        feats.loc[feats.best_score.isna()].drop('geometry', axis=1).to_csv("nadf.csv", index=False)

    # del list_scores, allArrays

    # Create a new column for all variables in new_colnames that selects the value of the optimal land use
    # for i in new_colnames:
    #     feats[i] = np.take_along_axis(feats[[l + new_colnames[i] for l in landuses]].values,
    #                                   feats['bestlu'].values[:, None], axis=1)
    for cname in ['production']:
        feats[cname] = np.take_along_axis(feats[[lu + new_colnames[cname] for lu in landuses]].values,
                                          feats['bestlu'].values[:, None], axis=1).flatten()
    # print('-> Production for cell id {}: {}'.format(4394871, feats.loc[feats.cell_id == 4394871][['production']]))

    return feats

def trade(feats, lam, landuses):
    """
    Function to update score based on trade costs and emissions

    Arguments:
    feats (pandas dataframe) -> Main dataframe
    lam (float)-> Lambda weight ([0,1])
    feed_option (str)-> folder where the output file is exported

    Output: returns a gridded dataframe with updated score
    """
    # if feed_option == 'v3':
    #     landuses = ['grazing', 'mixed']
    dressing = feats.merge(dressing_table, how='left', left_on='region', right_on='region')['dressing'].values

    transport_wage_prt = (feats["distance_port"].values / 60.) * feats[['ADM0_A3']].merge(wages, how='left')[
        'wage'].values

    for l in landuses:
        if l != 'c':
            # Calculate transport trips to export meat
            ntrips = (feats[l + '_meat'] / dressing / int(15) + 1) * 2

            # Calculate transport cost to nearest port
            trans_cost = (ntrips * feats["distance_port"] * \
                                        feats[['ADM0_A3']].merge(fuel_cost[['ADM0_A3', 'Diesel']],
                                                                 how='left', left_on='ADM0_A3',
                                                                 right_on='ADM0_A3')[
                                            'Diesel'].values * speed * fuel_efficiency + ntrips * transport_wage_prt) / 1000.

            # Calculate transport costs as a function of quantity traded
            exp_costs = feats[l + '_meat'] * feats[['ADM0_A3']].merge(sea_t_costs[['ADM0_A3', 'tcost']],
                                                                                    how='left')['tcost'].values

            # Transport emissions to port
            trans_emiss = ntrips * feats[
                "distance_port"] * speed * fuel_efficiency * truck_emission_factor / 1000.

            # Transport emissions by sea
            exp_emiss = feats[['ADM0_A3']].merge(sea_distances[['ADM0_A3', 'ave_distance']], how='left')[
                                          'ave_distance'].values * feats[l + '_meat'] * sea_emissions / 1000.

            process_energy = feats[l + '_meat'].values * process_pack * \
                                        feats[['ADM0_A3']].merge(energy_efficiency, how='left')['energy'].fillna(
                                            0).values

            feats[l + '_postfarm_cost'] = trans_cost + exp_costs
            feats[l + '_postfarm_emi'] = trans_emiss + exp_emiss + process_energy
            # Update total cost ('000 USD)
            cost_cols = ['_est_cost', '_cost', '_opp_cost', '_postfarm_cost']
            feats[l + '_tot_cost'] = np.nansum(
                feats[[l + c for c in cost_cols]].values, axis = 1, dtype = 'float64') -  np.nan_to_num(feats.aff_cost)

            # feats[l + '_exp_costs'] + feats[l + '_trans_cost']+ feats[l + '_compensation'])

            # Update annual emissions (t CO2 eq)
            emissions_cols = ['_n2o', '_meth', '_manure', '_postfarm_emi', '_agb_change', '_bgb_change']

                   # feats[l + '_trans_emiss'] + feats[l + '_exp_emiss'] + feats[l + '_process_energy']
            feats[l + '_ghg'] = np.nansum(
                feats[[l + c for c in emissions_cols]].values, axis = 1, dtype = 'float64') - np.nansum(
                feats[['opp_aff', 'opp_soc']].values, axis = 1, dtype = 'float64')

    # Drop rows where columns are all nas
    feats = feats.dropna(how='all', subset=[lu + '_score' for lu in landuses])

    # make sure that dataframe is not empty
    if feats.shape[0] > 0:

        # rel_ghg = np.where(feats[l + '_meat'] < 1, np.NaN, feats[l + '_ghg'] / (feats[l + '_meat']))
        # # Calculate relative Cost (GHG/meat)
        # rel_cost = np.where(feats[l + '_meat'] < 1, np.NaN,
        #                     feats[l + '_tot_cost'] / (feats[l + '_meat']))

        # feats['c_score'] = ((feats['c_ghg'].values / feats['c_meat'].values) * (1 - lam)) + (
        #         (feats['c_tot_cost'].values / feats['c_meat'].values) * lam)

        # Select lowest score
        feats['best_score'] = np.nanmin(feats[[l + '_score' for l in landuses]].values, axis=1)

        try:
            # Select position (land use) of lowest score
            feats['bestlu'] = np.nanargmin(feats[[l + '_score' for l in landuses]].values, axis=1)
        except:
            # If there is no best land use, export dataframe
            feats.loc[feats.best_score.isna()].drop('geometry', axis=1).to_csv("nadf.csv", index=False)

        # del list_scores, allArrays

        # Create a new column for all variables in new_colnames that selects the value of the optimal land use
        # for i in new_colnames:
        #     feats[i] = np.take_along_axis(feats[[l + new_colnames[i] for l in landuses]].values,
        #                                   feats['bestlu'].values[:, None], axis=1)
        for cname in ['production']:
            feats[cname] = np.take_along_axis(feats[[lu + new_colnames[cname] for lu in landuses]].values,
                                              feats['bestlu'].values[:, None], axis=1).flatten()

        return feats

def export_raster(grid, export_column, dst, spat_constraint, lam, aff_scenario):
    """
    Function to rasterize columns of a dataframe

    Arguments:
    grid (pandas dataframe)-> Dataframe to rasterize
    resolution (float)-> Resolution at which to rasterize
    export_column (list)-> list of columns to rasterize

    dst (str)-> folder where the output file is exported
    spat_constraint (str)-> Spatial constraint for beef production ('global', 'country', or 'subsistence')
    crop_yield (int)-> Scenario of crop yield (0 = current, 1 = no yield gap)
    beef_yield (str)-> Scenario of beef yield ('me_to_meat' = current, 'max_yield' = no yield gap)
    lam (float)-> Lambda weight ([0,1])
    demand_scenario (str)-> Scenario of beef demand ('SSP1-NoCC2010' or 'SSP1-NoCC2050')
    feed_option (str)-> folder where the output file is exported

    Output: Writes the grid as GPKG file
    """
    with rasterio.open('./rasters/grass.tif') as f:
        meta = f.meta
    meta.update(compress='lzw',
                nodata = 0)
    iddata = pd.DataFrame({'cell_id': range(meta['width'] * meta['height'])})

    for c in export_column:
        if c == 'cell_id':
            arr = np.asarray(iddata['cell_id'].values, dtype = 'int32')
        else:
            arr = iddata.merge(grid[['cell_id',  c]], how = 'left', left_on = 'cell_id', right_on = 'cell_id')[c].values

        # print("Array shape: {}".format(arr.shape))
        # print("height {}, width: {}".format(meta['height'], meta['width']))

        rast = arr.reshape(meta['height'], meta['width'])

        meta['dtype'] = arr.dtype
        out_fn = dst + '/' + c + "_" + spat_constraint + "_" + str(lam) + '_' + aff_scenario + ".tif"
        with rasterio.open(out_fn, 'w', **meta) as out:
            out.write_band(1, rast)

def main(grid, foddercrop_list, producer_prices, fertiliser_prices, energy_conversion, sc_change_opp, sc_change_exp,
         grain_max, stover_removal, horizon, lam=0.5, spat_constraint='global', aff_scenario="noaff", scope = None,
         logger = 'logger', scenario_id = 1, profit_margin_method = 0, dst = '.', simulation='main', trade_scenario='trade'):
    """
    Main function that optimises beef production for a given location and resolution, using a given number of cores.

    Arguments:
    dst (str)-> folder where the output file is exported
    lam (float)-> Lambda weight ([0,1])
    demand_scenario (str)-> Scenario of beef demand ('SSP1-NoCC2010' or 'SSP1-NoCC2050')
    crop_yield (int)-> Scenario of crop yield (0 = current, 1 = no yield gap)
    beef_yield (str)-> Scenario of beef yield ('me_to_meat' = current, 'max_yield' = no yield gap)
    spat_constraint (str)-> Spatial constraint for beef production ('global', 'country', or 'subsistence')
    feed_option (str)-> folder where the output file is exported
    trade_scenario (str)-> Trade scenario (if 'trade', apply trade based on country demand)

    Output: Writes the grid as GPKG file
    """

    # LOG_FORMAT = "%(asctime)s - %(message)s"
    #
    # logging.basicConfig(
    #     filename="/home/uqachare/model_file/whole_inf_{}_{}.log".format(scenario_id, lam),
    #     level=logging.INFO,
    #     format=LOG_FORMAT,
    #     filemode='w')
    #
    # logger = logging.getLogger('second logger')

    feed_option = 'v2'
    # horizon = 30
    production_yr = '2018'
    demand_scenario='SSP1-NoCC2010'
    crop_yield=0
    beef_yield = 'curr'

    grid['stover_bm'] = grid['stover_bm'].values * stover_removal

    if scope:
        ctry = scope
    else:
        ctry = 'global'

    logger.info("Start loading grid")
    logger.info('Current meat: {}'.format(grid.c_meat.sum()))
    logger.info('Current grid shape: {}'.format(grid.shape[0]))

    # for c in grid.columns:
    #     logger.info("   Min of {}".format(c))
    #     if c == 'region':
    #         logger.info(grid[c].dtype)
    #         logger.info(grid.region.unique())
    #
    #     logger.info("   Min of {}: {}".format(c,  np.nanmin(grid[c].values)))

    logger.info('Initial shape: {}'.format(grid.shape[0]))
    logger.info("Scope: {}".format(scope))

    logger.info("Simulation start")
    # logger.info('Me_to_meat scanerio: {}'.format(beef_yield))
    logger.info('Weight: {}'.format(lam))
    # logger.info('Feed option scenario: {}'.format(feed_option))
    logger.info('Constraint: {}'.format(spat_constraint))
    logger.info('Aff scenario: {}'.format(aff_scenario))


    # Set amount of beef to be produced based on the chosen location

    if spat_constraint == 'subsistence':
        logger.info('Shape of all grid: {}'.format(grid.shape))
        pastoral = grid.loc[grid.beef_gs > 0]
        logger.info('Shape of pastoral grid: {}'.format(pastoral.shape))
        grid = grid.loc[~(grid.beef_gs > 0)]
        logger.info('Shape of grid without pastoral: {}'.format(grid.shape))

        pastoral['production'] = pastoral['beef_gs'] * 0.01 * pastoral.suitable * 1e-3

        demand = beef_demand[demand_scenario].sum()
        logger.info('Total demand: {}'.format(demand))
        demand = demand - pastoral['production'].sum()
        logger.info('Non-pastoral demand: {}'.format(demand))
        logger.info('Pastoral production: {}'.format(pastoral['production'].sum()))
    else:
        if scope:
            demand = beef_demand.loc[beef_demand.ADM0_A3 == scope, demand_scenario].sum()
        else:
            demand = beef_demand[demand_scenario].sum()

        # demand = 2000000
        # demand = 2.899584e+05

    logger.info('Demand: {}'.format(demand))

    # Adjust other uses for future demand  Proportion of demand increase
    beef_demand['dem_increase'] = beef_demand[demand_scenario] / beef_demand['SSP1-NoCC2010']

    # logger.info('New demand for other uses before: {}'.format(grid[['diff_maize']].loc[grid.diff_maize > 0].head()))
    # other_uses = grid[['ADM0_A3']+['diff_' + i for i in crop_list]].merge(beef_demand[['ADM0_A3', 'dem_increase']])

    other_uses = grid[['ADM0_A3']].merge(beef_demand[['ADM0_A3', 'dem_increase']], how='left')['dem_increase'].values
    # grid[['diff_' + i for i in crop_list]] = grid[['diff_' + i for i in crop_list]].values * other_uses[:, None]
    grid['net_fodder_area'] = grid['net_fodder_area'].values * other_uses
    grid['stover_bm'] = grid['stover_bm'].values * other_uses
    grid['stover_energy'] = grid['stover_energy'].values * other_uses

    del other_uses

    landuses = grass_cols + ['grass_grain', 'stover_grass', 'stover_grain', 'c']  # landuses to include in the simulation

    grid['newarea'] = np.ones(shape = grid.shape[0], dtype = 'uint8')

    current = grid.loc[grid.c_meat > 0]

    current['newarea'] = np.zeros(shape = current.shape[0], dtype = 'uint8')

    logger.info('Current grid shape: {}, new grid shape: {}'.format(grid.shape[0], current.shape[0]))
    logger.info('Grid shape before concat: {}'.format(grid.shape[0]))
    grid.loc[grid['newarea'] == 1, ['c_meat', 'c_ghg', 'c_tot_cost']] = 0

    grid = pd.concat([grid, current])
    grid['id'] = range(0, grid.shape[0])

    logger.info('Grid shape after concat: {}'.format(grid.shape[0]))
    logger.info("Current meat before scoring: {}".format(grid['c_meat'].sum()))

    logger.info("sc_change_opp: {}".format(sc_change_opp))

    grid = opportunity_cost_carbon(grid, sc_change_opp, aff_scenario, logger, horizon, lam)

    grid = scoring(grid, foddercrop_list, producer_prices, fertiliser_prices, energy_conversion, sc_change_opp,
                   sc_change_exp, grain_max, stover_removal, crop_yield, lam, beef_yield, aff_scenario, logger, feed_option,
                   landuses, horizon)

    logger.info('Current meat: {}'.format(grid.c_meat.sum()))
    logger.info('Current grid shape: {}'.format(grid.shape[0]))

    logger.info("Current meat after scoring: {}".format(grid['c_meat'].sum()))
    logger.info('Grid shape after scoring: {}'.format(grid.shape[0]))
    logger.info("Done scoring")

    grid = grid.reset_index(drop=True)
    total_production = 0

    grid['changed'] = np.zeros_like(grid.ADM0_A3, dtype='int8')
    grid['destination'] = np.zeros_like(grid.ADM0_A3, dtype='int8')
    grid['exporting'] = np.zeros_like(grid.ADM0_A3, dtype='int8')

    # Get country-level domestic demand
    grid['dom_demand'] = grid[['ADM0_A3']].merge(beef_demand, how='left', left_on='ADM0_A3', right_on='ADM0_A3')['Demand'].values

    # grid['dom_production'] = grid[['ADM0_A3']].merge(
    #     beef_production, how='left', left_on='ADM0_A3', right_on=['ADM0_A3'])['curr_beef_meat'].values

    # Sort rows by increasing 'best score'
    grid = grid.sort_values('best_score')
    # Get cumulative country level production in order of increasing best score
    grid.loc[grid.changed == 0, 'cumdomprod'] = grid.groupby('ADM0_A3')['production'].transform(pd.Series.cumsum)

    # grid.to_csv("grid"+ str(lam) +".csv", index = False)

    logger.info('Shape before algo: {}'.format(grid.shape[0]))

    # nld_demand = beef_production.loc[beef_production.ADM0_A3 == 'NLD', 'curr_beef_meat'].iloc[0]
    # costs = tcost[nld_demand + grid.production > grid.production.cumsum()]

    logger.info('Grid shape before trade: {}'.format(grid.shape[0]))
    logger.info('trade_scenario: {} spat_constraint: {}'.format(trade_scenario, spat_constraint))
    logger.info('---')
    logger.info(grid[['bestlu']].dtypes)

    if trade_scenario == 'trade':
        if spat_constraint == 'country':
            grid = grid.sort_values('best_score')

            countrylist = [c for c in grid.ADM0_A3.unique() if c in beef_production.ADM0_A3.values]
            for country in countrylist:
                demand = beef_production.loc[beef_production.ADM0_A3 == country, 'curr_beef_meat'].iloc[0]
                cum_ctry_prod = grid.production[grid.ADM0_A3 == country].cumsum()
                if grid.loc[(grid['ADM0_A3'] == country) & (cum_ctry_prod > demand)].shape[0] > 0:

                    grid.loc[(grid['ADM0_A3'] == country) & (cum_ctry_prod > demand)] = trade(
                        grid.loc[(grid['ADM0_A3'] == country) & (cum_ctry_prod > demand)],
                        lam, landuses)
                logger.info('Grid shape during trade: {} for country: {}'.format(grid.shape[0], country))
                logger.info(grid[['bestlu']].dtypes)

        elif spat_constraint in ['global', 'subsistence']:


            # Set new production > 0 to compare old and new production to avoid new == old and infinite while loop
            new_production = 1

            countries_complete = []

            grid['total_prod'] = np.zeros(shape = grid.shape[0], dtype = 'float64')
            grid['supplied'] = np.zeros(shape = grid.shape[0], dtype = 'int8')
            grid['newcumprod'] = np.zeros(shape = grid.shape[0], dtype = 'float64')
            grid['cumprod'] = np.zeros(shape = grid.shape[0], dtype = 'float64')

            while total_production < demand and grid.loc[(grid.changed == 0)].shape[0] > 0 and new_production != 0:

                # Calculate old production to compare with new production
                old_production = grid.loc[grid.changed == 1, 'production'].sum()

                # Sort by increasing best score
                grid = grid.sort_values('best_score')

                # Recalculate cumulative production based on total production and according to sorted values
                grid.loc[grid.changed == 0, 'newcumprod'] = grid.loc[(grid.changed == 0) &
                                                                     ~grid.ADM0_A3.isin(
                                                                         countries_complete), 'production'].cumsum()
                grid['cumprod'] = grid['total_prod'] + grid['newcumprod']

                # Convert cells to production if (1) cells have not been changed yet, (2) cumulative domestic production is lower than domestic demand OR the country of the cell is already exporting,
                # (3) Cumulative production is lower than global demand and (4) best score is lower than the highest score meeting these conditions

                grid.loc[(grid['changed'] == 0) &
                         # ((grid['cumdomprod'] < grid['dom_demand']) | (grid['exporting'] == 1)) &
                         # ((grid['cumdomprod'] < grid['dom_demand'] + grid['production']) | (grid['exporting'] == 1)) &
                         ((grid['dom_demand'] + grid['production'] - grid['cumdomprod'] > 0) | (
                                     grid['exporting'] == 1)) &
                         ((demand + grid['production'] - grid['cumprod']) > 0) &
                         (grid['best_score'] <= grid.loc[(grid['changed'] == 0) &
                                                         ((demand + grid['production'] - grid['cumprod']) > 0) &
                                                         ((grid['dom_demand'] + grid['production'] - grid[
                                                             'cumdomprod'] > 0) | (grid[
                                                                                       'exporting'] == 1)), 'best_score'].max()), 'changed'] = 1

                ADM0_A3 = grid.loc[(grid.best_score <= grid.loc[grid.changed == 1].best_score.max()) &
                                   (grid.exporting == 0) &
                                   (grid.destination == 0) &
                                   (grid.cumdomprod > grid.dom_demand), 'ADM0_A3']

                grid.loc[(grid['changed'] == 0) & (grid['ADM0_A3'].isin(ADM0_A3)), 'exporting'] = 1

                if grid.loc[(grid['ADM0_A3'].isin(ADM0_A3)) & (grid['changed'] == 0)].shape[0] > 0:
                    grid.loc[(grid['ADM0_A3'].isin(ADM0_A3)) & (grid['changed'] == 0)] = trade(
                        grid.loc[(grid['ADM0_A3'].isin(ADM0_A3)) & (grid['changed'] == 0)], lam,
                        landuses)

                grid.loc[(grid['destination'] == 0) & (grid['changed'] == 1), 'destination'] = np.where(
                    grid.loc[(grid['destination'] == 0) & (grid['changed'] == 1), 'cumdomprod'] <
                    grid.loc[(grid['destination'] == 0) & (grid['changed'] == 1), 'dom_demand'], 1, 2)

                # Recalculate total production of converted cells
                total_production = grid.loc[grid.changed == 1, 'production'].sum()

                # Keep track of countries that have met their demand
                grid.loc[grid.ADM0_A3.isin(countries_complete), 'supplied'] = 1

                # Keep track of total production
                grid.loc[(grid.changed == 0) & (~grid.ADM0_A3.isin(countries_complete)), 'total_prod'] = total_production

                # Keep track of new production in loop to avoid looping with 0 new production
                new_production = round(total_production, 3) - round(old_production, 3)
                logger.info("Total production: {}".format(total_production))

    grid = grid.drop(['cumdomprod', 'dom_demand', 'exporting', 'destination'], axis = 1)

    logger.info('---')

    logger.info('Grid shape after trade: {}'.format(grid.shape[0]))

    grid = grid.sort_values('best_score')

    grid['changed'] = np.zeros_like(grid.best_score, dtype = 'int8')

    logger.info('Current meat: {}'.format(grid.c_meat.sum()))
    logger.info('Current grid shape: {}'.format(grid.shape[0]))

    if spat_constraint == 'global':

        grid['total_emission'] = np.take_along_axis(grid[[lu + '_ghg' for lu in landuses]].values,
                                                    grid['bestlu'].values[:, None], axis=1).flatten()
        grid['total_cost'] = np.take_along_axis(grid[[lu + '_tot_cost' for lu in landuses]].values,
                                                grid['bestlu'].values[:, None], axis=1).flatten()

        logger.info('Total costs post trade: {}'.format(
            grid.loc[(demand + grid['production'] > grid['production'].cumsum()), 'total_cost'].sum()))
        logger.info('Total production post trade: {}'.format(
            grid.loc[(demand + grid['production'] > grid['production'].cumsum()), 'production'].sum()))
        old_emissions = grid.loc[(demand + grid['production'] > grid['production'].cumsum()), 'total_emission'].sum()

        old_beef = 0

        old_costs = grid.loc[(demand + grid['production'] > grid['production'].cumsum()), 'total_cost'].sum()
        print('Beef production 1: {}'.format(
            grid.loc[(demand + grid['production'] > grid['production'].cumsum()), 'production'].sum()))

        grid = grid.reset_index(drop=True)
        nrows = grid.loc[(demand + grid['production'] - grid['production'].cumsum() > 0)].shape[0]
        sel_nrows = nrows + int(nrows * 0.01)
        unconv_grid = grid.iloc[sel_nrows:]
        grid = grid.iloc[:sel_nrows]

        ghg_cols = [lu + '_ghg' for lu in landuses]
        cost_cols = [lu + '_tot_cost' for lu in landuses]
        meat_cols = [lu + '_meat' for lu in landuses]
        score_cols = [lu + '_score' for lu in landuses]

        cols = ['best_score', 'bestlu', 'production', 'total_emission',
                'total_cost'] + ghg_cols + meat_cols + cost_cols + score_cols

        rest_grid = grid.drop(cols, axis=1)
        logger.info("Opp_aff in restgrid columns: {}".format('opp_aff' in rest_grid.columns))

        grid = grid[cols + ['id']]

        logger.info('Start of adaptive greedy')

        divisions = [100, 90, 80, 60, 40, 10, 2, 1]
        for div in divisions:

            logger.info('Number of divisions: {}'.format(div))

            grid['groups'] = np.arange(len(grid.index)) // (grid.shape[0] / div)

            newgrid = pd.DataFrame(data=None, columns=grid.columns)
            newgrid = newgrid.astype(grid.dtypes.to_dict())
            nloop = 0

            for g in grid['groups'].unique():

                tempgrid = grid[grid.groups == g]
                grid = grid[grid.groups != g]

                score_change = -1
                old_beef = 0
                new_beef = grid.loc[
                    (demand + grid['production'] > grid['production'].cumsum()), 'production'].sum()
                old_score = ((1 - lam) * old_emissions + (lam * old_costs)) / old_beef

                try:
                    while score_change != 0 and round(new_beef, 6) != round(old_beef, 6):
                        tempgrid = tempgrid.sort_values('best_score')

                        last_score = grid['best_score'].values[
                            demand + grid.production > newgrid.production.sum() + tempgrid.production.sum() + grid.production.cumsum()][
                            -1]

                        # Difference between demand and sum of beef without last cell
                        beef_needed = demand - (newgrid.production.sum() + tempgrid.production.sum() + np.nansum(
                            grid.production[(
                                    demand > newgrid.production.sum() + tempgrid.production.sum() + grid.production.cumsum())]))

                        delta_score = (1 - lam) * (
                                tempgrid[[lu + '_ghg' for lu in landuses]].values - tempgrid.total_emission.values[
                                                                                    :,
                                                                                    None]) + (lam) * (
                                              tempgrid[[lu + '_tot_cost' for lu in
                                                        landuses]].values - tempgrid.total_cost.values[:,
                                                                            None])

                        delta_meat = tempgrid[
                                         [lu + '_meat' for lu in landuses]].values - tempgrid.production.values[:,
                                                                                     None]

                        marginal_cost = delta_score / delta_meat

                        cond1 = tempgrid[[lu + '_meat' for lu in landuses]].values <= tempgrid.production.values[:,
                                                                                      None]

                        cond2 = marginal_cost >= last_score

                        masked = np.where(cond1 | cond2, np.nan,
                                          tempgrid[[lu + '_score' for lu in landuses]].values)

                        tempgrid['new_score'] = np.nanmin(masked, axis=1)

                        tempgrid['new_landuse'] = np.where(~np.isnan(tempgrid['new_score'].values),
                                                           np.nanargmin(
                                                               np.ma.masked_where(np.isnan(masked), masked),
                                                               axis=1), 0)

                        tempgrid['new_beef'] = np.where(~np.isnan(tempgrid['new_score'].values),
                                                        np.take_along_axis(
                                                            tempgrid[[lu + '_meat' for lu in landuses]].values,
                                                            tempgrid['new_landuse'].values[:, None],
                                                            axis=1).flatten(),
                                                        np.nan)

                        tempgrid['new_emissions'] = np.where(~np.isnan(tempgrid['new_score'].values),
                                                             np.take_along_axis(
                                                                 tempgrid[[lu + '_ghg' for lu in landuses]].values,
                                                                 tempgrid['new_landuse'].values[:, None],
                                                                 axis=1).flatten(),
                                                             np.nan)

                        tempgrid['new_costs'] = np.where(~np.isnan(tempgrid['new_score'].values),
                                                         np.take_along_axis(
                                                             tempgrid[[lu + '_tot_cost' for lu in landuses]].values,
                                                             tempgrid['new_landuse'].values[:, None],
                                                             axis=1).flatten(),
                                                         np.nan)

                        tempgrid['new_beef_diff'] = tempgrid.new_beef - tempgrid.production

                        tempgrid['switch'] = np.where(
                            (beef_needed + tempgrid.new_beef_diff > tempgrid.new_beef_diff.cumsum()), 1, 0)

                        tempgrid['production'] = np.where(tempgrid.switch == 1, tempgrid.new_beef,
                                                          tempgrid.production)

                        tempgrid['total_emission'] = np.where(tempgrid.switch == 1, tempgrid.new_emissions,
                                                              tempgrid.total_emission)
                        tempgrid['total_cost'] = np.where(tempgrid.switch == 1, tempgrid.new_costs,
                                                          tempgrid.total_cost)
                        tempgrid['best_score'] = np.where(tempgrid.switch == 1, tempgrid.new_score,
                                                          tempgrid.best_score)
                        tempgrid['bestlu'] = np.where(tempgrid.switch == 1, tempgrid.new_landuse, tempgrid.bestlu)

                        old_beef = new_beef

                        new_emissions = newgrid.total_emission.sum() + tempgrid.total_emission.sum() + np.nansum(
                            grid.total_emission[(
                                    demand + grid.production > newgrid.production.sum() + tempgrid.production.sum() + grid.production.cumsum())])

                        new_beef = newgrid.production.sum() + tempgrid.production.sum() + np.nansum(
                            grid.production[(
                                    demand + grid.production > newgrid.production.sum() + tempgrid.production.sum() + grid.production.cumsum())])

                        new_costs = newgrid.total_cost.sum() + tempgrid.total_cost.sum() + np.nansum(
                            grid.total_cost[(
                                    demand + grid.production > newgrid.production.sum() + tempgrid.production.sum() + grid.production.cumsum())])

                        new_score = ((1 - lam) * new_emissions + (lam * new_costs)) / new_beef

                        score_change = new_score - old_score

                        old_score = new_score
                        logger.info('   beef: {}, emissions: {}, costs: {}'.format(
                            # score_change,
                            new_beef,
                            # old_beef,
                            new_emissions,
                            new_costs))
                        nloop += 1
                except:
                    logger.info('No more cells')

                newgrid = pd.concat([newgrid, tempgrid])

            grid = newgrid.copy()

            logger.info('======> {} loops when grid divided in {} parts'.format(nloop, div))

        grid = grid.drop(
            ['new_beef_diff', 'new_beef', 'switch', 'new_costs', 'new_emissions', 'new_landuse', 'new_score'],
            axis=1)

        grid = rest_grid.merge(grid, left_on='id', right_on='id', how='left')
        grid = pd.concat([grid, unconv_grid])

        ########### Start last iteration ###########
        logger.info('======> Last loop <======')
        cols = ['best_score', 'bestlu', 'production', 'total_emission',
                'total_cost'] + ghg_cols + meat_cols + cost_cols + score_cols

        rest_grid = grid.drop(cols, axis=1)

        grid = grid[cols + ['id']]

        score_change = -1
        old_beef = 0
        new_beef = grid.loc[
            (demand + grid['production'] > grid['production'].cumsum()), 'production'].sum()
        old_score = ((1 - lam) * old_emissions + (lam * old_costs)) / old_beef
        while score_change != 0 and round(new_beef, 6) != round(old_beef, 6):
            grid = grid.sort_values('best_score')

            # Score and beef production of last converted cell
            last_score = grid['best_score'].values[demand + grid['production'].values > grid.production.cumsum()][
                -1]
            #                 logger.info('last_score')

            # Difference between demand and sum of beef without last cell
            beef_needed = demand - np.nansum(grid.production[(demand > grid.production.cumsum())])
            #                 logger.info('beef_needed')

            delta_score = (1 - lam) * (
                    grid[[lu + '_ghg' for lu in landuses]].values - grid.total_emission.values[:, None]) + (lam) * (
                                  grid[[lu + '_tot_cost' for lu in landuses]].values - grid.total_cost.values[:,
                                                                                       None])

            delta_meat = grid[[lu + '_meat' for lu in landuses]].values - grid.production.values[:, None]
            marginal_cost = delta_score / delta_meat
            cond1 = grid[[lu + '_meat' for lu in landuses]].values <= grid.production.values[:, None]
            cond2 = marginal_cost >= last_score
            masked = np.where(cond1 | cond2, np.nan, grid[[lu + '_score' for lu in landuses]].values)
            grid['new_score'] = np.nanmin(masked, axis=1)
            grid['new_landuse'] = np.where(~np.isnan(grid['new_score'].values),
                                           np.nanargmin(np.ma.masked_where(np.isnan(masked), masked), axis=1), 0)
            grid['new_beef'] = np.where(~np.isnan(grid['new_score'].values),
                                        np.take_along_axis(grid[[lu + '_meat' for lu in landuses]].values,
                                                           grid['new_landuse'].values[:, None], axis=1).flatten(),
                                        np.nan)
            grid['new_emissions'] = np.where(~np.isnan(grid['new_score'].values),
                                             np.take_along_axis(grid[[lu + '_ghg' for lu in landuses]].values,
                                                                grid['new_landuse'].values[:, None],
                                                                axis=1).flatten(),
                                             np.nan)
            grid['new_costs'] = np.where(~np.isnan(grid['new_score'].values),
                                         np.take_along_axis(grid[[lu + '_tot_cost' for lu in landuses]].values,
                                                            grid['new_landuse'].values[:, None], axis=1).flatten(),
                                         np.nan)

            grid['new_beef_diff'] = grid.new_beef - grid.production

            grid['switch'] = np.where((beef_needed + grid.new_beef_diff > grid.new_beef_diff.cumsum()), 1, 0)

            grid['production'] = np.where(grid.switch == 1, grid.new_beef, grid.production)
            grid['total_emission'] = np.where(grid.switch == 1, grid.new_emissions, grid.total_emission)
            grid['total_cost'] = np.where(grid.switch == 1, grid.new_costs, grid.total_cost)
            grid['best_score'] = np.where(grid.switch == 1, grid.new_score, grid.best_score)
            grid['bestlu'] = np.where(grid.switch == 1, grid.new_landuse, grid.bestlu)

            old_beef = new_beef

            new_emissions = np.nansum(grid.total_emission[(demand + grid.production > grid.production.cumsum())])
            new_beef = np.nansum(grid.production[(demand + grid.production > grid.production.cumsum())])
            new_costs = np.nansum(grid.total_cost[(demand + grid.production > grid.production.cumsum())])
            new_score = ((1 - lam) * new_emissions + (lam * new_costs)) / new_beef

            score_change = new_score - old_score

            old_score = new_score
            logger.info('   beef: {}, emissions: {}, costs: {}'.format(
                # score_change,
                new_beef,
                # old_beef,
                new_emissions,
                new_costs))

        logger.info('Beef production 2: {}'.format(
            grid.loc[(demand + grid['production'] > grid['production'].cumsum()), 'production'].sum()))

        grid = grid.drop(
            ['new_beef_diff', 'new_beef', 'switch', 'new_costs', 'new_emissions', 'new_landuse', 'new_score'],
            axis=1)

        grid = grid.merge(rest_grid, left_on='id', right_on='id', how='left')

        logger.info('Beef production 3: {}'.format(
            grid.loc[(demand + grid['production'] > grid['production'].cumsum()), 'production'].sum()))
        logger.info('Total costs: {}'.format(
            grid.loc[(demand + grid['production'] > grid['production'].cumsum()), 'total_cost'].sum()))
        logger.info('Total emissions: {}'.format(
            grid.loc[(demand + grid['production'] > grid['production'].cumsum()), 'total_emission'].sum()))

        grid['changed'] = np.where(demand + grid['production'] > grid['production'].cumsum(), 1, 0)

    if spat_constraint == 'country':

        grid['changed'] = 0
        grid = grid.sort_values('best_score')

        countrylist = [c for c in grid.ADM0_A3.unique() if c in beef_production.ADM0_A3.values]
        for country in countrylist:

            demand = beef_production.loc[beef_production.ADM0_A3 == country, 'curr_beef_meat'].iloc[0]
            logger.info('Country: {}, demand: {}'.format(country, demand))

            ghg_cols = [lu + '_ghg' for lu in landuses]
            cost_cols = [lu + '_tot_cost' for lu in landuses]
            meat_cols = [lu + '_meat' for lu in landuses]
            score_cols = [lu + '_score' for lu in landuses]

            cols = ['id', 'ADM0_A3', 'best_score', 'bestlu', 'production'] + ghg_cols + meat_cols + cost_cols + score_cols

            country_df = grid.loc[grid.ADM0_A3 == country][cols]

            country_df = country_df.reset_index(drop=True)

            country_df['total_emission'] = np.take_along_axis(country_df[[lu + '_ghg' for lu in landuses]].values,
                                                         country_df['bestlu'].values[:, None], axis=1).flatten()
            country_df['total_cost'] = np.take_along_axis(country_df[[lu + '_tot_cost' for lu in landuses]].values,
                                                     country_df['bestlu'].values[:, None], axis=1).flatten()


            old_emissions = country_df.loc[(demand + country_df['production'] > country_df['production'].cumsum()), 'total_emission'].sum()
            old_costs = country_df.loc[(demand + country_df['production'] > country_df['production'].cumsum()), 'total_cost'].sum()

            cols = ['best_score', 'bestlu', 'production', 'total_emission',
                    'total_cost'] + ghg_cols + meat_cols + cost_cols + score_cols

            country_df = country_df[cols + ['id']]

            logger.info('Start of adaptive greedy')
            logger.info(country_df.shape)

            divisions = [100, 90, 80, 60, 40, 10, 2, 1]
            for div in divisions:

                logger.info('Number of divisions: {}'.format(div))

                country_df['groups'] = np.arange(len(country_df.index)) // (country_df.shape[0] / div)

                newgrid = pd.DataFrame(data=None, columns=country_df.columns)
                newgrid = newgrid.astype(country_df.dtypes.to_dict())
                nloop = 0

                for g in country_df['groups'].unique():

                    tempgrid = country_df[country_df.groups == g]
                    country_df = country_df[country_df.groups != g]

                    score_change = -1
                    old_beef = 0
                    new_beef = country_df.loc[
                        (demand + country_df['production'] > country_df['production'].cumsum()), 'production'].sum()
                    old_score = ((1 - lam) * old_emissions + (lam * old_costs)) / old_beef

                    try:
                        while score_change != 0 and round(new_beef, 6) != round(old_beef, 6):
                            tempgrid = tempgrid.sort_values('best_score')

                            last_score = country_df['best_score'].values[
                                demand + country_df.production > newgrid.production.sum() + tempgrid.production.sum() + country_df.production.cumsum()][
                                -1]

                            # Difference between demand and sum of beef without last cell
                            beef_needed = demand - (newgrid.production.sum() + tempgrid.production.sum() + np.nansum(
                                country_df.production[(
                                        demand > newgrid.production.sum() + tempgrid.production.sum() + country_df.production.cumsum())]))

                            delta_score = (1 - lam) * (
                                    tempgrid[[lu + '_ghg' for lu in landuses]].values - tempgrid.total_emission.values[
                                                                                        :,
                                                                                        None]) + (lam) * (
                                                  tempgrid[[lu + '_tot_cost' for lu in
                                                            landuses]].values - tempgrid.total_cost.values[:,
                                                                                None])

                            delta_meat = tempgrid[
                                             [lu + '_meat' for lu in landuses]].values - tempgrid.production.values[:,
                                                                                         None]

                            marginal_cost = delta_score / delta_meat

                            cond1 = tempgrid[[lu + '_meat' for lu in landuses]].values <= tempgrid.production.values[:,
                                                                                          None]

                            cond2 = marginal_cost >= last_score

                            masked = np.where(cond1 | cond2, np.nan,
                                              tempgrid[[lu + '_score' for lu in landuses]].values)

                            tempgrid['new_score'] = np.nanmin(masked, axis=1)

                            tempgrid['new_landuse'] = np.where(~np.isnan(tempgrid['new_score'].values),
                                                               np.nanargmin(
                                                                   np.ma.masked_where(np.isnan(masked), masked),
                                                                   axis=1), 0)

                            tempgrid['new_beef'] = np.where(~np.isnan(tempgrid['new_score'].values),
                                                            np.take_along_axis(
                                                                tempgrid[[lu + '_meat' for lu in landuses]].values,
                                                                tempgrid['new_landuse'].values[:, None],
                                                                axis=1).flatten(),
                                                            np.nan)

                            tempgrid['new_emissions'] = np.where(~np.isnan(tempgrid['new_score'].values),
                                                                 np.take_along_axis(
                                                                     tempgrid[[lu + '_ghg' for lu in landuses]].values,
                                                                     tempgrid['new_landuse'].values[:, None],
                                                                     axis=1).flatten(),
                                                                 np.nan)

                            tempgrid['new_costs'] = np.where(~np.isnan(tempgrid['new_score'].values),
                                                             np.take_along_axis(
                                                                 tempgrid[[lu + '_tot_cost' for lu in landuses]].values,
                                                                 tempgrid['new_landuse'].values[:, None],
                                                                 axis=1).flatten(),
                                                             np.nan)

                            tempgrid['new_beef_diff'] = tempgrid.new_beef - tempgrid.production

                            tempgrid['switch'] = np.where(
                                (beef_needed + tempgrid.new_beef_diff > tempgrid.new_beef_diff.cumsum()), 1, 0)

                            tempgrid['production'] = np.where(tempgrid.switch == 1, tempgrid.new_beef,
                                                              tempgrid.production)

                            tempgrid['total_emission'] = np.where(tempgrid.switch == 1, tempgrid.new_emissions,
                                                                  tempgrid.total_emission)
                            tempgrid['total_cost'] = np.where(tempgrid.switch == 1, tempgrid.new_costs,
                                                              tempgrid.total_cost)
                            tempgrid['best_score'] = np.where(tempgrid.switch == 1, tempgrid.new_score,
                                                              tempgrid.best_score)
                            tempgrid['bestlu'] = np.where(tempgrid.switch == 1, tempgrid.new_landuse, tempgrid.bestlu)

                            old_beef = new_beef

                            new_emissions = newgrid.total_emission.sum() + tempgrid.total_emission.sum() + np.nansum(
                                country_df.total_emission[(
                                        demand + country_df.production > newgrid.production.sum() + tempgrid.production.sum() + country_df.production.cumsum())])

                            new_beef = newgrid.production.sum() + tempgrid.production.sum() + np.nansum(
                                country_df.production[(
                                        demand + country_df.production > newgrid.production.sum() + tempgrid.production.sum() + country_df.production.cumsum())])

                            new_costs = newgrid.total_cost.sum() + tempgrid.total_cost.sum() + np.nansum(
                                country_df.total_cost[(
                                        demand + country_df.production > newgrid.production.sum() + tempgrid.production.sum() + country_df.production.cumsum())])

                            new_score = ((1 - lam) * new_emissions + (lam * new_costs)) / new_beef

                            score_change = new_score - old_score

                            old_score = new_score
                            logger.info('   beef: {}, emissions: {}, costs: {}'.format(
                                # score_change,
                                new_beef,
                                # old_beef,
                                new_emissions,
                                new_costs))
                            nloop += 1
                    except:
                        logger.info('No more cells')

                    newgrid = pd.concat([newgrid, tempgrid])

                country_df = newgrid.copy()

                logger.info('======> {} loops when grid divided in {} parts'.format(nloop, div))
            try:
                country_df = country_df.drop(
                ['new_beef_diff', 'new_beef', 'switch', 'new_costs', 'new_emissions', 'new_landuse', 'new_score'],
                axis=1)
            except:
                logger.info('No change in country')

            ########### Start last iteration ###########
            logger.info('======> Last loop <======')
            cols = ['best_score', 'bestlu', 'production', 'total_emission',
                    'total_cost'] + ghg_cols + meat_cols + cost_cols + score_cols

            country_df = country_df[cols + ['id']]

            score_change = -1
            old_beef = 0
            new_beef = country_df.loc[
                (demand + country_df['production'] > country_df['production'].cumsum()), 'production'].sum()
            old_score = ((1 - lam) * old_emissions + (lam * old_costs)) / old_beef
            nrows = country_df.loc[(demand + country_df['production'] - country_df['production'].cumsum() > 0)].shape[0]
            old_score_change = 0

            logger.info(country_df.shape)

            score_list = []

            # Start last iteration
            while score_change != 0 and nrows > 0 and round(new_beef, 6) != round(old_beef,
                                                                                  6) and old_score_change != score_change * -1 and old_score not in score_list:
                country_df = country_df.sort_values('best_score')

                beef_needed = demand - country_df.loc[
                    (demand > country_df['production'].cumsum()), 'production'].sum()

                delta_score = (1 - lam) * (
                        country_df[[lu + '_ghg' for lu in landuses]].values - country_df.total_emission.values[:,
                                                                              None]) + (
                                  lam) * (
                                      country_df[
                                          [lu + '_tot_cost' for lu in landuses]].values - country_df.total_cost.values[:,
                                                                                          None])

                delta_meat = country_df[[lu + '_meat' for lu in landuses]].values - country_df.production.values[:,
                                                                                    None]

                marginal_cost = delta_score / delta_meat

                try:
                    lastbeef = country_df.loc[(demand < country_df['production'].cumsum()), 'production'].iloc[0]

                    last_score = (1 - lam) * (
                            country_df.loc[(demand < country_df['production'].cumsum()), 'total_emission'].iloc[
                                0] / lastbeef) + (lam *
                                                  country_df.loc[
                                                      (demand < country_df['production'].cumsum()), 'total_cost'].iloc[
                                                      0] / lastbeef)
                except:
                    logger.info("   *** Production does not meet target. ***")
                    lastbeef = country_df['production'].iloc[-1]

                    last_score = (1 - lam) * (country_df['total_emission'].iloc[-1] / lastbeef) + (lam *
                                                                                              country_df[
                                                                                                  'total_cost'].iloc[
                                                                                                  -1] / lastbeef)

                cond1 = country_df[[lu + '_meat' for lu in landuses]].values <= country_df.production.values[:,
                                                                                None]

                cond2 = marginal_cost >= last_score

                masked = np.where(cond1 | cond2, np.nan, country_df[[lu + '_score' for lu in landuses]].values)

                country_df['new_best_score'] = np.nanmin(masked, axis=1)

                country_df['new_bestlu'] = np.where(~np.isnan(country_df['new_best_score'].values),
                                                    np.nanargmin(np.ma.masked_where(np.isnan(masked), masked),
                                                                 axis=1),
                                                    0)

                country_df['new_production'] = np.where(~np.isnan(country_df['new_best_score'].values),
                                                        np.take_along_axis(
                                                            country_df[[lu + '_meat' for lu in landuses]].values,
                                                            country_df['new_bestlu'].values[:, None],
                                                            axis=1).flatten(),
                                                        np.nan)

                country_df['new_total_emission'] = np.where(~np.isnan(country_df['new_best_score'].values),
                                                       np.take_along_axis(
                                                           country_df[[lu + '_ghg' for lu in landuses]].values,
                                                           country_df['new_bestlu'].values[:, None],
                                                           axis=1).flatten(),
                                                       np.nan)

                country_df['new_total_cost'] = np.where(~np.isnan(country_df['new_best_score'].values),
                                                   np.take_along_axis(
                                                       country_df[[lu + '_tot_cost' for lu in landuses]].values,
                                                       country_df['new_bestlu'].values[:, None], axis=1).flatten(),
                                                   np.nan)

                country_df['new_production_diff'] = country_df['new_production'].values - country_df[
                    'production'].values

                country_df['switch'] = np.where(
                    (beef_needed + country_df['new_production_diff'].values > country_df[
                        'new_production_diff'].cumsum()), 1, 0)

                for i in ['production', 'total_emission', 'total_cost', 'best_score', 'bestlu']:
                    country_df[i] = np.where(country_df['switch'].values == 1,
                                             country_df['new_' + i].values,
                                             country_df[i])

                new_emissions = country_df.loc[
                    (demand + country_df['production'] > country_df['production'].cumsum()), 'total_emission'].sum()

                old_beef = new_beef
                new_beef = country_df.loc[
                    (demand + country_df['production'] > country_df['production'].cumsum()), 'production'].sum()
                new_costs = country_df.loc[
                    (demand + country_df['production'] > country_df['production'].cumsum()), 'total_cost'].sum()
                new_best_score = ((1 - lam) * new_emissions + (lam * new_costs)) / new_beef

                old_score_change = score_change
                score_change = new_best_score - old_score

                old_score = new_best_score

                score_list.append(new_best_score)
                if len(score_list) >= 10:
                    score_list.pop(0)

                logger.info('   Score change: {}, beef: {}, emissions: {}, costs: {}'.format(
                    score_change, round(new_beef, 2), round(new_emissions, 2), round(new_costs, 2)))
            # country_df.to_csv('./country_df_{}.csv'.format(round(lam,3)), index = False)
            country_df = country_df.loc[demand + country_df['production'] > country_df['production'].cumsum()]
            logger.info('grid size country_df: {}'.format(country_df.shape[0]))

            logger.info('Country beef: {}, new emissions: {}'.format(np.nansum(country_df['production'].values),
                                                                     np.nansum(country_df['total_emission'].values)))

            grid['changed'].values[(grid.ADM0_A3.values == country) & (grid.id.isin(country_df.id))] = 1

            temp = grid[['id']].merge(country_df[['id', 'best_score', 'production', 'bestlu']], how='left')
            for c in ['production', 'bestlu', 'best_score']:
                grid[c].values[(grid['ADM0_A3'].values == country) & (grid.id.isin(country_df.id))] = \
                temp[c].values[(grid['ADM0_A3'].values == country) & (grid.id.isin(country_df.id))]
            del temp

            logger.info(
                'After merging beef: {}, new emissions: {}'.format(np.nansum(country_df['production'].values),
                                                                   np.nansum(country_df['total_emission'].values)))


    ######### Export #########

    opp_aff_curr = grid.loc[grid.changed == 0, 'opp_aff'].sum()
    aff_costs_curr = grid.loc[grid.changed == 0, 'aff_cost'].sum()
    opp_soc_curr = grid.loc[grid.changed == 0, 'opp_soc'].sum()
    refor_area = grid.loc[(grid.changed == 0) & (grid.newarea == 0) & (grid.opp_aff != 0), 'regrowth_area'].sum()

    regrowth_cells = grid.loc[(grid.changed == 0) & (grid.regrowth_area > 0), 'best_regrowth'].values

    try:
        percent_manaff = regrowth_cells[regrowth_cells == 2].shape[0] / regrowth_cells.shape[0]
    except:
        percent_manaff = 0
    try:
        percent_nataff = regrowth_cells[regrowth_cells == 1].shape[0] / regrowth_cells.shape[0]
    except:
        percent_nataff = 0
    try:
        percent_noaff = regrowth_cells[regrowth_cells == 0].shape[0] / regrowth_cells.shape[0]
    except:
        percent_noaff = 0

    del regrowth_cells
    # Calculate how much current beef production is lost
    # production_loss = grid.loc[grid.changed == 0, 'cell_area'] * grid.loc[grid.changed == 0, 'bvmeat'].values * 1e-5

    # unconv_cells = grid.loc[grid['changed'] == 0][['ADM0_A3', 'opp_aff', 'opp_soc', 'aff_cost', 'regrowth_area']].groupby('ADM0_A3',
    #                                                                                                      as_index=False).sum()
    unconv_cells = grid.loc[(grid.changed == 0) &
                            (grid.newarea == 0) &
                            (np.nan_to_num(grid.opp_aff) != 0)][
        ['ADM0_A3', 'opp_aff', 'opp_soc', 'aff_cost', 'regrowth_area']].groupby('ADM0_A3', as_index=False).sum()

    unconv_grid = grid.loc[(grid['changed'] == 0)][['cell_id', 'opp_aff', 'opp_soc']].groupby('cell_id', as_index = False).sum()

    # Export raster of opportunity of afforestation
    export_raster(unconv_grid, ['opp_aff', 'opp_soc'], dst, spat_constraint, str(round(lam, 3)), aff_scenario)
    if lam in [0, 1]:


        if lam == 0:
            grid['tot_emi'] = np.take_along_axis(grid[[lu + new_colnames['total_emission'] for lu in landuses]].values,
                                                 grid['bestlu'].values[:, None], axis=1).flatten()
            g = grid[['cell_id', 'production', 'tot_emi']].groupby('cell_id', as_index=False).mean()
            g['score'] = g['tot_emi'].values / g['production'].values
        elif lam == 1:
            grid['tot_cost'] = np.take_along_axis(grid[[lu + new_colnames['total_cost'] for lu in landuses]].values,
                                                 grid['bestlu'].values[:, None], axis=1).flatten()
            g = grid[['cell_id', 'production', 'tot_cost']].groupby('cell_id', as_index=False).mean()
            g['score'] = g['tot_cost'].values / g['production'].values

        # export_raster(g, ['score'], dst, spat_constraint, lam, horizon, aff_scenario)
        export_raster(g, ['score'], dst, spat_constraint, round(lam, 3), aff_scenario)

    logger.info('grid size before subset: {}'.format(grid.shape[0]))
    grid = grid.loc[grid['changed'] == 1]

    if lam in [0, 1]:
        export_raster(unconv_grid, ['opp_aff', 'opp_soc'], dst, spat_constraint, str(round(lam, 3)), aff_scenario)
        if lam == 0:
            grid['tot_emi'] = np.take_along_axis(grid[[lu + new_colnames['total_emission'] for lu in landuses]].values,
                                                 grid['bestlu'].values[:, None], axis=1).flatten()
            g = grid[['cell_id', 'production', 'tot_emi']].groupby('cell_id', as_index=False).mean()
            g['score_selected'] = g['tot_emi'].values / g['production'].values
        elif lam == 1:
            grid['tot_cost'] = np.take_along_axis(grid[[lu + new_colnames['total_cost'] for lu in landuses]].values,
                                                  grid['bestlu'].values[:, None], axis=1).flatten()
            g = grid[['cell_id', 'production', 'tot_cost']].groupby('cell_id', as_index=False).mean()
            g['score_selected'] = g['tot_cost'].values / g['production'].values
        export_raster(g, ['score_selected'], dst, spat_constraint, round(lam, 3), aff_scenario)

    logger.info('grid size after subset: {}'.format(grid.shape[0]))

    # rasters = glob('./rasters/current_data/*.tif')
    # c_list = [r.split('/')[-1].split('.')[0] for r in rasters]

    for i in new_colnames:
        grid[i] = np.take_along_axis(grid[[lu + new_colnames[i] for lu in landuses]].values,
                                     grid['bestlu'].values[:, None], axis=1).flatten()
        # if i not in ['production', 'total_cost', 'total_emission']:
            # if 'c' + new_colnames[i] in c_list:
            #     with rasterio.open('./rasters/current_data/c{}.tif'.format(new_colnames[i])) as f:
            #         # if int(production_yr) == 2018:
            #         #     grid['c' + new_colnames[i]] = f.read(1).flatten()[grid.cell_id] * dem_inc_factor
            #         # else:
            #         grid['c' + new_colnames[i]] = f.read(1).flatten()[grid.cell_id]
            # else:
            #     grid['c' + new_colnames[i]] = np.zeros(shape = grid.shape[0], dtype = 'uint8')

    defor_area = grid.loc[(grid.changed == 1) & (grid.newarea == 1), 'beef_area'].sum()

    # Record biomass based on land uses
    grid['grass_BM'] = np.select([grid.bestlu == 9, grid.bestlu == 10, grid.bestlu == 11, grid.bestlu == 12],
                                 [grid.grain_grassBM, grid.stover_grass_grassBM, 0, np.nan_to_num(grid.c_graz) + np.nan_to_num(grid.c_occa)],
                                            default=grid.biomass)

    grid['grain_BM'] = np.select([grid.bestlu == 9, grid.bestlu == 11, grid.bestlu == 12],
                                 [grid.grain_grainBM, grid.stover_grain_grainBM, grid.c_grain],
                                            default=0)

    grid['stover_BM'] = np.select([grid.bestlu == 10, grid.bestlu == 11, grid.bestlu == 12],
                                  [grid.stover_grass_stoverBM, grid.stover_grain_stoverBM, grid.c_stover], default=0)

    infcost = np.select([grid.bestlu == 9, grid.bestlu == 11],
                        [grid.grass_grain_infcost, grid.stover_grain_infcost], default=0)

    # grid['grass_meat'] = np.select([grid.bestlu == 9, grid.bestlu == 10, grid.bestlu == 11],
    #                                [grid.grain_grass_meat, grid.stover_grass_grass_meat, 0], default=grid.production)
    # grid['grain_meat'] = np.select([grid.bestlu == 9, grid.bestlu == 11],
    #                                [grid.grain_grain_meat, grid.stover_grain_grain_meat], default=0)
    # grid['stover_meat'] = np.select([grid.bestlu == 10, grid.bestlu == 11],
    #                                 [grid.stover_grass_stover_meat, grid.stover_grain_stover_meat], default=0)

    # beef_price = grid.loc[grid.changed == 0][['ADM0_A3']].merge(beefprices, how='left')['price'].values / 1000.
    # total_compensation = grid.loc[grid.changed == 1,'compensation'].sum() + np.nansum(beef_price * production_loss)
    # logger.info("Compensation in changed cells: {}".format(grid.loc[grid['changed'] == 1, 'compensation'].sum()))
    # logger.info("Compensation for loss income: {}".format(np.nansum(production_loss)))
    # logger.info("Maximum compensation: {}".format(np.nansum(grid['cell_area'].values * grid['bvmeat'].values * 1e-5)))

    # If scenario is keep subsistence production: add subsistence production, impacts and costs to grid
    # if spat_constraint == 'subsistence':
    #     logger.info('Production before merging: {}'.format(grid.production.sum()))
    #     grid = pd.concat([grid, gpd.read_file('global_south_results.gpkg')])
    #     logger.info('Production after merging: {}'.format(grid.production.sum()))

    grid['total_costs'] = np.where(grid['bestlu'] != 12,
                                   np.nansum(grid[['establish_cost', 'production_cost', 'opportunity_cost', 'postfarm_cost']].values, axis = 1),
                                   # + grid['transp_cost'] + grid['export_cost']
                                   np.nansum(grid[['c_cost', 'c_opp_cost', 'c_postfarm_cost']].values, axis = 1))
    # + grid['aff_cost']
    grid['total_emissions'] = np.where(grid['bestlu'] != 12,
                                       np.nansum(grid[['agb_change', 'bgb_change', 'n2o_emissions', 'enteric', 'manure',
                                             'postfarm_emi']].values, axis =1),
                                       # grid['transp_emission'] + grid['processing_energy'] + grid['export_emissions'],
                                       np.nansum(grid[['c_meth', 'c_manure', 'c_n2o', 'c_postfarm_emi']].values, axis = 1))

    grid['beef_area'] = np.where(grid.bestlu == 12,
                                 grid['current_grazing'].fillna(0).values + grid['current_cropping'].fillna(0).values,
                                 grid.beef_area)
    # grid.to_csv('grid_{}.csv'.format(lam))

    totaldf = pd.DataFrame({"suitable": grid.suitable.sum()* grid.cell_area.sum(),
                            "grass_BM": grid.grass_BM.sum(),
                            "grain_BM": grid.grain_BM.sum(),
                            "stover_BM": grid.stover_BM.sum(),
                            "beef_area": grid.beef_area.sum(),
                            "production": grid.production.sum(),
                            "enteric": grid.enteric.sum(),
                            "manure": grid.manure.sum(),
                            "n2o_emissions": grid.n2o_emissions.sum(),
                            "agb_change": grid.agb_change.sum(),
                            "bgb_change": grid.bgb_change.sum(),
                            'aff_cost': [aff_costs_curr],
                            'unconvaff_cost': [aff_costs_curr],
                            'unconvopp_aff': [opp_aff_curr],
                            'unconvopp_soc': [opp_soc_curr],
                            "emissions": grid.total_emissions.sum(),
                            "costs": grid.total_costs.sum(),
                            "total_emissions": grid.total_emissions.sum() + opp_aff_curr + opp_soc_curr,
                            "total_costs": grid.total_costs.sum() + aff_costs_curr,
                            "est_cost": grid.establish_cost.sum(),
                            'infras_cost': np.nansum(infcost),
                            "production_cost": grid.production_cost.sum(),
                            "opportunity_cost": grid.opportunity_cost.sum(),
                            "postfarm_emissions": grid.postfarm_emi.sum(),
                            "postfarm_costs": grid.postfarm_cost.sum(),
                            "perc_expansion": np.nansum(grid.beef_area[grid.newarea == 1])/np.nansum(grid.beef_area),
                            "perc_new_qty": np.nansum(grid.production[grid.newarea == 1])/np.nansum(grid.production),
                            "refor_area": [refor_area],
                            "defor_area": [defor_area],
                            'percent_manaff': percent_manaff,
                            'percent_nataff': percent_nataff,
                            'percent_noaff': percent_noaff,
                            "spat_constraint": [spat_constraint],
                            "weight": [str(lam)],
                            "grain_max": [grain_max],
                            "stover_removal": [stover_removal],
                            'profit_margin': [profit_margin_method],
                            "aff_scenario": [aff_scenario],
                            'scenario_id':[scenario_id],
                            })

    # grid.to_csv("test_" + str(lam) + ".csv", index = False)

    logger.info('Final beef production: {}'.format(totaldf.production.iloc[0]))
    logger.info('Final emissions: {}'.format(totaldf.total_emissions.iloc[0]))
    logger.info('Final costs: {}'.format(totaldf.total_costs.iloc[0]))
    logger.info('Final bgb change: {}'.format(totaldf.bgb_change.iloc[0]))

    totaldf.to_csv(dst + '/total_' + spat_constraint + "_" + str(round(lam,3)) + '_' + aff_scenario + ".csv", index=False)

    logger.info("Exporting CSV finished")

    country_data = grid[['ADM0_A3', 'total_costs', 'total_emissions', 'production', 'biomass', 'beef_area', "enteric",
                        "manure", "postfarm_emi", 'postfarm_cost', 'production_cost', 'establish_cost', 'opportunity_cost',
                         # "export_emissions", "transp_emission",  "processing_energy",
                         "n2o_emissions", "agb_change", "bgb_change","grass_BM","grain_BM", "stover_BM",
                         # 'grain_meat',     'stover_meat'
                         ]].groupby('ADM0_A3', as_index=False).sum()

    defor_area = grid.loc[(grid.changed == 1) & (grid.newarea == 1)][['ADM0_A3', 'beef_area']].groupby('ADM0_A3', as_index = False).sum()
    defor_area.rename(columns={'beef_area': 'deforested_area'}, inplace=True)

    country_data = country_data.merge(unconv_cells, how='left', left_on='ADM0_A3', right_on='ADM0_A3')
    country_data = country_data.merge(defor_area, how='left', left_on='ADM0_A3', right_on='ADM0_A3')

    country_data['total_costs'] = country_data['total_costs'] + country_data['aff_cost'].fillna(0).values
    country_data['total_emissions'] = country_data['total_emissions'] + country_data['opp_aff'].fillna(0).values + \
                                      country_data['opp_soc'].fillna(0).values

    country_data['horizon'] = horizon
    country_data['weight'] = lam
    country_data['aff_scenario'] = aff_scenario
    country_data['spat_constraint'] = spat_constraint
    country_data["prod_year"] = production_yr

    # if simulation == 'main':
    country_data.to_csv(dst + '/countrylevel_' + spat_constraint + "_" + str(round(lam,3)) + '_'  + aff_scenario + ".csv", index=False)

    # logger.info("Countries in country level: {}".format(country_data.ADM0_A3))
    #
    # # Export raster
    # # grid = grid[['cell_id', 'production', 'agb_change', 'bgb_change',"grass_BM","grain_BM", "stover_BM"]].groupby('cell_id', as_index=False).sum()
    # # export_raster(grid, ['production', 'agb_change', "grass_BM","grain_BM", "stover_BM"], dst, spat_constraint, lam, horizon, aff_scenario)
    #
    # logger.info("Exporting raster finished")
    #
    # if lam == 1:
    #     logger.info(grid.shape)
    #     for c in grid.columns:
    #         logger.info('Dtype of {}: {}'.format(c, grid[c].dtype))
    # return totaldf
    grid['new_production'] = np.where(grid.newarea == 1, grid.production, 0)
    grid['expansion_area'] = np.where(grid.newarea == 1, grid.beef_area, 0)

    # grid.loc[grid.ADM0_A3 == 'DEU'].to_csv('grid_' + spat_constraint + "_" + str(round(lam,3)) + "_" + aff_scenario + ".csv", index=False)
    for i in ['c_graz', 'c_stover', 'c_grain', 'c_meat']:
        grid[i] = np.nan_to_num(grid[i].values)
        grid.loc[grid.newarea == 1, i] = 0
    # grid.to_csv('./grid.csv', index = False)
    grid2 = grid[['cell_id', 'production', 'c_meat', 'n2o_emissions', 'c_graz', 'c_stover', 'c_grain','opportunity_cost',
                  'postfarm_cost', 'production_cost', 'establish_cost',
                  'agb_change', 'bgb_change',
                  'total_costs', 'total_emissions',
                  'expansion_area', "grass_BM","grain_BM", "stover_BM", 'new_production', 'beef_area']].groupby('cell_id', as_index=False).sum()
    # grid2.merge(grid.loc[grid.new_production > 0][['cell_id', 'new_production']], how = 'left')

    grid2['hotspot_factor'] = np.select(
        [grid2.production.astype(int) == grid2.c_meat.astype(int), # same production
         grid2.production < grid2.c_meat, # less production
         grid2.new_production > 0, # expansion
         ((grid2.c_grain + grid2.c_stover == 0) & (grid2.stover_BM + grid2.grain_BM > 0)), # grazing to mixed
         (grid2.c_grain + grid2.c_stover > 0) & (grid2.stover_BM + grid2.grain_BM == 0), # Mixed to grazing
         (grid2.c_graz < grid2.grass_BM) & (grid2.c_grain + grid2.c_stover + grid2.grain_BM + grid2.stover_BM == 0),# grazing to grazing, biomass increase
         (grid2.c_graz > grid2.grass_BM) & (grid2.c_grain + grid2.c_stover + grid2.grain_BM + grid2.stover_BM == 0),# grazing to grazing, biomass decrease
         (grid2.c_grain + grid2.c_stover > 0) & (grid2.grain_BM + grid2.stover_BM > 0) & (grid2.c_grain + grid2.c_stover + grid2.c_graz < grid2.grain_BM + grid2.stover_BM + grid2.grass_BM),# Mixed to mixed, biomass increase
        (grid2.c_grain + grid2.c_stover > 0) & (grid2.grain_BM + grid2.stover_BM > 0) & (grid2.c_grain + grid2.c_stover + grid2.c_graz > grid2.grain_BM + grid2.stover_BM + grid2.grass_BM)],  # Mixed to mixed, biomass decrease
    [1,2, 3, 4, 5, 6, 7, 8, 9], default = 10)

    if simulation == 'main':
        if lam in [0,1]:
            # grid2['newlanduse'] = np.select(
            #     [(grid2.grass_BM > 0) & (grid2.stover_BM + grid2.grain_BM == 0) & (grid2.n2o_emissions > 0),
            #      # unimproved pasture
            #      (grid2.grass_BM > 0) & (grid2.stover_BM + grid2.grain_BM == 0) & (grid2.n2o_emissions == 0),
            #      # improved pasture
            #      (grid2.grass_BM > 0) & (grid2.stover_BM > 0) & (grid2.grain_BM == 0),  # grass + stover,
            #
            #      grid2.grain_BM > 0], # grain-based
            #      # (grid2.grass_BM > 0) & (grid2.stover_BM == 0) & (grid2.grain_BM > 0),  # grass + grain
            #      # (grid2.grass_BM == 0) & (grid2.stover_BM > 0) & (grid2.grain_BM > 0)],  # stover + grain
            #     [1, 2, 3, 4, 5], default=6)
            grid2['newlanduse'] = np.select(
                [(grid2.grass_BM > 0) & (grid2.stover_BM + grid2.grain_BM == 0) & (grid2.n2o_emissions > 0),  # unimproved pasture
                 (grid2.grass_BM > 0) & (grid2.stover_BM + grid2.grain_BM == 0) & (grid2.n2o_emissions == 0), # improved pasture
                 (grid2.grass_BM > 0) & (grid2.stover_BM > 0) & (grid2.grain_BM == 0),# grass + stover

                 (grid2.grain_BM > 0) & (grid2.grain_BM < (grid2.grass_BM + grid2.stover_BM)),# grain main feed
                (grid2.grain_BM > 0) & (grid2.grain_BM > (grid2.grass_BM + grid2.stover_BM))],  # grain minority feed

            # (grid2.grass_BM > 0) & (grid2.stover_BM == 0) & (grid2.grain_BM > 0),# grass + grain
                 # (grid2.grass_BM == 0 )& (grid2.stover_BM  > 0) & (grid2.grain_BM > 0)],# stover + grain
                 [1, 2, 3, 4, 5], default = 6)
            col_list = ['production','hotspot_factor',
                'newlanduse', 'expansion_area',
                        'agb_change','bgb_change',
                        # 'cell_id',
                         "grass_BM","grain_BM", "stover_BM",
                        # 'beef_area',
                        'total_emissions','total_costs'
                                      # 'opportunity_cost','postfarm_cost', 'production_cost', 'establish_cost'
                        ]

            exp = grid.loc[grid.newarea == 1][['cell_id', 'bestlu']].rename(columns = {'bestlu':'prod_exp'})
            curr = grid.loc[grid.newarea == 0][['cell_id', 'bestlu']].rename(columns = {'bestlu':'prod_curr'})

            exp['prod_exp'] +=1
            curr['prod_curr'] +=1

            export_raster(exp, ['prod_exp'], dst, spat_constraint, round(lam, 3), aff_scenario)
            export_raster(curr, ['prod_curr'], dst, spat_constraint, round(lam, 3), aff_scenario)
            export_raster(grid2, col_list, dst, spat_constraint, round(lam,3), aff_scenario)

        else:
            col_list = [
                'production',
                # "grass_BM","grain_BM", "stover_BM",
                'expansion_area'
                                                    # 'total_costs','opportunity_cost','postfarm_cost', 'production_cost', 'establish_cost'
                        # 'agb_change', 'expansion_area',
                        ]

            export_raster(grid2, col_list, dst, spat_constraint, round(lam,3), aff_scenario)

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)
    l.handlers.pop(1)
    print('Handlers: ')
    print(l.handlers)

def parallelise(dst,  job, step, job_name, suit_area, graz_options, crop_options, feed_source, horizon, graz_cap,
                feed_area,
                scope, sim=None):

    simulation = job_name.split('-')[0]
    pnas_inputs = job_name.split('-')[1]
    # feed_inputs = job_name.split('-')[2]

    horizon_dict = {1: 20,
                    2: 30,
                    3: 50}
    horizon = horizon_dict[horizon]

    feed_source_dict = {1: 'country',
                        2: 'GLW2_country',
                        3: 'original',
                        4: 'threshold1head',
                        5: 'USA_GLW2_country',
                        6: 'countryGLW3',
                        7: 'country_GLPS_GLW3'}

    pnas_inputs = feed_source_dict[feed_source]

    setup_logger('log1', '/home/uqachare/model_file/current_whole_{}.log'.format(crop_options))
    log1 = logging.getLogger('log1')
    global grass_cols
    grass_cols = []

    if graz_options == 1:
        # Create list of grazing options
        for i in ["0250", "0375", "0500"]:
            for n in ["000", "050", "200"]:
                grass_cols.append("grass_" + i + "_N" + n)
    elif graz_options == 2:
        grass_cols = ['grass_0375_N000',  'grass_0375_N050']

    global foddercrop_area
    if crop_options == 1:
        foddercrop_list = ['barley', 'maize', 'rapeseed', 'rice', 'sorghum', 'soybean', 'wheat']
    elif crop_options == 2:
        foddercrop_list = ['barley', 'maize', 'soybean', 'wheat']
        feed_area_list = [i + '_area' for i in foddercrop_list]
        foddercrop_area = foddercrop_area[['ADM0_A3'] + feed_area_list]
        foddercrop_area[feed_area_list] = foddercrop_area[feed_area_list] / foddercrop_area[feed_area_list].sum(axis=1).values[:, None]

    elif crop_options == 3:
        foddercrop_list = ['maize', 'wheat']
        feed_area_list = [i + '_area' for i in foddercrop_list]
        foddercrop_area = foddercrop_area[['ADM0_A3'] + feed_area_list]
        foddercrop_area[feed_area_list] = foddercrop_area[feed_area_list] / foddercrop_area[feed_area_list].sum(axis=1).values[:, None]

    elif crop_options == 4:
        foddercrop_list = ['maize']
        feed_area_list = [i + '_area' for i in foddercrop_list]
        foddercrop_area = foddercrop_area[['ADM0_A3'] + feed_area_list]
        foddercrop_area[feed_area_list] = foddercrop_area[feed_area_list] / foddercrop_area[feed_area_list].sum(axis=1).values[:, None]
    else:
        print('Crop option scenario not in options')

    log1.info('foddercrop_area: {}'.format(foddercrop_area))

    grain_composition = grain_composition_clean(foddercrop_list)

    log1.info('grain_composition: {}'.format(grain_composition))

    domestic_feed = domestic_feed_clean(foddercrop_list)

    log1.info('domestic_feed: {}'.format(domestic_feed))

    FAO_producer_prices = producer_prices_clean(foddercrop_list)

    if simulation == 'uncertainty':
        profit_margin_method = random.randint(0, 1)
        grain_max = random.uniform(0.4, 0.8)
        stover_removal = random.uniform(0.2, 0.8)
    else:
        profit_margin_method = 0
        grain_max = 0.8
        stover_removal = 0.4

    init_grid = create_grid(foddercrop_list, profit_margin_method, pnas_inputs, simulation, suit_area, feed_area)
    feed_list = [c for c in init_grid.columns if 'grass' in c or c in foddercrop_list]
    producer_prices = parameter_sampling(init_grid, FAO_producer_prices, foddercrop_list, log1, simulation)
    fertiliser_prices = fertiliser_sampling(simulation)

    if scope:
        init_grid = init_grid.loc[init_grid.ADM0_A3 == scope]
        ctry = scope
    else:
        ctry = 'global'

    current_grid = current_state(init_grid,grain_composition, domestic_feed, producer_prices, fertiliser_prices,
                                 foddercrop_list, profit_margin_method, graz_cap, feed_area, scope, log1, job)

    log1.info('Current meat: {}'.format(current_grid.c_meat.sum()))
    log1.info('Current grid shape: {}'.format(current_grid.shape[0]))
    log1.info('Horizon: {}'.format(horizon))

    current_grid['net_fodder_area'] = np.where(current_grid['sum_area'] - current_grid['current_cropping'] < 0,
                                       0, current_grid['sum_area'] - current_grid['current_cropping'])

    current_grid = current_grid.drop(['sum_area'], axis = 1)

    current_grid = current_grid.loc[((current_grid[feed_list].sum(axis=1) > 0) & (~current_grid.region.isna()) &
                    (current_grid['suitable'].values * current_grid['cell_area'].values - current_grid['net_fodder_area'].values > 0)) |
                    (current_grid['c_meat'].values > 0)]

    if simulation == 'main':
        export_raster(current_grid, ['c_meat','c_opp_cost','c_tot_cost', 'c_ghg', 'c_cost','c_area'
                                     # 'export_costs',
                                     # 'loc_trans_cost', 'grass_cost', 'trans_feed_cost', 'curr_grain_cost'
                                     # 'c_grain','c_graz','c_occa',
                                     # , 'c_ghg'
        ], dst, '', '', '')

    index = 1
    scenarios = {}
    for spat_cons in ['global', 'country']:
            for a in ['noaff', 'regrowth']:
                scenarios[index] = [spat_cons, a]
                index += 1
    # step # step = 5

    aff_scenario = scenarios[job][1]
    spat = scenarios[job][0]

    # ME_conv_df = energy_conversion_sampling()
    ME_conv_df = pd.read_csv("tables/energy_conversion.csv")

    opp_soc = opp_soc_change_sampling('opp', aff_scenario, simulation, log1),
    exp_soc = opp_soc_change_sampling('expansion', aff_scenario, simulation, log1)

    if type(opp_soc) == tuple:
        opp_soc = opp_soc[0]

    log1.info('Start loop')

    if sim == 'cprice':

        weight_list = findcprice(np.nansum(current_grid.c_ghg), np.nansum(current_grid.c_tot_cost))
        for w in weight_list:
            setup_logger('log{}'.format(round(w, 3)),
                         "/home/uqachare/model_file/addweights_{}_{}.log".format(pnas_inputs, round(w, 3)))
            logaddweights = logging.getLogger('log{}'.format(round(w, 3)))
            pool = multiprocessing.Process(target=main,
                                           args=(current_grid.copy(),
                                                 foddercrop_list,
                                                 producer_prices,
                                                 fertiliser_prices,
                                                 ME_conv_df,
                                                 opp_soc,
                                                 exp_soc,
                                                 grain_max,
                                                 stover_removal,
                                                 horizon,
                                                 w,  # Weight
                                                 spat,  # Spatial constraint
                                                 aff_scenario,  # Afforestation scenario
                                                 scope,
                                                 logaddweights,
                                                 job,
                                                 profit_margin_method,
                                                 dst,
                                                 simulation))
            pool.start()

    else:
        for w in np.append(1 - np.logspace(0, 1, step - 1, base=0.01), 1):

            setup_logger('log{}'.format(round(w,3)), "/home/uqachare/model_file/opt_{}_{}.log".format(pnas_inputs, round(w,3)))
            log2 = logging.getLogger('log{}'.format(round(w,3)))

            pool = multiprocessing.Process(target=main,
                                           args=(current_grid.copy(),
                                                  foddercrop_list,
                                                  producer_prices,
                                                  fertiliser_prices,
                                                  ME_conv_df,
                                                  opp_soc,
                                                  exp_soc,
                                                  grain_max,
                                                  stover_removal, horizon,
                                                  w,  # Weight
                                                spat,  # Spatial constraint
                                                aff_scenario,  # Afforestation scenario
                                                scope,
                                                log2,
                                                 job,
                                                 profit_margin_method))
            pool.start()

if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('dst', help='Destination directory where results should be exported')
    argparser.add_argument('job', help="Job number defining scenarios", type=int)
    argparser.add_argument('step',
                           help="Number of weights to run in parallel, distributed between 0 and 1)", type=int)
    argparser.add_argument('job_name', help='Method for running the corrective algorithm: "GLW2_country" or "original"')
    argparser.add_argument('suit_area', help='Method for running the corrective algorithm: "GLW2_country" or "original"', type=int)
    argparser.add_argument('graz_options', help='Method for running the corrective algorithm: "GLW2_country" or "original"', type=int)
    argparser.add_argument('crop_options', help='Method for running the corrective algorithm: "GLW2_country" or "original"', type=int)
    argparser.add_argument('feed_source', help='Method for running the corrective algorithm: "GLW2_country" or "original"', type=int)
    argparser.add_argument('horizon', help='Time horizon to annualise', type=int)
    argparser.add_argument('graz_cap', help='Method for running the corrective algorithm: "GLW2_country" or "original"', type=int)
    argparser.add_argument('feed_area', help='Method for running the corrective algorithm: "GLW2_country" or "original"', type=int)

    argparser.add_argument('--scope', help='Optional country argument (using ISO3 country code) to run model, if None then simulation is run globally')
    argparser.add_argument('--sim', help='Type of simulation, None then run weight to generate frontier, if "cprice" then run model only for weight of the carbon price')

    args = argparser.parse_args()
    dst = args.dst
    job = args.job
    step = args.step
    job_name = args.job_name
    suit_area = args.suit_area
    graz_options = args.graz_options
    crop_options = args.crop_options
    feed_source = args.feed_source
    horizon = args.horizon
    graz_cap = args.graz_cap
    feed_area = args.feed_area

    scope = args.scope
    sim = args.sim

    parallelise(dst, job, step, job_name,
                suit_area,
                graz_options,
                crop_options,
                feed_source,
                horizon,
                graz_cap,
                feed_area,
                scope, sim)