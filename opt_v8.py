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

###########################################################################
### V8: All feeds & Take crop production from total attainable - other uses ###
###########################################################################


# import os
# os.environ["PROJ_LIB"] = "C:/Program Files/Anaconda3/envs/myenvironment/Library/share"
# from functools import wraps
# from memory_profiler import profile

######################### Load tables #########################

grouped_ME = pd.read_csv("tables/nnls_group_ME.csv") # Load country groups
grass_energy = pd.read_csv("tables/grass_energy.csv") # Load energy in grasses
beef_production = pd.read_csv("tables/beef_production.csv", index_col="Code") # Load country-level beef supply
fertiliser_prices = pd.read_csv("tables/fertiliser_prices.csv") # Load fertiliser prices
nutrient_req_grass = pd.read_csv("tables/nutrient_req_grass.csv") # Load nutrient requirement for grasses
beef_demand = pd.read_csv("tables/beef_demand.csv", index_col="ADM0_A3") # Load country-level beef demand
sea_distances = pd.read_csv("tables/sea_distances.csv") # Load averaged distances between countries
sea_t_costs = pd.read_csv("tables/sea_t_costs.csv") # Load transport costs
energy_conversion = pd.read_csv("tables/energy_conversion.csv") # Load transport costs

crop_area = pd.read_csv("tables/crop_area.csv") # proportion of crop areas by country
feed_energy = pd.read_csv("tables/feed_energy.csv") # ME in different feeds
partner_me = pd.read_csv("tables/partner_me.csv") # Weighted average of ME to meat conversion factor in export partner countries
potential_yields = pd.read_csv("tables/potential_yields.csv") # Potential yields by climate bins
yield_fraction = pd.read_csv("tables/yield_fraction.csv") # Fraction yield gap
percent_exported = pd.read_csv("tables/percent_exported.csv") # Fraction of exported feed
feedprices = pd.read_csv("tables/feedprices.csv") # Crop prices
crop_emissions_factors = pd.read_csv("tables/emissions_factors.csv") # N2O emission factors from N for crops
feedpartners = pd.read_csv("tables/feedpartners.csv") # Trade partners for each feed
expcosts = pd.read_csv("tables/expcosts.csv") # Export cost of feeds
sea_dist = pd.read_csv("tables/sea_dist.csv") # Sea distances matrix
exp_access = pd.read_csv("tables/partner_access.csv") # Access to market in importing country
fuel_partner = pd.read_csv("tables/fuel_partner.csv") # Fuel cost in partner countries
fertiliser_requirement = pd.read_csv("tables/fertiliser_requirement.csv") # fertiliser requirement per crop production
energy_efficiency = pd.read_csv("tables/energy_efficiency.csv") # Energy efficiency
crop_residues= pd.read_csv("tables/crop_residues.csv") # Energy efficiency
residue_energy= pd.read_csv("tables/residue_energy.csv") # Energy efficiency
stover_frac = pd.read_csv("tables/stover_frac.csv") # Fraction of stover feed for beef cattle vs all livestock

stover_other_uses = 0.6

index = 1
scenarios = {}
for spat_cons in ['global', 'country']:
    for j in ['me_to_meat', 'max_yield']:
        for k in [0, 100]:
            scenarios[index] = [spat_cons, j, k]
            index += 1

grass_cols = []
for i in ["0250", "0375", "0500"]:
    for n in ["000", "050", "200"]:
        grass_cols.append("grass_" + i + "_N" + n)
# for i in ["0250", "0500"]:
#     for n in ["000", "200"]:
#         grass_cols.append("grass_" + i + "_N" + n)
######################### Set parameters #########################

# N20 emission_factors from N application from Gerber et al 2016
emission_factors = {"maize": 0.0091, "soybean": 0.0066, "wheat": 0.008, "grass": 0.007}

# landuses = ['grass_low', 'grass_high', 'alfalfa_high', 'maize', 'soybean', 'wheat'] # k landuses to include in the simulation
# landuses = grass_cols + ['maize', 'soybean', 'wheat'] # k landuses to include in the simulation
landuses = grass_cols + ['cropland', 'stover'] # k landuses to include in the simulation

# Regression coefficients to get fertiliser needed for a given yield
grain_fertiliser = {'intercept' : {'maize' :3.831, 'soybean':2.287, 'wheat': 6.18018},
                    'coefficent': {'maize' :0.02416, 'soybean':0.01642, 'wheat': 0.03829}}

# me_forrage = {'maize': 9.6, 'wheat': 9.6,  'soybean': 9.2, 'alfalfa': 9.4} # Energy in DM from feedipedia.com (MJ/t)

fuel_efficiency = 0.4 # in l/km
# pasture_utilisation = 0.3 # Proportion of grazing biomass consumed
truck_emission_factor = 2.6712 # Emissions factor for heavy trucks (kg CO2/l)
sea_emissions =  0.048  # Emissions factor for heavy trucks (kg CO2/ton-km)
dressing = 0.625 # dressing percentage

# Energy consumption related to processing and packaging, MJ·kg CW-1,  from GLEAM
process_pack = 1.45

# Create list on column names for monthly effective temperature
months = ["efftemp0" + str(i) for i in range(1, 10)] + ["efftemp10", "efftemp11", "efftemp12"]
# column names for optimal costs/emissions sources
new_colnames = {'production': '_meat',
                'enteric': '_meth',
                'manure': '_manure',
                'export_emissions': '_exp_emiss',
                'export_cost': '_exp_costs',
                'transp_emission': '_trans_emiss',
                'transp_cost': '_trans_cost',
                'total_cost': '_tot_cost',
                'total_emission': '_ghg',
                'n2o_emissions': '_n2o',
                'production_cost': '_cost',
                'agb_change': '_agb_change',
                'opportunity_cost': '_opp_cost',
                'bgb_change': '_bgb_change',
                'processing_energy': '_process_energy',
                'beef_area': '_area'}

# Function used to determine suitable area based on the land cover raster
crop_list = ['barley', 'cassava', 'groundnut', 'maize', 'millet', 'oilpalm', 'potato', 'rapeseed', 'rice', 'rye',
             'sorghum', 'soybean', 'sugarbeet', 'sugarcane', 'sunflower', 'wheat']

def eac(price, rate = 0.05, lifespan = 100.):
    if rate == 0:
        return price/lifespan
    else:
        return(price * rate)/(1-(1+rate)**-lifespan)

#@profile
def scoring(feats, optimisation_method, gap_reduce, lam, me_to_meat, logger):
    """
    Finds the best landuse for each cell in the partition and returns the partition

    Arguments:
    feats (array)-> partion of dataframe

    Output: returns a gridded dataframe
    """
    # for i in ["maize_ratio", "soybean_ratio", "wheat_ratio"]:
    #     yield_ratios[i] = np.where(yield_ratios[i] + int(gap_reduce) > 100, 100, yield_ratios[i] + int(gap_reduce))

    yield_fraction[crop_list] = yield_fraction[crop_list] + gap_reduce
    yield_fraction[crop_list] = yield_fraction[crop_list].where(~(yield_fraction[crop_list] > 1), other = 1)

    # Initialise columns
    # cols = ['_meat','_meth', "_manure", '_cost','_trans_cost', '_trans_emiss', '_cstock','_n2o','_tot_cost', '_ghg',
    #         '_rel_ghg', '_rel_cost','_process_energy', '_opp_cost', "_area"]
    # for l in landuses:
    #     for c in cols:
    #       feats[l+c]=0.
    # feats['destination'] = 0
    feats['destination'] = pd.Series(np.zeros_like(feats.ADM0_A3, dtype=int), dtype='int8')

    # pd.Series([1, 2], dtype='int32')

    for l in grass_cols:

        # GLPS coding: 1 = Temperate, 2 = Arid, 3 = Humid
        # Grass productivity in t/ha is multiplied by the suitable area (ha)
        # Calculate biomass consumed (ton) = (grazed biomass (t/ha) * area (ha))
        biomass_consumed = feats[l].values * feats['suitable_area'].values

        # Calculate energy consumed (MJ) = biomass (kg) * energy content (MJ/t)
        # energy = biomass_consumed * feats.merge(grass_energy, how='left',
        #                                        left_on=['region', 'glps'], right_on=['region', 'glps'])['ME'].values

        # Subset energy conversion table to keep grazing systems and ME to meat conversion column
        me_table = energy_conversion.loc[energy_conversion.feed == 'grazing'][['region', 'climate', me_to_meat]]

        # Calculate meat produced (ton) = biomass consumed (t) * energy in grass (MJ/kg) * conversion factor (kg/MJ)
        energy = biomass_consumed * \
                             feats.merge(grass_energy, how='left',
                                               left_on=['region', 'glps'], right_on=['region', 'glps'])['ME'].values


        meat = energy * feats[['group', 'glps']].merge(me_table, how='left', left_on = ['group', 'glps'],
                                                       right_on = ['region', 'climate'])[me_to_meat].values
        # int, object
        # Adjust meat prodution based on effective temperature * dressing (%)
        feats[l + '_meat'] = np.sum(np.where(feats[months] < -1,
                                             (meat[:, None] - (meat[:, None] * (-0.0182*feats[months] - 0.0182)))/12.,
                                             meat[:, None]/12.), axis = 1) * dressing
        # Subset energy conversion table to keep grazing systems and biomass to methane conversion column
        me_table = energy_conversion.loc[energy_conversion.feed == 'grazing'][['region', 'climate', 'me_to_meth']]

        # Calculate methane production (ton CO2eq) = biomass consumed (t) * conversion factor (ton CO2eq/ton biomass)
        feats[l+'_meth'] = biomass_consumed * feats[['group', 'glps']].merge(me_table, how='left', left_on = ['group', 'glps'],
                                                       right_on = ['region', 'climate'])['me_to_meth'].values

        # Calculate N2O from manure from energy consumed with coefficients (ton CO2eq) = biomass consumed (ton) * conversion factor (ton CO2eq/tom DM)
        feats[l+'_manure'] = biomass_consumed * feats[['group']].merge(grouped_ME, how='left')['nitrous'].values

        # Calculate fertiliser application in tons (0 for rangeland, assuming no input)
        n_applied = int(l.split("_N")[1])

        if n_applied == 0:
            k_applied = 0
            p_applied = 0
        else:
            k_applied = feats['suitable_area'] * feats[['nutrient_availability']].merge(
                nutrient_req_grass, how='left')['K'].values * 2.2 / 1000.

            p_applied = feats['suitable_area'] * feats[['nutrient_availability']].merge(
                nutrient_req_grass,  how='left')['P'].values * 1.67 / 1000.

        fert_costs = feats[['ADM0_A3']].merge(fertiliser_prices, how='left')

        feats[l + '_cost'] = n_applied * fert_costs['n'].values + k_applied * fert_costs['k'].values + p_applied * \
                             fert_costs['p'].values

        # Calculate N20 emissions based on N application
        feats[l + '_n2o'] = (n_applied * emission_factors["grass"]) * 298

        # Number of trips to market; assuming 15 tons per trip, return
        # ntrips = (feats[l+'_meat']/int(15)+1)*2
        ntrips = np.ceil(feats[l+'_meat']/ int(15)) * 2

        # Transport cost to market: number of trips * transport cost ('000 US$)
        feats[l+'_trans_cost'] = ntrips * feats['transport_cost']/1000.
        # Transport emissions: number of trips * emissions per trip (tons CO2 eq)
        feats[l+'_trans_emiss'] = ntrips * feats['transport_emissions']/1000.
        # Estimate carbon content as 47.5% of remaining grass biomass. Then convert to CO2 eq (*3.67)
        feats[l+'_cstock'] = 0.475 * (1-(int(l.split("_")[1])/1000.)) * biomass_consumed * 3.67
        feats[l + '_area'] = feats['suitable_area']
        feats[l + '_opp_cost'] = feats['opp_cost'].astype(float) / 1000. * feats[l + '_area']

    # del biomass_consumed, ntrips, n_applied, p_applied, k_applied, meat, energy, biomass_consumed
    logger.info("Done with grass columns")

    for l in ['cropland']:

        #### Local feed consumption ####
        # Area cultivated (ha) = Suitable area (ha) x fraction area of feed per country
        areas_cultivated = feats['suitable_area'].values[:, None] * \
                           feats[['ADM0_A3']].merge(crop_area, how="left").drop('ADM0_A3', axis=1).values

        # Potential production (t) = Area cultivated (ha) x potential yields (t/ha)
        potential_prod = areas_cultivated * feats[['climate_bin']].merge(potential_yields, how="left").drop('climate_bin',
                                                                                                       axis=1).values
        # Actual production (t) = Potential production (t) x fraction yield gap x fraction of total grain production going to beef cattle

        actual_prod = potential_prod * feats[['ADM0_A3']].merge(yield_fraction, how="left").drop('ADM0_A3', axis=1).values - feats[['diff_' + i for i in crop_list]].values
        actual_prod = np.where(actual_prod < 0, 0 , actual_prod)

        feats[l + '_area'] = np.nansum(actual_prod / (feats[['climate_bin']].merge(potential_yields, how="left").drop('climate_bin',
                                                                                                       axis=1).values * feats[['ADM0_A3']].merge(yield_fraction, how="left").drop('ADM0_A3', axis=1).values), axis = 1)

        # Biomass consumed for domestic production (t) = actual production (t) x (1 - fraction exported feed)
        biomass_dom = actual_prod * (
                    1 - feats[['ADM0_A3']].merge(percent_exported, how="left").drop('ADM0_A3', axis=1).values)

        # Biomass consumed for domestic production (t) = actual production (t) x fraction exported feed
        biomass_exported = actual_prod * feats[['ADM0_A3']].merge(percent_exported, how="left").drop('ADM0_A3',
                                                                                                   axis=1).values

        # ME in conversion per region and climate
        me_table = energy_conversion.loc[energy_conversion.feed == 'mixed'][['region', 'climate', me_to_meat]]

        # Meat production (t) = sum across feeds (Domestic biomass (t) x ME in feed (MJ/kd DM)) x ME to beef conversion ratio
        meat = np.nansum(biomass_dom * feed_energy.iloc[0].values[None, :], axis=1) * \
               feats[['group', 'glps']].merge(me_table, how='left', left_on=['group', 'glps'],
                                              right_on=['region', 'climate'])[me_to_meat].values
        # Update meat production after climate penalty
        local_meat = np.sum(np.where(feats[months] < -1,
                                             (meat[:, None] - (meat[:, None] * (-0.0182*feats[months] - 0.0182)))/12.,
                                             meat[:, None]/12.), axis = 1) * dressing

        # Get methane conversion factor based on region and climate
        me_table = energy_conversion.loc[energy_conversion.feed == 'mixed'][['region', 'climate', 'me_to_meth']]

        # Calculate methane produced from local beef production (ton) = biomass consumed (ton) x biomass-methane conversion (ton/ton)
        local_methane = np.nansum(biomass_dom, axis=1) * \
                                 feats[['group', 'glps']].merge(me_table, how='left', left_on=['group', 'glps'],
                                                                right_on=['region', 'climate'])['me_to_meth'].values

        # Calculate N2O from manure from energy consumed with coefficients (ton CO2eq) = biomass consumed (ton) * conversion factor (ton CO2eq/tom DM)
        local_manure = np.nansum(biomass_dom, axis=1) * feats[['group']].merge(grouped_ME, how='left')[
            'nitrous'].values

        # Calculate nitrous N2O (ton) = Actual production (ton) x fertiliser requirement (kg) x crop_emission factors (% per thousand)
        feats[l + '_n2o'] = np.nansum(actual_prod * fertiliser_requirement['fertiliser'].values[None, :] * (
                    crop_emissions_factors['factor'].values[None, :] / 100), axis=1)
        logger.info("Done with local meat production")

        ##### Exported feed #####
        # Suitable area x fraction of feed for domestic use x potential yields x yield gap fraction
        meat_abroad = np.zeros_like(feats.ADM0_A3, dtype = float)
        methane_abroad = np.zeros_like(feats.ADM0_A3, dtype = float)
        manure_abroad = np.zeros_like(feats.ADM0_A3, dtype = float)
        exp_costs = np.zeros_like(feats.ADM0_A3, dtype = float)
        sea_emissions_ls = np.zeros_like(feats.ADM0_A3, dtype = float)
        emissions_partner_ls = np.zeros_like(feats.ADM0_A3, dtype = float)
        trancost_partner_ls = np.zeros_like(feats.ADM0_A3, dtype = float)

        for f in feedpartners['feed'].unique():
            ### Meat produced abroad
            # Quantity of feed f exported
            qty_exported = ((feats['suitable_area'].values * \
                           feats[['ADM0_A3']].merge(crop_area[['ADM0_A3', f + '_area']], how="left").drop('ADM0_A3',
                                                                                                          axis=1)[
                               f + '_area'].values * \
                           feats[['climate_bin']].merge(potential_yields[['climate_bin', f + '_potential']],
                                                        how="left").drop('climate_bin', axis=1)[
                               f + '_potential'].values * \
                           feats[['ADM0_A3']].merge(yield_fraction, how="left").drop('ADM0_A3', axis=1)[f].values) - feats['diff_' + f].values) *\
                           feats[['ADM0_A3']].merge(percent_exported[['ADM0_A3', f]], how="left").drop('ADM0_A3',
                                                                                                       axis=1)[f].values
            qty_exported = np.where(qty_exported < 0, 0, qty_exported)

            meat_abroad = meat_abroad + np.nansum(qty_exported[:, None] * \
                                      feats[['ADM0_A3']].merge(feedpartners.loc[feedpartners.feed == f],
                                                               how='left').drop(['ADM0_A3', 'feed'], axis=1).values * \
                                      feed_energy[f].iloc[0] * partner_me[me_to_meat].values[None, :], axis=1)

            ### Methane emitted abroad
            methane_abroad = methane_abroad + np.nansum(qty_exported[:, None] * \
                                      feats[['ADM0_A3']].merge(feedpartners.loc[feedpartners.feed == f],
                                                               how='left').drop(['ADM0_A3', 'feed'], axis=1).values * \
                                      partner_me["me_to_meth"].values[None, :], axis=1)

            ### N20 emitted from manure abroad
            manure_abroad = manure_abroad + np.nansum(qty_exported[:, None] * \
                                        feats[['ADM0_A3']].merge(feedpartners.loc[feedpartners.feed == f],
                                                                 how='left').drop(['ADM0_A3', 'feed'], axis=1).values * \
                                        partner_me["nitrous"].values[None, :], axis=1)

            ### Export cost
            exp_costs = exp_costs + np.nansum(qty_exported[:, None] * \
                                      feats[['ADM0_A3']].merge(feedpartners.loc[feedpartners.feed == f],
                                                               how='left').drop(['ADM0_A3', 'feed'], axis=1).values * \
                                      feats[['ADM0_A3']].merge(expcosts.loc[expcosts.feed == f], how='left').drop(
                                          ['ADM0_A3', 'feed'], axis=1).values,
                                      axis=1)

            ### sea emissions (ton)
            sea_emissions_ls = sea_emissions_ls + np.nansum(qty_exported[:, None] * \
                                feats[['ADM0_A3']].merge(feedpartners.loc[feedpartners.feed == f], how='left').drop(
                                    ['ADM0_A3', 'feed'], axis=1).values * \
                                feats[['ADM0_A3']].merge(sea_dist, how='left').drop(['ADM0_A3'],
                                                                                    axis=1).values * sea_emissions,
                                axis=1) / 1000.

            ### Local transport cost in importing country
            ntrips_local_transp = qty_exported[:, None] * \
                                  feats[['ADM0_A3']].merge(feedpartners.loc[feedpartners.feed == f], how='left').drop(
                                      ['ADM0_A3', 'feed'], axis=1).values / int(15) * 2

            trancost_partner_ls = trancost_partner_ls + np.nansum(ntrips_local_transp * exp_access['access'].values[None, :] * fuel_partner[
                                                                                                'Diesel'].values[None,
                                                                                            :] * fuel_efficiency / 1000., axis=1)

            emissions_partner_ls = emissions_partner_ls + np.nansum(ntrips_local_transp * exp_access['access'].values[None,
                                                      :] * fuel_efficiency * truck_emission_factor / 1000., axis=1)
            logger.info("   Done with {}".format(f))

            ### Local transport emissions in importing country
        logger.info("Done looping through feeds")

        local_cost = np.nansum(
            biomass_dom * feats[['ADM0_A3']].merge(feedprices, how="left").drop("ADM0_A3", axis=1).values, axis=1)

        # Get price from trade database
        # Cost of producing feed to be exported

        # Number of trips to bring feed to port
        ntrips_feed_exp = np.nansum(biomass_exported, axis=1) / int(15) * 2
        ntrips_feed_exp = np.where(ntrips_feed_exp < 0, 0, ntrips_feed_exp)
        # Cost of sending feed to port
        feed_to_port_cost = ntrips_feed_exp * feats["distance_port"] * feats['Diesel'] * fuel_efficiency / 1000.

        # Total cost of exporting feed
        # Emissions from transporting feed to nearest port (tons)
        feed_to_port_emis = ntrips_feed_exp * feats[
            'distance_port'] * fuel_efficiency * truck_emission_factor / 1000.

        # Number of trips to markets
        ntrips_beef_mkt = local_meat / int(15) * 2
        ntrips_beef_mkt = np.where(ntrips_beef_mkt < 0, 0, ntrips_beef_mkt)

        # Emissions from transporting beef to market (tons)
        beef_trans_emiss = ntrips_beef_mkt * feats['transport_emissions'] / 1000.

        # Cost from transporting beef to market ('000 dollars)
        beef_trans_cost = ntrips_beef_mkt * feats['transport_cost'] / 1000.
        logger.info("Done calculating costs and emissions")

        feats[l + '_meat'] = meat_abroad + local_meat
        feats[l + '_meth'] = methane_abroad + local_methane
        feats[l + '_manure'] = manure_abroad + local_manure
        feats[l + '_cost'] = local_cost
        feats[l + '_trans_cost'] = beef_trans_cost + feed_to_port_cost + exp_costs + trancost_partner_ls
        feats[l + '_trans_emiss'] = beef_trans_emiss + feed_to_port_emis + sea_emissions_ls + emissions_partner_ls
        feats[l + '_cstock'] = 0

        feats[l + '_opp_cost'] = feats['opp_cost'].astype(float) / 1000. * feats[l + '_area']

        logger.info("Done writing cropland columns")

        del beef_trans_emiss , feed_to_port_emis , sea_emissions_ls , emissions_partner_ls,\
            beef_trans_cost , feed_to_port_cost , exp_costs , trancost_partner_ls, local_cost, manure_abroad , \
            local_manure, methane_abroad , local_methane, meat_abroad , local_meat, ntrips_beef_mkt, ntrips_feed_exp,\
            meat, biomass_dom, areas_cultivated

    for l in ['stover']:
        # stover_production = potential_prod * \
        #                     feats[['ADM0_A3']].merge(yield_fraction, how="left").drop('ADM0_A3', axis=1).values * \
        #                     crop_residues.iloc[0].values[None,:] * \
        #                     feats[['ADM0_A3']].merge(stover_frac, how="left")['beef_frac'].values[: , None] * \
        #                     stover_other_uses
        stover_production = feats[['diff_' + i for i in crop_list]].values * crop_residues.iloc[0].values[None,:] * stover_other_uses

        stover_energy = np.nansum(stover_production * residue_energy.iloc[0].values[None,:], axis = 1)

        me_table = energy_conversion.loc[energy_conversion.feed == 'mixed'][['region', 'climate', me_to_meat]]
        meat_stover = stover_energy * feats[['group', 'glps']].merge(me_table, how='left', left_on=['group', 'glps'],
                                              right_on=['region', 'climate'])[me_to_meat].values
        # Update meat production after climate penalty
        feats[l + '_meat'] = np.sum(np.where(feats[months] < -1,
                                             (meat_stover[:, None] - (meat_stover[:, None] * (-0.0182*feats[months] - 0.0182)))/12.,
                                             meat_stover[:, None]/12.), axis = 1) * dressing

        # Get methane conversion factor based on region and climate
        me_table = energy_conversion.loc[energy_conversion.feed == 'mixed'][['region', 'climate', 'me_to_meth']]

        # Calculate nitrous N2O (ton) = Actual production (ton) x fertiliser requirement (kg) x crop_emission factors (% per thousand)
        feats[l + '_n2o'] = np.nansum(actual_prod * fertiliser_requirement['fertiliser'].values[None, :] * (
                    crop_emissions_factors['factor'].values[None, :] / 100), axis=1)
        feats[l + '_meth'] = np.nansum(stover_production, axis=1) * \
                                 feats[['group', 'glps']].merge(me_table, how='left', left_on=['group', 'glps'],
                                                                right_on=['region', 'climate'])['me_to_meth'].values
        feats[l + '_manure'] = np.nansum(stover_production, axis=1) * feats[['group']].merge(grouped_ME, how='left')[
            'nitrous'].values

        ntrips_beef_mkt = feats[l + '_meat'] / int(15) * 2
        ntrips_beef_mkt = np.where(ntrips_beef_mkt < 0, 0, ntrips_beef_mkt)

        feats[l + '_trans_cost'] = ntrips_beef_mkt * feats['transport_cost'] / 1000.
        feats[l + '_trans_emiss'] = ntrips_beef_mkt * feats['transport_emissions'] / 1000.
        feats[l + '_cost'] = 0
        feats[l + '_cstock'] = 0
        feats[l + '_opp_cost'] = 0
        feats[l + '_area'] = 0

    # Drop monthly temperature and other crop uses columns
    feats = feats.drop(months + ['diff_' + crop for crop in crop_list], axis = 1)

    # Only keep cells where at least 1 feed option produces meat
    # logger.info(feats.shape[0])
    feats = feats.loc[feats[[l + '_meat' for l in landuses]].sum(axis=1) > 0]

    feats['est_cost'] = eac(feats['est_cost'])
    logger.info("Done calculating establishment cost")

    ### SCENARIOS

    for l in landuses:

        # soc_change= {'crop': {'from_tree':-0.42, 'from_pasture': -0.59},
        #              'pasture': {'from_crop': 0.19,'from_tree': 0.08}}

        feats[l + '_process_energy'] = feats[l+'_meat'].values * process_pack * feats[['ADM0_A3']].merge(energy_efficiency, how = 'left')['energy'].fillna(0).values
        # For all landuse, calculate total costs over 20 years

        feats[l + '_tot_cost'] = feats['est_cost'] + \
                                 (feats[l + '_cost'] + feats[l + '_trans_cost'] + feats[l + '_opp_cost'])
        # For all landuse, calculate emissions over 20 years
        flow = feats[l + '_n2o'] + feats[l + '_meth'] + feats[l + '_manure'] + feats[l + '_trans_emiss'] + feats[l + '_process_energy']
        # For all landuse, calculate change in carbon stock
        agb_change = feats['carbon_stock'] * feats['suitable_area'] * 3.67 - feats[l + '_cstock']

        if l in ['cropland', 'stover']:
            bgb_change = ((-0.59 * feats['pasture_area'] * feats['soil_carbon10km']) + (-0.42 * feats['tree_area'] * feats['soil_carbon10km'])) * 3.67 *-1
        else:
            bgb_change = ((0.19 * feats['crop_area'] * feats['soil_carbon10km']) + (0.08 * feats['tree_area'] * feats['soil_carbon10km'])) * 3.67 *-1

        # Calculate total loss of carbon
        feats[l + '_agb_change'] = eac(agb_change, rate = 0)
        feats[l + '_bgb_change'] = eac(bgb_change, rate = 0)

        # feats[l + '_ghg'] = feats[l + '_agb_change'] + feats[l + '_bgb_change'] + flow
        feats[l + '_ghg'] = feats[l + '_agb_change'] + feats[l + '_bgb_change'] + flow - feats['opp_aff'] - feats[
            'opp_soc']

        del agb_change, bgb_change

        # Calculate costs and emissions per unit of meat produced for each land use. Convert 0 to NaN to avoid error
        feats[l + '_exp_emiss'] = 0
        feats[l + '_exp_costs'] = 0
        # Calculate relative GHG and costs

        feats[l+'_rel_ghg'] = np.where(feats[l+'_meat'] == 0, np.NaN, feats[l+'_ghg']/(feats[l+'_meat']))
        feats[l+'_rel_cost'] = np.where(feats[l+'_meat'] == 0, np.NaN, feats[l+'_tot_cost']/(feats[l+'_meat']))
        
        logger.info("Done calculating rel cost & emissions for  {}".format(l))

    feats = feats.dropna(how='all', subset=[l+'_rel_ghg' for l in landuses] + [l+'_rel_cost' for l in landuses])

    if optimisation_method == 'carbon_price':
        lam = lam/1000.
    # for each landuse, append to a list an array with the lowest weighted sum of rel ghg and rel cost per cell
    ### new weighted sum
    list_scores = [(feats[l + '_rel_ghg'].values * (1 - lam)) +
                   (feats[l + '_rel_cost'].values * lam) for l in landuses]

    # Stack arrays
    allArrays = np.stack(list_scores, axis=-1)

    # feats.drop('geometry', axis = 1).to_csv("allgrid.csv", index = False)

    feats['best_score'] = np.nanmin(allArrays, axis=1)
    # feats['bestlu'] = pd.Series(np.nanargmin(allArrays, axis=1), dtype='int8')
    feats['bestlu'] = np.nanargmin(allArrays, axis=1)

    logger.info(feats['bestlu'].dtype)
    logger.info(feats['bestlu'].head())

    del list_scores, allArrays

    # Column suffixes for landuse specific costs/emissions sources
    for i in new_colnames:
        feats[i] = np.take_along_axis(feats[[l + new_colnames[i] for l in landuses]].values,
                                                 feats['bestlu'].values[:, None], axis=1)
    logger.info(feats.shape[0])
    return feats

def trade(feats, optimisation_method, lam):

    for l in landuses:
        # For all landuse, calculate total costs over 20 years

        ntrips = (feats[l + '_meat'] / int(15) + 1) * 2

        # Calculate transport cost to nearest port
        feats[l + '_trans_cost'] = ntrips * feats["distance_port"] * feats['Diesel'] * fuel_efficiency/ 1000.

        # Calculate international transport costs based on average sea distance (km), transport cost to port, used for FOB ('000$) and transport cost percentage ($/(FOB*km))

        # Calculate transport costs as a function of quantity traded
        feats[l + '_exp_costs'] =  feats[l + '_meat'] * feats[['ADM0_A3']].merge(sea_t_costs[['ADM0_A3', 'tcost']], how='left')['tcost'].values

        # Transport emissions to port
        feats[l + '_trans_emiss'] = ntrips * feats["distance_port"] * fuel_efficiency * truck_emission_factor / 1000.
        # Transport emissions by sea
        feats[l + '_exp_emiss'] =  feats[['ADM0_A3']].merge(sea_distances[['ADM0_A3', 'ave_distance']], how='left')['ave_distance'].values * feats[l + '_meat'] * sea_emissions / 1000.

        feats[l + '_tot_cost'] = feats['est_cost'] + (feats[l + '_cost'] + feats[l + '_trans_cost'] + feats[l + '_opp_cost'] + feats[l + '_exp_costs'])
        # Calculate emissions over 20 years (t CO2 eq)
        flow = feats[l + '_n2o'] + feats[l + '_meth'] + feats[l + '_manure'] + feats[l + '_trans_emiss'] + feats[l + '_exp_emiss'] +feats[l + '_process_energy']

        # Calculate total loss of carbon (t CO2 eq)
        feats[l + '_ghg'] = feats[l + '_agb_change'] + feats[l + '_bgb_change'] + flow
        # feats[l + '_ghg'] = feats[l + '_agb_change'] + flow

        # Calculate costs and emissions per unit of meat produced for each land use. Convert 0 to NaN to avoid error

        # Calculate relative GHG and costs
        feats[l+'_rel_ghg'] = np.where(feats[l+'_meat'] == 0, np.NaN, feats[l+'_ghg']/(feats[l+'_meat']))
        feats[l+'_rel_cost'] = np.where(feats[l+'_meat'] == 0, np.NaN, feats[l+'_tot_cost']/(feats[l+'_meat']))

    feats = feats.dropna(how='all', subset=[l+'_rel_ghg' for l in landuses] + [l+'_rel_cost' for l in landuses])

    if feats.shape[0] > 0:
        if optimisation_method == 'carbon_price':
            lam = lam/1000.

        # New weighted sum
        list_scores = [(feats[i + '_rel_ghg'].values * (1 - lam)) +
                       (feats[i + '_rel_cost'].values * lam) for i in landuses]
        # Stack arrays horizontally
        allArrays = np.stack(list_scores, axis=-1)
        # Take lowest score across feed options
        feats['best_score'] = np.nanmin(allArrays, axis=1)
        # Get best feed option based on position of best score
        # feats['bestlu'] = pd.Series(np.nanargmin(allArrays, axis=1), dtype='int8')
        feats['bestlu'] = np.nanargmin(allArrays, axis=1)

        del list_scores, allArrays
        for i in new_colnames:
            feats[i] = np.take_along_axis(feats[[l + new_colnames[i] for l in landuses]].values,
                                          feats['bestlu'].values[:, None], axis=1)
        return feats

def export_raster(grid, resolution, export_column, export_folder, optimisation_method, gap_reduce, me_to_meat, lam, dem):

    bounds = list(grid.total_bounds)
    resolution = float(resolution)
    width = abs(int((bounds[2] - bounds[0]) / resolution))
    heigth = abs(int((bounds[3] - bounds[1]) / resolution))
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
            'transform': Affine(resolution, 0.0, bounds[0],
                                0.0, -resolution, bounds[3]),
            'compress': 'lzw',
            }
        # for m in meta: print(m, meta[m])
        out_fn = export_folder + '/' + optimisation_method + "_" + str(gap_reduce) + '_' + me_to_meat + '_' + str(lam)+ '_' + dem + ".tif"
          
        with rasterio.open(out_fn, 'w', **meta) as out:
            # Create a generator for geom and value pairs
            grid_cell = ((geom, value) for geom, value in zip(grid.geometry, grid[i]))

            burned = features.rasterize(shapes=grid_cell, fill=0, out_shape=out_shape, dtype = dt,
                                        transform=Affine(resolution, 0.0, bounds[0],
                                                         0.0, -resolution, bounds[3]))
            print("Burned value dtype: {}".format(burned.dtype))
            out.write_band(1, burned)

def main(export_folder ='.', optimisation_method= 'weighted_sum', lam = 0.5, demand_scenario = 'Demand',
         gap_reduce = 0, me_to_meat = 'me_to_meat', constraint = 'global', trade_scenario = 'trade',
         exp_global_cols = ['best_score', 'bestlu'], exp_changed_cols = ['best_score', 'bestlu', 'production'],
         rate = 0.05, lifespan = 20):
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

    LOG_FORMAT = "%(asctime)s - %(message)s"
    try:
        logging.basicConfig(
            # filename="/home/uqachare/model_file/logs_opt/opt_" + constraint + "_" + str(gap_reduce) + "_" + me_to_meat + "_" + str(lam) + '_' + dem +".log",
            filename="/home/uqachare/model_file/test.log",
            level=logging.INFO,
            format=LOG_FORMAT,
            filemode='w')
    except:
        logging.basicConfig(
            level=logging.INFO,
            format=LOG_FORMAT,
            filemode='w')
    logger = logging.getLogger()

    logger.info("Start loading grid")
    # logger.info(sea_dist.dtypes)
    grid = gpd.read_file("grid.gpkg")
    grid = grid.loc[grid.area < 1000000]

    for i in ['soilmoisture', "gdd", 'accessibility', 'ls_opp_cost', 'agri_opp_cost', 'est_area']:
        if i in grid.columns:
            grid = grid.drop(i, axis = 1)

    logger.info("Simulation start")
    logger.info('Me_to_meat scanerio: {}'.format(me_to_meat))
    logger.info('Weight scenario: {}'.format(lam))

    # Set amount of beef to be produced based on the chosen location

    logger.info('Constraint: {}'.format(constraint))
    if constraint == 'subsistence':
        logger.info('Shape of all grid: {}'.format(grid.shape))
        pastoral = grid.loc[grid.beef_gs > 0]
        logger.info('Shape of pastoral grid: {}'.format(pastoral.shape))
        grid = grid.loc[~(grid.beef_gs > 0)]
        logger.info('Shape of grid without pastoral: {}'.format(grid.shape))

        pastoral['production'] = pastoral['beef_gs'] * 0.01 * pastoral.suitable_area * 1e-3

        demand = beef_demand[demand_scenario].sum()
        logger.info('Total demand: {}'.format(demand))
        demand = demand - pastoral['production'].sum()
        logger.info('Non-pastoral demand: {}'.format(demand))
        logger.info('Pastoral production: {}'.format(pastoral['production'].sum()))

    else:
        demand = beef_demand[demand_scenario].sum()

    logger.info('Demand: {}'.format(demand))

    start_module = time.time()

    # Parallelise the scoring
    start = time.time()

    # Adjust other uses for future demand
    # Proportion of demand increase
    beef_demand['dem_increase'] = beef_demand[demand_scenario]/beef_demand['SSP1-NoCC2010']

    logger.info('New demand for other uses before: {}'.format(grid[['diff_maize']].loc[grid.diff_maize > 0].head()))
    other_uses = grid[['ADM0_A3']+['diff_' + i for i in crop_list]].merge(beef_demand[['dem_increase']], left_on = 'ADM0_A3', right_index = True)
    grid[['diff_' + i for i in crop_list]] = other_uses[['diff_' + i for i in crop_list]].values * other_uses['dem_increase'].values[:,None]
    del other_uses

    logger.info('New demand for other uses after: {}'.format(grid[['diff_maize']].loc[grid.diff_maize > 0].head()))

    grid = scoring(grid, optimisation_method, gap_reduce, lam, me_to_meat, logger = logger)

    print('### Done scoring in {} seconds'.format(time.time()-start))
    logger.info("Done scoring")

    start = time.time()

    grid = grid.reset_index(drop=True)
    total_production = 0

    ser = np.zeros_like(grid.ADM0_A3, dtype=int)

    grid['changed'] = pd.Series(ser, dtype='int8')
    grid['destination'] = pd.Series(ser, dtype='int8')
    grid['exporting'] = pd.Series(ser, dtype='int8')

    logger.info("Dtype of 'changed' column: {}".format(grid['changed'].dtype))
    # Get country-level domestic demand
    grid['dom_demand'] = grid.merge(beef_demand, how='left', left_on='ADM0_A3', right_index=True)['Demand']
    grid['dom_production'] = grid[['ADM0_A3']].merge(beef_production, how='left', left_on='ADM0_A3', right_on = ['Code'])['prop'].values * demand


    # Sort rows by increasing 'best score'
    grid = grid.sort_values('best_score')
    # Get cumulative country level production in order of increasing best score
    grid.loc[grid.changed == 0, 'cumdomprod'] = grid.groupby('ADM0_A3')['production'].transform(pd.Series.cumsum)
    print(grid.shape[0])
    print(grid['best_score'])

    if trade_scenario == 'trade':
        # Create original best score to compare scores for domestic vs international destination
        # grid['orig_bestscore'] = grid['best_score']
        # grid['orig_bestlu'] = grid['bestlu']

        # Set new production > 0 to compare old and new production to avoid new == old and infinite while loop
        new_production = 1
        countries_complete = []

        grid['tempdomprod'] = 0
        grid['prev_dom_prod'] = 0
        grid['total_prod'] = 0

        while total_production < demand and grid.loc[(grid.changed == 0)].shape[0] > 0 and new_production != 0:
            print('entered while loop')

            # Calculate old production to compare with new production
            old_production = grid.loc[grid.changed == 1, 'production'].sum()

            # Sort by increasing best score
            grid = grid.sort_values('best_score')

            # Recalculate cumulative production based on total production and according to sorted values
            grid.loc[grid.changed == 0, 'newcumprod'] = grid.loc[(grid.changed == 0) &
                                                              ~grid.ADM0_A3.isin(countries_complete), 'production'].cumsum()
            grid['cumprod'] = grid['total_prod'] + grid['newcumprod']

            # Convert cells to production if (1) cells have not been changed yet, (2) cumulative domestic production is lower than domestic demand OR the country of the cell is already exporting,
            # (3) Cumulative production is lower than global demand and (4) best score is lower than the highest score meeting these conditions

            if constraint in ['global', 'subsistence']:
                grid.loc[(grid['changed'] == 0) &
                         ((grid['cumdomprod'] < grid['dom_demand']) | (grid['exporting'] == 1)) &
                         ((demand + grid['production'] - grid['cumprod']) > 0) &
                         (grid['best_score'] <= grid.loc[(grid['changed'] == 0) &
                                                         ((demand + grid['production'] - grid['cumprod']) > 0) &
                                                         ((grid['cumdomprod'] < grid['dom_demand']) | (grid[
                                                                                                           'exporting'] == 1)), 'best_score'].max()), 'changed'] = 1
            elif constraint == 'country':
                if grid.loc[grid.changed == 1].empty:
                    print("No cell converted yet")
                    # grid['dom_production'] = grid[['ADM0_A3']].merge(beef_production, how='left', left_on='ADM0_A3', right_on = ['Code'])['Value']

                else:
                    # Get aggregated production of converted cells grouped by country
                    total_country_production = grid.loc[grid.changed == 1][['ADM0_A3', 'production']].groupby(
                        ['ADM0_A3'], as_index=False).sum()
                    total_country_production.loc[~total_country_production.ADM0_A3.isin(
                        grid.loc[(grid.changed == 1), 'ADM0_A3']), 'production'] = 0

                    grid.loc[grid.changed == 0, 'tempdomprod'] = grid.loc[grid.changed == 0].groupby(['ADM0_A3'])[
                        'production'].apply(lambda x: x.cumsum())
                    grid.loc[grid.changed == 0, 'prev_dom_prod'] = \
                    grid.loc[grid.changed == 0][['ADM0_A3']].merge(total_country_production,
                                                                   how='left', left_on='ADM0_A3',
                                                                   right_on='ADM0_A3')['production'].values
                    grid['prev_dom_prod'] = grid['prev_dom_prod'].fillna(0)
                    grid['cumdomprod'] = grid['tempdomprod'] + grid['prev_dom_prod']

                grid.loc[(grid['changed'] == 0) &
                         ((grid['cumdomprod'] < grid['dom_demand']) | (grid['exporting'] == 1)) &
                         ((demand + grid['production'] - grid['cumprod']) > 0) &
                         ((grid['dom_production'] + grid['production'] - grid['cumdomprod']) > 0) &
                         (~grid['ADM0_A3'].isin(countries_complete)) &
                         (grid['best_score'] <= grid.loc[(grid['changed'] == 0) &
                                                         ((demand + grid['production'] - grid['cumprod']) > 0) &
                                                         ((grid['cumdomprod'] < grid['dom_demand']) | (grid[
                                                                                                           'exporting'] == 1)), 'best_score'].max()), 'changed'] = 1

                test = grid.loc[(grid.changed == 1)][['ADM0_A3', 'dom_production', 'production']].groupby(
                    by='ADM0_A3', as_index=False).agg({'dom_production': 'mean',
                                                       'production': 'sum'})
                test.loc[test.production > test.dom_production, 'supplied'] = 1
                countries_complete = test.loc[test.production > test.dom_production, 'ADM0_A3'].values.tolist()
                # logger.info('Countries supplying their demand: ')
                # logger.info(countries_complete)

            ADM0_A3 = grid.loc[(grid.best_score <= grid.loc[grid.changed == 1].best_score.max()) &
                               (grid.exporting == 0) &
                               (grid.destination == 0) &
                               (grid.cumdomprod > grid.dom_demand), 'ADM0_A3']

            grid.loc[(grid['changed'] == 0) & (grid['ADM0_A3'].isin(ADM0_A3)), 'exporting'] = 1

            if grid.loc[(grid['ADM0_A3'].isin(ADM0_A3)) & (grid['changed'] == 0)].shape[0] > 0:
                grid.loc[(grid['ADM0_A3'].isin(ADM0_A3)) & (grid['changed'] == 0)] = trade(
                    grid.loc[(grid['ADM0_A3'].isin(ADM0_A3)) & (grid['changed'] == 0)], optimisation_method, lam)

            grid.loc[(grid['destination'] == 0) &
                     (grid['changed'] == 1), 'destination'] = np.where(grid.loc[(grid['destination'] == 0) &
                                                                                (grid[
                                                                                     'changed'] == 1), 'cumdomprod'] <
                                                                       grid.loc[(grid['destination'] == 0) & (
                                                                               grid['changed'] == 1), 'dom_demand'],
                                                                       1,
                                                                       2)
            total_production = grid.loc[grid.changed == 1, 'production'].sum()
            grid.loc[grid.ADM0_A3.isin(countries_complete), 'supplied'] = 1
            grid.loc[(grid.changed == 0) &
                     (~grid.ADM0_A3.isin(countries_complete)), 'total_prod'] = grid.loc[
                grid.changed == 1, 'production'].sum()
            print('total production: ', total_production)

            new_production = round(total_production,3) - round(old_production,3)
            logger.info("Total production: {}".format(total_production))

            # elif constraint == 'country':
            #     grid.loc[grid.changed == 0, 'newcumprod'] = grid.loc[(grid.changed == 0) &
            #                                                          ~grid.ADM0_A3.isin(
            #                                                              countries_complete), 'production'].cumsum()
            #     grid['cumprod'] = grid['total_prod'] + grid['newcumprod']
            #
            #
            #     if grid.loc[grid.changed == 1].empty:
            #         grid['dom_production'] = grid[['ADM0_A3']].merge(beef_production, how='left', left_on='ADM0_A3', right_on = 'Code')['Value']
            #
            #     else:
            #         total_country_production = grid.loc[grid.changed == 1].groupby(['ADM0_A3'])['production'].sum()
            #         new_cumdom_prod = grid.loc[grid.changed == 0].groupby('ADM0_A3')['production'].transform(
            #             pd.Series.cumsum)
            #         merged = grid.loc[grid.changed == 0][['ADM0_A3']].merge(total_country_production, how='left',
            #                                                                 left_on='ADM0_A3', right_index=True)
            #         grid.loc[grid.changed == 0, 'cumdomprod'] = merged['production'] + new_cumdom_prod
            #
            #     logger.info("Domestic prodution")
            #     logger.info(grid[['ADM0_A3', 'dom_production']].head(20))
            #     grid.loc[(grid['changed'] == 0) &
            #              ((grid['cumdomprod'] < grid['dom_demand']) | (grid['exporting'] == 1)) &
            #              ((demand + grid['production'] - grid['cumprod']) > 0) &
            #              ((grid['dom_production'] + grid['production'] - grid['cumdomprod']) > 0) &
            #              (~grid['ADM0_A3'].isin(countries_complete)) &
            #              (grid['best_score'] <= grid.loc[(grid['changed'] == 0) &
            #                                              ((demand + grid['production'] - grid['cumprod']) > 0) &
            #                                              ((grid['cumdomprod'] < grid['dom_demand']) | (grid[
            #                                                                                                'exporting'] == 1)), 'best_score'].max()), 'changed'] = 1
            #     # countries_complete = grid.loc[(grid.changed == 1) &
            #     #                               grid.cumdomprod < grid.dom_production, 'ADM0_A3'].unique()
            #     countries_complete = grid.loc[(grid.changed == 1) &
            #                                   (grid.cumdomprod > grid.dom_production), 'ADM0_A3']
            # # Select all countries that have been converted and that are not yet exporting for which we recalculate costs
            # ADM0_A3 = grid.loc[(grid.best_score <= grid.loc[grid.changed == 1].best_score.max()) &
            #                    (grid.exporting == 0) &
            #                    (grid.destination == 0) &
            #                    (grid.cumdomprod > grid.dom_demand), 'ADM0_A3']
            #
            # # Set these countries to exporting (0 = not exporting; 1 = exporting)
            # grid.loc[(grid['changed'] == 0) & (grid['ADM0_A3'].isin(ADM0_A3)), 'exporting'] = 1
            #
            # start = time.time()
            # # Recalculate costs and emissions of cells from listed countries
            # if grid.loc[(grid['ADM0_A3'].isin(ADM0_A3)) & (grid['changed'] == 0)].shape[0] > 0:
            #     grid.loc[(grid['ADM0_A3'].isin(ADM0_A3)) & (grid['changed'] == 0)] = trade(
            #         grid.loc[(grid['ADM0_A3'].isin(ADM0_A3)) & (grid['changed'] == 0)], optimisation_method, lam, c_price)
            # logger.info("Trade done in {}".format(time.time() - start))
            # # Set destination of production depending on whether domestic demand is met (1 = local; 2 = international)
            # grid.loc[(grid['destination'] == 0) &
            #          (grid['changed'] == 1), 'destination'] = np.where(grid.loc[(grid['destination'] == 0) &
            #                                                                     (grid['changed'] == 1), 'cumdomprod'] <
            #                                                            grid.loc[(grid['destination'] == 0) & (
            #                                                                        grid['changed'] == 1), 'dom_demand'], 1,
            #                                                            2)
            # # Recalculate total production
            # total_production = grid.loc[grid.changed == 1, 'production'].sum()
            #
            # new_production = total_production - old_production
            # logger.info("Total production: {}".format(total_production))
    else:
        # Calculate cumulative beef production by increasing score
        grid['cumprod'] = grid['production'].cumsum()
        # Convert cells as long as the targeted demand has not been achieved
        grid.loc[(demand + grid['production'] - grid['cumprod']) > 0, 'changed'] = 1
    # # Reset index which had duplicates somehow

    print('### Done sorting and selecting cells in  {} seconds. ###'.format(time.time() - start))
    print('### Main simulation finished in {} seconds. ###'.format(time.time()-start_module))
    logger.info("Main simulation finished")

    ######### Export #########
    cols = ['geometry','ADM0_A3', 'region', 'group','climate_bin', 'glps','suitable_area',  'best_score', 'bestlu', 'destination',
                                        'opp_cost', 'est_cost'] + list(new_colnames.keys())
           # +  ["diff_" + crop for crop in crop_list]

    grid = grid.loc[grid['changed'] == 1]

    if constraint == 'subsistence':
        logger.info('Production before merging: {}'.format(grid.production.sum()))
        grid = pd.concat([grid, gpd.read_file('global_south_results.gpkg')])
        logger.info('Production after merging: {}'.format(grid.production.sum()))

    newdf = pd.DataFrame({"production": grid.production.sum(),
                          "cost" : grid.total_cost.sum(),
                          "enteric": grid.enteric.sum(),
                          "manure": grid.manure.sum(),
                          "export_emissions": grid.export_emissions.sum(),
                          "transp_emission": grid.transp_emission.sum(),
                          "n2o_emissions": grid.n2o_emissions.sum(),
                          "agb_change": grid.agb_change.sum(),
                          "bgb_change": grid.bgb_change.sum(),
                          "processing_energy": grid.processing_energy.sum(),
                          "emissions": grid.total_emission.sum(),
                          "est_cost": grid.est_cost.sum(),
                          "production_cost": grid.production_cost.sum(),
                          "export_cost": grid.export_cost.sum(),
                          "transp_cost": grid.transp_cost.sum(),
                          "opportunity_cost": grid.opportunity_cost.sum(),
                          "beef_area": grid.beef_area.sum(),
                          "suitable_area": grid.suitable_area.sum(),
                          "optimisation_method": [optimisation_method],
                          "constraint": [constraint],
                          "weight" : [str(lam)],
                          "crop_gap": [str(gap_reduce)],
                          "beef_gap": [str(me_to_meat)]})

    dem = demand_scenario.split('NoCC')[1]
    newdf.to_csv(export_folder + '/' + constraint + "_" + str(gap_reduce) + "_" + me_to_meat + "_" + str(lam) + '_' + dem + ".csv", index=False)
    logger.info("Exporting CSV finished")
    grid[cols].to_file(export_folder + '/' + constraint + "_" + str(gap_reduce) + '_' + me_to_meat + '_' + str(lam) + '_' + dem + ".gpkg", driver="GPKG")
    logger.info("Exporting GPKG finished")
    export_raster(grid, 0.0833, ['production'], export_folder, constraint, gap_reduce, me_to_meat, lam, dem)
    logger.info("Exporting raster finished")

def parallelise(export_folder, optimisation_method, job_nmr):

    # scenarios_dict = {1: 'SSP1-NoCC2010', 2: 'SSP1-NoCC2030', 3: 'SSP1-NoCC2050', 4: 'SSP2-NoCC2010',
    #                   5: 'SSP2-NoCC2030', 6: 'SSP2-NoCC2050', 7: 'SSP3-NoCC2010', 8: 'SSP3-NoCC2030', 9: 'SSP3-NoCC2050'}
    # demand_scenario = scenarios_dict[int(job_nmr)]
    # demand_scenario = scenarios_dict[1]

    index = 1
    scenarios = {}
    for spat_cons in ['global', 'country', 'subsistence']:
    # for spat_cons in ['subsistence']:

        for j in ['me_to_meat', 'max_yield']:
            for k in [0, 1]:
                for d in ['SSP1-NoCC2010', 'SSP1-NoCC2050']:
                    scenarios[index] = [spat_cons, j, k, d]
                    scenarios[index] = [spat_cons, j, k, d]
                    index += 1

    for w in range(0, 110, 10):
    # for w in [50]:

        if optimisation_method == 'weighted_sum':
            w = w/100.
        pool = multiprocessing.Process(target=main, args = (export_folder, optimisation_method, w, scenarios[job_nmr][3],
                                                     scenarios[job_nmr][2],scenarios[job_nmr][1],scenarios[job_nmr][0],))
        pool.start()

if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('export_folder', help='Name of exported file')
    argparser.add_argument('optimisation_method', help='Which scenario of optimisation to run ("weighted_sum", "carbon_price", "cscc", "costs", "ghg")')
    argparser.add_argument('job_nmr', help="Job number", type=int)
    argparser.add_argument('--crop_gap', help="Optional argument: Percentage reduction in yield gap", default=4, type=int)
    argparser.add_argument('--me_to_meat', default='me_to_meat',
                           help="Optional argument: conversion of ME to meat, 'me_to_meat' or 'max_yield'")
    argparser.add_argument('--constraint', default='global',
                           help="Optional argument: Constraint on production, 'global' for no constraint or 'country' for keeping country-specific production")

    args = argparser.parse_args()
    export_folder = args.export_folder
    optimisation_method = args.optimisation_method
    crop_gap = args.crop_gap
    me_to_meat = args.me_to_meat
    job_nmr = args.job_nmr
    constraint = args.constraint

    parallelise(export_folder, optimisation_method, job_nmr)
