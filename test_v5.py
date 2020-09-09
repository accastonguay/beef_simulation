import geopandas as gpd
import pandas as pd
import numpy as np
import time
import multiprocessing
import logging
from affine import Affine
from rasterio import features
import rasterio

######################### Load tables #########################

grouped_ME = pd.read_csv("tables/nnls_group_ME.csv")  # Load country groups
grass_energy = pd.read_csv("tables/grass_energy.csv")  # Load energy in grasses
beef_production = pd.read_csv("tables/beef_production.csv", index_col="Code")  # Load country-level beef supply
fertiliser_prices = pd.read_csv("tables/fertiliser_prices.csv")  # Load fertiliser prices
nutrient_req_grass = pd.read_csv("tables/nutrient_req_grass.csv")  # Load nutrient requirement for grasses
beef_demand = pd.read_csv("tables/beef_demand.csv", index_col="ADM0_A3")  # Load country-level beef demand
sea_distances = pd.read_csv("tables/sea_distances.csv")  # Load averaged distances between countries
sea_t_costs = pd.read_csv("tables/sea_t_costs.csv")  # Load sea transport costs
energy_conversion = pd.read_csv("tables/energy_conversion.csv")  # Load energy conversion table
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
crop_residues= pd.read_csv("tables/crop_residues.csv")  # Residue to product ratio
residue_energy= pd.read_csv("tables/residue_energy.csv")  # Energy in crop residues
stover_frac = pd.read_csv("tables/stover_frac.csv")  # Fraction of stover feed for beef cattle vs all livestock
sc_change = pd.read_csv("tables/sc_change.csv")  # Fraction of stover feed for beef cattle vs all livestock
feed_composition= pd.read_csv("tables/feed_composition.csv")  # Energy efficiency

######################### Set parameters #########################

# Creat list of grazing options
grass_cols = []
for i in ["0250", "0375", "0500"]:
    for n in ["000", "050", "200"]:
        grass_cols.append("grass_" + i + "_N" + n)

landuses = grass_cols + ['cropland', 'stover'] # landuses to include in the simulation

stover_availability = 0.6  # Availability of crop residues

# N20 emission_factors from N application from Gerber et al 2016
emission_factors = {"grass": 0.007}

fuel_efficiency = 0.4 # fuel efficiency in l/km
truck_emission_factor = 2.6712 # Emissions factor for heavy trucks (kg CO2/l)
sea_emissions =  0.048  # Emissions factor for heavy trucks (kg CO2/ton-km)
dressing = 0.625 # dressing percentage

# Energy consumption related to processing and packaging, MJ·kg CW-1,  from GLEAM
process_pack = 1.45

# Create list monthly effective temperature column names
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

# Liat of crops
crop_list = ['barley', 'cassava', 'groundnut', 'maize', 'millet', 'oilpalm', 'potato', 'rapeseed', 'rice', 'rye',
             'sorghum', 'soybean', 'sugarbeet', 'sugarcane', 'sunflower', 'wheat']

def eac(cost, rate = 0.05, lifespan = 100.):
    """
    Function to annualize a cost based on a discount rate and a lifespan

    Arguments:
    cost (float) -> One-off cost to annualise
    rate (float)-> Discount rate
    lifespan (float)-> lifespan

    Output: returns the annualised cost as a float
    """

    if rate == 0: # For emissions, no discount rate
        return cost/lifespan
    else:
        return(cost * rate)/(1-(1+rate)**-lifespan)

def scoring(feats, optimisation_method, crop_yield, lam, beef_yield, logger, feed_option):
    """
    Finds the best landuse for each cell in the partition and returns the partition

    Arguments:
    feats (pandas dataframe) -> Main dataframe
    optimisation_method (str)-> Method for optimisation ('weighted_sum' or 'carbon_price')
    crop_yield (int)-> Scenario of crop yield (0 = current, 1 = no yield gap)
    lam (float)-> Lambda weight ([0,1])
    beef_yield (str)-> Scenario of beef yield ('me_to_meat' = current, 'max_yield' = no yield gap)
    logger (RootLogger) -> logger defined in main function
    feed_option (str)-> folder where the output file is exported

    Output: returns a gridded dataframe with scores
    """

    # Adjust yield fraction based on yield gap reduction scenario
    yield_fraction[crop_list] = yield_fraction[crop_list] + crop_yield

    # Cap yield fraction to 1 (cannot be high than attainable yield)
    yield_fraction[crop_list] = yield_fraction[crop_list].where(~(yield_fraction[crop_list] > 1), other=1)

    ### If using feed options 1 or 2, convert all cell or use difference between max yield and other uses
    if feed_option in ['v1', 'v2']:
        landuses = grass_cols + ['cropland', 'stover']

        for l in grass_cols:

            # Calculate biomass consumed (ton) = (grazed biomass (t/ha) * area (ha))
            biomass_consumed = feats[l].values * feats['suitable_area'].values

            # Subset energy conversion table to keep grazing systems and ME to meat conversion column.
            # Climate coding: 1 = Temperate, 2 = Arid, 3 = Humid
            meat_table = energy_conversion.loc[energy_conversion.feed == 'grazing'][['region', 'climate', beef_yield]]

            # Calculate energy consumed ('000 MJ) = biomass consumed (t) * energy in grass (MJ/kg)
            energy = biomass_consumed * \
                     feats.merge(grass_energy, how='left',
                                 left_on=['region', 'glps'], right_on=['region', 'glps'])['ME'].values

            # Meat production (t) = energy consumed ('000 MJ) * energy conversion (kg/MJ) * dressing (%)
            meat = energy * feats[['group', 'glps']].merge(meat_table, how='left', left_on=['group', 'glps'],
                                                           right_on=['region', 'climate'])[beef_yield].values * dressing

            # Adjust meat prodution based on effective temperature
            feats[l + '_meat'] = np.sum(np.where(feats[months] < -1,
                                                 (meat[:, None] - (
                                                             meat[:, None] * (-0.0182 * feats[months] - 0.0182))) / 12.,
                                                 meat[:, None] / 12.), axis=1)

            # Subset table to keep grazing systems and biomass to methane conversion column
            methane_table = energy_conversion.loc[energy_conversion.feed == 'grazing'][
                ['region', 'climate', 'me_to_meth']]

            # Calculate methane production (ton CO2eq) = biomass consumed (t) * conversion factor (ton CO2eq/ton biomass)
            feats[l + '_meth'] = biomass_consumed * \
                                 feats[['group', 'glps']].merge(methane_table, how='left', left_on=['group', 'glps'],
                                                                right_on=['region', 'climate'])['me_to_meth'].values

            # Calculate N2O from manure from energy consumed with coefficients (ton CO2eq) = biomass consumed (ton) * conversion factor (ton CO2eq/tom DM)
            feats[l + '_manure'] = biomass_consumed * feats[['group']].merge(grouped_ME, how='left')['nitrous'].values

            # Calculate fertiliser application in tons (0 for rangeland, assuming no N, P, K inputs)
            # Extract N application from column name, convert to ton
            n = int(l.split("_N")[1]) / 1000.

            if n == 0:
                n_applied = 0
                k_applied = 0
                p_applied = 0
            else:
                n_applied = int(l.split("_N")[1]) / 1000. * feats['suitable_area'].values

                k_applied = feats['suitable_area'] * feats[['nutrient_availability']].merge(
                    nutrient_req_grass, how='left')['K'].values * 2.2 / 1000.

                p_applied = feats['suitable_area'] * feats[['nutrient_availability']].merge(
                    nutrient_req_grass, how='left')['P'].values * 1.67 / 1000.

            # Get cost of fertilisers per country (USD/ton)
            fert_costs = feats[['ADM0_A3']].merge(fertiliser_prices, how='left')

            # Get total cost of fertilisers (USD) (N content in nitrate = 80%)
            feats[l + '_cost'] = n_applied * 1.2 * fert_costs['n'].values + k_applied * fert_costs[
                'k'].values + p_applied * \
                                 fert_costs['p'].values

            # Calculate N20 emissions based on N application = N application (ton) * grass emission factor (%) * CO2 equivalency
            feats[l + '_n2o'] = (n_applied * emission_factors["grass"]) * 298

            # Number of trips to market; assuming 15 tons per trip, return
            ntrips = np.ceil(feats[l + '_meat'] / int(15)) * 2

            # Transport cost to market: number of trips * transport cost ('000 US$)
            feats[l + '_trans_cost'] = ntrips * feats[['ADM0_A3']].merge(fuel_cost[['ADM0_A3', 'Diesel']],
                                                                         how='left',
                                                                         left_on='ADM0_A3',
                                                                         right_on='ADM0_A3')['Diesel'].values * \
                                       feats["accessibility"] * fuel_efficiency / 1000.
            # Transport emissions: number of trips * emissions per trip (tons CO2 eq)
            feats[l + '_trans_emiss'] = ntrips * feats[
                "accessibility"] * fuel_efficiency * truck_emission_factor / 1000.

            # Estimate carbon content as 47.5% of remaining grass biomass. Then convert to CO2 eq (*3.67)
            grazing_intensity = (1 - (int(l.split("_")[1]) / 1000.))
            feats[l + '_cstock'] = 0.475 * (biomass_consumed / grazing_intensity * (1 - grazing_intensity)) * 3.67

            # For grazing, convert all area
            feats[l + '_area'] = feats['suitable_area']

            # Calculate opportunity cot in '000 USD
            feats[l + '_opp_cost'] = feats['opp_cost'].astype(float) / 1000. * feats[l + '_area']

            # Change in soil carbon (t CO2 eq) = change from land use to grassland (%) * current land use area * current soil carbon (t/ha) * C-CO2 conversion * emission (-1)
            bgb_change = ((0.19 * feats['crop_area'] * feats['soil_carbon10km']) + (
                    0.08 * feats['tree_area'] * feats['soil_carbon10km'])) * 3.67 * -1

            # Annualise change in soil carbon
            feats[l + '_bgb_change'] = eac(bgb_change, rate=0)

        logger.info("Done with grass columns")

        for l in ['cropland']:

            #### Local feed consumption ####
            # Area cultivated (ha) = Suitable area (ha) x fraction area of feed per country
            areas_cultivated = feats['suitable_area'].values[:, None] * \
                               feats[['ADM0_A3']].merge(crop_area, how="left").drop('ADM0_A3', axis=1).values

            # Potential production (t) = Area cultivated (ha) x potential yields (t/ha)
            potential_prod = areas_cultivated * feats[['climate_bin']].merge(potential_yields, how="left").drop(
                'climate_bin', axis=1).values
            # Actual production (t) = Potential production (t) x fraction yield gap x fraction of total grain production going to beef cattle

            if feed_option == 'v1':
                # For feed option v1, convert 100% of cell
                actual_prod = potential_prod * feats[['ADM0_A3']].merge(yield_fraction, how="left").drop('ADM0_A3',
                                                                                                         axis=1).values

            elif feed_option == 'v2':
                # For feed option v2, keep other uses constant

                # Actual production (t) = Potential production (t) x fraction yield gap x fraction of total grain production going to beef cattle
                actual_prod = potential_prod * feats[['ADM0_A3']].merge(yield_fraction, how="left").drop(
                    'ADM0_A3', axis=1).values - feats[['diff_' + i for i in crop_list]].values

            actual_prod = np.where(actual_prod < 0, 0, actual_prod)

            feats[l + '_area'] = np.nansum(actual_prod / (
                        feats[['climate_bin']].merge(potential_yields, how="left").drop('climate_bin', axis=1).values *
                        feats[
                            ['ADM0_A3']].merge(yield_fraction, how="left").drop('ADM0_A3', axis=1).values), axis=1)

            # Biomass consumed for domestic production (t) = actual production (t) x (1 - fraction exported feed)
            biomass_dom = actual_prod * (
                    1 - feats[['ADM0_A3']].merge(percent_exported, how="left").drop('ADM0_A3', axis=1).values)

            # Biomass consumed for domestic production (t) = actual production (t) x fraction exported feed
            biomass_exported = actual_prod * feats[['ADM0_A3']].merge(percent_exported, how="left").drop('ADM0_A3',
                                                                                                         axis=1).values

            # Subset ME in conversion per region and climate
            meat_table = energy_conversion.loc[energy_conversion.feed == 'mixed'][['region', 'climate', beef_yield]]

            # Meat production (t) = sum across feeds (Domestic biomass (t) x ME in feed (MJ/kd DM)) x ME to beef conversion ratio * dressing (%)
            meat = np.nansum(biomass_dom * feed_energy.iloc[0].values[None, :], axis=1) * \
                   feats[['group', 'glps']].merge(meat_table, how='left', left_on=['group', 'glps'],
                                                  right_on=['region', 'climate'])[beef_yield].values * dressing

            # Update meat production after climate penalty
            local_meat = np.sum(np.where(feats[months] < -1,
                                         (meat[:, None] - (meat[:, None] * (-0.0182 * feats[months] - 0.0182))) / 12.,
                                         meat[:, None] / 12.), axis=1)
            # Get methane conversion factor based on region and climate
            methane_table = energy_conversion.loc[energy_conversion.feed == 'mixed'][
                ['region', 'climate', 'me_to_meth']]

            # Calculate methane produced from local beef production (ton) = biomass consumed (ton) x biomass-methane conversion (ton/ton)
            local_methane = np.nansum(biomass_dom, axis=1) * \
                            feats[['group', 'glps']].merge(methane_table, how='left', left_on=['group', 'glps'],
                                                           right_on=['region', 'climate'])['me_to_meth'].values

            # Calculate N2O from manure from energy consumed with coefficients (ton CO2eq) = biomass consumed (ton) * conversion factor (ton CO2eq/tom DM)
            local_manure = np.nansum(biomass_dom, axis=1) * feats[['group']].merge(grouped_ME, how='left')[
                'nitrous'].values

            # Calculate nitrous N2O (ton) = Actual production (ton) x fertiliser requirement (kg) x crop_emission factors (% per thousand)
            feats[l + '_n2o'] = np.nansum(actual_prod * fertiliser_requirement['fertiliser'].values[None, :] * (
                    crop_emissions_factors['factor'].values[None, :] / 100), axis=1)
            logger.info("Done with local meat production")

            ##### Exported feed #####
            # Create empty arrays to fill in
            meat_abroad = np.zeros_like(feats.ADM0_A3, dtype='float32')
            methane_abroad = np.zeros_like(feats.ADM0_A3, dtype='float32')
            manure_abroad = np.zeros_like(feats.ADM0_A3, dtype='float32')
            exp_costs = np.zeros_like(feats.ADM0_A3, dtype='float32')
            sea_emissions_ls = np.zeros_like(feats.ADM0_A3, dtype='float32')
            emissions_partner_ls = np.zeros_like(feats.ADM0_A3, dtype='float32')
            trancost_partner_ls = np.zeros_like(feats.ADM0_A3, dtype='float32')

            for f in feedpartners['feed'].unique():  # Loop though feeds
                ### Meat produced abroad
                # Quantity of feed f exported
                if feed_option == "v1":
                    # Qty exported (t) = Suitable area (ha) * crop area fraction * crop yield (t/ha) * yield gap (%) * export fraction
                    qty_exported = ((feats['suitable_area'].values * feats[['ADM0_A3']].merge(
                        crop_area[['ADM0_A3', f + '_area']], how="left").drop('ADM0_A3', axis=1)[f + '_area'].values * \
                                     feats[['climate_bin']].merge(potential_yields[['climate_bin', f + '_potential']],
                                                                  how="left").drop('climate_bin', axis=1)[
                                         f + '_potential'].values * \
                                     feats[['ADM0_A3']].merge(yield_fraction, how="left").drop('ADM0_A3', axis=1)[
                                         f].values)) * \
                                   feats[['ADM0_A3']].merge(percent_exported[['ADM0_A3', f]], how="left").drop(
                                       'ADM0_A3',
                                       axis=1)[f].values
                if feed_option == "v2":
                    # Qty exported (t) = (Suitable area (ha) * crop area fraction * crop yield (t/ha) * yield gap (%)) - production for other uses (t) * export fraction
                    qty_exported = ((feats['suitable_area'].values * \
                                     feats[['ADM0_A3']].merge(crop_area[['ADM0_A3', f + '_area']], how="left").drop(
                                         'ADM0_A3',
                                         axis=1)[
                                         f + '_area'].values * \
                                     feats[['climate_bin']].merge(potential_yields[['climate_bin', f + '_potential']],
                                                                  how="left").drop('climate_bin', axis=1)[
                                         f + '_potential'].values * \
                                     feats[['ADM0_A3']].merge(yield_fraction, how="left").drop('ADM0_A3', axis=1)[
                                         f].values) - feats['diff_' + f].values) * \
                                   feats[['ADM0_A3']].merge(percent_exported[['ADM0_A3', f]],
                                                            how="left").drop('ADM0_A3', axis=1)[f].values

                qty_exported = np.where(qty_exported < 0, 0, qty_exported)

                # Meat produced from exported feed (t) = Exported feed (t) * partner fraction (%) * energy in feed ('000 MJ/t) * energy conversion in partner country (t/'000 MJ) * dressing (%)
                meat_abroad = meat_abroad + np.nansum(qty_exported[:, None] * \
                                                      feats[['ADM0_A3']].merge(feedpartners.loc[feedpartners.feed == f],
                                                                               how='left').drop(['ADM0_A3', 'feed'],
                                                                                                axis=1).values * \
                                                      feed_energy[f].iloc[0] * partner_me[beef_yield].values[None, :],
                                                      axis=1) * dressing

                ### Methane emitted abroad (t CO2 eq) = Exported feed (t) * partner fraction (%) * methane emissions per biomass consumed (t/t)
                methane_abroad = methane_abroad + np.nansum(qty_exported[:, None] * \
                                                            feats[['ADM0_A3']].merge(
                                                                feedpartners.loc[feedpartners.feed == f],
                                                                how='left').drop(['ADM0_A3', 'feed'], axis=1).values * \
                                                            partner_me["me_to_meth"].values[None, :], axis=1)

                ### N2O from manure emitted abroad (t CO2 eq) = Exported feed (t) * partner fraction (%) * N2O emissions per biomass consumed (t/t)
                manure_abroad = manure_abroad + np.nansum(qty_exported[:, None] * \
                                                          feats[['ADM0_A3']].merge(
                                                              feedpartners.loc[feedpartners.feed == f],
                                                              how='left').drop(['ADM0_A3', 'feed'], axis=1).values * \
                                                          partner_me["nitrous"].values[None, :], axis=1)

                ### Export cost ('000 USD) = Exported feed (t) * partner fraction (%) * value of exporting crop c to partner p ('000 USD/t)
                exp_costs = exp_costs + np.nansum(qty_exported[:, None] * \
                                                  feats[['ADM0_A3']].merge(feedpartners.loc[feedpartners.feed == f],
                                                                           how='left').drop(['ADM0_A3', 'feed'],
                                                                                            axis=1).values * \
                                                  feats[['ADM0_A3']].merge(expcosts.loc[expcosts.feed == f],
                                                                           how='left').drop(
                                                      ['ADM0_A3', 'feed'], axis=1).values,
                                                  axis=1)

                ### Sea emissions (t CO2 eq) = Exported feed (t) * partner fraction (%) * sea distance from partner p (km) * sea emissions (kg CO2 eq/t-km) * kg-t conversion
                sea_emissions_ls = sea_emissions_ls + np.nansum(qty_exported[:, None] * \
                                                                feats[['ADM0_A3']].merge(
                                                                    feedpartners.loc[feedpartners.feed == f],
                                                                    how='left').drop(
                                                                    ['ADM0_A3', 'feed'], axis=1).values * \
                                                                feats[['ADM0_A3']].merge(sea_dist, how='left').drop(
                                                                    ['ADM0_A3'],
                                                                    axis=1).values * sea_emissions,
                                                                axis=1) / 1000.

                ### Number of local transport cost in importing country
                ntrips_local_transp = qty_exported[:, None] * \
                                      feats[['ADM0_A3']].merge(feedpartners.loc[feedpartners.feed == f],
                                                               how='left').drop(
                                          ['ADM0_A3', 'feed'], axis=1).values / int(15) * 2

                ### Transport cost in partner country ('000 USD) = trips * accessibility to market in partner country (km) * fuel cost in partner country * fuel efficiency * USD-'000 USD conversion
                trancost_partner_ls = trancost_partner_ls + np.nansum(
                    ntrips_local_transp * exp_access['access'].values[None, :] * fuel_partner[
                                                                                     'Diesel'].values[None,
                                                                                 :] * fuel_efficiency / 1000., axis=1)

                ### Transport emissions in partner country (t CO2 eq) = trips * accessibility to market in partner country (km) *
                # fuel efficiency (l/km) * truck emission factor (kg CO2 eq/l) * kg-ton conversion
                emissions_partner_ls = emissions_partner_ls + np.nansum(
                    ntrips_local_transp * exp_access['access'].values[None,
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
            feed_to_port_cost = ntrips_feed_exp * feats["distance_port"] * \
                                feats[['ADM0_A3']].merge(fuel_cost[['ADM0_A3', 'Diesel']],
                                                         how='left',
                                                         left_on='ADM0_A3',
                                                         right_on='ADM0_A3')['Diesel'].values * fuel_efficiency / 1000.

            # Total cost of exporting feed
            # Emissions from transporting feed to nearest port (tons)
            feed_to_port_emis = ntrips_feed_exp * feats[
                'distance_port'] * fuel_efficiency * truck_emission_factor / 1000.

            # Number of trips to markets
            ntrips_beef_mkt = local_meat / int(15) * 2
            ntrips_beef_mkt = np.where(ntrips_beef_mkt < 0, 0, ntrips_beef_mkt)

            beef_trans_cost = ntrips_beef_mkt * feats[['ADM0_A3']].merge(fuel_cost[['ADM0_A3', 'Diesel']],
                                                                         how='left',
                                                                         left_on='ADM0_A3',
                                                                         right_on='ADM0_A3')['Diesel'].values * \
                              feats["accessibility"] * fuel_efficiency / 1000.

            # Transport emissions: number of trips * emissions per trip (tons CO2 eq)
            beef_trans_emiss = ntrips_beef_mkt * feats[
                "accessibility"] * fuel_efficiency * truck_emission_factor / 1000.
            logger.info("Done calculating costs and emissions")

            feats[l + '_meat'] = meat_abroad + local_meat
            feats[l + '_meth'] = methane_abroad + local_methane
            feats[l + '_manure'] = manure_abroad + local_manure
            feats[l + '_cost'] = local_cost
            feats[l + '_trans_cost'] = beef_trans_cost + feed_to_port_cost + exp_costs + trancost_partner_ls
            feats[l + '_trans_emiss'] = beef_trans_emiss + feed_to_port_emis + sea_emissions_ls + emissions_partner_ls
            feats[l + '_cstock'] = pd.Series(np.zeros_like(feats.ADM0_A3, dtype=int), dtype='int8')
            logger.info('Cropland cstock : {}'.format(feats[l + '_cstock'] ))
            logger.info('Cropland cstock dtype: {}'.format(feats[l + '_cstock'].dtype))

            feats[l + '_opp_cost'] = feats['opp_cost'].astype(float) / 1000. * feats[l + '_area']

            bgb_change = (((-0.59 * feats['pasture_area'] * feats['soil_carbon10km']) + (
                        -0.42 * feats['tree_area'] * feats['soil_carbon10km'])) * 3.67 * -1) * feats[l + '_area'] / \
                         feats['suitable_area']
            feats[l + '_bgb_change'] = eac(bgb_change, rate=0)

            logger.info("Done writing cropland columns")

            del beef_trans_emiss, feed_to_port_emis, sea_emissions_ls, emissions_partner_ls, \
                beef_trans_cost, feed_to_port_cost, exp_costs, trancost_partner_ls, local_cost, manure_abroad, \
                local_manure, methane_abroad, local_methane, meat_abroad, local_meat, ntrips_beef_mkt, ntrips_feed_exp, \
                meat, biomass_dom, areas_cultivated

        for l in ['stover']:
            # Stover production (t) = crop production for other uses (t) * stover availability (%)
            stover_production = feats[['diff_' + i for i in crop_list]].values * crop_residues.iloc[0].values[None,
                                                                                 :] * stover_availability

            # Stover energy ('000 MJ) = sum across rows of Stover production (t) * stover energy (MJ/kg dm)
            stover_energy = np.nansum(stover_production * residue_energy.iloc[0].values[None, :], axis=1)

            # Subset meat table for mixed systems
            meat_table = energy_conversion.loc[energy_conversion.feed == 'mixed'][['region', 'climate', beef_yield]]

            # Meat production (t) = Stover energy ('000 MJ) * energy converion (kg/MJ) * dressing percentage
            meat_stover = stover_energy * \
                          feats[['group', 'glps']].merge(meat_table, how='left', left_on=['group', 'glps'],
                                                         right_on=['region', 'climate'])[beef_yield].values * dressing

            # Update meat production after climate penalty
            feats[l + '_meat'] = np.sum(np.where(feats[months] < -1,
                                                 (meat_stover[:, None] - (meat_stover[:, None] * (
                                                             -0.0182 * feats[months] - 0.0182))) / 12.,
                                                 meat_stover[:, None] / 12.), axis=1)

            # Subset methane table for mixed systems
            methane_table = energy_conversion.loc[energy_conversion.feed == 'mixed'][
                ['region', 'climate', 'me_to_meth']]

            # Methane emissions (t CO2 eq) = Biomass consumed (t) * CH4 emissions (t CO2 eq/t)
            feats[l + '_meth'] = np.nansum(stover_production, axis=1) * \
                                 feats[['group', 'glps']].merge(methane_table, how='left', left_on=['group', 'glps'],
                                                                right_on=['region', 'climate'])['me_to_meth'].values

            # Manure N20 (t CO2 eq) = Biomass consumed (t) * N2O emissions (t CO2 eq/t)
            feats[l + '_manure'] = np.nansum(stover_production, axis=1) * \
                                   feats[['group']].merge(grouped_ME, how='left')[
                                       'nitrous'].values

            # Trips to market
            ntrips_beef_mkt = feats[l + '_meat'] / int(15) * 2
            ntrips_beef_mkt = np.where(ntrips_beef_mkt < 0, 0, ntrips_beef_mkt)

            # Transport emissions = number of trips to nearest market * distance to market (km) * fuel efficeincy (l/km) * Diesel cost (USD/l)
            feats[l + '_trans_cost'] = ntrips_beef_mkt * feats["accessibility"] * fuel_efficiency * feats[
                ['ADM0_A3']].merge(fuel_cost[['ADM0_A3', 'Diesel']], how='left', left_on='ADM0_A3',
                                   right_on='ADM0_A3')['Diesel'].values / 1000.

            # Transport emissions = number of trips to nearest market * distance to market (km) * fuel efficeincy (l/km) * emissions factor (kg CO2/l) * kg/t conversion
            feats[l + '_trans_emiss'] = ntrips_beef_mkt * feats[
                "accessibility"] * fuel_efficiency * truck_emission_factor / 1000.

            # Make the following columns = 0
            feats[l + '_cost'] = pd.Series(np.zeros_like(feats.ADM0_A3, dtype=int), dtype='int8')
            feats[l + '_cstock'] = pd.Series(np.zeros_like(feats.ADM0_A3, dtype=int), dtype='int8')
            feats[l + '_opp_cost'] = pd.Series(np.zeros_like(feats.ADM0_A3, dtype=int), dtype='int8')
            feats[l + '_area'] = pd.Series(np.zeros_like(feats.ADM0_A3, dtype=int), dtype='int8')
            feats[l + '_n2o'] = pd.Series(np.zeros_like(feats.ADM0_A3, dtype=int), dtype='int8')
            feats[l + '_bgb_change'] = 0
    ### If using v3 feed option, i.e., feed mix from PNAS
    elif feed_option == 'v3':
        landuses = ['grazing', 'mixed']  # two types of land uses: grazing or mixed

        for l in landuses:
            for feed in ["grass", "grain", "stover"]:
                # Calculate area of feed for land use l (ha) = suitable area (ha) x percentage of feed for land use l
                feats[l + '_' + feed + '_area'] = feats['suitable_area'].values * feats[['region', 'glps']].merge(
                    feed_composition.loc[feed_composition.system == l], how='left',
                    left_on=['region', 'glps'], right_on=['region', 'climate'])[feed].values

                if feed == 'grass':
                    # Calculate grazing biomass consumption based on medium grazing intensity
                    biomass_consumed = feats['grass_0375_N050'].values * feats[l + '_' + feed + '_' + "area"]

                    # Subset energy conversion table for graing feed
                    me_table = energy_conversion.loc[energy_conversion.feed == 'grazing'][
                        ['region', 'climate', beef_yield]]

                    # Calculate energy consumed
                    energy = biomass_consumed * \
                             feats.merge(grass_energy, how='left',
                                         left_on=['region', 'glps'], right_on=['region', 'glps'])['ME'].values

                    meat = energy * feats[['group', 'glps']].merge(me_table, how='left', left_on=['group', 'glps'],
                                                                   right_on=['region', 'climate'])[
                        beef_yield].values * dressing
                    feats[l + '_' + feed + '_meat'] = np.sum(np.where(feats[months] < -1,
                                                                      (meat[:, None] - (meat[:, None] * (
                                                                              -0.0182 * feats[months] - 0.0182))) / 12.,
                                                                      meat[:, None] / 12.), axis=1)
                    me_table = energy_conversion.loc[energy_conversion.feed == 'grazing'][
                        ['region', 'climate', 'me_to_meth']]
                    feats[l + '_' + feed + '_meth'] = biomass_consumed * \
                                                      feats[['group', 'glps']].merge(me_table, how='left',
                                                                                     left_on=['group', 'glps'],
                                                                                     right_on=['region', 'climate'])[
                                                          'me_to_meth'].values

                    feats[l + '_' + feed + '_manure'] = biomass_consumed * \
                                                        feats[['group']].merge(grouped_ME, how='left')[
                                                            'nitrous'].values

                    n_applied = 50 / 1000.

                    if n_applied == 0:
                        k_applied = 0
                        p_applied = 0
                    else:
                        k_applied = feats[l + '_' + feed + '_' + "area"] * feats[['nutrient_availability']].merge(
                            nutrient_req_grass, how='left')['K'].values * 2.2 / 1000.

                        p_applied = feats[l + '_' + feed + '_' + "area"] * feats[['nutrient_availability']].merge(
                            nutrient_req_grass, how='left')['P'].values * 1.67 / 1000.

                    fert_costs = feats[['ADM0_A3']].merge(fertiliser_prices, how='left')

                    feats[l + '_' + feed + '_cost'] = n_applied * 1.2 * fert_costs['n'].values + k_applied * fert_costs[
                        'k'].values + p_applied * fert_costs['p'].values

                    # Calculate N20 emissions based on N application
                    feats[l + '_' + feed + '_n2o'] = (n_applied * emission_factors["grass"]) * 298
                    # feats[l + '_' + feed + '_cstock'] = 0.475 * 0.5 * biomass_consumed * 3.67

                    grazing_intensity = 375 / 1000.
                    feats[l + '_' + feed + '_cstock'] = 0.475 * (
                            biomass_consumed / grazing_intensity * (1 - grazing_intensity)) * 3.67

                    feats[l + '_' + feed + '_feed_cost'] = 0
                    feats[l + '_' + feed + '_feed_emission'] = 0

                    feats[l + '_' + feed + '_opp_cost'] = feats['opp_cost'].astype(float) / 1000. * feats[
                        l + '_' + feed + '_area']

                    bgb_change = ((0.19 * feats['crop_area'] * feats['soil_carbon10km']) + (
                            0.08 * feats['tree_area'] * feats['soil_carbon10km'])) * 3.67 * -1
                    feats[l + '_' + feed + '_bgb_change'] = eac(bgb_change, rate=0)
                    feats[l + '_' + feed + '_beefarea'] = feats[l + '_' + feed + '_area']

                elif feed == 'grain':
                    areas_cultivated = feats[l + '_' + feed + '_' + "area"].values[:, None] * \
                                       feats[['ADM0_A3']].merge(crop_area, how="left").drop('ADM0_A3', axis=1).values

                    # Potential production (t) = Area cultivated (ha) x potential yields (t/ha)
                    potential_prod = areas_cultivated * feats[['climate_bin']].merge(potential_yields, how="left").drop(
                        'climate_bin', axis=1).values

                    actual_prod = potential_prod * feats[['ADM0_A3']].merge(yield_fraction, how="left").drop('ADM0_A3',
                                                                                                             axis=1).values - \
                                  feats[['diff_' + i for i in crop_list]]
                    actual_prod = np.where(actual_prod < 0, 0, actual_prod)

                    biomass_dom = actual_prod * (
                            1 - feats[['ADM0_A3']].merge(percent_exported, how="left").drop('ADM0_A3', axis=1).values)

                    biomass_exported = actual_prod * feats[['ADM0_A3']].merge(percent_exported, how="left").drop(
                        'ADM0_A3',
                        axis=1).values

                    me_table = energy_conversion.loc[energy_conversion.feed == 'mixed'][
                        ['region', 'climate', beef_yield]]

                    meat = np.nansum(biomass_dom * feed_energy.iloc[0].values[None, :], axis=1) * \
                           feats[['group', 'glps']].merge(me_table, how='left', left_on=['group', 'glps'],
                                                          right_on=['region', 'climate'])[beef_yield].values
                    local_meat = np.sum(np.where(feats[months] < -1,
                                                 (meat[:, None] - (
                                                         meat[:, None] * (-0.0182 * feats[months] - 0.0182))) / 12.,
                                                 meat[:, None] / 12.), axis=1) * dressing

                    # Get methane conversion factor based on region and climate
                    me_table = energy_conversion.loc[energy_conversion.feed == 'mixed'][
                        ['region', 'climate', 'me_to_meth']]

                    # Calculate methane produced from local beef production (ton) = biomass consumed (ton) x biomass-methane conversion (ton/ton)
                    local_methane = np.nansum(biomass_dom, axis=1) * \
                                    feats[['group', 'glps']].merge(me_table, how='left', left_on=['group', 'glps'],
                                                                   right_on=['region', 'climate'])['me_to_meth'].values

                    # Calculate N2O from manure from energy consumed with coefficients (ton CO2eq) = biomass consumed (ton) * conversion factor (ton CO2eq/tom DM)
                    local_manure = np.nansum(biomass_dom, axis=1) * feats[['group']].merge(grouped_ME, how='left')[
                        'nitrous'].values

                    # Calculate nitrous N2O (ton) = Actual production (ton) x fertiliser requirement (kg) x crop_emission factors (% per thousand)
                    feats[l + '_' + feed + '_n2o'] = np.nansum(
                        actual_prod * fertiliser_requirement['fertiliser'].values[None, :] * (
                                crop_emissions_factors['factor'].values[None, :] / 100), axis=1)

                    ##### Exported feed #####
                    # Suitable area x fraction of feed for domestic use x potential yields x yield gap fraction
                    meat_abroad = np.zeros_like(feats.ADM0_A3, dtype=float)
                    methane_abroad = np.zeros_like(feats.ADM0_A3, dtype=float)
                    manure_abroad = np.zeros_like(feats.ADM0_A3, dtype=float)
                    exp_costs = np.zeros_like(feats.ADM0_A3, dtype=float)
                    sea_emissions_ls = np.zeros_like(feats.ADM0_A3, dtype=float)
                    emissions_partner_ls = np.zeros_like(feats.ADM0_A3, dtype=float)
                    trancost_partner_ls = np.zeros_like(feats.ADM0_A3, dtype=float)

                    for f in feedpartners['feed'].unique():
                        ### Meat produced abroad
                        # Quantity of feed f exported
                        qty_exported = ((feats['suitable_area'].values * \
                                         feats[['ADM0_A3']].merge(crop_area[['ADM0_A3', f + '_area']], how="left").drop(
                                             'ADM0_A3', axis=1)[f + '_area'].values * feats[['climate_bin']].merge(
                                    potential_yields[['climate_bin', f + '_potential']],
                                    how="left").drop('climate_bin', axis=1)[f + '_potential'].values * \
                                         feats[['ADM0_A3']].merge(yield_fraction, how="left").drop('ADM0_A3', axis=1)[
                                             f].values) - feats['diff_' + f].values) * \
                                       feats[['ADM0_A3']].merge(percent_exported[['ADM0_A3', f]], how="left").drop(
                                           'ADM0_A3', axis=1)[f].values

                        qty_exported = np.where(qty_exported < 0, 0, qty_exported)

                        meat_abroad = meat_abroad + np.nansum(qty_exported[:, None] * \
                                                              feats[['ADM0_A3']].merge(
                                                                  feedpartners.loc[feedpartners.feed == f],
                                                                  how='left').drop(['ADM0_A3', 'feed'], axis=1).values * \
                                                              feed_energy[f].iloc[0] * partner_me[beef_yield].values[
                                                                                       None,
                                                                                       :], axis=1)

                        ### Methane emitted abroad
                        methane_abroad = methane_abroad + np.nansum(qty_exported[:, None] * \
                                                                    feats[['ADM0_A3']].merge(
                                                                        feedpartners.loc[feedpartners.feed == f],
                                                                        how='left').drop(['ADM0_A3', 'feed'],
                                                                                         axis=1).values * \
                                                                    partner_me["me_to_meth"].values[None, :], axis=1)

                        ### N20 emitted from manure abroad
                        manure_abroad = manure_abroad + np.nansum(qty_exported[:, None] * \
                                                                  feats[['ADM0_A3']].merge(
                                                                      feedpartners.loc[feedpartners.feed == f],
                                                                      how='left').drop(['ADM0_A3', 'feed'],
                                                                                       axis=1).values * \
                                                                  partner_me["nitrous"].values[None, :], axis=1)

                        ### Export cost
                        exp_costs = exp_costs + np.nansum(qty_exported[:, None] * \
                                                          feats[['ADM0_A3']].merge(
                                                              feedpartners.loc[feedpartners.feed == f],
                                                              how='left').drop(['ADM0_A3', 'feed'],
                                                                               axis=1).values * \
                                                          feats[['ADM0_A3']].merge(expcosts.loc[expcosts.feed == f],
                                                                                   how='left').drop(
                                                              ['ADM0_A3', 'feed'], axis=1).values,
                                                          axis=1)

                        ### sea emissions (ton)
                        sea_emissions_ls = sea_emissions_ls + np.nansum(
                            qty_exported[:, None] * feats[['ADM0_A3']].merge(
                                feedpartners.loc[feedpartners.feed == f], how='left').drop(['ADM0_A3', 'feed'],
                                                                                           axis=1).values * \
                            feats[['ADM0_A3']].merge(sea_dist, how='left').drop(
                                ['ADM0_A3'], axis=1).values * sea_emissions,
                            axis=1) / 1000.

                        ### Local transport cost in importing country
                        ntrips_local_transp = qty_exported[:, None] * \
                                              feats[['ADM0_A3']].merge(feedpartners.loc[feedpartners.feed == f],
                                                                       how='left').drop(
                                                  ['ADM0_A3', 'feed'], axis=1).values / int(15) * 2

                        trancost_partner_ls = trancost_partner_ls + np.nansum(
                            ntrips_local_transp * exp_access['access'].values[None, :] * fuel_partner[
                                                                                             'Diesel'].values[None,
                                                                                         :] * fuel_efficiency / 1000.,
                            axis=1)

                        emissions_partner_ls = emissions_partner_ls + np.nansum(
                            ntrips_local_transp * exp_access['access'].values[None,
                                                  :] * fuel_efficiency * truck_emission_factor / 1000., axis=1)

                        ### Local transport emissions in importing country

                    local_cost = np.nansum(
                        biomass_dom * feats[['ADM0_A3']].merge(feedprices, how="left").drop("ADM0_A3", axis=1).values,
                        axis=1)

                    # Get price from trade database
                    # Cost of producing feed to be exported

                    # Number of trips to bring feed to port
                    ntrips_feed_exp = np.nansum(biomass_exported, axis=1) / int(15) * 2
                    ntrips_feed_exp = np.where(ntrips_feed_exp < 0, 0, ntrips_feed_exp)
                    # Cost of sending feed to port
                    feed_to_port_cost = ntrips_feed_exp * feats["distance_port"] * feats[['ADM0_A3']].merge(
                        fuel_cost[['ADM0_A3', 'Diesel']], how='left', left_on='ADM0_A3',
                        right_on='ADM0_A3')['Diesel'].values * fuel_efficiency / 1000.

                    # Total cost of exporting feed
                    # Emissions from transporting feed to nearest port (tons)
                    feed_to_port_emis = ntrips_feed_exp * feats[
                        'distance_port'] * fuel_efficiency * truck_emission_factor / 1000.

                    feats[l + '_' + feed + '_meat'] = meat_abroad + local_meat
                    feats[l + '_' + feed + '_meth'] = methane_abroad + local_methane
                    feats[l + '_' + feed + '_manure'] = manure_abroad + local_manure
                    feats[l + '_' + feed + '_cost'] = local_cost
                    feats[l + '_' + feed + '_feed_cost'] = feed_to_port_cost
                    feats[l + '_' + feed + '_feed_emission'] = feed_to_port_emis
                    feats[l + '_' + feed + '_cstock'] = 0
                    feats[l + '_' + feed + '_beefarea'] = np.nansum(
                        actual_prod / (feats[['climate_bin']].merge(potential_yields, how="left").drop('climate_bin',
                                                                                                       axis=1).values *
                                       feats[['ADM0_A3']].merge(yield_fraction, how="left").drop('ADM0_A3',
                                                                                                 axis=1).values),
                        axis=1)

                    feats[l + '_' + feed + '_opp_cost'] = feats['opp_cost'].astype(float) / 1000. * feats[
                        l + '_' + feed + '_beefarea']

                    bgb_change = (((-0.59 * feats['pasture_area'] * feats['soil_carbon10km']) + (
                            -0.42 * feats['tree_area'] * feats['soil_carbon10km'])) * 3.67 * -1) * feats[l + '_' + feed + '_beefarea'] / feats['suitable_area']
                    feats[l + '_' + feed + '_bgb_change'] = eac(bgb_change, rate=0)

                elif feed == 'stover':
                    stover_production = feats[['diff_' + i for i in crop_list]].values * crop_residues.iloc[0].values[
                                                                                         None, :] * stover_availability
                    stover_energy = np.nansum(stover_production * residue_energy.iloc[0].values[None, :], axis=1)

                    me_table = energy_conversion.loc[energy_conversion.feed == 'mixed'][
                        ['region', 'climate', beef_yield]]
                    meat_stover = stover_energy * \
                                  feats[['group', 'glps']].merge(me_table, how='left', left_on=['group', 'glps'],
                                                                 right_on=['region', 'climate'])[beef_yield].values

                    local_meat = np.sum(np.where(feats[months] < -1,
                                                 (meat_stover[:, None] - (
                                                         meat_stover[:, None] * (
                                                         -0.0182 * feats[months] - 0.0182))) / 12.,
                                                 meat_stover[:, None] / 12.), axis=1) * dressing

                    me_table = energy_conversion.loc[energy_conversion.feed == 'mixed'][
                        ['region', 'climate', 'me_to_meth']]

                    local_methane = np.nansum(stover_production, axis=1) * \
                                    feats[['group', 'glps']].merge(me_table, how='left', left_on=['group', 'glps'],
                                                                   right_on=['region', 'climate'])['me_to_meth'].values

                    local_manure = np.nansum(stover_production, axis=1) * \
                                   feats[['group']].merge(grouped_ME, how='left')[
                                       'nitrous'].values
                    feats[l + '_' + feed + '_meat'] = local_meat
                    feats[l + '_' + feed + '_meth'] = local_methane
                    feats[l + '_' + feed + '_manure'] = local_manure
                    feats[l + '_' + feed + '_cost'] = 0
                    feats[l + '_' + feed + '_cstock'] = 0
                    feats[l + '_' + feed + '_n2o'] = 0
                    feats[l + '_' + feed + '_feed_cost'] = 0
                    feats[l + '_' + feed + '_feed_emission'] = 0
                    feats[l + '_' + feed + '_opp_cost'] = 0
                    feats[l + '_' + feed + '_beefarea'] = 0
                    feats[l + '_' + feed + '_bgb_change'] = 0

        for l in ['grazing', 'mixed']:
            feats[l + '_bgb_change'] = feats[
                [l + '_' + feed + '_bgb_change' for feed in ["grass", "grain", "stover"]]].sum(axis=1)
            feats[l + '_meat'] = feats[[l + '_' + feed + '_meat' for feed in ["grass", "grain", "stover"]]].sum(axis=1)
            feats[l + '_manure'] = feats[[l + '_' + feed + '_manure' for feed in ["grass", "grain", "stover"]]].sum(
                axis=1)
            feats[l + '_meth'] = feats[[l + '_' + feed + '_meth' for feed in ["grass", "grain", "stover"]]].sum(axis=1)
            feats[l + '_cstock'] = feats[[l + '_' + feed + '_cstock' for feed in ["grass", "grain", "stover"]]].sum(
                axis=1)
            feats[l + '_cost'] = feats[[l + '_' + feed + '_cost' for feed in ["grass", "grain", "stover"]]].sum(axis=1)
            feats[l + '_n2o'] = feats[[l + '_' + feed + '_n2o' for feed in ["grass", "grain", "stover"]]].sum(axis=1)
            feats[l + '_feed_cost'] = feats[
                [l + '_' + feed + '_feed_cost' for feed in ["grass", "grain", "stover"]]].sum(axis=1)
            feats[l + '_feed_emission'] = feats[
                [l + '_' + feed + '_feed_emission' for feed in ["grass", "grain", "stover"]]].sum(axis=1)
            feats[l + '_opp_cost'] = feats[[l + '_' + feed + '_opp_cost' for feed in ["grass", "grain", "stover"]]].sum(
                axis=1)
            feats[l + '_area'] = feats[[l + '_' + feed + '_beefarea' for feed in ["grass", "grain", "stover"]]].sum(
                axis=1)

            ntrips = np.ceil(feats[l + '_meat'] / int(15)) * 2

            feats[l + '_trans_cost'] = ntrips * feats[['ADM0_A3']].merge(fuel_cost[['ADM0_A3', 'Diesel']],
                                                                         how='left',
                                                                         left_on='ADM0_A3',
                                                                         right_on='ADM0_A3')['Diesel'].values * \
                                       feats["accessibility"] * fuel_efficiency / 1000.
            feats[l + '_trans_emiss'] = ntrips * feats[
                "accessibility"] * fuel_efficiency * truck_emission_factor / 1000.

    else:
        logger.inf('Feed option {} not in choices'.format(feed_option))
    # Drop monthly temperature and other crop uses columns
    feats = feats.drop(months + ['diff_' + crop for crop in crop_list], axis=1)

    # Only keep cells where at least 1 feed option produces meat
    feats = feats.loc[feats[[l + '_meat' for l in landuses]].sum(axis=1) > 0]

    # Annualise establishment cost
    feats['est_cost'] = eac(feats['est_cost'])
    logger.info("Done calculating establishment cost")

    ############################## SOC for current beef to no beef #############################

    # Get percentage change of grassland and cropland to 'original' ecoregion
    soc_change = feats[['ecoregions']].merge(sc_change[['code', 'grassland', 'cropland']],
                                             how='left',
                                             left_on='ecoregions',
                                             right_on='code')

    # 'Opportunity cost' of soil carbon = sum over land uses of current area of land use * current soil carbon * percentage change * negative emission (-1)
    opp_soc = feats['current_grazing'].fillna(0).values * feats['soil_carbon10km'].values * soc_change[
        'grassland'].values * -1 + \
              feats['current_cropping'].fillna(0).values * feats['soil_carbon10km'].values * soc_change[
                  'cropland'].values * -1

    # Make sure that opportunity cost of soil carbon is only where there currently is beef production
    opp_soc = np.where(feats.bvmeat.values > 0, opp_soc, 0)

    # Annualize opportunity cost of soil carbon
    feats['opp_soc'] = eac(opp_soc, rate=0)
    feats['opp_soc'] = feats['opp_soc'].fillna(0)
    del soc_change, opp_soc

    # Opportunity cost of afforestation = current beef area (ha) * carbon in potential vegetation (t C/ha) * negative emission (-1)
    opp_aff = (feats['current_grazing'].fillna(0).values + feats['current_cropping'].fillna(0).values) * feats[
        'potential_carbon'].values * -1

    # Make sure that opportunity cost of afforestation is greater than 0
    opp_aff = np.where(feats.bvmeat.values > 0, opp_aff, 0)
    # Annualise opportunity cost of afforestation
    feats['opp_aff'] = eac(opp_aff, rate=0)
    del opp_aff

    ### Calculate score

    for l in landuses:
        # Processing energy = meat (t) * process energy (MJ/kg) * energy efficiency (kg CO2/kg)
        feats[l + '_process_energy'] = feats[l + '_meat'].values * process_pack * \
                                       feats[['ADM0_A3']].merge(energy_efficiency, how='left')['energy'].fillna(
                                           0).values

        # Total costs ('000 USD) = Establishment cost (annualised '000 USD) + production cost + transport cost + opportunity cost
        feats[l + '_tot_cost'] = feats['est_cost'] + \
                                 (feats[l + '_cost'] + feats[l + '_trans_cost'] + feats[l + '_opp_cost'])

        # Annual emissions (t CO2 eq) = Fertiliser N2O + Enteric CH4 + Manure N2O + transport CO2 + Processing CO2
        flow = feats[l + '_n2o'] + feats[l + '_meth'] + feats[l + '_manure'] + feats[l + '_trans_emiss'] + feats[
            l + '_process_energy']

        # Carbon stock = Current C stock (t/ha) * area (ha) * C-CO2 conversion - remaining C stock (t)
        agb_change = feats['carbon_stock'] * feats['suitable_area'] * 3.67 - feats[l + '_cstock']

        # Annualise the loss of carbon stock
        feats[l + '_agb_change'] = eac(agb_change, rate=0)

        # Total GHG emissions = Above ground carbon change + below ground carbon change + annual emissions - afforestation potential - soil carbon potential
        feats[l + '_ghg'] = feats[l + '_agb_change'] + feats[l + '_bgb_change'] + flow - feats['opp_aff'] - feats[
            'opp_soc']
        del agb_change

        # Set export cost and emissions to 0
        feats[l + '_exp_emiss'] = 0
        feats[l + '_exp_costs'] = 0
        # Calculate relative GHG and costs

        # Relative emissions and costs = Emissions (t CO2 eq)/Meat (ton) ; Cost ('000 USD)/Meat (ton)
        feats[l + '_rel_ghg'] = np.where(feats[l + '_meat'] < 1, np.NaN, feats[l + '_ghg'] / (feats[l + '_meat']))
        feats[l + '_rel_cost'] = np.where(feats[l + '_meat'] < 1, np.NaN,
                                          feats[l + '_tot_cost'] / (feats[l + '_meat']))

        logger.info("Done calculating rel cost & emissions for  {}".format(l))

    # Drop rows where all columns starting with 'rel_ghg' and 'rel_cost' are NAs
    feats = feats.dropna(how='all', subset=[l + '_rel_ghg' for l in landuses] + [l + '_rel_cost' for l in landuses])

    # If optimisation uses carbon pricing, convert weight to USD
    if optimisation_method == 'carbon_price':
        lam = lam / 1000.

    # for each landuse, append to a list an array with the weighted sum of rel ghg and rel cost per cell
    list_scores = [(feats[l + '_rel_ghg'].values * (1 - lam)) +
                   (feats[l + '_rel_cost'].values * lam) for l in landuses]

    # Stack horizontally 1d arrays to get a 2d array
    allArrays = np.stack(list_scores, axis=-1)

    # Select lowest score
    feats['best_score'] = np.nanmin(allArrays, axis=1)

    try:
        # Select position (land use) of lowest score
        feats['bestlu'] = np.nanargmin(allArrays, axis=1)
    except:
        # If there is no best land use, export dataframe
        feats.loc[feats.best_score.isna()].drop('geometry', axis=1).to_csv("nadf.csv", index=False)

    del list_scores, allArrays

    # Create a new column for all variables in new_colnames that selects the value of the optimal land use
    for i in new_colnames:
        feats[i] = np.take_along_axis(feats[[l + new_colnames[i] for l in landuses]].values,
                                      feats['bestlu'].values[:, None], axis=1)
    logger.info(feats.shape[0])
    return feats

def trade(feats, optimisation_method, lam, feed_option):
    """
    Function to update score based on trade costs and emissions

    Arguments:
    feats (pandas dataframe) -> Main dataframe
    optimisation_method (str)-> Method for optimisation ('weighted_sum' or 'carbon_price')
    lam (float)-> Lambda weight ([0,1])
    feed_option (str)-> folder where the output file is exported

    Output: returns a gridded dataframe with updated score
    """
    if feed_option in ['v1', 'v2']:
        landuses = grass_cols + ['cropland', 'stover']

    elif feed_option == 'v3':
        landuses = ['grazing', 'mixed']
    else:
        print('Feed option {} not in choices'.format(feed_option))

    for l in landuses:

        # Calculate transport trips to export meat
        ntrips = (feats[l + '_meat'] / int(15) + 1) * 2

        # Calculate transport cost to nearest port
        feats[l + '_trans_cost'] = ntrips * feats["distance_port"] * \
                                   feats[['ADM0_A3']].merge(fuel_cost[['ADM0_A3', 'Diesel']],
                                                            how='left', left_on='ADM0_A3',
                                                            right_on='ADM0_A3')['Diesel'].values * fuel_efficiency/1000.

        # Calculate transport costs as a function of quantity traded
        feats[l + '_exp_costs'] = feats[l + '_meat'] * feats[['ADM0_A3']].merge(sea_t_costs[['ADM0_A3', 'tcost']],
                                                                                how = 'left')['tcost'].values

        # Transport emissions to port
        feats[l + '_trans_emiss'] = ntrips * feats["distance_port"] * fuel_efficiency * truck_emission_factor / 1000.

        # Transport emissions by sea
        feats[l + '_exp_emiss'] = feats[['ADM0_A3']].merge(sea_distances[['ADM0_A3', 'ave_distance']], how='left')['ave_distance'].values * feats[l + '_meat'] * sea_emissions / 1000.

        # Update total cost ('000 USD)
        feats[l + '_tot_cost'] = feats['est_cost'] + (feats[l + '_cost'] + feats[l + '_trans_cost'] + feats[l + '_opp_cost'] + feats[l + '_exp_costs'])

        # Update annual emissions (t CO2 eq)
        flow = feats[l + '_n2o'] + feats[l + '_meth'] + feats[l + '_manure'] + feats[l + '_trans_emiss'] + feats[l + '_exp_emiss'] + feats[l + '_process_energy']

        # Update total emissions (t CO2 eq)
        feats[l + '_ghg'] = feats[l + '_agb_change'] + feats[l + '_bgb_change'] + flow - feats['opp_aff'] - feats[
            'opp_soc']

        # Update relative GHG and costs
        feats[l+'_rel_ghg'] = np.where(feats[l+'_meat'] < 1, np.NaN, feats[l+'_ghg']/(feats[l+'_meat']))
        feats[l+'_rel_cost'] = np.where(feats[l+'_meat'] < 1, np.NaN, feats[l+'_tot_cost']/(feats[l+'_meat']))

    # Drop rows where columns are all nas
    feats = feats.dropna(how='all', subset=[lu+'_rel_ghg' for lu in landuses] + [lu+'_rel_cost' for lu in landuses])

    # make sure that dataframe is not empty
    if feats.shape[0] > 0:
        if optimisation_method == 'carbon_price':
            lam = lam/1000.

        # Calculate new weighted sum
        list_scores = [(feats[i + '_rel_ghg'].values * (1 - lam)) +
                       (feats[i + '_rel_cost'].values * lam) for i in landuses]

        # Stack arrays horizontally
        allArrays = np.stack(list_scores, axis=-1)

        # Take lowest score across feed options
        feats['best_score'] = np.nanmin(allArrays, axis=1)

        # Get best feed option based on position of best score
        feats['bestlu'] = np.nanargmin(allArrays, axis=1)

        del list_scores, allArrays

        # Write new columns according to optimal land use
        for cname in new_colnames:
            feats[cname] = np.take_along_axis(feats[[lu + new_colnames[cname] for lu in landuses]].values,
                                              feats['bestlu'].values[:, None], axis=1)
        return feats

def export_raster(grid, resolution, export_column, export_folder, constraint, crop_yield, beef_yield,
                  lam, demand_scenario, feed_option):
    """
    Function to rasterize columns of a dataframe

    Arguments:
    grid (pandas dataframe)-> Dataframe to rasterize
    resolution (float)-> Resolution at which to rasterize
    export_column (list)-> list of columns to rasterize

    export_folder (str)-> folder where the output file is exported
    constraint (str)-> Spatial constraint for beef production ('global', 'country', or 'subsistence')
    crop_yield (int)-> Scenario of crop yield (0 = current, 1 = no yield gap)
    beef_yield (str)-> Scenario of beef yield ('me_to_meat' = current, 'max_yield' = no yield gap)
    lam (float)-> Lambda weight ([0,1])
    demand_scenario (str)-> Scenario of beef demand ('SSP1-NoCC2010' or 'SSP1-NoCC2050')
    feed_option (str)-> folder where the output file is exported

    Output: Writes the grid as GPKG file
    """
    bounds = list(grid.total_bounds)
    resolution = float(resolution)
    width = abs(int((bounds[2] - bounds[0]) / resolution))
    heigth = abs(int((bounds[3] - bounds[1]) / resolution))
    out_shape = (heigth, width)
    grid['bestlu'] = np.array(grid['bestlu'], dtype='uint8')
    for i in export_column:

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
        out_fn = export_folder + '/' + constraint + "_" + str(crop_yield) + '_' + beef_yield + '_' + str(lam) + '_' + demand_scenario + '_' + feed_option + ".tif"
          
        with rasterio.open(out_fn, 'w', **meta) as out:
            # Create a generator for geom and value pairs
            grid_cell = ((geom, value) for geom, value in zip(grid.geometry, grid[i]))

            burned = features.rasterize(shapes=grid_cell, fill=0, out_shape=out_shape, dtype = dt,
                                        transform=Affine(resolution, 0.0, bounds[0],
                                                         0.0, -resolution, bounds[3]))
            print("Burned value dtype: {}".format(burned.dtype))
            out.write_band(1, burned)

def main(export_folder ='.', optimisation_method= 'weighted_sum', lam = 0.5, demand_scenario = 'Demand',
         crop_yield = 0, beef_yield ='me_to_meat', constraint ='global', feed_option ='v1', trade_scenario ='trade'):
    """
    Main function that optimises beef production for a given location and resolution, using a given number of cores.
    
    Arguments:
    export_folder (str)-> folder where the output file is exported
    optimisation_method (str)-> Method for optimisation ('weighted_sum' or 'carbon_price')
    lam (float)-> Lambda weight ([0,1])
    demand_scenario (str)-> Scenario of beef demand ('SSP1-NoCC2010' or 'SSP1-NoCC2050')
    crop_yield (int)-> Scenario of crop yield (0 = current, 1 = no yield gap)
    beef_yield (str)-> Scenario of beef yield ('me_to_meat' = current, 'max_yield' = no yield gap)
    constraint (str)-> Spatial constraint for beef production ('global', 'country', or 'subsistence')
    feed_option (str)-> folder where the output file is exported
    trade_scenario (str)-> Trade scenario (if 'trade', apply trade based on country demand)

    Output: Writes the grid as GPKG file
    """

    LOG_FORMAT = "%(asctime)s - %(message)s"
    try:
        logging.basicConfig(
            # filename="/home/uqachare/model_file/logs_opt/opt_" + constraint + "_" + str(crop_yield) + "_" + me_to_meat + "_" + str(lam) + '_' + dem +".log",
            filename="/home/uqachare/model_file/test_" + feed_option + ".log",
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
    grid = gpd.read_file("grid.gpkg")
    logger.info("Done loading grid, memory usage of grid = {}".format(grid.memory_usage().sum()*1e-6))

    dt_dict = {np.dtype('float64'): np.dtype('float32'),
               np.dtype('int64'): np.dtype('int32')}
    datatypes = pd.DataFrame({'dtypes': grid.dtypes,
                              'newtypes': [dt_dict[dt] if dt in dt_dict else dt for dt in grid.dtypes]})

    for id in datatypes.index:
        grid[id] = grid[id].astype(datatypes.loc[id, 'newtypes'])

    logger.info("End changing datatypes, memory usage of grid = {}".format(grid.memory_usage().sum()*1e-6))

    for i in ['soilmoisture', "gdd",  'ls_opp_cost', 'agri_opp_cost', 'est_area']:
        if i in grid.columns:
            grid = grid.drop(i, axis = 1)

    logger.info("Simulation start")
    logger.info('Me_to_meat scanerio: {}'.format(beef_yield))
    logger.info('Weight scenario: {}'.format(lam))
    logger.info('Feed option scenario: {}'.format(feed_option))

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

    # Adjust other uses for future demand  Proportion of demand increase
    beef_demand['dem_increase'] = beef_demand[demand_scenario]/beef_demand['SSP1-NoCC2010']

    logger.info('New demand for other uses before: {}'.format(grid[['diff_maize']].loc[grid.diff_maize > 0].head()))
    other_uses = grid[['ADM0_A3']+['diff_' + i for i in crop_list]].merge(beef_demand[['dem_increase']], left_on = 'ADM0_A3', right_index = True)
    grid[['diff_' + i for i in crop_list]] = other_uses[['diff_' + i for i in crop_list]].values * other_uses['dem_increase'].values[:,None]
    del other_uses

    logger.info('New demand for other uses after: {}'.format(grid[['diff_maize']].loc[grid.diff_maize > 0].head()))

    grid = scoring(grid, optimisation_method, crop_yield, lam, beef_yield, logger, feed_option)

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
    grid['dom_demand'] = pd.Series(grid.merge(beef_demand, how='left', left_on='ADM0_A3', right_index=True)['Demand'], dtype = 'int32')
    grid['dom_production'] = pd.Series(grid[['ADM0_A3']].merge(beef_production, how='left', left_on='ADM0_A3', right_on = ['Code'])['prop'].values * demand, dtype = 'int32')

    # Sort rows by increasing 'best score'
    grid = grid.sort_values('best_score')
    # Get cumulative country level production in order of increasing best score
    grid.loc[grid.changed == 0, 'cumdomprod'] = grid.groupby('ADM0_A3')['production'].transform(pd.Series.cumsum)
    # print(grid.shape[0])
    # print(grid['best_score'])

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
                    grid.loc[(grid['ADM0_A3'].isin(ADM0_A3)) & (grid['changed'] == 0)], optimisation_method, lam, feed_option)

            grid.loc[(grid['destination'] == 0) & (grid['changed'] == 1),  'destination'] = np.where(
                grid.loc[(grid['destination'] == 0) & (grid['changed'] == 1), 'cumdomprod'] <
                grid.loc[(grid['destination'] == 0) & (grid['changed'] == 1), 'dom_demand'], 1, 2)

            # Recalculate total production of converted cells
            total_production = grid.loc[grid.changed == 1, 'production'].sum()

            # Keep track of countries that have met their demand
            grid.loc[grid.ADM0_A3.isin(countries_complete), 'supplied'] = 1

            # Keep track of total production
            # grid.loc[(grid.changed == 0) & (~grid.ADM0_A3.isin(countries_complete)), 'total_prod'] = grid.loc[
            #     grid.changed == 1, 'production'].sum()
            grid.loc[(grid.changed == 0) & (~grid.ADM0_A3.isin(countries_complete)), 'total_prod'] = total_production

            print('total production: ', total_production)

            # Keep track of new production in loop to avoid looping with 0 new production
            new_production = round(total_production,3) - round(old_production,3)
            logger.info("Total production: {}".format(total_production))

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

    total_opp_aff = grid['opp_aff'].sum()
    total_opp_soc = grid['opp_soc'].sum()

    grid = grid.loc[grid['changed'] == 1]

    if constraint == 'subsistence':
        logger.info('Production before merging: {}'.format(grid.production.sum()))
        grid = pd.concat([grid, gpd.read_file('global_south_results.gpkg')])
        logger.info('Production after merging: {}'.format(grid.production.sum()))

    newdf = pd.DataFrame({"suitable_area": grid.suitable_area.sum(),
                          "beef_area": grid.beef_area.sum(),
                          "production": grid.production.sum(),
                          "enteric": grid.enteric.sum(),
                          "manure": grid.manure.sum(),
                          "export_emissions": grid.export_emissions.sum(),
                          "transp_emission": grid.transp_emission.sum(),
                          "n2o_emissions": grid.n2o_emissions.sum(),
                          "agb_change": grid.agb_change.sum(),
                          "bgb_change": grid.bgb_change.sum(),
                          "processing_energy": grid.processing_energy.sum(),
                          'opp_aff': [total_opp_aff],
                          'opp_soc': [total_opp_soc],
                          "emissions": grid.total_emission.sum(),
                           "cost": grid.total_cost.sum(),
                           "est_cost": grid.est_cost.sum(),
                           "production_cost": grid.production_cost.sum(),
                           "export_cost": grid.export_cost.sum(),
                           "transp_cost": grid.transp_cost.sum(),
                           "opportunity_cost": grid.opportunity_cost.sum(),
                           "optimisation_method": [optimisation_method],
                           "constraint": [constraint],
                           "weight" : [str(lam)],
                           "crop_gap": [str(crop_yield)],
                           "beef_gap": [str(beef_yield)]})

    dem = demand_scenario.split('NoCC')[1]
    newdf.to_csv(export_folder + '/' + constraint + "_" + str(crop_yield) + "_" + beef_yield + "_" + str(lam) + '_' + dem + "_" + feed_option + ".csv", index=False)
    logger.info("Exporting CSV finished")
    grid[cols].to_file(export_folder + '/' + constraint + "_" + str(crop_yield) + '_' + beef_yield + '_' + str(lam) + '_' + dem + "_" + feed_option + ".gpkg", driver="GPKG")
    logger.info("Exporting GPKG finished")
    export_raster(grid, 0.0833, ['production'], export_folder, constraint, crop_yield, beef_yield, lam, dem, feed_option)
    logger.info("Exporting raster finished")

def parallelise(export_folder, optimisation_method, job_nmr, feed_option):

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
        pool = multiprocessing.Process(target=main,
                                       args=(export_folder,
                                             optimisation_method,  # Optimisation method (weighted sum vs carbon price)
                                             w,  # Weight
                                             scenarios[job_nmr][3],  # Demand
                                             scenarios[job_nmr][2],  # Crop yield
                                             scenarios[job_nmr][1],  # Beef yield
                                             scenarios[job_nmr][0],  # Spatial constraint
                                             feed_option,  # Feed option
                                             ))
        pool.start()

if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('export_folder', help='Name of exported file')
    argparser.add_argument('optimisation_method', help='Which scenario of optimisation to run ("weighted_sum", "carbon_price", "cscc", "costs", "ghg")')
    argparser.add_argument('job_nmr', help="Job number", type=int)
    argparser.add_argument('feed_option', help="Options for calculating grain: v1 -> convert all cell to grain, v6 -> convert the difference between attainable yield and production for non-beef uses")

    args = argparser.parse_args()
    export_folder = args.export_folder
    optimisation_method = args.optimisation_method
    job_nmr = args.job_nmr
    feed_option = args.feed_option

    parallelise(export_folder, optimisation_method, job_nmr, feed_option)