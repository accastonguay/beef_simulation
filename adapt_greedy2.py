import pandas as pd
import numpy as np
import logging
import multiprocessing
import time

pd.set_option('display.max_columns', 20)

beef_production = pd.read_csv("./beef_production.csv")  # Load country-level beef demand
beef_demand = pd.read_csv("./beef_demand.csv")  # Load country-level beef demand

def main(spat_const, aff_scenario, weight, iterations, method = 'loop'):
    LOG_FORMAT = "%(asctime)s - %(message)s"
    try:
        logging.basicConfig(
            # filename="/home/uqachare/model_file/logs_opt/opt_" + constraint + "_" + str(crop_yield) + "_" + me_to_meat + "_" + str(lam) + '_' + dem +".log",
            filename="/home/uqachare/model_file/simpleopt_" + str(spat_const) +str(weight) + ".log",
            level=logging.INFO,
            format=LOG_FORMAT,
            filemode='w')
    except:
        logging.basicConfig(
            level=logging.INFO,
            format=LOG_FORMAT,
            filemode='w')
    logger = logging.getLogger()
    grid = pd.read_csv('./score_init.csv')
    # grid = grid.loc[grid.ADM0_A3 == 'EGY']

    demand = 69477745
    # demand = 456359

    grass_cols = []

    for i in ["0250", "0375", "0500"]:
        for n in ["000", "050", "200"]:
            grass_cols.append("grass_" + i + "_N" + n)

    landuses = grass_cols + ['grass_grain', 'stover_grass', 'stover_grain']

    #--------------
    if spat_const == 'global':

        allweights = pd.DataFrame()
        for w in range(0,11,1):

            for l in landuses:
                rel_ghg = np.where(grid[l + '_meat'] < 1, np.NaN, grid[l + '_ghg'] / (grid[l + '_meat']))
                rel_cost = np.where(grid[l + '_meat'] < 1, np.NaN,
                                    grid[l + '_tot_cost'] / (grid[l + '_meat']))
                grid[l + '_score'] = (rel_ghg * (1 - w/10)) + (rel_cost * w/10)

            grid['score'] = np.nanmin(grid[[l + '_score' for l in landuses]].values, axis=1)

            try:
                grid['lu'] = np.nanargmin(grid[[l + '_score' for l in landuses]].values, axis=1)
            except:
                print(grid.loc[grid.score.isna()][[lu + '_meat' for lu in landuses] + [lu + '_ghg' for lu in landuses] + [lu + '_tot_cost' for lu in landuses]])

            grid['beef'] = np.take_along_axis(grid[[lu + '_meat' for lu in landuses]].values,
                                              grid['lu'].values[:, None], axis=1).flatten()
            grid['emissions'] = np.take_along_axis(grid[[lu + '_ghg' for lu in landuses]].values,
                                                   grid['lu'].values[:, None], axis=1).flatten()
            grid['costs'] = np.take_along_axis(grid[[lu + '_tot_cost' for lu in landuses]].values,
                                               grid['lu'].values[:, None], axis=1).flatten()

            grid = grid.sort_values('score')

            # logger.info('Beef at weight {}: {}'.format(weight, grid.loc[(demand + grid['beef'] - grid['beef'].cumsum() > 0), 'beef'].sum() * 1e-6))
            # logger.info('Emissions: {}'.format(grid.loc[(demand + grid['beef'] - grid['beef'].cumsum() > 0), 'emissions'].sum() * 1e-6))
            # logger.info('Costs: {}'.format(grid.loc[(demand + grid['beef'] - grid['beef'].cumsum() > 0), 'costs'].sum() * 1e-6))
            # logger.info('Number of converted cells: {}'.format(grid.loc[(demand + grid['beef'] - grid['beef'].cumsum() > 0)].shape[0]))

            total = pd.DataFrame({'beef':[grid.loc[(demand + grid['beef'] > grid['beef'].cumsum()), 'beef'].sum()],
                                  'costs':[grid.loc[(demand + grid['beef'] > grid['beef'].cumsum()), 'costs'].sum()],
                                  'emissions':[grid.loc[(demand + grid['beef'] > grid['beef'].cumsum()), 'emissions'].sum()],
                                  'weight': [w/10],
                                  'cells':[grid.loc[(demand + grid['beef'] > grid['beef'].cumsum()), 'emissions'].shape[0]]})
            allweights = pd.concat([allweights, total])

        ##### Find weight to use as initial condition #####

        dico_weight = {}

        logger.info(allweights)

        for i in allweights.weight:
            init_costs = allweights.loc[allweights.weight == i, 'costs'].iloc[0]
            init_emissions = allweights.loc[allweights.weight == i, 'emissions'].iloc[0]
            dico_weight[i] = i
            for index, row in allweights.iterrows():
                if ((row['costs'] < init_costs) and (row['emissions'] < init_emissions)):
                    new_weight = row['weight']
                    init_costs = row['costs']
                    init_emissions = row['emissions']
                    dico_weight[i] = new_weight
        for i in dico_weight:
            logger.info('weight: {}, initial: {}'.format(i, dico_weight[i]))

        ##### Re-initialise order of cells #####

        for l in landuses:
            rel_ghg = np.where(grid[l + '_meat'] < 1, np.NaN, grid[l + '_ghg'] / (grid[l + '_meat']))
            rel_cost = np.where(grid[l + '_meat'] < 1, np.NaN,
                                grid[l + '_tot_cost'] / (grid[l + '_meat']))
            # grid[l + '_score'] = (rel_ghg * (1 - dico_weight[weight/10])) + (rel_cost * dico_weight[weight/10])
            grid[l + '_score'] = (rel_ghg * (1 - weight/10)) + (rel_cost * weight/10)

        grid['score'] = np.nanmin(grid[[l + '_score' for l in landuses]].values, axis=1)

        try:
            grid['lu'] = np.nanargmin(grid[[l + '_score' for l in landuses]].values, axis=1)
        except:
            logger.info(grid.loc[grid.score.isna()])

        grid['beef'] = np.take_along_axis(grid[[lu + '_meat' for lu in landuses]].values,
                                          grid['lu'].values[:, None], axis=1).flatten()
        grid['emissions'] = np.take_along_axis(grid[[lu + '_ghg' for lu in landuses]].values,
                                               grid['lu'].values[:, None], axis=1).flatten()
        grid['costs'] = np.take_along_axis(grid[[lu + '_tot_cost' for lu in landuses]].values,
                                           grid['lu'].values[:, None], axis=1).flatten()
        grid['export_emissions'] = np.take_along_axis(grid[[lu + '_exp_emiss' for lu in landuses]].values,
                                           grid['lu'].values[:, None], axis=1).flatten()
        grid['export_costs'] = np.take_along_axis(grid[[lu + '_exp_costs' for lu in landuses]].values,
                                           grid['lu'].values[:, None], axis=1).flatten()
        grid = grid.sort_values('score')
        total_optimised = pd.DataFrame(
            {'beef': [grid.loc[(demand + grid['beef'] - grid['beef'].cumsum() > 0), 'beef'].sum()],
             'costs': [grid.loc[(demand + grid['beef'] - grid['beef'].cumsum() > 0), 'costs'].sum()],
             'emissions': [grid.loc[(demand + grid['beef'] - grid['beef'].cumsum() > 0), 'emissions'].sum()],
             'weight': [weight / 10.],
             'iteration': ['Initial']})

        if method == "loop":
            # start = time.time()

            logger.info('Weight: {}'.format(weight / 10.))
            logger.info('Initial sorting based on weight: {}'.format(dico_weight[weight / 10]))
            logger.info('Demand: {}'.format(demand))

            for iteration in range(iterations):
                current_prod = 0
                current_emi = 0
                current_cost = 0

                nrows = grid.loc[(demand + grid['beef'] - grid['beef'].cumsum() > 0)].shape[0]
                sel_nrows = nrows + int(nrows * 0.01)

                logger.info('Iteration: {}'.format(iteration))
                logger.info('Number of total rows: {}'.format(grid.shape[0]))
                logger.info('Number of converted rows: {}'.format(nrows))
                logger.info('Number of selected rows: {}'.format(sel_nrows))

                grid = grid.iloc[0:sel_nrows]
                grid = grid.reset_index(drop=True)
                exporting_countries = []
                country_demand = {}

                for index, row in grid.iterrows():

                    dico = {}
                    for lu_id, m, e, c in zip(range(12), [lu + '_meat' for lu in landuses],
                                              [lu + '_ghg' for lu in landuses],
                                              [lu + '_tot_cost' for lu in landuses]):

                        if row[m] > 0:
                            grid.at[index, 'beef'] = row[m]
                            grid.at[index, 'emissions'] = row[e]
                            grid.at[index, 'costs'] = row[c]

                            cumprod = current_prod + grid['beef'].values[index:None].cumsum()
                            cumemi = current_emi + grid['emissions'].values[index:None].cumsum()
                            cumcosts = current_cost + grid['costs'].values[index:None].cumsum()

                            if aff_scenario == 'noaff':
                                cum_score = ((1 - weight / 10.) * cumemi[np.argwhere(cumprod > demand)[0][0]]) + (
                                            weight / 10. * cumcosts[np.argwhere(cumprod > demand)[0][0]])

                            elif aff_scenario == 'nataff':
                                aff = np.nansum(grid.opp_nataff[np.argwhere(cumprod > demand)[0][0]:None])
                                soc = np.nansum(grid.opport_soc[np.argwhere(cumprod > demand)[0][0]:None])

                                cum_score = ((1 - weight / 10.) * (cumemi[np.argwhere(cumprod > demand)[0][0]] + aff + soc)) + (
                                            weight / 10. * cumcosts[np.argwhere(cumprod > demand)[0][0]])

                            elif aff_scenario == 'manaff':
                                aff = np.nansum(grid.opp_manaff[np.argwhere(cumprod > demand)[0][0]:None])
                                soc = np.nansum(grid.opport_soc[np.argwhere(cumprod > demand)[0][0]:None])
                                affcost = np.nansum(grid.affor_cost[np.argwhere(cumprod > demand)[0][0]:None])

                                cum_score = ((1 - weight / 10.) * (cumemi[np.argwhere(cumprod > demand)[0][0]] + aff + soc)) + (
                                            weight / 10. * (cumcosts[np.argwhere(cumprod > demand)[0][0]] + affcost))

                            dico[cum_score] = [row[m], lu_id, row[e], row[c]]

                    grid.at[index, 'beef'] = dico[min(dico)][0]
                    grid.at[index, 'lu'] = dico[min(dico)][1]
                    grid.at[index, 'emissions'] = dico[min(dico)][2]
                    grid.at[index, 'costs'] = dico[min(dico)][3]
                    current_prod += grid.at[index, 'beef']
                    current_cost += grid.at[index, 'costs']
                    current_emi += grid.at[index, 'emissions']

                    # If country is not yet exporting
                    if row['ADM0_A3'] not in exporting_countries:
                        if row['ADM0_A3'] not in country_demand:
                            country_demand[row['ADM0_A3']] = grid.at[index, 'beef']
                        else:
                            country_demand[row['ADM0_A3']] += grid.at[index, 'beef']

                        # If cumulative beef production in country is exceeding domestic demand, include country in list of exporting countries
                        if country_demand[row['ADM0_A3']] >= \
                                beef_demand.loc[beef_demand.ADM0_A3 == row['ADM0_A3'], 'SSP1-NoCC2010'].iloc[0]:
                            exporting_countries.append(row['ADM0_A3'])
                            grid['costs'] = np.where((grid.index > index) & (grid.ADM0_A3 == row['ADM0_A3']),
                                                     grid.costs + grid.export_costs, grid.costs)
                            grid['emissions'] = np.where((grid.index > index) & (grid.ADM0_A3 == row['ADM0_A3']),
                                                         grid.emissions + grid.export_emissions, grid.emissions)
                            # grid.loc[grid.ADM0_A3 == row['ADM0_A3'], 'costs'] = grid.loc[grid.ADM0_A3 == row['ADM0_A3'], 'costs'] + grid.loc[grid.ADM0_A3 == row['ADM0_A3'], 'export_costs']
                            # grid.loc[grid.ADM0_A3 == row['ADM0_A3'], 'emissions'] = grid.loc[grid.ADM0_A3 == row['ADM0_A3'], 'emissions'] + grid.loc[grid.ADM0_A3 == row['ADM0_A3'], 'emissions']

                    if current_prod > demand:
                        break

                total = pd.DataFrame(
                    {'beef': [grid.loc[(demand + grid['beef'] - grid['beef'].cumsum() > 0), 'beef'].sum()],
                     'costs': [grid.loc[(demand + grid['beef'] - grid['beef'].cumsum() > 0), 'costs'].sum()],
                     'emissions': [grid.loc[(demand + grid['beef'] - grid['beef'].cumsum() > 0), 'emissions'].sum()],
                     'weight': [weight / 10.],
                     'iteration': [iteration]})

                total_optimised = pd.concat([total_optimised, total])

            # df.to_csv('./dynprog_' + str(weight) + '.csv', index=False)

            # grid.loc[grid.changed == 1].to_csv('./dynprog_' + str(weight) + '.csv', index = False)

    if spat_const == 'country':
        total_original = pd.DataFrame()

        for w in range(0, 11, 1):
            # weight = w / 10.

            for l in landuses:
                rel_ghg = np.where(grid[l + '_meat'] < 1, np.NaN, grid[l + '_ghg'] / (grid[l + '_meat']))
                rel_cost = np.where(grid[l + '_meat'] < 1, np.NaN,
                                    grid[l + '_tot_cost'] / (grid[l + '_meat']))
                grid[l + '_score'] = (rel_ghg * (1 - w / 10.)) + (rel_cost * w / 10.)

            grid['score'] = np.nanmin(grid[[l + '_score' for l in landuses]].values, axis=1)

            try:
                grid['lu'] = np.nanargmin(grid[[l + '_score' for l in landuses]].values, axis=1)
            except:
                print(grid.loc[grid.score.isna()])

            grid['beef'] = np.take_along_axis(grid[[lu + '_meat' for lu in landuses]].values,
                                              grid['lu'].values[:, None], axis=1).flatten()
            grid['emissions'] = np.take_along_axis(grid[[lu + '_ghg' for lu in landuses]].values,
                                                   grid['lu'].values[:, None], axis=1).flatten()
            grid['costs'] = np.take_along_axis(grid[[lu + '_tot_cost' for lu in landuses]].values,
                                               grid['lu'].values[:, None], axis=1).flatten()

            grid = grid.sort_values('score')

            allcountries = pd.DataFrame()
            for country in beef_production.Code:
                demand = beef_production.loc[beef_production.Code == country, 'Value'].iloc[0]
                country_df = grid.loc[grid.ADM0_A3 == country]
                country_total = pd.DataFrame({'beef': [country_df.loc[(demand + country_df['beef'] > country_df['beef'].cumsum()), 'beef'].sum()],
                                  'costs': [country_df.loc[(demand + country_df['beef'] > country_df['beef'].cumsum()), 'costs'].sum()],
                                  'emissions': [
                                      country_df.loc[(demand + country_df['beef'] > country_df['beef'].cumsum()), 'emissions'].sum()]})
                allcountries = pd.concat([allcountries, country_total])

            allcountries_total = pd.DataFrame(
                {'beef': [allcountries.beef.sum()],
                 'costs': [allcountries.costs.sum()],
                 'emissions': [allcountries.emissions.sum()],
                 'weight': [w / 10.]})

            total_original = pd.concat([total_original, allcountries_total])

        ##### Find weight to use as initial condition #####

        dico_weight = {}

        for i in total_original.weight:
            init_costs = total_original.loc[total_original.weight == i, 'costs'].iloc[0]
            init_emissions = total_original.loc[total_original.weight == i, 'emissions'].iloc[0]
            dico_weight[i] = i
            for index, row in total_original.iterrows():
                if ((row['costs'] < init_costs) and (row['emissions'] < init_emissions)):
                    new_weight = row['weight']
                    init_costs = row['costs']
                    init_emissions = row['emissions']
                    dico_weight[i] = new_weight

        ##### Re-initialise order of cells #####
        #####______________________________#####
        for l in landuses:
            rel_ghg = np.where(grid[l + '_meat'] < 1, np.NaN, grid[l + '_ghg'] / (grid[l + '_meat']))
            rel_cost = np.where(grid[l + '_meat'] < 1, np.NaN,
                                grid[l + '_tot_cost'] / (grid[l + '_meat']))
            # grid[l + '_score'] = (rel_ghg * (1 - dico_weight[weight / 10])) + (rel_cost * dico_weight[weight / 10])
            grid[l + '_score'] = (rel_ghg * (1 - weight / 10)) + (rel_cost * weight / 10)

        grid['score'] = np.nanmin(grid[[l + '_score' for l in landuses]].values, axis=1)

        try:
            grid['lu'] = np.nanargmin(grid[[l + '_score' for l in landuses]].values, axis=1)
        except:
            print(grid.loc[grid.score.isna()])

        grid['beef'] = np.take_along_axis(grid[[lu + '_meat' for lu in landuses]].values,
                                          grid['lu'].values[:, None], axis=1).flatten()
        grid['emissions'] = np.take_along_axis(grid[[lu + '_ghg' for lu in landuses]].values,
                                               grid['lu'].values[:, None], axis=1).flatten()
        grid['costs'] = np.take_along_axis(grid[[lu + '_tot_cost' for lu in landuses]].values,
                                           grid['lu'].values[:, None], axis=1).flatten()
        grid['export_emissions'] = np.take_along_axis(grid[[lu + '_exp_emiss' for lu in landuses]].values,
                                           grid['lu'].values[:, None], axis=1).flatten()
        grid['export_costs'] = np.take_along_axis(grid[[lu + '_exp_costs' for lu in landuses]].values,
                                           grid['lu'].values[:, None], axis=1).flatten()

        grid = grid.sort_values('score')
        # total_optimised = pd.DataFrame()

        ### Adaptive greedy ###
        logger.info('Weight: {}'.format(weight / 10.))
        logger.info('Initial sorting based on weight: {}'.format(dico_weight[weight / 10]))
        logger.info('Demand: {}'.format(demand))

        initial = pd.DataFrame(
            {'beef': [grid.loc[(demand + grid['beef'] - grid['beef'].cumsum() > 0), 'beef'].sum()],
             'costs': [grid.loc[(demand + grid['beef'] - grid['beef'].cumsum() > 0), 'costs'].sum()],
             'emissions': [grid.loc[(demand + grid['beef'] - grid['beef'].cumsum() > 0), 'emissions'].sum()],
             'weight': [weight / 10.],
             'iteration': ['Initial']})

        allcountries = pd.DataFrame()
        for country in beef_production.Code:
            demand = beef_production.loc[beef_production.Code == country, 'Value'].iloc[0]
            # logger.info('Demand: {}'.format(demand))

            country_df = grid.loc[grid.ADM0_A3 == country]


            country_df = country_df.reset_index(drop=True)
            logger.info('Country: {}'.format(country))

            for iteration in range(3):
                nrows = country_df.loc[(demand + country_df['beef'] - country_df['beef'].cumsum() > 0)].shape[0]
                sel_nrows = nrows + int(nrows * 0.01)
                country_df = country_df.iloc[0:sel_nrows]

                logger.info('Iteration: {}'.format(iteration))
                logger.info('Number of total rows: {}'.format(country_df.shape[0]))
                logger.info('Number of converted rows: {}'.format(nrows))
                logger.info('Number of selected rows: {}'.format(sel_nrows))

                current_prod = 0
                current_emi = 0
                current_cost = 0
                # logger.info('Demand: {}'.format(demand))

                exporting_countries = []
                country_demand = {}
                # logger.info('country: {}'.format(country))
                # logger.info('country column: {}'.format(grid.ADM0_A3))
                # logger.info('country_df shape: {}'.format(country_df.shape[0]))
                # logger.info('Number of total rows: {}'.format(grid.shape[0]))
                # logger.info('Number of converted rows: {}'.format(nrows))
                # logger.info('Number of selected rows: {}'.format(sel_nrows))

                for index, row in country_df.iterrows():

                    dico = {}
                    for lu_id, m, e, c in zip(range(12), [lu + '_meat' for lu in landuses],
                                              [lu + '_ghg' for lu in landuses],
                                              [lu + '_tot_cost' for lu in landuses]):

                        if row[m] > 0:
                            country_df.at[index, 'beef'] = row[m]
                            country_df.at[index, 'emissions'] = row[e]
                            country_df.at[index, 'costs'] = row[c]

                            cumprod = current_prod + country_df['beef'].values[index:None].cumsum()
                            cumemi = current_emi + country_df['emissions'].values[index:None].cumsum()
                            cumcosts = current_cost + country_df['costs'].values[index:None].cumsum()

                            try:
                                cum_score = ((1 - weight / 10.) * cumemi[np.argwhere(cumprod > demand)[0][0]]) + (
                                        weight / 10. * cumcosts[np.argwhere(cumprod > demand)[0][0]])
                            except:
                                logger.info('***************')
                                logger.info('Country {}'.format(country))
                                logger.info('Iteration {}'.format(iteration))
                                logger.info('Demand for {}: {}'.format(country, demand))
                                logger.info('Max production limited for {}: {}'.format(country, country_df['beef'].sum()))
                                logger.info('Max production for {}: {}'.format(country, grid.loc[grid.ADM0_A3 == 'EGY', 'beef'].sum()))

                                logger.info('***************')
                                cum_score = ((1 - weight / 10.) * (current_emi + country_df['emissions'].sum())) + (
                                        weight / 10. * (current_cost + country_df['costs'].sum()))

                            dico[cum_score] = [row[m], lu_id, row[e], row[c]]

                    country_df.at[index, 'beef'] = dico[min(dico)][0]
                    country_df.at[index, 'lu'] = dico[min(dico)][1]
                    country_df.at[index, 'emissions'] = dico[min(dico)][2]
                    country_df.at[index, 'costs'] = dico[min(dico)][3]
                    current_prod += country_df.at[index, 'beef']
                    current_cost += country_df.at[index, 'costs']
                    current_emi += country_df.at[index, 'emissions']

                    # If country is not yet exporting
                    if row['ADM0_A3'] not in exporting_countries:
                        if row['ADM0_A3'] not in country_demand:
                            country_demand[row['ADM0_A3']] = country_df.at[index, 'beef']
                        else:
                            country_demand[row['ADM0_A3']] += country_df.at[index, 'beef']

                        # If cumulative beef production in country is exceeding domestic demand, include country in list of exporting countries
                        if country_demand[row['ADM0_A3']] >= \
                                beef_demand.loc[beef_demand.ADM0_A3 == row['ADM0_A3'], 'SSP1-NoCC2010'].iloc[0]:
                            exporting_countries.append(row['ADM0_A3'])
                            country_df['costs'] = np.where((country_df.index > index) & (country_df.ADM0_A3 == row['ADM0_A3']),
                                                     country_df.costs + country_df.export_costs, country_df.costs)
                            country_df['emissions'] = np.where((country_df.index > index) & (country_df.ADM0_A3 == row['ADM0_A3']),
                                                         country_df.emissions + country_df.export_emissions, country_df.emissions)
                            # grid.loc[grid.ADM0_A3 == row['ADM0_A3'], 'costs'] = grid.loc[grid.ADM0_A3 == row['ADM0_A3'], 'costs'] + grid.loc[grid.ADM0_A3 == row['ADM0_A3'], 'export_costs']
                            # grid.loc[grid.ADM0_A3 == row['ADM0_A3'], 'emissions'] = grid.loc[grid.ADM0_A3 == row['ADM0_A3'], 'emissions'] + grid.loc[grid.ADM0_A3 == row['ADM0_A3'], 'emissions']

                    if current_prod > demand:
                        break

                country_total = pd.DataFrame(
                    {'beef': [country_df.loc[(demand + country_df['beef'] > country_df['beef'].cumsum()), 'beef'].sum()],
                     'costs': [country_df.loc[(demand + country_df['beef'] > country_df['beef'].cumsum()), 'costs'].sum()],
                     'emissions': [
                         country_df.loc[(demand + country_df['beef'] > country_df['beef'].cumsum()), 'emissions'].sum()],
                     'country': [country],
                     'iteration': [iteration]})
                # logger.info("Countr {}: {}".format(country, country_total))

                allcountries = pd.concat([allcountries, country_total])
            # totals = pd.DataFrame({'beef': [allcountries.beef.sum()],
            #                                 'costs': [allcountries.costs.sum()],
            #                                 'emissions': [allcountries.emissions.sum()],
            #                                 'iteration': [iteration]})
            #
            # total_optimised = pd.concat([total_optimised, totals])
        total_optimised = allcountries[['iteration', 'beef', 'costs', 'emissions']].groupby('iteration', as_index=False).sum()
        total_optimised['weight'] = weight/10.
    # total_original.to_csv('./total_original_' + str(spat_const) + '_' + str(weight) + '.csv', index=False)
        pd.concat([initial, total_optimised])
    total_optimised.to_csv('./total_optimised_' + str(spat_const) + '_' + str(aff_scenario) + '_' +  str(weight) + '.csv', index=False)


def parallelise(job_nmr, iterations):

    # Loop through all scenarios to create a dictionary with scenarios and scenario id
    index = 1
    scenarios = {}
    for spat_cons in ['global', 'country']:
        for a in ['noaff', 'nataff', 'manaff']:
            scenarios[index] = [spat_cons, a]
            index += 1

    for w in range(0, 110, 10):

        pool = multiprocessing.Process(target=main,
                                       args=(scenarios[job_nmr][0],  # Spatial constraint
                                             scenarios[job_nmr][1],  # Afforestation scenario
                                             w,
                                             iterations
                                             ))
        pool.start()

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('job_nmr')
    argparser.add_argument('iterations')

    args = argparser.parse_args()
    job_nmr = args.job_nmr
    iterations = args.iterations

    parallelise(job_nmr, iterations)
