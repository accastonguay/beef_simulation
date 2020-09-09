import pandas as pd
import numpy as np
import logging
import multiprocessing



def main(grid, weight, c):
    # logger.info('Simulation start')
    # logger.info('Finished loading file')

    grass_cols = []
    for i in ["0250", "0375", "0500"]:
        for n in ["000", "050", "200"]:
            grass_cols.append("grass_" + i + "_N" + n)

    landuses = grass_cols + ['grass_grain', 'stover_grass', 'stover_grain']

    demand = 69000000

    ### Select random land use
    # grid['lu'] = np.random.randint(0,12, size=grid.shape[0])
    u = np.random.rand(grid.shape[0], 1)
    grid['lu'] = (u < c).argmax(axis=1)
    # logger.info('Calculate land use')

    ### Select beef from land use
    grid['beef'] = np.take_along_axis(grid[[lu + '_meat' for lu in landuses]].values,
                                     grid['lu'].values[:, None], axis=1).flatten()
    # logger.info('Number of cells with 0 beef: {}'.format(grid.loc[grid['beef'] == 0].shape[0]))
    # while grid.loc[grid['beef'] == 0].shape[0] > 0:
    #
    #     grid['lu'] = np.where(grid['beef'].values == 0,
    #                           np.random.randint(0, 12, size=grid.shape[0]),
    #                           grid['lu'].values)
    #     grid['beef'] = np.take_along_axis(grid[[lu + '_meat' for lu in landuses]].values,
    #                                       grid['lu'].values[:, None], axis=1).flatten()
    #     logger.info('   N cells remaining: {}'.format(grid.loc[grid['beef'] == 0].shape[0]))
    # logger.info('   N cells: {}'.format(grid.loc[grid['beef'] == 0].shape[0]))

    grid['emissions'] = np.take_along_axis(grid[[lu + '_ghg' for lu in landuses]].values,
                                           grid['lu'].values[:, None], axis=1).flatten()
    grid['costs'] = np.take_along_axis(grid[[lu + '_tot_cost' for lu in landuses]].values,
                                       grid['lu'].values[:, None], axis=1).flatten()
    # logger.info('Calculate beef, costs and emissions')

    # grid['lu'] = np.where(grid['beef'].values == 0,
    #                       np.random.randint(0,12, size=grid.shape[0]),
    #                      grid['lu'].values)

    grid['score'] = grid['costs'].values / grid['beef'].values * weight + grid['emissions'].values / grid['beef'].values * (
                1 - weight)
    # logger.info('Calculate score')

    grid = grid.sort_values('score')
    # logger.info('Sort')

    grid['cumprod'] = grid['beef'].cumsum()
    # logger.info('Cumsum')

    totaldf = pd.DataFrame({ "emissions": [grid.loc[(demand + grid['beef'] - grid['cumprod']) > 0, 'emissions'].sum()],
                            "costs": [grid.loc[(demand + grid['beef'] - grid['cumprod']) > 0, 'costs'].sum()],
                            "production": [grid.loc[(demand + grid['beef'] - grid['cumprod']) > 0, 'beef'].sum()]})

    # logger.info('Calculate total')

    return totaldf

    # logger.info('Export')

def parallelise(job, cpus, job_nmr):

    for loop_nmr in range(0, int(cpus)):

        dict_scenarios = {}
        index = 1
        for i in range(job_nmr+1):
            for j in range(0, int(cpus)):
                dict_scenarios[i, j] = index
                index += 1
        scenario_id = dict_scenarios[job,loop_nmr]
        pool = multiprocessing.Process(target=loop, args = (scenario_id, ))
        pool.start()


def loop(scenario_id):
    import os
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    LOG_FORMAT = "%(asctime)s - %(message)s"
    try:
        logging.basicConfig(
            # filename="/home/uqachare/model_file/logs_opt/opt_" + constraint + "_" + str(crop_yield) + "_" + me_to_meat + "_" + str(lam) + '_' + dem +".log",
            filename="/home/uqachare/model_file/test_mc" + scenario_id + ".log",
            level=logging.INFO,
            format=LOG_FORMAT,
            filemode='w')
    except:
        logging.basicConfig(
            level=logging.INFO,
            format=LOG_FORMAT,
            filemode='w')
    logger = logging.getLogger()

    demand = 69000000

    grass_cols = []
    for i in ["0250", "0375", "0500"]:
        for n in ["000", "050", "200"]:
            grass_cols.append("grass_" + i + "_N" + n)

    landuses = grass_cols + ['grass_grain', 'stover_grass', 'stover_grain']
    grid = pd.read_csv('./score_init.csv')



    for l in landuses:
        grid[l + '_score'] = np.where(grid[l + '_meat'] > 0,
                                      grid[l + '_ghg'] / grid[l + '_meat'],
                                      np.nan)
    grid['score'] = np.nanmin(grid[[l + '_score' for l in landuses]].values, axis=1)
    grid = grid.sort_values('score')
    bestlu = np.nanargmin(grid[[l + '_score' for l in landuses]].values, axis=1)

    beef = np.take_along_axis(grid[[lu + '_meat' for lu in landuses]].values,
                                      bestlu[:, None], axis=1).flatten()

    grid = grid.loc[(demand + beef - beef.cumsum()) > 0]


    nmin = np.nanmin(grid[[l + '_score' for l in landuses]].values, axis=1)
    nmax = np.nanmax(grid[[l + '_score' for l in landuses]].values, axis=1)

    for l in landuses:
        grid[l + '_norm'] = 1 - (grid[l + '_score'].values - nmin) / (nmax - nmin)
    for l in landuses:
        grid[l + '_prob'] = grid[l + '_norm'] / np.nansum(grid[[l + '_norm' for l in landuses]].values, axis=1)

    p = grid[[l + '_prob' for l in landuses]].values
    c = p.cumsum(axis=1)

    alldat = pd.DataFrame()
    for i in range(5):
       temp = main(grid, 0, c)
       alldat = pd.concat([alldat, temp])
       logger.info('end')
    alldat.to_csv('total' + str(scenario_id) + '.csv', index=False)

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('job', help="Job id", type=int)
    argparser.add_argument('cpus', help="Number of cpus", type=int)
    argparser.add_argument('job_nmr', help="Job number", type=int)

    args = argparser.parse_args()
    job = args.job
    cpus = args.cpus
    job_nmr = args.job_nmr

    parallelise(job, cpus, job_nmr)
