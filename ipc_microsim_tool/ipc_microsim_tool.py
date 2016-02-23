# -*- coding: utf-8 -*-
"""
------------------------
IPC microsimulation tool
for SDG-based planning
------------------------
Version 0
January 2016
Written by Rafael Guerreiro Osorio
Instituto de Pesquisa Econômica Aplicada - www.ipea.gov.br
International Policy Centre for Inclusive Growth - www.ipc-undp.org
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import random
import statsmodels.formula.api as smf


class IPCmicrosimTool(object):
    """
    #TODO: docstring
    """

    class population(object):
        """
        Population projection object

        On this version, the population projection comes from

        United Nations
        Department of Economic and Social Affairs
        Population Division (2015)
        World Population Prospects: The 2015 Revision, DVD Edition.
        (downloaded November 2015)

        This was imported by read_WPP.py to two CSV files,
        one with past estimates, and other with the projections

        WPP-ESTIMATES-1950-2015.tab.txt
        WPP-PROJECTIONS-2015-2100.tab.txt
        """

        def __init__(self):
            """
            projections is a pandas dataframe
            indexed by year, with a column named year containing
            years; second column is the total population, next
            columns are projections for sex, age, regions, to
            make simulations by groups (in the future...)
            """
            self.projection = None
            self.country = None
            self.description = None

        def get_WPP_countries(self, begins=None):
            """
            get a list of country or region names
            as in the WPP dataset

            initial - first letters
            """
            estimates = pd.read_csv('WPP-ESTIMATES-1950-2015.tab.txt',
                                    sep='\t', index_col=None, na_values='')
            countries = estimates.geoname.values
            if not begins is None:
                begins = begins.capitalize()
                countries = [country for country in countries if
                             country[0:len(begins)] == begins]
            return countries

        def get_WPP_projection(self, country, variant):
            """
            country or region name as in the WPP dataset

            variants available in the WPP dataset:

            0 'Low variant'
            1 'Medium variant'
            2 'High variant'
            3 'Constant-fertility'
            4 'Instant-replacement'
            5 'Zero-migration'
            6 'Constant-mortality'
            7 'No change'
            """
            variants = ('Low variant' , 'Medium variant',
                        'High variant' , 'Constant-fertility',
                        'Instant-replacement', 'Zero-migration',
                        'Constant-mortality', 'No change')
            try:
                int(variant)
                variant = variants[variant]
            except:
                if str(variant).isalnum():
                    raise IndexError(variant)
                if not variant in variants:
                    raise ValueError(variant)

            estimates = pd.read_csv('WPP-ESTIMATES-1950-2015.tab.txt',
                                    sep='\t', index_col=None, na_values='')

            try:
                res1 = estimates[estimates.geoname == country]
            except:
                raise ValueError(country)

            self.country = country
            self.description = 'WPP 2015 - {}'.format(variant)

            projections = pd.read_csv('WPP-PROJECTIONS-2015-2100.tab.txt',
                                      sep='\t', index_col=None, na_values='')

            res2 = projections[projections.geoname == country] \
                              [projections.variant == variant]

            del(estimates, projections)
            todrop = ['index', 'variant', 'geoname', 'geocode']

            # TODO: get period from data
            res1.drop(todrop, axis=1, inplace=True)
            res1 = res1.T[:-1]
            res1.columns = [country]
            res1.index = range(1950, 2015)

            res2.drop(todrop, axis=1, inplace=True)
            res2 = res2.T
            res2.columns = [country]
            res2.index = range(2015, 2101)

            res = pd.concat((res1, res2))
            self.projection = res
            return '{} - {}'.format(country, variant)

    class resource_access(object):
        """
        at the moment this is very basic
        """
        def __init__(self):
            """
            RAM is a pandas dataframe
            indexed by year, with a column named year containing
            years; second column is the access rate, next
            columns are projections for sex, age, regions, to
            make simulations by groups (in the future...)
            """
            self.RAM = None
            self.period = None
            self.description = None

    class microsim(object):
        """
        Microsimulation object
        """
        def __init__(self, dataframe):
            if len(dataframe) == 0 and not isinstance(dataframe, pd.DataFrame):
                raise Exception("Needs a populated pandas dataframe")
            self.dataset = dataframe[:]
            self.results = {}
            self.seedvars = list(dataframe.columns)
            self.cursim = None

        def add_results(self, data, name='newvar'):
            """
            add results generated with microsim methods
            to the results dic - such as:
            pov = ms.poverty('inc', 'wgt', [1.9, 3.1, 5, 10])
            ms.add_results(pov)

            if result is a Series a name is needed, otherwise
            will create column with name 'newvar' and overwrite
            if already a column thus named
            """
            if isinstance(data, pd.Series):
                self.results[self.cursim]['dataset'][name] = data
            elif isinstance(data, pd.DataFrame):
                self.results[self.cursim]['dataset'] = pd.concat(
                    [self.results[self.cursim]['dataset'], data], axis=1)
            else:
                raise TypeError(data)

        def elast_calc(self, key, Y, X, P, stub='', parts=100):
            """
            Add elasticities using log-log quantile regressions

            number of elasticities will be that of hypothetical delimiters
            in parts i.e parts-1

            key - household key
            Y - dependent variable - resource consumption
            X - independent variable - income
            P - household population weights
            stub - sufix to name variables containing quantiles
                   and elasticities
            parts - number of parts
            """
            dt = self.dataset
            quantstub = 'quant' + stub
            elaststub = 'elast' + stub
            print '\nElasticity calculator started - please be patient'

            # take the logs of Y and X
            dt['__lnY'] = np.log(dt[Y])
            dt['__lnX'] = np.log(dt[X])

            # log of 0 is -infinite, replace with missing (NaN)
            dt['__lnY'][dt[Y] == 0] = np.NaN
            dt['__lnX'][dt[X] == 0] = np.NaN

            # rescale and round weights to inform replication
            dt['__' + P] = dt[P]/dt[P].min()
            dt['__rdwgt'] = dt['__' + P].round()

            # define quantiles based on parts and mark
            dt.sort(Y, inplace=True)
            dt[quantstub] = (dt['__' + P].cumsum() /
                             dt['__' + P].sum() *
                             parts).astype(int) / float(parts)
            dt.sort(key, inplace=True)

            # the quantile of the regression, can't be 0 or 1
            # unique() is sorted as dt, get the smallest non zero quantile
            # and the larger < 1
            quantiles = dt[quantstub].unique()
            quantiles.sort()
            quantiles = quantiles[1:-1]
            dt[quantstub][dt[quantstub] == 0] = quantiles[0]
            dt[quantstub][dt[quantstub] == 1] = quantiles[-1]

            # dataframe with replications
            print 'Replicating observations, {} to {}...'.format(
                dt['__rdwgt'].count(), int(dt['__rdwgt'].sum()))
            lnY, lnX = pd.Series(), pd.Series()
            for i in xrange(len(dt)):
                lnY = lnY.append(pd.Series((dt['__lnY'][i],) *
                                 int(dt['__rdwgt'][i])))
                lnX = lnX.append(pd.Series((dt['__lnX'][i],) *
                                 int(dt['__rdwgt'][i])))
            estdt = pd.DataFrame()
            estdt['lnY'] = lnY
            estdt['lnX'] = lnX
            del lnY, lnX

            # calculate elasticities
            print 'Fitting models...'
            model = smf.quantreg('lnY ~ lnX', estdt)
            elastseries = ()
            #elasterrors = ()
            print 'Quantile\telasticity\tse_elast\tintercept\tse_intercept'
            for quantile in quantiles:
                elast = model.fit(quantile)
                elastseries += (elast.params[1],)
                print '{}\t{:8.6f}\t{:8.6f}\t{:8.6f}\t{:8.6f}'.format(
                    quantile, elast.params[1], elast.bse[1], elast.params[0],
                    elast.bse[0],)
            elastdt = pd.DataFrame()
            elastdt[quantstub] = quantiles
            elastdt[elaststub] = elastseries

            # add elasticities and clean dataset
            todrop = [var for var in dt.keys() if '__' in var]
            self.dataset = pd.merge(dt, elastdt, on=quantstub)
            self.dataset.sort(key, inplace=True)
            self.dataset.reset_index(drop=True, inplace=True)
            self.dataset.drop(todrop, axis=1, inplace=True)
            self.seedvars += [quantstub, elaststub]

        def __reset__(self):
            """
            keep the seed variables and drop all others from the dataset
            """
            todrop = [col for col in list(self.dataset.columns) if
                      col not in self.seedvars]
            self.dataset.drop(todrop, axis=1, inplace=True)

        def simulate(self, name, period, X, Y, P, key):
            """
            name - a name for the simulation
            period - tuple base year, end year eg. (2010,2030)
            X - income tuple (variable name, stub,
                              'random'/'order', growth object)
            Y - [(resourcevar, stub, elasticityvar)]
            P - weight variable
            """

            # resets the dataset, results of previous simulations will
            # remain in results
            self.__reset__()
            self.cursim = name
            if name in self.results.keys():
                rettext = 'Simulation {} results overwritten'.format(name)
                del(self.results[name])
            else:
                rettext = 'Simulation {} results written'.format(name)
            self.results[name] = {'name': name}
            self.results[name]['period'] = period
            self.results[name]['income'] = (X[0], X[1])
            self.results[name]['growth'] = X[3]
            for y in range(len(Y)):
                self.results[name]['resvar{}'.format(y)] = (Y[y][0], Y[y][1])
                self.results[name]['reselast{}'.format(y)] = Y[y][2]
            self.results[name]['dataset'] = pd.DataFrame({'year': range(
                period[0], period[1] + 1)}, range(period[0], period[1] + 1))

            # first year of the period is the base year
            # a duplicate is generated as stubbaseyear
            dt = self.dataset
            dt[X[1] + str(period[0])] = dt[X[0]]
            for y in Y:
                dt[y[1] + str(period[0])] = dt[y[0]]

            # a list to register the growth pattern
            # 'none' for base year
            grwtpatt = ['none']

            # simulations begin
            order = 0
            for year in xrange(period[0] + 1, period[1] + 1):

                # previous income distribution - base year
                prvinc = X[1] + str(year - 1)

                # sort by previous income distribution
                # repeating allows income mobility
                sorted_dt = dt.sort(prvinc)

                # partition and tag cases by income quantiles
                # each part will receive a growth rate
                # TODO: make sure there is no variable named as growth.key
                #       in the microsim dataset
                sorted_dt[X[3].key] = (sorted_dt[P].cumsum() /
                                     sorted_dt[P].sum() *
                                     X[3].parts).astype(int)
                sorted_dt[X[3].key][sorted_dt[X[3].key]
                                  == X[3].parts] = X[3].parts - 1
                dt[X[3].key] = sorted_dt[X[3].key]
                del sorted_dt
                # get and distribute growth rates
                if X[2] == 'random':
                    # choose a column from growth.dataset, excluding
                    # key from choice
                    grwtrates = random.choice([col for col in
                                               X[3].dataset.columns if
                                               col != X[3].key])
                elif X[2] == 'order':
                    # choose a column as they appear on the dataset
                    # left to right after key; if there are less columns
                    # than periods, start over
                    orlst = [col for col in X[3].dataset.columns if
                             col != X[3].key]
                    grwtrates = orlst[order]
                    order += 1
                    if order == len(orlst):
                        order = 0
                elif X[2] in X[3].dataset.columns:
                    # a column name was passed - this specific
                    # single pattern will be repeated - same as
                    # choosing ordered with just one column
                    grwtrates = X[2]
                else:
                    raise ValueError(X[2])

                # register the name of the growth pattern (column name)
                # this goes to results.dataset
                grwtpatt += [grwtrates]

                # prepare a dataset with the growth pattern and the key
                # and merge it with the simulation dataset distributing
                # the growth rates by income quantiles
                tomerge = X[3].dataset[[X[3].key, grwtrates]]
                self.dataset = pd.merge(dt, tomerge, on=X[3].key)
                self.dataset.sort(key, inplace=True)
                self.dataset.reset_index(drop=True, inplace=True)
                dt = self.dataset


                # THIS IS IMPORTANT: the rate should be Xt1/Xt0-1
                dt[grwtrates] = dt[grwtrates] + 1.0

                # new income variable
                dt[X[1] + str(year)] = dt[prvinc] * dt[grwtrates]

                # new resource variables
                for y in Y:
                    dt[y[1] + str(year)] = dt[y[1] + str(year - 1)] * \
                        (dt[y[2]] * (dt[grwtrates] - 1.0) + 1.0)
                self.dataset.drop([X[3].key, grwtrates], axis=1, inplace=True)

            # simulation is over, besides information about the
            # simulation, means are stored  in results, and also
            # the name of the growth pattern for income
            self.results[name]['dataset']['grwtpatt'] = grwtpatt
            col = self.results[name]['income'][1]
            self.results[name]['dataset']['mean_{}'.format(col)] = self.mean(
                X[1], P)
            for y in range(len(Y)):
                col = self.results[name]['resvar{}'.format(y)][1]
                self.results[name]['dataset']['mean_{}'.format(col)] = \
                    self.mean(Y[y][1], P, nozero=True)
            return rettext

        def totaldemand(self, stub, pop, ram, correct=1, unit=1e6):
            period = self.results[self.cursim]['period']
            totdem = pd.Series(0.0, range(period[0], period[1] + 1))
            resmean = 'mean_{}'.format(stub)
            # TODO: improve this
            # below it is assuming data average is for months
            # and annualizes; improve indices - why pop is string
            cmp1 = pop.projection.icol(0).loc[period[0]:period[1]]
            cmp2 = ram.RAM.icol(0)
            cmp3 = self.results[self.cursim]['dataset'][resmean]
            totdem = (cmp1 * cmp2 * cmp3 * 12 * correct) / unit
            return totdem

        def mean(self, stub, weight, nozero=False):
            if nozero:
                rescale = True
            else:
                rescale = False
            period = self.results[self.cursim]['period']
            mean = pd.Series(0.0, range(period[0], period[1] + 1))
            for year in xrange(period[0], period[1] + 1):
                mean[year] = ((self.dataset['{}{}'.format(stub, year)] *
                               self.dataset[weight]).sum() /
                               self.dataset[weight].sum())
                if rescale:
                    zeropop = self.dataset[weight][self.dataset[
                        '{}{}'.format(stub, year)] == 0].sum()
                    poptot = self.dataset[weight].sum() - zeropop
                    mean[year] = (mean[year] * self.dataset[weight].sum() /
                                   poptot)
            return mean

        def variance(self, stub, weight):
            period = self.results[self.cursim]['period']
            means = self.mean(stub, weight)
            variance = pd.Series(0.0, range(period[0], period[1] + 1))
            for year in xrange(period[0], period[1] + 1):
                variance[year] = (((self.dataset['{}{}'.format(stub, year)] -
                                    means[year]) ** 2
                                   * self.dataset[weight]).sum() /
                                  (self.dataset[weight].sum() - 1))
            return variance

        def inequality_ge(self, stub, weight, theta=1.0, nozero=False):
            period = self.results[self.cursim]['period']
            if theta <= 0 or theta == 1:
                    nozero = True
            means = self.mean(stub, weight, nozero=nozero)
            ge = pd.Series(0.0, range(period[0], period[1] + 1))
            for year in xrange(period[0], period[1] + 1):
                if nozero:
                    zeropop = self.dataset[weight][self.dataset[
                        '{}{}'.format(stub, year)] == 0].sum()
                    poptot = self.dataset[weight].sum() - zeropop
                else:
                    poptot = self.dataset[weight].sum()

                self.dataset['__ratio'] = (
                    self.dataset['{}{}'.format(stub, year)] / means[year])
                if float(theta) == 0.0:
                    self.dataset['__ratio'] = np.log(
                        self.dataset['__ratio'] ** -1)
                    self.dataset['__ratio'][self.dataset['__ratio']
                                            == np.inf] = 0
                    ge[year] = (self.dataset['__ratio'] *
                                self.dataset[weight]).sum() / poptot
                elif float(theta) == 1.0:
                    self.dataset['__ratio'] = self.dataset['__ratio'] * np.log(
                        self.dataset['__ratio'])
                    self.dataset['__ratio'][self.dataset['__ratio']
                                            == np.inf] = 0
                    ge[year] = (self.dataset['__ratio'] *
                                self.dataset[weight]).sum() / poptot
                else:
                    self.dataset['__ratio'] = self.dataset['__ratio'] ** theta
                    self.dataset['__ratio'][self.dataset['__ratio']
                                            == np.inf] = 0
                    ge[year] = (((self.dataset['__ratio'] *
                                  self.dataset[weight]).sum() / poptot - 1) /
                                (theta ** 2 - theta))
            if 0 in self.dataset['{}{}'.format(stub, year)]:
                print '\nThere were zeroes in {}xxxx'.format(stub)
                if nozero:
                    print 'Ge({}) did not consider those obs.'.format(theta)
            return ge

        def inequality_gini(self, stub, weight):
            """
            stub - variable stub existent in microsim.dataset
            weight - weight variable in microsim.dataset
            """
            period = self.results[self.cursim]['period']
            gini = pd.Series(0.0, range(period[0], period[1] + 1))
            for year in xrange(period[0], period[1] + 1):
                """
                the Gini index is calculated as twice the area between
                the Lorenz Curve and the diagonal (equality line)
                """
                curvar = '{}{}'.format(stub, year)
                self.dataset.sort(curvar, inplace=True)
                self.dataset['__cumpop'] = (self.dataset[weight].cumsum() /
                                            self.dataset[weight].sum())
                self.dataset['__cumren'] = ((self.dataset[curvar] *
                                             self.dataset[weight]).cumsum() /
                                            (self.dataset[curvar] *
                                             self.dataset[weight]).sum())
                self.dataset['__polyarea'] = (self.dataset['__cumpop'] -
                    self.dataset['__cumpop'].shift(1)) * (
                    self.dataset['__cumren'] +
                    self.dataset['__cumren']).shift(1)
                gini[year] = 1 - self.dataset['__polyarea'].sum()
            todrop = [c for c in self.dataset.columns if '__' in c]
            self.dataset.drop(todrop, axis=1, inplace=True)
            return gini

        def poverty(self, stub, weight, plines):
            """
            stub - variable stub existent in microsim.dataset
            weight - weight variable in microsim.dataset
            plines - list or tuple with poverty lines
            """
            period = self.results[self.cursim]['period']
            perang = range(period[0], period[1] + 1)
            povind = pd.DataFrame({'__temp': pd.Series(0.0, perang)})
            for pline in plines:
                povind['p0({})'.format(pline)] = 0.0
                povind['p1({})'.format(pline)] = 0.0
                povind['p2({})'.format(pline)] = 0.0
                povind['pgap({})'.format(pline)] = 0.0
                povind['pge2({})'.format(pline)] = 0.0
            povind.drop('__temp', axis=1, inplace=True)

            for year in xrange(period[0], period[1] + 1):
                curvar = '{}{}'.format(stub, year)
                dt = self.dataset
                # process a list of poverty lines
                for pline in plines:
                    povind['p0({})'.format(pline)][year] = (
                        ((dt[curvar] < pline) *
                          dt[weight]).sum() /
                         dt[weight].sum())

                    povind['pgap({})'.format(pline)][year] = (
                        (((dt[curvar] < pline) *
                          (pline - dt[curvar]) / pline) *
                         dt[weight]).sum() /
                        dt[weight][dt[curvar] < pline].sum())

                    povind['p1({})'.format(pline)][year] = (
                        povind['p0({})'.format(pline)][year] *
                        povind['pgap({})'.format(pline)][year])

                    mnge2 = (((dt[curvar] < pline) * dt[curvar] *
                        dt[weight]).sum() /
                        dt[weight][dt[curvar] < pline].sum())
                    stge2 = (((dt[curvar] < pline) * (dt[curvar] -
                        mnge2) ** 2 * dt[weight]).sum() /
                        dt[weight][dt[curvar] < pline].sum()) ** 0.5
                    ge2 = (stge2/mnge2) ** 2 / 2
                    povind['pge2({})'.format(pline)][year] = ge2

                    povind['p2({})'.format(pline)][year] = (
                        povind['p0({})'.format(pline)][year] *
                        (povind['pgap({})'.format(pline)][year] ** 2 +
                         (1 - povind['pgap({})'.format(pline)][year]) ** 2 *
                        ge2 * 2))

            return povind

        def visualize(self, data, xcol='year', cols='all', subplotcols=2):
            if xcol not in data.columns:
                raise Exception('{} not in dataset'.format(xcol))
            if cols == 'all':
                toplot = [col for col in data.columns if col != xcol
                          and type(data[col].iloc[0]) != str]
            elif isinstance(cols, tuple) or isinstance(cols, list):
                for col in cols:
                    if col not in data.columns:
                        raise Exception('{} not in dataset'.format(col))
                    elif type(data[col].iloc[0]) == str:
                        raise Exception('{} is a string variable'.format(col))
                toplot = cols
            elif isinstance(cols, str):
                if cols not in data.columns:
                    raise Exception('{} not in dataset'.format(cols))
                elif type(data[cols].iloc[0]) == str:
                    raise Exception('{} is a string variable'.format(cols))
                toplot = [cols]
            else:
                raise TypeError()

            plt.figure(1)
            plt.clf()

            if len(toplot) > subplotcols:
                gridrows = len(toplot) // subplotcols
                if len(toplot) % subplotcols > 0:
                    gridrows += 1
                gridcols = subplotcols
            elif len(toplot) < subplotcols:
                gridrows = 1
                gridcols = len(toplot)
            else:
                gridrows = 1
                gridcols = subplotcols

            for nplot in range(len(toplot)):
                plot = toplot[nplot]
                plt.subplot(gridrows, gridcols, nplot)
                #plt.subplot.axes.get_xaxis().set_ticks([])
                #plt.subplot.axes.get_yaxis().set_ticks([])
                plt.title(plot)
                plt.locator_params(axis='both', tight=True, nbins=7)
                plt.plot(data[xcol],
                         data[plot])

            plt.tight_layout(pad=1.5)
            plt.show()

        class growth(object):
            """
            Growt object
            dataset
            key column name
            # of parts

            this is the growth rate for every partition
            of the income distribution

            simulate will partition the income distribution of
            microsim.dataset based on the growth object # of parts
            and the identifier is key column name

            the dataset has the growth rates, if the dataset has
            more than one growth rate, simulate will need an order
            instruction (such as random.choice)
            """

            def __init__(self, parts=100, key='key'):
                self.dataset = pd.DataFrame({key: range(parts)})
                self.parts = parts
                self.key = key
                self.__logn = 1
                self.__pwrn = 1
                self.__coln = 1

            def add_columns(self, data, stub='newcol'):
                """
                Will get columns from a pandas dataframe
                same rules that apply to csv files
                csv reads to pandas dataframe and calls add_columns
                """
                if isinstance(data, pd.Series) or \
                   isinstance(data, tuple) or \
                   isinstance(data, list):
                    if not self.__invalid(data):
                        name =stub
                        while name in self.dataset.columns:
                            name = '{}{}'.format(stub, self.__logn)
                            self.__coln += 1
                    dataframe = pd.DataFrame({name: data})
                else:
                    dataframe = data[:]
                for col in dataframe.columns:
                    if self.__invalid(dataframe[col]):
                        raise TypeError('{} is invalid'.format(col))
                    if col in self.dataset.columns:
                        raise Exception('{} exists in dataset'.format(col))
                    self.dataset[col] = dataframe[col]

            def add_log(self, stub='log',
                        average=0.01,
                        shift=0,
                        flip=False,
                        alpha=1):
                """
                will add a log growth pattern with average = average

                if flip the log function is mirrored and the
                growth rate will correlate negatively with
                income - FLIP for PRO-POOR GROWTH
                """

                self.dataset['__x'] = self.dataset.key + alpha

                if flip:
                    X = self.dataset['__x'][:]
                    X.sort(ascending=False)
                    X = X.reset_index()
                    self.dataset['__x'] = X['__x']

                self.dataset['__lnx'] = np.log(self.dataset['__x'])

                name = stub
                while name in self.dataset.columns:
                    name = '{}{}'.format(stub, self.__logn)
                    self.__logn += 1

                average = average * self.parts
                self.dataset[name] = self.dataset['__lnx'] / \
                                     self.dataset['__lnx'].sum() * average + shift
                self.dataset.drop(['__x', '__lnx'], axis=1, inplace=True)

            def add_power(self, stub='pwr',
                          power=0.2,
                          average=0.01,
                          shift=0,
                          flip=False,
                          alpha=1):
                """
                will add a power growth pattern with average = average

                if flip the power function is mirrored and the
                growth rate will correlate negatively with
                income - FLIP for PRO-POOR GROWTH
                """

                self.dataset['__x'] = self.dataset.key + alpha

                if flip:
                    X = self.dataset['__x'][:]
                    X.sort(ascending=False)
                    X = X.reset_index()
                    self.dataset['__x'] = X['__x']

                self.dataset['__pwrx'] = self.dataset['__x'] ** float(power)

                name = stub
                while name in self.dataset.columns:
                    name = '{}{}'.format(stub, self.__pwrn)
                    self.__pwrn += 1
                average = average * self.parts
                self.dataset[name] = self.dataset['__pwrx'] / \
                                     self.dataset['__pwrx'].sum() * average + shift
                self.dataset.drop(['__x', '__pwrx'], axis=1, inplace=True)

            def load_csv(self, csvfile, delimiter='\t'):
                """
                reads to pandas dataframe and calls add_columns
                csv file has to conform:

                # lines should be equal to parts
                column names in first row (valid for pandas)
                growth rate columns only, in proportions, 0.1 = 10%

                if one growth rate column per simulation year, first
                column should be the growth rate from base year to
                second year
                """
                data = pd.read_csv(csvfile, sep=delimiter,
                                   index_col=False, na_values='')
                self.add_columns(data)

            def __invalid(self, X):
                if isinstance(X, pd.Series):
                    if tuple(X.index) != tuple(self.dataset.index):
                        raise Exception('Idxs differ - should be zero to self.parts')
                    return False
                elif isinstance(X, tuple) or isinstance(X, list):
                    if len(X) != self.parts:
                        raise Exception('Sizes differ: len(X) != self.parts')
                    return False
                else:
                    return True


# if main, runs the example
if __name__ == '__main__':
    pass




