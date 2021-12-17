from qso_toolbox import catalog_tools as ct, image_tools as it, photometry_tools as pt, utils as ut

import time

import pandas as pd
from astropy.io import fits
from astropy.table import Table


test_set = pd.read_hdf('./data/stripe82_milliquas_190210.hdf5',key='data')
print(test_set.shape)


surveys = ['ukidss', 'ukidss', 'ukidss']
bands = ['Y', 'H', 'K']
fovs = [100, 100, 100]

start_time = time.time()
ct.get_photometry(test_set[:50], 'mq_ra', 'mq_dec', surveys, bands,
                  './cutouts/', fovs, verbosity=2)
print("Elapsed time: {0:.2f} sec" .format(
                                time.time() - start_time))

# start_time = time.time()
# ct.get_photometry_mp(test_set[:1000], 'mq_ra', 'mq_dec', surveys, bands,
#                      './cutouts/', fovs, verbosity=2, n_jobs=8)
# print("MP Elapsed time: {0:.2f} sec" .format(
#                                 time.time() - start_time))

