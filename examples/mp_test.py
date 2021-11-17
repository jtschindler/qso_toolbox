import pandas as pd
import multiprocessing as mp
from qso_toolbox import catalog_tools as ct
from qso_toolbox import imview_gui as imview

def f(x):
    return x*x

# Hack to avoid being stuck on waiter.acquire()
# I have no idea why it works and what is wrong with the implementation
# below. It seems to be specific to OSX as the code works fine on astro-node3.
pool = mp.Pool(3)
print(pool.map(f,[1,2,3]))
pool.close()



catalog_filename = 'data/stripe82_milliquas_190210.hdf5'


table = pd.read_hdf(catalog_filename)

ct.get_photometry_mp(table[0:5], 'mq_ra', 'mq_dec', ['ps1'], ['i'],
                     './cutouts/', [80], n_jobs=2,  verbosity=2)


# Run a simple example
# imview.run(catalog_filename, './cutouts/', 'mq_ra',
#                  'mq_dec', ['ps1'], ['g'], verbosity=2)
