

import pandas as pd
from qso_toolbox import imview_gui as imview
from qso_toolbox import catalog_tools as ct
import multiprocessing as mp




#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# GLOBAL USER INPUT -- TO CHECK BEFORE STARTING THE PYTHON ROUTINE
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# INPUT Arguments
#------------------------------------------------------------------------------

# Input File (hdf5, astropy fits table, ...)
catalog_filename = 'data/stripe82_milliquas_190210.hdf5'

# Image path
image_path = './cutouts/'
# Coordinate column names, either string or list of strings with length N
ra_column_name = 'mq_ra'
dec_column_name = 'mq_dec'
# List of surveys, list with length N
surveys = ['ps1', 'ps1', 'ps1', 'skymapper', 'skymapper', 'skymapper',
           'skymapper',
           'vlass', 'unwise-neo6', 'unwise-neo6']
# List of survey bands, list with length N
bands = ['i','z','y','g','r','i','z','3GHz','w1','w2']

# List of field of views for downloading the images
fovs = [100]*10

#------------------------------------------------------------------------------
# INPUT Keyword Arguments
#------------------------------------------------------------------------------

# List of psf sizes, either None, float or list with length N
psf_size = None
# List of aperture sizes, either None (automatic) or list with length N
apertures = None


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

table = pd.read_hdf(catalog_filename)

table.query('340 < mq_ra < 350 and -1.26 < mq_dec < 0', inplace=True)

# Get only the first 10 objects for this example
n = 10

# Currently only the normal get_photometry function works for VLASS and
# skymapper. The multiprocessing function needs to be updated before it can
# be used.
ct.get_photometry(table[:n], ra_column_name, dec_column_name, surveys,
                     bands, './cutouts/', fovs, verbosity=2)


table[:n].to_hdf('temp.hdf5', 'data')

catalog_filename = 'temp.hdf5'

# Run a simple example
imview.run(catalog_filename, image_path, ra_column_name,
                 dec_column_name, surveys, bands, verbosity=2)




