

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
surveys = ['ps1', 'ps1', 'ps1', 'vhsdr6', 'vhsdr6', 'unwise-neo3',
           'unwise-neo3']
# List of survey bands, list with length N
bands = ['i','z','y','J','Ks','w1','w2']

#------------------------------------------------------------------------------
# INPUT Keyword Arguments
#------------------------------------------------------------------------------

# List of psf sizes, either None, float or list with length N
psf_size = None
# List of aperture sizes, either None (automatic) or list with length N
apertures = None

# List of magnitude column names, list with length N
mag_column_names = [None, None, None, 'VHS_mag_J',
                  'VHS_mag_K', 'UNWISE_mag_w1',
                  'UNWISE_mag_w2']
# List of magnitude error column names, list with length N
magerr_column_names = [None, None, None, 'vhs_magerr_j', 'vhs_magerr_k',
                       'unwise_magerr_w1', 'unwise_magerr_w2']
# List of S/N column names, list with length N
sn_column_names = None

# List of forced magnitude column names, list with length N
forced_mag_column_names = ['DES_mag_i', 'DES_mag_z','DES_mag_y', None, None,
                           'forced_unwise-neo3_mag_w1',
                           'forced_unwise-neo3_mag_w2']
# List of forced magnitude error column names, list with length N
forced_magerr_column_names = ['forced_desdr1_magerr_i',
                              'forced_desdr1_magerr_z',
                              'forced_desdr1_magerr_Y',
                              None, None, 'forced_unwise-neo3_magerr_w1',
                           'forced_unwise-neo3_magerr_w2']
# List of S/N column names, list with length N
forced_sn_column_names = None

# List of custom visual classification classes (default is point, extended,
# bad pixel, artifact, other)
# visual_classes = ['ydrop', 'point', 'ext', 'badpix', 'blend',
#                   'DES_artifcat', 'UNWISE_artifact', 'no_UNWISE_source',
#                   'no_VHS_source', 'VHS_artifact', 'review']
visual_classes = ['great','really bad']

# add_info_list = [('color', 'JK', 'VHS_mag_J', 'VHS_mag_K'),
#                  ('color', 'KW1', 'VHS_mag_K', 'UNWISE_mag_w1'),
#                  ('color', 'W1W2', 'UNWISE_mag_w1', 'UNWISE_mag_w2'),
#                  ('column','desdr1_nepochs_i','desdr1_nepochs_i'),
#                  ('column','desdr1_nepochs_z','desdr1_nepochs_z'),
#                  ('column','desdr1_nepochs_y','desdr1_nepochs_y'),
#                  ('column', 'VHS-J Pixel X', 'vhs_xpos_j'),
#                  ('column', 'VHS-J Pixel Y', 'vhs_ypos_j'),
#                  ('column', 'VHS-K Pixel X', 'vhs_xpos_k'),
#                  ('column', 'VHS-K Pixel Y', 'vhs_ypos_k'),]

add_info_list = [('column', 'Milliquas citation', 'mq_cite')]

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Suppress warnings!!! THIS IS EXTREMELY DANGEROUS!!!
# import warnings
# warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

table = pd.read_hdf(catalog_filename)

# List of surveys, list with length N
surveys = ['ps1', 'ps1', 'ps1', 'vhsdr6', 'vhsdr6', 'unwise-neo3',
           'unwise-neo3']
# List of survey bands, list with length N
bands = ['i','z','y','J','Ks','w1','w2']


fovs = [80, 80, 80, 80, 80, 80, 80]

n = 10




# Download all the necessary photometry
# If the code gets stuck on this try the non multiprocessing version (below)

# ct.get_photometry_mp(table[:n], ra_column_name, dec_column_name, surveys,
#                      bands, './cutouts/', fovs, n_jobs=3, verbosity=2)



ct.get_photometry(table[:n], ra_column_name, dec_column_name, surveys,
                     bands, './cutouts/', fovs, verbosity=2)



table[:n].to_hdf('temp.hdf5', 'data')

catalog_filename = 'temp.hdf5'


# Run a simple example
imview.run(catalog_filename, image_path, ra_column_name,
                 dec_column_name, surveys, bands, add_info_list=add_info_list,
                 visual_classes=visual_classes, verbosity=2)


# Run a more complex example
