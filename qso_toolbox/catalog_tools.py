#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import time
import os

import tarfile

import pandas as pd

import numpy as np
from scipy.optimize import curve_fit

import astropy.units as u
from astropy.utils.console import ProgressBar
from astropy.io import fits, ascii
from astropy.table import Table, Column
from astropy import wcs
from astropy.stats import sigma_clipped_stats
from astropy.nddata.utils import Cutout2D
from astropy.io import ascii, fits


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from photutils import aperture_photometry, CircularAperture
from photutils import Background2D, MedianBackground, make_source_mask


try:
  from urllib2 import urlopen #python2
  from httplib import IncompleteRead
  from urllib2 import HTTPError
except ImportError:
  from urllib.request import urlopen #python3
  from urllib.error import HTTPError
  from http.client import IncompleteRead

#SIA
from pyvo.dal import sia

import gzip
import shutil

from qso_toolbox import utils as ut

import itertools
import multiprocessing as mp

from astropy.nddata.utils import Cutout2D





# ------------------------------------------------------------------------------
#  Input table manipulation
# ------------------------------------------------------------------------------

# copied from http://docs.astropy.org/en/stable/_modules/astropy/io/fits/column.html
# L: Logical (Boolean)
# B: Unsigned Byte
# I: 16-bit Integer
# J: 32-bit Integer
# K: 64-bit Integer
# E: Single-precision Floating Point
# D: Double-precision Floating Point
# C: Single-precision Complex
# M: Double-precision Complex
# A: Character
fits_to_numpy = {'L': 'i1', 'B': 'u1', 'I': 'i2', 'J': 'i4', 'K': 'i8',
                'E': 'f4',
              'D': 'f8', 'C': 'c8', 'M': 'c16', 'A': 'a'}


def fits_to_hdf(filename):
    """ Convert fits data table to hdf5 data table.

    :param filename:
    :return:
    """
    hdu = fits.open(filename)
    filename = os.path.splitext(filename)[0]
    df = pd.DataFrame()

    format_list = ['D', 'J']

    dtype_dict = {}

    # Cycle through all columns in the fits file
    for idx, column in enumerate(hdu[1].data.columns):

        # Check whether the column is in a multi-column format
        if len(column.format) > 1 and column.format[-1] in format_list:
            n_columns = int(column.format[:-1])

            # unWISE specific solution
            if column.name[:6] == 'unwise' and n_columns == 2:
                passbands = ['w1', 'w2']

                for jdx, passband in enumerate(passbands):
                    new_column_name = column.name + '_' + passband

                    print(new_column_name)

                    df[new_column_name] = hdu[1].data[column.name][:, jdx]

                    numpy_type = fits_to_numpy[column.format[-1]]
                    dtype_dict.update({new_column_name: numpy_type})

            # SOLUTIONS FOR OTHER SURVEYS MAY BE APPENDED HERE

        # else for single columns
        else:
            print(column.name)
            df[column.name] = hdu[1].data[column.name]
            numpy_type = fits_to_numpy[column.format[-1]]
            dtype_dict.update({column.name: numpy_type})

    # update the dtype for the DataFrame
    print(dtype_dict)
    df.astype(dtype_dict, inplace=True)

    df.to_hdf(filename+'.hdf5', 'data', format='table')



def check_if_table_is_pandas_dataframe(table):
    """
    Check whether the supplied table is a pandas Dataframe and convert to it
    if necessary.

    This function also returns the original file type. Current file types
    implemented include:
    - astropy tables
    - fits record arrays

    :param table: object
    :return: pd.DataFrame, string
    """

    if type(table) == pd.DataFrame:
        return table, 'pandas_dataframe'

    elif type(table) == Table:
        return table.to_pandas(), 'astropy_table'

    elif type(table) == fits.fitsrec.FITS_rec:
        return Table(table).to_pandas(), 'fits_rec'


def convert_table_to_format(table, format):
    """ Convert a pandas Dataframe back to an original format.

    Conversions to the following file types are possible:
    -astropy table

    :param table: pd.DataFrame
    :param format: string
    :return: object
    """

    if format == 'astropy_table':
        return Table.from_pandas(table)
    elif format == 'fits_rec':
        print('Warning: You entered a fits record array. However, this code '
               'does not support this data type. Your table is returned as an'
               'astropy table!')
        return Table.from_pandas(table)
    else:
        return table


# ------------------------------------------------------------------------------
#  Download catalog images
# ------------------------------------------------------------------------------

def get_photometry(table, ra_col_name, dec_col_name, surveys, bands, image_path,
                   fovs,
                   verbosity=0):
    """Download photometric images for all objects in the given input table.

    Lists need to be supplied to specify the survey, the photometric passband
    and the field of fiew. Each entry in the survey list corresponds to one
    entry in the passband and field of view lists.

    The list of field of views will be accurately downloaded for desdr1. For
    the download of the unWISE image cutouts the field of views will be
    converted to number of pixels with npix = fov / 60. /4. * 100, with an
    upper limit of 256 pixels.

    :param table: table object
        Input data table with at least RA and Decl. columns
    :param ra_col_name: string
        Exact string for the RA column in the table
    :param dec_col_name: string
        Exact string for the Decl. column in the table
    :param surveys: list of strings
        List of survey names, length has to be equal to bands and fovs
    :param bands: list of strings
        List of band names, length has to be equal to surveys and fovs
    :param image_path: string
        Path to the directory where all the images will be stored
    :param fovs: list of floats
        Field of view in arcseconds, length has be equal to surveys and bands
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution
    :return: None
    """

    table, table_format = check_if_table_is_pandas_dataframe(table)

    table['temp_object_name'] = ut.coord_to_name(table[ra_col_name].values,
                                                 table[dec_col_name].values,
                                                 epoch="J")

    for jdx, band in enumerate(bands):

        survey = surveys[jdx]
        fov = fovs[jdx]


        for idx in table.index:
            ra = table[ra_col_name].values[idx]
            dec = table[dec_col_name].values[idx]

            if survey == "desdr1" and band in ["g", "r", "i", "z", "Y"]:

                img_name = table.temp_object_name[idx] + "_" + survey + "_" + \
                           band + "_fov" + '{:d}'.format(fov)

                file_path = image_path + '/' + img_name + '.fits'
                file_exists = os.path.isfile(file_path)

                if file_exists is not True:
                    url = get_desdr1_deepest_image_url(ra, dec, fov=fov,
                                                   band=band,
                                                   verbosity=verbosity)
                else:
                    url = None

            elif survey.split("-")[0] == "unwise" and band in ["w1", "w2",
                                                               "w3", "w4"]:

                # Hack to create npix from fov approximately
                npix = fov/60./4. * 100

                img_name = table.temp_object_name[idx] + "_" + survey + "_" + \
                           band + "_fov" + '{:d}'.format(fov)

                file_path = image_path + '/' + img_name + '.fits'
                file_exists = os.path.isfile(file_path)

                data_release = survey.split("-")[1]
                wband = band[1]


                if file_exists is not True:
                    url = get_unwise_image_url(ra, dec, npix, wband,
                                               data_release)
                else:
                    url = None

            else:
                raise ValueError("Survey and band name not recognized: {} {}. "
                                 "\n "
                                 "Possible survey names include: desdr1, "
                                 "unwise-allwise, unwise-neo1, unwise-neo2, "
                                 "unwise-neo3".format(survey, band))

            if url is not None:
                download_image(url, image_name=img_name, image_path=image_path,
                               verbosity=verbosity)


def get_photometry_mp(table, ra_col_name, dec_col_name, surveys, bands,
                      image_path, fovs, n_jobs=2, verbosity=0):
    """

    :param table: table object
        Input data table with at least RA and Decl. columns
    :param ra_col_name: string
        Exact string for the RA column in the table
    :param dec_col_name: string
        Exact string for the Decl. column in the table
    :param surveys: list of strings
        List of survey names, length has to be equal to bands and fovs
    :param bands: list of strings
        List of band names, length has to be equal to surveys and fovs
    :param image_path: string
        Path to the directory where all the images will be stored
    :param fovs: list of floats
        Field of view in arcseconds, length has be equal to surveys and bands
    :param n_jobs : integer
        Number of cores to be used
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution
    :return: None
    """

    table, table_format = check_if_table_is_pandas_dataframe(table)

    table['temp_object_name'] = ut.coord_to_name(table[ra_col_name].values,
                                                 table[dec_col_name].values,
                                                 epoch="J")

    for jdx, band in enumerate(bands):

        survey = surveys[jdx]
        fov = fovs[jdx]

        mp_args = list(zip(table[ra_col_name].values,
                           table[dec_col_name].values,
                           itertools.repeat(survey),
                           itertools.repeat(band),
                           itertools.repeat(fov),
                           itertools.repeat(image_path),
                           table['temp_object_name'].values,
                           itertools.repeat(verbosity)))


        # alternative idea: get all urls first and then download them.

        with mp.Pool(n_jobs) as pool:
            pool.starmap(_mp_photometry_download, mp_args)


def _mp_photometry_download(ra, dec, survey, band,  fov, image_path,
                            temp_object_name, verbosity):
    """Download one photometric image.

    This function is designed to be an internal function to be called by the
    multiprocessing module using pool.map() .


    :param ra: float
        Right Ascension of the target
    :param dec: float
        Declination of the target
    :param survey: string
        Survey name
    :param band: string
        Passband name
    :param fov: float
        Field of view in arcseconds
    :param image_path: string
        Path to where the image will be stored
    :param temp_object_name:
        Temporary object name specifying the coordinates of the target
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution
    :return: None
    """

    if survey == "desdr1":

        img_name = temp_object_name + "_" + survey + "_" + \
                   band + "_fov" + '{:d}'.format(fov)

        file_path = image_path + '/' + img_name + '.fits'
        file_exists = os.path.isfile(file_path)

        if file_exists is not True:
            url = get_desdr1_deepest_image_url(ra, dec, fov=fov,
                                               band=band,
                                               verbosity=verbosity)
        else:
            url = None

    else:
        raise ValueError("Survey name not recognized: {} . \n "
                         "Possible survey names include: desdr1".format(
            survey))

    if url is not None:
        try:
            download_image(url, image_name=img_name, image_path=image_path,
                       verbosity=verbosity)
        except:
            # if verbosity >0:
            print('Download error')


def download_image(url, image_name, image_path, verbosity=0):
    """Download an image cutout specified by the given url.

    :param url: str
        URL to image cutout
    :param image_name: str
        Unique name of the image: "Coordinate"+"Survey"+"Band"
    :param image_path: str
        Path to where the image is saved to
    :param verbosity: int
        Verbosity > 0 will print verbose statements during the execution
    :return:
    """

    # Check if download directory exists. If not, create it
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    survey = image_name.split("_")[1]

    # Try except clause for downloading the image
    try:
        datafile = urlopen(url)

        check_ok = datafile.msg == 'OK'

        if check_ok:

            if survey == 'desdr1':

                file = datafile.read()

                output = open(image_path+'/'+image_name+'.fits', 'wb')
                output.write(file)
                output.close()
                if verbosity > 0:
                    print("Download of {} to {} completed".format(image_name,
                                                                  image_path))

            elif survey == "unwise-allwise" or survey == "unwise-neo1" or \
                    survey == "unwise-neo2" or survey == "unwise-neo3":

                datafile = urlopen(url)
                file = datafile.read()
                tmp_name = "tmp.tar.gz"
                tmp = open(tmp_name, "wb")
                tmp.write(file)
                tmp.close()

                tar = tarfile.open(tmp_name, "r:gz")
                file_name = tar.firstmember.name
                untar = tar.extractfile(file_name)
                untar = untar.read()

                output = open(image_path + '/' + image_name + '.fits', 'wb')
                output.write(untar)
                output.close()
                if verbosity > 0:
                    print("Download of {} to {} completed".format(image_name,
                                                                  image_path))

        else:
            if verbosity > 0:
                print("Download of {} unsuccessful".format(image_name))

    except (IncompleteRead, HTTPError, AttributeError, ValueError) as err:
        print(err)
        if verbosity > 0:
            print("Download of {} unsuccessful".format(image_name))

# ------------------------------------------------------------------------------
#  Catalog specific URL construction
# ------------------------------------------------------------------------------


def get_desdr1_deepest_image_url(ra, dec, fov=6, band='g', verbosity=0):
    """Return the url from where the DES DR1 cutout can be downloaded.

    :param ra: float
        Right ascension of target
    :param dec: float
        Declination of target
    :param fov: float
        Field of view in arcseconds
    :param band: str
        Passband of the image
    :param verbosity: int
        Verbosity > 0 will print verbose statements during the execution
    :return: str
        Returns the url to the DES DR1 image cutout
    """

    # Set the DES DR1 NOAO sia url
    def_access_url = "https://datalab.noao.edu/sia/des_dr1"
    svc = sia.SIAService(def_access_url)

    fov = fov / 3600.

    try:
        img_table = svc.search((ra, dec), (fov/np.cos(dec*np.pi/180), fov),
                               verbosity=verbosity).to_table()
    except:
        img_table = svc.search((ra, dec),
                               (fov / np.cos(dec * np.pi / 180), fov),
                               verbosity=verbosity).table

    if verbosity > 0:
        print("The full image list contains", len(img_table), "entries")

    sel_band = img_table['obs_bandpass'].astype(str) == band

    sel = sel_band & ((img_table['proctype'].astype(str) == 'Stack') &
                      (img_table['prodtype'].astype(str) == 'image'))

    # basic selection
    table = img_table[sel]  # select

    if len(table) > 0:
        row = table[np.argmax(table['exptime'].data.data.astype(
            'float'))]  # pick image with longest exposure time
        url = row['access_url'].decode()  # get the download URL

        if verbosity > 0:
            print('downloading deepest stacked image...')

    else:
        if verbosity > 0:
            print('No image available.')
        url = None

    return url


def get_unwise_image_url(ra, dec, npix, band, data_release, filetype="image"):
    """ Construct the UNWISE specific URL to download UNWISE cutouts.

    :param ra: float
        Right ascension of target
    :param dec: float
        Declination of target
    :param npix: float
        Cutout image size in pixels
    :param band: str
        Passband of the image
    :param data_release: str
        String specifying the unwise data release. Possible values are: neo1,
        neo2, neo3, allwise
    :param verbosity: int
        Verbosity > 0 will print verbose statements during the execution
    :return: str
        Returns the url to the DES DR1 image cutout
    """

    # Maximum cutout size for unWISE cutouts is 256 pixel
    if npix >=256:
        npix=256

    datatype = {"image":"&file_img_m=on",
                "std":"&file_std_m=on",
                "invvar":"&file_invvar_m=on"}

    file_type = datatype[filetype]

    basedr = dict(neo1="http://unwise.me/cutout_fits?version=neo1&",
                neo2="http://unwise.me/cutout_fits?version=neo2&",
                neo3="http://unwise.me/cutout_fits?version=neo3&",
                 allwise="http://unwise.me/cutout_fits?version=allwise&")
    base = basedr[data_release]
    ra = "ra={:0}&".format(ra)
    dec = "dec={:0}&".format(dec)
    size = "size={:0}&".format(npix)
    band = "bands={0:s}".format(band)

    url = base + ra + dec + size + band + file_type

    return url

# ------------------------------------------------------------------------------
#  Alternative helper functions for non-automatic cutout downloads
# ------------------------------------------------------------------------------


def make_vsa_upload_file(table, ra_col_name, dec_col_name,
                         filename='vsa_upload.csv'):
    """Create the Vista Science Archive upload file (csv format) from the input
    data table.

    :param table: table object
        Input data table with at least RA and Decl. columns
    :param ra_col_name: string
        Exact string for the RA column in the table
    :param dec_col_name: string
        Exact string for the Decl. column in the table
    :param filename:
        Output filename, should include ".csv"
    :return: None
    """

    table, format = check_if_table_is_pandas_dataframe(table)

    table[[ra_col_name, dec_col_name]].to_csv(filename, index=False,
                                              header=False)


def download_vhs_wget_cutouts(wget_filename, image_path, survey_name='vhsdr6',
                              fov=None, verbosity=0):
    """Download the image cutouts based on the VSA 'wget' file specified.

    :param wget_filename: str
        Filepath and filename to the "wget" Vista Science Archive file
    :param image_path: str
        Path to where the image will be stored
    :param survey_name: str
        Name of the Vista Science Archive survey and data release queried.
        This name will be used in further manipulation of the image data. An
        example for VHS DR6 is 'vhsdr6'
    :param fov: float
        Field of view in arcseconds
    :param verbosity: int
        Verbosity > 0 will print verbose statements during the execution
    :return: None
    """

    data = np.genfromtxt(wget_filename, delimiter=' ', dtype=("str"))

    for idx, url in enumerate(data[:, 1]):

        url = url[1:-1]
        vhs_download_name = data[idx, 3]
        vhs_name_list = vhs_download_name.split(".")[0].split("_")

        position_name = vhs_name_list[1]
        band = vhs_name_list[2]

        # Create image name
        if fov is not None:
            image_name = "J" + position_name + "_" + survey_name + "_" + \
                         band + "_fov" + str(fov)
        else:
            image_name = "J" + position_name + "_" + survey_name + "_" + band

        # Check if file is in folder
        file_path = image_path + '/' + image_name + '.fits'
        file_exists = os.path.isfile(file_path)

        if file_exists:
            if verbosity > 0:
                print("Image of {} already exists in {}.".format(image_name,
                                                              image_path))
        else:
            datafile = urlopen(url)
            check_ok = datafile.msg == 'OK'

            if check_ok:

                file = datafile.read()
                tmp_name = "tmp.gz"
                tmp = open(tmp_name, "wb")
                tmp.write(file)
                tmp.close()



                with gzip.open('tmp.gz', 'rb') as f_in:
                    with open(image_path + '/' + image_name + '.fits',
                              'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                if verbosity > 0:
                    print("Download of {} to {} completed".format(image_name,
                                                                  image_path))
            else:
                if verbosity > 0:
                    print("Download of {} unsuccessful".format(image_name))



def make_png(filename, ra, dec, size=20, aperture=2, band='filter',
             forced_mag=None, forced_magerr=None, forced_sn=None,
             catmag=None, caterr=None, catsn=None, output='stamp.png', \
                                                        title=None):
    '''

    make  a png file of the Xarcsec x Xarcsec stamp and plot
    an 6arcsecs aperture

    '''

    data, hdr = fits.getdata(filename, header=True)

    wcs_img = wcs.WCS(hdr)
    pixcrd = wcs_img.wcs_world2pix(ra, dec, 0)
    positions = (np.float(pixcrd[0]), np.float(pixcrd[1]))
    print(pixcrd, positions)
    overlap = True
    try:
        img_stamp = Cutout2D(data, positions, size=size * u.arcsec,
                             wcs=wcs_img)
    except:
        print("Source not in image")
        overlap = False

    if overlap:
        img_stamp = img_stamp.data
        (x,y) = img_stamp.shape


    if overlap:
        plt.cla()
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect=1)

        # norm = ImageNormalize(img_stamp, interval=ZScaleInterval())
        # simg = ax.imshow(img_stamp, origin='lower', norm=norm, cmap='gray')

        zscaleimg = img_stamp.copy()
        #Not considering NanS
        mask = np.isnan(zscaleimg)
        median = np.nanmedian(zscaleimg)

        zscaleimg[mask] = median
        z1, z2 = zscale(zscaleimg)

        simg = ax.imshow(img_stamp, origin='lower', cmap='gray',
                                    vmin=z1, vmax=z2)

        fig.colorbar(simg)

        # Plot circular aperture (forced photometry flux)
        (yy, xx) = img_stamp.shape
        circx = (xx * 0.5) #+ 1
        circy = (yy * 0.5) #+ 1
        aper_pix = aperture_inpixels(aperture, hdr)

        circle = plt.Circle((circx, circy), aper_pix, color='y', fill=False,
                            lw=1.5)
        fig.gca().add_artist(circle)

        # Plot rectangular aperture (error region)
        twenty_arcsec = aperture_inpixels(20., hdr)
        square=plt.Rectangle((circx-twenty_arcsec * 0.5,
                              circy-twenty_arcsec * 0.5),
                             twenty_arcsec, twenty_arcsec,
                             color='y', fill=False, lw=1.5)
        fig.gca().add_artist(square)

        # Create forced photometry label
        if (forced_mag is not None) & (forced_magerr is not None) & (
                forced_sn is not None) :
            forcedlabel = r'${0:s} = {1:.2f} \pm {2:.2f} (SN={3:.1f})$'.format(
                band+"_{forced}", forced_mag, forced_magerr, forced_sn)
            ax.text(xx*0.01, yy * 0.14, forcedlabel, color='black',
                    weight='bold', fontsize='large',
                    bbox=dict(facecolor='white', alpha=0.6))

        # Create catalog magnitude label
        if (catmag is not None) & (catsn is not None) & (caterr is not None):
            catlabel = r'${0:s} = {1:.2f} \pm {3:.2f}  (SN={2:.2f})$'.format(
                band+"_{cat}", catmag, catsn, caterr)

            ax.text(xx*0.02, yy * 0.03, catlabel, color='black', weight='bold',
                    fontsize='large', bbox=dict(facecolor='white', alpha=0.6))

        if title is not None:
            plt.title(title)

        plt.savefig(output, dpi=100)
        print("Image ", output, " created")
        plt.close('all')
    else:
        print("Couldn't create png. Object outside the image")


