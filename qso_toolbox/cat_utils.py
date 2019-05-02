#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import time
import os

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


from qso_toolbox import utils as ut

import itertools
import multiprocessing as mp



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
        print ('Warning: You entered a fits record array. However, this code '
               'does not support this data type. Your table is returned as an'
               'astropy table!')
        return Table.from_pandas(table)
    else:
        return table


def get_photometry(table, ra_col_name, dec_col_name, surveys, bands, image_path,
                   fovs,
                   verbosity=0):
    """Download photometric images for all objects in the given input table.

    Lists need to be supplied to specify the survey, the photometric passband
    and the field of fiew. Each entry in the survey list corresponds to one
    entry in the passband and field of view lists.

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

            if survey== "desdr1":

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

            else:
                raise ValueError("Survey name not recognized: {} . \n "
                                 "Possible survey names include: desdr1".format(
                    survey))

            if url is not None:
                download_image(url, image_name=img_name, image_path=image_path,
                           verbosity=verbosity)


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
            print ('Download error')


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


        # idea: get all urls first and then download them.

        with mp.Pool(n_jobs) as pool:
            pool.starmap(_mp_photometry_download, mp_args)






def get_desdr1_deepest_image_url(ra, dec, fov=6, band='g', verbosity=0):
    """Returns the url from where the DES DR1 cutout can be downloaded.

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
    DEF_ACCESS_URL = "https://datalab.noao.edu/sia/des_dr1"
    svc = sia.SIAService(DEF_ACCESS_URL)

    fov = fov / 3600.

    try:
        img_table = svc.search((ra,dec), (fov/np.cos(dec*np.pi/180), fov),
                           verbosity=verbosity).to_table()
    except:
        img_table = svc.search((ra, dec),
                               (fov / np.cos(dec * np.pi / 180), fov),
                               verbosity=verbosity).table

    if verbosity>0:
        print("The full image list contains", len(img_table), "entries")

    sel_band = img_table['obs_bandpass'].astype(str) == band


    sel = sel_band & ((img_table['proctype'].astype(str) == 'Stack') & \
                      (img_table['prodtype'].astype(str) == 'image'))
          # basic selection


    table = img_table[sel]  # select

    if len(table) > 0:
        row = table[np.argmax(table['exptime'].data.data.astype(
            'float'))]  # pick image with longest exposure time
        url = row['access_url'].decode()  # get the download URL

        if verbosity>0:
            print('downloading deepest stacked image...')

    else:
        if verbosity>0:
            print('No image available.')
        url = None

    return url



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

    # Try except clause for downloading the image
    try:
        datafile = urlopen(url)

        check_ok = datafile.msg == 'OK'

        if check_ok:
            file = datafile.read()
            # fits_name = '{0}_desdr1_{1}.fits'.format(ps1name_i, args.bands.lower())
            output = open(image_path+'/'+image_name+'.fits', 'wb')
            output.write(file)
            output.close()
            if verbosity>0:
                print("Download of {} to {} completed".format(image_name,
                                                              image_path))
        else:
            if verbosity>0:
                print("Download of {} unsuccessful".format(image_name))

    except (IncompleteRead, HTTPError, AttributeError, ValueError) as err:
        print(err)
        if verbosity > 0:
            print("Download of {} unsuccessful".format(image_name))


def get_forced_photometry(table, ra_col_name, dec_col_name, surveys,
                          bands, apertures, fovs, image_path,
                          auto_download=True,
                          verbosity=0):
    """

    :param table:
    :param ra_col_name:
    :param dec_col_name:
    :param surveys:
    :param bands:
    :param apertures:
    :param fovs:
    :param image_path:
    :param auto_download: Boolean
    :param verbosity:
    :return:
    """


    # Check if table is pandas DataFrame otherwise convert to one
    table, format = check_if_table_is_pandas_dataframe(table)
    # Add a column to the table specifying the object name used
    # for the image name
    table['temp_object_name'] = ut.coord_to_name(table[ra_col_name].values,
                                            table[dec_col_name].values,
                                            epoch="J")



    for jdx, survey in enumerate(surveys):

        band = bands[jdx]
        aperture = apertures[jdx]
        fov = fovs[jdx]

        for idx in table.index:

            ra = table[ra_col_name].values[idx]
            dec = table[dec_col_name].values[idx]



            img_name = table.temp_object_name[idx]+"_"+survey+"_"+ \
                       band+"_fov"+'{:d}'.format(fov)

            # Check if file is in folder
            file_path = image_path + '/' + img_name + '.fits'
            file_exists = os.path.isfile(file_path)

            if file_exists is not True and auto_download is True:

                if survey == "desdr1":
                    url = get_desdr1_deepest_image_url(ra,
                                                       dec,
                                                       fov=fov,
                                                       band=band,
                                                       verbosity=verbosity)

                else:
                    raise ValueError("Survey name not recognized: {} . \n "
                                     "Possible survey names include: desdr1".format(
                        survey))

                if url is not None:
                    download_image(url, image_name=img_name,
                                   image_path=image_path,
                                   verbosity=verbosity)


                    file_path = image_path + '/' + img_name + '.fits'
                    file_exists = os.path.isfile(file_path)


            file_size_sufficient = False
            if file_exists is True:
                # Check if file is sufficient
                file_size_sufficient = check_image_size(img_name,
                                                        file_path,
                                                        verbosity)

            if file_exists is True and file_size_sufficient is True:

                mag, flux, sn, err = \
                    calculate_forced_aperture_photometry(file_path,
                                                         ra, dec, survey,
                                                         aperture,
                                                         verbosity=verbosity)
                table.loc[idx, 'forced_{}_mag_{}'.format(survey,band)] = mag
                table.loc[idx, 'forced_{}_flux_{}'.format(survey, band)] = flux
                table.loc[idx, 'forced_{}_sn_{}'.format(survey, band)] = sn
                table.loc[idx, 'forced_{}_magerr_{}'.format(survey, band)] = \
                    err
                table.loc[idx, 'forced_{}_{}_comment'.format(survey, band)] =\
                    'ap_{}'.format(aperture)

            if file_exists is True and file_size_sufficient is not True:

                table.loc[idx, 'forced_{}_{}_comment'.format(survey, band)] = \
                    'image_too_small'.format(aperture)

            if file_exists is not True:

                table.loc[idx, 'forced_{}_{}_comment'.format(survey, band)] = \
                    'image_not_available'.format(aperture)



    table.drop(columns='temp_object_name')

    table = convert_table_to_format(table, format)

    return table


def calculate_forced_aperture_photometry(filepath, ra, dec, survey, aperture,
                                         verbosity=0):
    """

    :param filepath:
    :param ra:
    :param dec:
    :param survey:
    :param aperture:
    :param verbosity:
    :return:
    """

    # Open the fits image
    data, header = fits.getdata(filepath, header=True)

    # Convert radius from arcseconds to pixel
    pixelscale = get_pixelscale(header)
    aperture_pixel = aperture / pixelscale  # pixels

    # Transform coordinates of target position to pixel scale
    wcs_img = wcs.WCS(header)
    pixel_coordinate = wcs_img.wcs_world2pix(ra, dec, 1)

    # Get photometry
    positions = (pixel_coordinate[0], pixel_coordinate[1])
    apertures = CircularAperture(positions, r=aperture_pixel)
    f = aperture_photometry(data, apertures)
    flux = np.ma.masked_invalid(f['aperture_sum'])

    # Get the noise
    rmsimg, mean_noise, empty_flux = get_noiseaper(data, aperture_pixel)

    sn = flux[0] / rmsimg

    if verbosity > 0:
        print("flux: ", flux[0], "sn: ", sn)

    if sn < 0:
        flux[0] = rmsimg
        err = -1
        mags = flux_to_magnitude(flux, survey)[0]
    else:
        mags = flux_to_magnitude(flux, survey)[0]
        err = mag_err(1. / sn, verbose=False)

    if verbosity > 0:
        print("mag: ", mags)

    if mags is np.ma.masked:
        mags = -999
    if sn is np.ma.masked:
        sn = np.nan
    if err is np.ma.masked:
        err = np.nan
    if flux[0] is np.ma.masked:
        flux = np.nan
    else:
        flux = flux[0]

    return mags, flux, sn, err


def check_image_size(image_name, file_path, verbosity):
    """

    :param image_name:
    :param file_path:
    :param verbosity:
    :return:
    """

    shape = fits.getdata(file_path).shape
    min_axis = np.min(shape)
    if min_axis < 50 and verbosity >0:
        print("Minimum image dimension : {} (pixels)".format(min_axis))
        print("Too few pixels in one axis (<50). Skipping {}".format(
            image_name))

    if min_axis < 50:
        return False
    else:
        return True


# magnitude calculation

def flux_to_magnitude(flux, survey):
    """

    :param flux:
    :param survey:
    :return:
    """
    if survey == "desdr1":
        zpt=30.
    else:
        raise ValueError("Survey name not recgonized: {}".format(survey))

    return -2.5 * np.log10(flux) + zpt


#  OLD CODE FROM EDUARDO - NOT MODIFIED

def aperture_inpixels(aperture, hdr):
    '''
    receives aperture in arcsec. Returns aperture in pixels
    '''
    pixelscale=get_pixelscale(hdr)
    aperture /= pixelscale #pixels

    return aperture

def zscale(zscaleimg):

    z1 = np.amin(zscaleimg)
    z2 = np.amax(zscaleimg)

    return z1, z2

def make_png(img, ra, dec, size=20, aperture=2, band='filter', forced_mag='',
             forced_magerr='', forced_sn='', output='stamp.png', title=''):
    '''

    make  a png file of the Xarcsec x Xarcsec stamp and plot
    an 6arcsecs aperture

    '''
    data, hdr = fits.getdata(img, header=True)

    wcs_img = wcs.WCS(hdr)
    pixcrd = wcs_img.wcs_world2pix(ra, dec, 0)
    positions = (np.float(pixcrd[0]), np.float(pixcrd[1]))

    overlap = True
    try:
        img_stamp = Cutout2D(data, positions,
                        size=size * u.arcsec, wcs=wcs_img)
    except:
        print("Source not in image")
        overlap=False

    if overlap:
        img_stamp = img_stamp.data
        (x,y) = img_stamp.shape

    #if (x>20) & (y>20) & overlap:
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
        median  = np.nanmedian(zscaleimg)

        zscaleimg[mask] = median
        z1, z2 = zscale(zscaleimg)
            #print(z1, z2)

        simg = ax.imshow(img_stamp, origin='lower', cmap='gray',
                                    vmin=z1, vmax=z2)


        fig.colorbar(simg)
        (yy, xx) = img_stamp.shape
        circx = (xx * 0.5) #+ 1
        circy = (yy * 0.5) #+ 1
        aper_pix = aperture_inpixels(aperture, hdr)
        twenty_arcsec = aperture_inpixels(20., hdr)
        circle=plt.Circle((circx, circy),
                                 aper_pix, color='y', fill=False, lw=1.5)
        fig.gca().add_artist(circle)
        square=plt.Rectangle((circx-twenty_arcsec *0.5, circy-twenty_arcsec *0.5),
                                 twenty_arcsec, twenty_arcsec, color='y', fill=False, lw=1.5)
        fig.gca().add_artist(square)
        #ax.scatter(circx, circy, c='y', marker='+', s=180)

        forcedlabel = r'${0:s} = {1:.2f} \pm {2:.2f} (SN={3:.1f})$'.format(
                                                 band+"_{forced}", forced_mag,
            forced_magerr, forced_sn)
        ax.text(xx*0.01, yy * 0.14, forcedlabel, color='black', weight='bold',
         fontsize='large', bbox=dict(facecolor='white', alpha=0.6))

        if (catmag is not None) & (catsn is not None):
            catlabel = r'${0:s} = {1:.2f} \pm {3:.2f}  (SN={2:.2f})$'.format(
                                                 band+"_{cat}", catmag, catsn,
                caterr)
            ax.text(xx*0.02, yy * 0.03, catlabel, color='black', weight='bold',
         fontsize='large', bbox=dict(facecolor='white', alpha=0.6))


        plt.title(title)

        plt.savefig(output, dpi=100)
        print("Image ", output, " created" )
        plt.close('all')
    else:
        print("Couldn't create png. Object outside the image")


def mag_err(noise_flux_ratio, verbose=True):
    '''
    Calculates the magnitude error from the input noise_flux_ratio
    which is basically the inverse of the Signal-to-Noise ratio
    '''
    err = (2.5 / np.log(10)) * noise_flux_ratio
    if verbose:
        print(err)
    return err




def get_pixelscale(hdr):
    '''
    Get pixelscale from header and return in it in arcsec/pixel
    '''
    if 'CDELT1' in hdr.keys():
        CD1 = hdr['CDELT1']
        CD2 = hdr['CDELT2']
    elif 'CD1_1' in hdr.keys():
        CD1 = hdr['CD1_1']
        CD2 = hdr['CD2_2']
    else:
        print('pixel scale unknown. Using 1 pix/arcsec')
        CD1 = CD2 = 1

    scale = 0.5 * (np.abs(CD1) + np.abs(CD2)) * 3600

    return scale



def get_noiseaper(data, radius):
    print("estimating noise in aperture: ", radius)

    sources_mask = make_source_mask(data, snr=2.5, npixels=3,
                                     dilate_size=15, filter_fwhm=4.5)


    N=5100
    ny, nx = data.shape
    x1= np.int(nx * 0.09)
    x2 = np.int(nx * 0.91)
    y1= np.int(ny * 0.09)
    y2= np.int(ny * 0.91)
    xx = np.random.uniform(x1, x2, N)
    yy = np.random.uniform(y1, y2, N)

    mask = sources_mask[np.int_(yy), np.int_(xx)]
    xx= xx[~mask]
    yy = yy[~mask]

    positions = (xx, yy)
    apertures = CircularAperture(positions, r=radius)
    f = aperture_photometry(data, apertures, mask=sources_mask)
    f = np.ma.masked_invalid(f['aperture_sum'])
    m1 = np.isfinite(f) #& (f!=0)
    empty_fluxes = f[m1]
    emptyapmeanflux, emptyapsigma = gaussian_fit_to_histogram(empty_fluxes)

    return emptyapsigma, emptyapmeanflux, empty_fluxes



def gaussian_fit_to_histogram(dataset):
    """ fit a gaussian function to the histogram of the given dataset
    :param dataset: a series of measurements that is presumed to be normally
       distributed, probably around a mean that is close to zero.
    :return: mean, mu and width, sigma of the gaussian model fit.

    Taken from

    https://github.com/djones1040/PythonPhot/blob/master/PythonPhot/photfunctions.py
    """
    def gauss(x, mu, sigma):
        return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    if np.ndim(dataset) == 2:
        musigma = np.array([gaussian_fit_to_histogram(dataset[:, i])
                            for i in range(np.shape(dataset)[1])])
        return musigma[:, 0], musigma[:, 1]

    dataset = dataset[np.isfinite(dataset)]
    ndatapoints = len(dataset)
    stdmean, stdmedian, stderr, = sigma_clipped_stats(dataset, sigma=5.0)
    nhistbins = max(10, int(ndatapoints / 20))
    histbins = np.linspace(stdmedian - 5 * stderr, stdmedian + 5 * stderr,
                           nhistbins)
    yhist, xhist = np.histogram(dataset, bins=histbins)
    binwidth = np.mean(np.diff(xhist))
    binpeak = float(np.max(yhist))
    param0 = [stdmedian, stderr]  # initial guesses for gaussian mu and sigma
    xval = xhist[:-1] + (binwidth / 2)
    yval = yhist / binpeak
    try:
        minparam, cov = curve_fit(gauss, xval, yval, p0=param0)
    except RuntimeError:
        minparam = -99, -99
    mumin, sigmamin = minparam
    return mumin, sigmamin



