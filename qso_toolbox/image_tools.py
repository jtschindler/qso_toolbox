#!/usr/bin/env python

import os
import astropy.units as u
from astropy.utils.console import ProgressBar
from astropy.io import fits, ascii
from astropy.table import Table, Column
from astropy import wcs
from astropy.stats import sigma_clipped_stats
from astropy.nddata.utils import Cutout2D
from astropy.io import ascii, fits

import pandas as pd

from scipy.optimize import curve_fit

from photutils import aperture_photometry, CircularAperture
from photutils import Background2D, MedianBackground, make_source_mask

import itertools
import multiprocessing as mp

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import glob

from qso_toolbox import utils as ut
from qso_toolbox import catalog_tools as ct

import math


# ------------------------------------------------------------------------------
#  Plotting functions for image_cutouts
# ------------------------------------------------------------------------------


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
        if (forced_mag is not None):
            if (forced_sn is not None) & (forced_magerr is not None):
                forcedlabel = r'${0:s} = {1:.2f} \pm {2:.2f} (SN=' \
                              r'{3:.1f})$'.format(band + "_{forced}",
                                                  forced_mag,
                                                  forced_magerr,
                                                  forced_sn)
            elif forced_magerr is not None:
                forcedlabel = r'${0:s} = {1:.2f} \pm {2:.2f}$'.format(
                    band + "_{forced}", forced_mag, forced_magerr)
            else:
                forcedlabel = r'${0:s} = {1:.2f}$'.format(
                    band + "_{forced}", forced_mag)

            ax.text(xx * 0.01, yy * 0.14, forcedlabel, color='black',
                     weight='bold', fontsize='large',
                     bbox=dict(facecolor='white', alpha=0.6))

        # Create catalog magnitude label
        if catmag is not None:
            if (catsn is not None) & (caterr is not None):
                catlabel = r'${0:s} = {1:.2f} \pm {2:.2f}  (SN=' \
                           r'{3:.2f})$'.format(
                    band + "_{cat}", catmag, caterr, catsn)
            elif caterr is not None:
                catlabel = r'${0:s} = {1:.2f} \pm {2:.2f}$'.format(
                    band + "_{cat}", catmag, caterr)
            else:
                catlabel = r'${0:s} = {1:.2f}$'.format(
                    band + "_{cat}", catmag)

            ax.text(xx * 0.02, yy * 0.03, catlabel, color='black',
                     weight='bold',
                     fontsize='large',
                     bbox=dict(facecolor='white', alpha=0.6))

        if title is not None:
            plt.title(title)

        plt.savefig(output, dpi=100)
        print("Image ", output, " created")
        plt.close('all')
    else:
        print("Couldn't create png. Object outside the image")




def make_mult_png_fig(ra, dec, surveys, bands,
                  fovs, apertures, square_sizes, image_path, mag_list=None,
                  magerr_list=None, sn_list=None,
                  forced_mag_list=None, forced_magerr_list=None,
                  forced_sn_list=None, n_col=3,
                  n_sigma=3, color_map_name='viridis', verbosity=0):
    """Create figure to plot cutouts for one source in all specified surveys
    and bands.

    :param ra:
    :param dec:
    :param surveys:
    :param bands:
    :param fovs:
    :param apertures:
    :param square_sizes:
    :param image_path:
    :param mag_list:
    :param magerr_list:
    :param sn_list:
    :param forced_mag_list:
    :param forced_magerr_list:
    :param forced_sn_list:
    :param n_col:
    :param n_sigma:
    :param color_map_name:
    :param verbosity:
    :return:
    """


    n_images = len(surveys)

    n_row = int(math.ceil(n_images / n_col))

    fig = plt.figure(figsize=(5*n_col, 5*n_row))

    fig = _make_mult_png_axes(fig, n_row, n_col, ra, dec, surveys, bands,
                  fovs, apertures, square_sizes, image_path, mag_list,
                  magerr_list, sn_list,
                  forced_mag_list, forced_magerr_list,
                  forced_sn_list, n_sigma, color_map_name, verbosity)

    coord_name = ut.coord_to_name(np.array([ra]),
                                  np.array([dec]),
                                  epoch="J")

    fig.suptitle(coord_name[0])

    return fig


def _make_mult_png_axes(fig, n_row, n_col, ra, dec, surveys, bands,
                  fovs, apertures, square_sizes, image_path, mag_list=None,
                  magerr_list=None, sn_list=None,
                  forced_mag_list=None, forced_magerr_list=None,
                  forced_sn_list=None,
                  n_sigma=3, color_map_name='viridis', verbosity=0):
    """ Create axes components to plot one source in all specified surveys
    and bands.

    :param fig:
    :param n_row:
    :param n_col:
    :param ra:
    :param dec:
    :param surveys:
    :param bands:
    :param fovs:
    :param apertures:
    :param square_sizes:
    :param image_path:
    :param mag_list:
    :param magerr_list:
    :param sn_list:
    :param forced_mag_list:
    :param forced_magerr_list:
    :param forced_sn_list:
    :param n_sigma:
    :param color_map_name:
    :param verbosity:
    :return:
    """

    for idx, survey in enumerate(surveys):
        band = bands[idx]
        fov = fovs[idx]
        aperture = apertures[idx]
        size = square_sizes[idx]

        if mag_list is not None:
            catmag = mag_list[idx]
        else:
            catmag = None
        if magerr_list is not None:
            caterr = magerr_list[idx]
        else:
            caterr = None
        if sn_list is not None:
            catsn = sn_list[idx]
        else:
            catsn = None
        if forced_mag_list is not None:
            forced_mag = forced_mag_list[idx]
        else:
            forced_mag = None
        if forced_magerr_list is not None:
            forced_magerr = forced_magerr_list[idx]
        else:
            forced_magerr = None
        if forced_sn_list is not None:
            forced_sn = forced_sn_list[idx]
        else:
            forced_sn = None

        # Get the correct filename, accept larger fovs
        if survey == 'vhsdr6':
            coord_name = ut.coord_to_vhsname(np.array([ra]), np.array([dec]),
                                             epoch="J")
        else:
            coord_name = ut.coord_to_name(np.array([ra]), np.array([dec]),
                                          epoch="J")

        filename = image_path + '/' + coord_name[0] + "_" + survey + "_" + \
                   band + "*.fits"
        filenames_available = glob.glob(filename)

        file_found = False
        for filename in filenames_available:

            try:
                file_fov = int(filename.split("_")[3].split(".")[0][3:])
            except:
                file_fov = 0

            if fov <= file_fov:
                data, hdr = fits.getdata(filename, header=True)
                file_found = True

        if file_found:
            if verbosity > 0:
                print("Opened {} with a fov of {} "
                      "arcseconds".format(filename, file_fov))
        else:
            if verbosity > 0:
                print("File {} in folder {} not found. Target with RA {}"
                      " and Decl {}".format(
                    filename, image_path, ra, dec))

            # Old plotting routine to modify, currently it only plots images for
            # surveys and bands that it could open, no auto download implemented
        if file_found:
            wcs_img = wcs.WCS(hdr)
            axs = fig.add_subplot(n_row, n_col, idx + 1, projection=wcs_img)

            pixcrd = wcs_img.wcs_world2pix(ra, dec, 0)
            positions = (np.float(pixcrd[0]), np.float(pixcrd[1]))
            overlap = True
            try:
                img_stamp = Cutout2D(data, positions, size=fov * u.arcsec,
                                     wcs=wcs_img)
            except:
                print("Source not in image")
                overlap = False

            if overlap:
                img_stamp = img_stamp.data
                (x, y) = img_stamp.shape
            try:
                cm = plt.get_cmap(color_map_name)
            except ValueError:
                print('Color map argument is not a color map. Setting '
                      'default: viridis')
                cm = plt.get_cmap('viridis')

            img = axs.imshow(img_stamp, origin='lower', cmap=cm)

            # Sigma-clipping of the color scale
            mean = np.mean(img_stamp[~np.isnan(img_stamp)])
            std = np.std(img_stamp[~np.isnan(img_stamp)])
            upp_lim = mean + n_sigma * std
            low_lim = mean - n_sigma * std
            img.set_clim(low_lim, upp_lim)

            # Adjusting axis directions
            if survey == 'vhsdr6':
                axs.invert_xaxis()
                axs.invert_yaxis()
            elif survey.split("-")[0] == "unwise":
                axs.invert_xaxis()


            # Plot circular aperture (forced photometry flux)
            (yy, xx) = img_stamp.shape
            circx = (xx * 0.5)  # + 1
            circy = (yy * 0.5)  # + 1
            aper_pix = aperture_inpixels(aperture, hdr)

            circle = plt.Circle((circx, circy), aper_pix, color='y', fill=False,
                                lw=1.5)
            fig.gca().add_artist(circle)

            # Plot rectangular aperture (error region)
            twenty_arcsec = aperture_inpixels(size, hdr)
            square = plt.Rectangle((circx - twenty_arcsec * 0.5,
                                    circy - twenty_arcsec * 0.5),
                                   twenty_arcsec, twenty_arcsec,
                                   color='y', fill=False, lw=1.5)
            fig.gca().add_artist(square)

            # Create forced photometry label
            if (forced_mag is not None):
                if (forced_sn is not None) & (forced_magerr is not None):
                    forcedlabel = r'${0:s} = {1:.2f} \pm {2:.2f} (SN=' \
                                  r'{3:.1f})$'.format(band + "_{forced}",
                                                      forced_mag,
                                                      forced_magerr,
                                                      forced_sn)
                elif forced_magerr is not None:
                    forcedlabel = r'${0:s} = {1:.2f} \pm {2:.2f}$'.format(
                        band + "_{forced}", forced_mag, forced_magerr)
                else:
                    forcedlabel = r'${0:s} = {1:.2f}$'.format(
                        band + "_{forced}", forced_mag)

                axs.text(0.03, 0.16, forcedlabel, color='black',
                         weight='bold', fontsize='large',
                         bbox=dict(facecolor='white', alpha=0.6),
                         transform=axs.transAxes)

            # Create catalog magnitude label
            if catmag is not None:
                if (catsn is not None) & (caterr is not None):
                    maglabel = r'${0:s} = {1:.2f} \pm {2:.2f}  (SN=' \
                               r'{3:.2f})$'.format(
                        band + "_{cat}", catmag, caterr, catsn)
                elif caterr is not None:
                    maglabel = r'${0:s} = {1:.2f} \pm {2:.2f}$'.format(
                        band + "_{cat}", catmag, caterr)
                else:
                    maglabel = r'${0:s} = {1:.2f}$'.format(
                        band + "_{cat}", catmag)

                axs.text(0.03, 0.04, maglabel, color='black',
                         weight='bold',
                         fontsize='large',
                         bbox=dict(facecolor='white', alpha=0.6),
                         transform=axs.transAxes)

            axs.set_title(survey + " " + band)

    return fig


# ------------------------------------------------------------------------------
#  Determine forced photometry for sources in cutouts.
# ------------------------------------------------------------------------------


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
    table, format = ct.check_if_table_is_pandas_dataframe(table)
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

            ra = table.loc[idx, ra_col_name]
            dec = table.loc[idx, dec_col_name]

            img_name = table.temp_object_name[idx]+"_"+survey+"_" + \
                       band+"_fov"+'{:d}'.format(fov)

            # Check if file is in folder
            file_path = image_path + '/' + img_name + '.fits'
            file_exists = os.path.isfile(file_path)

            if file_exists is not True and auto_download is True:

                if survey == "desdr1":
                    url = ct.get_desdr1_deepest_image_url(ra,
                                                       dec,
                                                       fov=fov,
                                                       band=band,
                                                       verbosity=verbosity)
                elif survey.split("-")[0] == "unwise" and band in ["w1",
                                                                   "w2",
                                                                   "w3",
                                                                   "w4"]:
                    # Hack to create npix from fov approximately
                    npix = int(round(fov / 60. / 4. * 100))

                    data_release = survey.split("-")[1]
                    wband = band[1]

                    url = ct.get_unwise_image_url(ra, dec, npix, wband,
                                                   data_release)

                else:
                    raise ValueError(
                        "Survey and band name not recognized: {} {}. "
                        "\n "
                        "Possible survey names include: desdr1, "
                        "unwise-allwise, unwise-neo1, unwise-neo2, "
                        "unwise-neo3".format(survey, band))

                if url is not None:
                    ct.download_image(url, image_name=img_name,
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

                mag, flux, sn, err, comment = \
                    calculate_forced_aperture_photometry(file_path,
                                                         ra, dec, survey,
                                                         aperture,
                                                         verbosity=verbosity)
                table.loc[idx, 'forced_{}_mag_{}'.format(survey, band)] = mag
                table.loc[idx, 'forced_{}_flux_{}'.format(survey, band)] = flux
                table.loc[idx, 'forced_{}_sn_{}'.format(survey, band)] = sn
                table.loc[idx, 'forced_{}_magerr_{}'.format(survey, band)] = \
                    err
                table.loc[idx, 'forced_{}_{}_comment'.format(survey, band)] =\
                    comment

            if file_exists is True and file_size_sufficient is not True:

                table.loc[idx, 'forced_{}_{}_comment'.format(survey, band)] = \
                    'image_too_small'.format(aperture)

            if file_exists is not True:

                table.loc[idx, 'forced_{}_{}_comment'.format(survey, band)] = \
                    'image_not_available'.format(aperture)

    table.drop(columns='temp_object_name')

    table = ct.convert_table_to_format(table, format)

    return table


def get_forced_photometry_mp(table, ra_col_name, dec_col_name, surveys,
                          bands, apertures, fovs, image_path, n_jobs=8,
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
    :param n_jobs:
    :param auto_download:
    :param verbosity:
    :return:
    """

    # Check if table is pandas DataFrame otherwise convert to one
    table, format = ct.check_if_table_is_pandas_dataframe(table)
    # Add a column to the table specifying the object name used
    # for the image name
    table['temp_object_name'] = ut.coord_to_name(table[ra_col_name].values,
                                                 table[dec_col_name].values,
                                                 epoch="J")

    for jdx, survey in enumerate(surveys):
        band = bands[jdx]
        aperture = apertures[jdx]
        fov = fovs[jdx]

        # Create list with image names
        ra = table[ra_col_name].values
        dec = table[dec_col_name].values
        index = table.index

        img_names = table.temp_object_name + "_" + survey + "_" + \
                    band + "_fov" + '{:d}'.format(fov)

        mp_args = list(zip(index,
                           ra,
                           dec,
                           itertools.repeat(survey),
                           itertools.repeat(band),
                           itertools.repeat(aperture),
                           itertools.repeat(fov),
                           itertools.repeat(image_path),
                           img_names,
                           itertools.repeat(auto_download),
                           itertools.repeat(verbosity)))

        # Start multiprocessing pool
        with mp.Pool(n_jobs) as pool:
            results = pool.starmap(_mp_get_forced_photometry, mp_args)



        for result in results:
            idx, mag, flux, sn, err, comment = result
            table.loc[idx, 'forced_{}_mag_{}'.format(survey, band)] = mag
            table.loc[idx, 'forced_{}_flux_{}'.format(survey, band)] = flux
            table.loc[idx, 'forced_{}_sn_{}'.format(survey, band)] = sn
            table.loc[idx, 'forced_{}_magerr_{}'.format(survey, band)] = \
                err
            table.loc[idx, 'forced_{}_{}_comment'.format(survey, band)] = \
                comment

    table.drop(columns='temp_object_name')

    table = ct.convert_table_to_format(table, format)

    return table


def _mp_get_forced_photometry(index, ra, dec, survey,
                          band, aperture, fov, image_path, img_name,
                          auto_download=True,
                          verbosity=0):
    """

    :param index:
    :param ra:
    :param dec:
    :param survey:
    :param band:
    :param aperture:
    :param fov:
    :param image_path:
    :param img_name:
    :param auto_download:
    :param verbosity:
    :return:
    """

    # Check if file is in folder
    file_path = image_path + '/' + img_name + '.fits'
    file_exists = os.path.isfile(file_path)

    if file_exists is not True and auto_download is True:

        if survey == "desdr1":
            url = ct.get_desdr1_deepest_image_url(ra,
                                               dec,
                                               fov=fov,
                                               band=band,
                                               verbosity=verbosity)

        else:
            raise ValueError("Survey name not recognized: {} . \n "
                             "Possible survey names include: desdr1".format(
                survey))

        if url is not None:
            ct.download_image(url, image_name=img_name,
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
        mag, flux, sn, err, comment = \
            calculate_forced_aperture_photometry(file_path,
                                                 ra, dec, survey,
                                                 aperture,
                                                 verbosity=verbosity)

        return index, mag, flux, sn, err, comment

    if file_exists is True and file_size_sufficient is not True:
        comment = 'image_too_small'.format(aperture)

        return index, np.nan, np.nan, np.nan, np.nan, comment

    if file_exists is not True:
        comment = 'image_not_available'.format(aperture)

        return index, np.nan, np.nan, np.nan, np.nan, comment


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

    # QUICKFIX to stop aperture photometry from crashing
    try:
        # Get photometry
        positions = (pixel_coordinate[0], pixel_coordinate[1])
        apertures = CircularAperture(positions, r=aperture_pixel)
        f = aperture_photometry(data, apertures)
        flux = np.ma.masked_invalid(f['aperture_sum'])

        # Get the noise
        rmsimg, mean_noise, empty_flux = get_noiseaper(data, aperture_pixel)

        sn = flux[0] / rmsimg

        comment = 'ap_{}'.format(aperture)

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
            comment = 'masked'
        if sn is np.ma.masked:
            sn = np.nan
        if err is np.ma.masked:
            err = np.nan
        if flux[0] is np.ma.masked:
            flux = np.nan
        else:
            flux = flux[0]

        return mags, flux, sn, err, comment

    except ValueError:
        return -999, np.nan, np.nan, np.nan, 'crashed'


# ------------------------------------------------------------------------------
#  Image utility functions for forced photometry
#  (mostly from Eduardo and not modified)
# ------------------------------------------------------------------------------


def check_image_size(image_name, file_path, verbosity):
    """

    :param image_name:
    :param file_path:
    :param verbosity:
    :return:
    """

    shape = fits.getdata(file_path).shape
    min_axis = np.min(shape)

    if min_axis < 50 and verbosity > 0:
        print("Minimum image dimension : {} (pixels)".format(min_axis))
        print("Too few pixels in one axis (<50). Skipping {}".format(
            image_name))

    if min_axis < 50:
        return False
    else:
        return True


def aperture_inpixels(aperture, hdr):
    '''
    receives aperture in arcsec. Returns aperture in pixels
    '''
    pixelscale=get_pixelscale(hdr)
    aperture /= pixelscale #pixels

    return aperture


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


def mag_err(noise_flux_ratio, verbose=True):
    '''
    Calculates the magnitude error from the input noise_flux_ratio
    which is basically the inverse of the Signal-to-Noise ratio
    '''
    err = (2.5 / np.log(10)) * noise_flux_ratio
    if verbose:
        print(err)
    return err


def get_noiseaper(data, radius):
    # print("estimating noise in aperture: ", radius)

    sources_mask = make_source_mask(data, snr=2.5, npixels=3,
                                     dilate_size=15, filter_fwhm=4.5)


    N=5100
    ny, nx = data.shape
    x1 = np.int(nx * 0.09)
    x2 = np.int(nx * 0.91)
    y1 = np.int(ny * 0.09)
    y2 = np.int(ny * 0.91)
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


def flux_to_magnitude(flux, survey):
    """

    :param flux:
    :param survey:
    :return:
    """
    if survey == "desdr1":
        zpt = 30.
    elif survey.split("-")[0] == "unwise":
        zpt = 22.5
    else:
        raise ValueError("Survey name not recgonized: {}".format(survey))

    return -2.5 * np.log10(flux) + zpt


def nmgy2abmag(flux, flux_ivar=None):
    """
    Conversion from nanomaggies to AB mag as used in the DECALS survey
    flux_ivar= Inverse variance oF DECAM_FLUX (1/nanomaggies^2)
    """
    lenf = len(flux)
    if lenf > 1:
        ii = np.where(flux>0)
        mag = 99.99 + np.zeros_like(flux)
        mag[ii] = 22.5 - 2.5*np.log10(flux[ii])
    else:
        mag = 22.5 - 2.5*np.log10(flux)

    if flux_ivar is None:
        return mag
    elif lenf>1:
        err = np.zeros_like(mag)
        df = np.sqrt(1./flux_ivar)
        err[ii] = mag_err(df[ii]/flux[ii], verbose=False)
    else:
        df = np.sqrt(1./flux_ivar)
        err = mag_err(df/flux, verbose=False)

    return mag, err



