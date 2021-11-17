#!/usr/bin/env python
import astropy.units as u
from astropy.utils.console import ProgressBar
from astropy.io import fits, ascii
from astropy.table import Table, Column
from astropy import wcs
from astropy.stats import sigma_clipped_stats
from astropy.nddata.utils import Cutout2D
from astropy.io import ascii, fits

import pandas as pd

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import glob

from qso_toolbox import utils as ut

import math


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


        if file_found == True:
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

            # fig.colorbar(simg, cax=axs, orientation='vertical')

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


def zscale(zscaleimg):

    z1 = np.amin(zscaleimg)
    z2 = np.amax(zscaleimg)

    return z1, z2

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




