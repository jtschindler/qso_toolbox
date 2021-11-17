#!/usr/bin/env python

import os
import time
import requests
import regex as re
import numpy as np
import pandas as pd

from astropy import wcs
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy import units

try:
  from urllib2 import urlopen #python2
  from httplib import IncompleteRead
  from urllib2 import HTTPError
except ImportError:
  from urllib.request import urlopen #python3
  from urllib.error import HTTPError
  from http.client import IncompleteRead

from qso_toolbox import utils as ut
from qso_toolbox import catalog_tools as ct

def get_tile_dataframe():

    quicklook_summary = urlopen('https://archive-new.nrao.edu/vlass/VLASS_dyn_summary.php')
    # open('quicklook_summary_temp.txt', 'wb').write(quicklook_summary.read())

    lines = quicklook_summary.readlines()

    columns = ['name', 'dec_min', 'dec_max',
               'ra_min', 'ra_max', 'obdata', 'epoch']

    df = pd.DataFrame(columns=columns)

    for line in lines[3:]:
        linedata = line.decode('utf-8').split()
        linedata = np.array([val.strip() for val in linedata])

        if len(linedata) == 9:
            df = df.append({'name': linedata[0],
                           'dec_min': float(linedata[1]),
                           'dec_max': float(linedata[2]),
                           'ra_min': float(linedata[3]),
                           'ra_max': float(linedata[4]),
                           'obsdate': linedata[6],
                           'epoch': linedata[5],
                           }, ignore_index=True)

    df.to_hdf('vlass_quicklook_summary.hdf5', 'data')

    return df



def search_tiles(tiles_df, coord, verbosity=0, mode='recent'):

    ra_hr = coord.ra.hour
    dec_deg = coord.dec.deg

    tile = tiles_df.query('dec_min < {} < dec_max and '
                   'ra_min < {} < ra_max'.format(dec_deg, ra_hr))
    tile = tile.sort_values(by='obsdate')

    if tile.shape[0] > 1:
        if verbosity > 1:
            print('[INFO] Source appears in more than one tile')
        for idx in tile.index:
            if verbosity > 0:
                print('[INFO] tile {}, '
                      'obsdate {}, epoch {}'.format(tile.loc[idx, 'name'],
                                                    tile.loc[idx, 'obsdate'],

                                                    tile.loc[idx, 'epoch']))
        if mode == 'recent':
            tile = tile.loc[[tile.index[-1]]]
            if verbosity > 1:
                print('[INFO] Choosing tile with most recent observation '
                      'date: {}'.format(tile.loc[tile.index[0], 'name']))
        elif mode == 'all':
            if verbosity > 1:
                print('[INFO] Returning all tiles:')
                print('{}'.format(tile.loc[:, 'name']))
        else:
            raise ValueError('[ERROR] Search mode not understood. "mode" can '
                             'be "recent" or "all".')

    elif tile.shape[0] == 0:
        if verbosity > 1:
            print('[INFO] No tile found for this source')
        tile = None

    else:
        if verbosity > 1:
            print('[INFO] One tile found for this source')

    return tile

def get_closest_subtile_url(tile, coord, verbosity=0):

    vlass_urls = []

    for idx in tile.index:

        name = tile.loc[idx, 'name']
        epoch = tile.loc[idx, 'epoch']


        url_base = 'https://archive-new.nrao.edu' \
              '/vlass/quicklook/{:s}/{:s}/'.format(epoch, name)

        html = requests.get(url_base).content
        subtile_df = pd.read_html(html)[-1]
        # Remove the NaN and from the 'Last modified' column
        subtile_df.dropna(subset=['Last modified'], inplace=True)

        designations = [subtile_df.loc[idx, 'Name'].split('.')[4] for idx in
                       subtile_df.index]

        ra_list = []
        dec_list = []
        for designation in designations:

            ra_deg, dec_deg = ut.designation_to_coord(designation)
            ra_list.append(ra_deg)
            dec_list.append(dec_deg)

        subtile_coords = SkyCoord(np.array(ra_list), np.array(dec_list),
                                 frame='icrs', unit='deg')

        dist = coord.separation(subtile_coords)
        subtile_name = subtile_df.loc[subtile_df.index[np.argmin(dist)], 'Name']

        if verbosity > 1:
            print('[INFO] {} is closest subtile'.format(subtile_name))
            print('[INFO] with a source distance of {:.5f} '.format(np.min(dist)))

        url_base = url_base + '/{}/'.format(subtile_name)

        image_name = '{}.I.iter1.image.pbcor.tt0.subim.fits'.format(subtile_name[ 0:-1])

        vlass_urls.append(url_base+image_name)


    return vlass_urls


def search_vlass_quicklook(coord, update_summary=False,
                           mode='recent', verbosity=0):

    if not os.path.isfile('vlass_quicklook_summary.hdf5') or update_summary \
            is True:
        tiles_df = get_tile_dataframe()
    elif update_summary == 'auto':
        now = time.time()
        if now-os.path.getmtime('./vlass_quicklook_summary.hdf5') > 24 * 3600:
            if verbosity > 1:
                print('[INFO] Downloading VLASS quicklook summary table.')
            tiles_df = get_tile_dataframe()
        else:
            tiles_df = pd.read_hdf('vlass_quicklook_summary.hdf5', 'data')
    else:
        tiles_df = pd.read_hdf('vlass_quicklook_summary.hdf5', 'data')

    tile = search_tiles(tiles_df, coord, mode=mode, verbosity=verbosity)


    return tile


def get_quicklook_url(ra, dec,
                      epoch = None,
                      update_summary='auto',
                      mode='all',
                      verbosity=0):

    coord = SkyCoord(ra, dec, unit='deg', frame='icrs')

    tile = search_vlass_quicklook(coord,
                                  update_summary=update_summary,
                                  mode=mode,
                                  verbosity=verbosity)

    if epoch is not None:
        tile = tile.query('epoch == "{}"'.format(epoch))

    if tile is not None:
        vlass_url = get_closest_subtile_url(tile, coord, verbosity=verbosity)

        return vlass_url
    else:
        return None


def make_vlass_cutout(ra_deg, dec_deg, fov, raw_image_name, image_name,
                      image_folder_path='.', verbosity=0):

    if verbosity > 1:
        print('[INFO] Generate VLASS cutout centered on '
              '{:.2f} {:.2f}'.format(ra_deg, dec_deg))

    filename = image_folder_path + '/' + raw_image_name + '.fits'

    data, hdr = fits.getdata(filename, header=True)
    wcs_img = wcs.WCS(hdr)

    pixcrd = wcs_img.wcs_world2pix([[ra_deg, dec_deg, 0, 0]], 0)

    positions = (np.float(pixcrd[0, 0]), np.float(pixcrd[0, 1]))
    overlap = True

    if verbosity >= 4:
        print("[DIAGNOSTIC] Image file shape {}".format(data.shape))
    try:
        wcs_img = wcs_img.dropaxis(2)
        wcs_img = wcs_img.dropaxis(2)

        img_stamp = Cutout2D(data[0,0], positions, size=fov * units.arcsec,
                             wcs=wcs_img)

        if verbosity >= 4:
            print("[DIAGNOSTIC] Cutout2D file shape {}".format(
                img_stamp.shape))

    except:
        print("[WARNING] Cutout could not be generated.")
        overlap = False
        img_stamp = None

    if img_stamp is not None:

        new_hdr = img_stamp.wcs.to_header()
        new_hdr['BPA'] = hdr['BPA']
        new_hdr['BMIN'] = hdr['BMIN']
        new_hdr['BMAJ'] = hdr['BMAJ']

        filename = image_folder_path + '/' + image_name + '.fits'

        if verbosity > 1:
            print("[INFO] Cutout with a FOV of {} generated.".format(fov))
            print("[INFO] Cutout saved as {}".format(filename))
        if overlap:
            img_stamp = img_stamp.data

        empty_primary = fits.PrimaryHDU(header=new_hdr)
        imhdu = fits.ImageHDU(data=img_stamp, header=new_hdr)
        hdul = fits.HDUList([empty_primary, imhdu])

        hdul.writeto(filename)



def download_vlass_images(ra, dec, fov, image_folder_path,
                          update_summary='auto', verbosity=2,
                          outputfile_basename=None):
    """

    :param ra: R.A. of source in decimal degrees
    :param dec: Decl. of source in decimal degrees
    :param fov: Field of view in arcseconds
    :param image_folder_path: Name of the path where downloaded images are
    stored
    :param update_summary: "False" only read summary file, "True" always
    download and read summary file, "auto" donwload summary file if older
    than 24 hours otherwise just read.
    :param verbosity: Verbose output
    :return: None
    """

    ra_deg = np.array([ra])
    dec_deg = np.array([dec])
    coord = SkyCoord(ra, dec, unit='deg', frame='icrs')

    tile = search_vlass_quicklook(coord,
                                  update_summary=update_summary,
                                  mode='all',
                                  verbosity=0)

    survey = 'vlass'
    band = '3GHz'

    for tdx in tile.index:
        temp_object_name = ut.coord_to_name(ra_deg,
                           dec_deg,
                           epoch="J")

        epoch = tile.loc[tdx, 'epoch']

        if outputfile_basename is None:
            outputfile_basename = temp_object_name[0]

        vlass_img_name = outputfile_basename + "_" + survey + \
                         "_" + band + '_{}'.format(epoch) + "_fov" + \
                         '{:d}'.format(fov)

        # Introduce different naming for raw file, possibly use default name
        # (tilename, subtilename, etc.)
        raw_img_name = temp_object_name[0] + "_" + survey + "_" + \
                   band + '_{}'.format(epoch) + '_raw'


        file_path = image_folder_path + '/' + raw_img_name + '.fits'
        raw_file_exists = os.path.isfile(file_path)


        file_path = image_folder_path + '/' + vlass_img_name + '.fits'
        file_exists = os.path.isfile(file_path)

        if raw_file_exists is False:

            if verbosity > 1:
                print('[INFO] Downloading raw VLASS images.')

            url = get_quicklook_url(ra, dec,
                  epoch=epoch,
                  update_summary=update_summary,
                  verbosity=verbosity)

            ct.download_image(url[0], raw_img_name, image_folder_path,
                              verbosity=verbosity)

        if file_exists is False:

            if verbosity > 1:
                print('[INFO] Generating VLASS cutout image.')

            make_vlass_cutout(ra, dec, fov, raw_img_name,
                              vlass_img_name,
                              image_folder_path=
                              image_folder_path,
                              verbosity=verbosity)

        else:
            if verbosity > 1:
                print('[INFO] File already exists')


# print(ut.coord_to_name([30.2], [-20.2]))


# def get_cutout(imname, name, c, epoch):
#     print("Generating cutout")
#     # Position of source
#     ra_deg = c.ra.deg
#     dec_deg = c.dec.deg
#
#     print("Cutout centered at position %s,%s" % (ra_deg, dec_deg))
#
#     # Open image and establish coordinate system
#     hdu = pyfits.open(imname)
#     im = hdu[0].data[0, 0]
#     hdr = hdu[0].header
#     # im = pyfits.open(imname)[0].data[0,0]
#     # w = WCS(imname)
#     w = WCS(hdr)
#
#     # Find the source position in pixels.
#     # This will be the center of our image.
#     src_pix = w.wcs_world2pix([[ra_deg, dec_deg, 0, 0]], 0)
#     x = src_pix[0, 0]
#     y = src_pix[0, 1]
#
#     # Check if the source is actually in the image
#     # pix1 = pyfits.open(imname)[0].header['CRPIX1']
#     # pix2 = pyfits.open(imname)[0].header['CRPIX2']
#     pix1 = hdr['CRPIX1']
#     pix2 = hdr['CRPIX2']
#     badx = np.logical_or(x < 0, x > 2 * pix1)
#     bady = np.logical_or(y < 0, y > 2 * pix2)
#     if np.logical_or(badx, bady):
#         print("Tile has not been imaged at the position of the source")
#         return None
#
#     else:
#         # Set the dimensions of the image
#         # Say we want it to be 30 arcseconds on a side,
#         # to match the DES images
#         delt1 = hdr['CDELT1']
#         delt2 = hdr['CDELT2']
#         cutout_size = 30. / 3600  # in degrees
#         dside1 = -cutout_size / 2. / delt1
#         dside2 = cutout_size / 2. / delt2
#
#         vmin = -1e-4
#         vmax = 1e-3
#
#         im_plot_raw = im[int(y - dside1):int(y + dside1),
#                       int(x - dside2):int(x + dside2)]
#         im_plot = np.ma.masked_invalid(im_plot_raw)
#
#         # 3-sigma clipping
#         rms_temp = np.ma.std(im_plot)
#         keep = np.ma.abs(im_plot) <= 3 * rms_temp
#         rms = np.ma.std(im_plot[keep])
#
#         # peak_flux = np.ma.max(im.flatten())
#         peak_flux = np.ma.max(im_plot.flatten())
#
#         plt.imshow(
#             np.flipud(im_plot),
#             extent=[-0.5 * cutout_size * 3600., 0.5 * cutout_size * 3600.,
#                     -0.5 * cutout_size * 3600., 0.5 * cutout_size * 3600],
#             vmin=vmin, vmax=vmax, cmap='YlOrRd')
#
#         peakstr = "Peak Flux %s mJy" % (np.round(peak_flux * 1e3, 3))
#         rmsstr = "RMS Flux %s mJy" % (np.round(rms * 1e3, 3))
#         plt.title(name + ": %s; %s" % (peakstr, rmsstr))
#         plt.xlabel("Offset in RA (arcsec)")
#         plt.ylabel("Offset in Dec (arcsec)")
#
#         # pyfits.writeto(name + '_' + epoch + ".fits", im_plot_raw, overwrite=True)
#         filename = name + '_' + epoch
#         plt.savefig(filename + ".png")
#         print(filename + ".png", " created")
#         save_fits_cutout(im, c, 4 * u.arcmin, w, hdr, filename + '.fits')
#         # print(name + '_' + epoch + ".fits", " created")
#         return peak_flux, rms




