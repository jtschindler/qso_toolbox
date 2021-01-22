#!/usr/bin/env python
from __future__ import print_function, division

import math
import glob
import aplpy
import numpy as np
import itertools
import multiprocessing as mp


from astropy.wcs.utils import proj_plane_pixel_scales
from astropy import wcs
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from astropy.nddata.utils import Cutout2D
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from scipy.optimize import curve_fit

from photutils import aperture_photometry, CircularAperture
from photutils import Background2D, MedianBackground, make_source_mask

from qso_toolbox import utils as ut
from qso_toolbox import catalog_tools as ct
from qso_toolbox import photometry_tools as pt


def show_rectangles(fig, xw, yw, width, height, angle=0, layer=False,
                    zorder=None, coords_frame='world', **kwargs):
    """
    Overlay rectangles on the current plot.

    ATTENTION! THIS IS A MODIFIED VERSION OF THE ORIGINAL APLPY ROUTINE THAT
    CORRECTLY ROTATES THE RECTANGLE AROUND ITS CENTER POSITION.
    see https://github.com/aplpy/aplpy/pull/327

    Parameters
    ----------
    xw : list or `~numpy.ndarray`
        The x positions of the centers of the rectangles (in world coordinates)
    yw : list or `~numpy.ndarray`
        The y positions of the centers of the rectangles (in world coordinates)
    width : int or float or list or `~numpy.ndarray`
        The width of the rectangle (in world coordinates)
    height : int or float or list or `~numpy.ndarray`
        The height of the rectangle (in world coordinates)
    angle : int or float or list or `~numpy.ndarray`, optional
        rotation in degrees (anti-clockwise). Default
        angle is 0.0.
    layer : str, optional
        The name of the rectangle layer. This is useful for giving
        custom names to layers (instead of rectangle_set_n) and for
        replacing existing layers.
    coords_frame : 'pixel' or 'world'
        The reference frame in which the coordinates are defined. This is
        used to interpret the values of ``xw``, ``yw``, ``width``, and
        ``height``.
    kwargs
        Additional keyword arguments (such as facecolor, edgecolor, alpha,
        or linewidth) are passed to Matplotlib
        :class:`~matplotlib.collections.PatchCollection` class, and can be
        used to control the appearance of the rectangles.
    """

    xw, yw, width, height, angle = aplpy.core.uniformize_1d(xw, yw, width,
                                                      height, angle)

    if 'facecolor' not in kwargs:
        kwargs.setdefault('facecolor', 'none')

    if layer:
        fig.remove_layer(layer, raise_exception=False)

    if coords_frame not in ['pixel', 'world']:
        raise ValueError("coords_frame should be set to 'pixel' or 'world'")

    # While we could plot the shape using the get_transform('world') mode
    # from WCSAxes, the issue is that the rotation angle is also measured in
    # world coordinates so will not be what the user is expecting. So we
    # allow the user to specify the reference frame for the coordinates and
    # for the rotation.

    if coords_frame == 'pixel':
        x, y = xw, yw
        w = width
        h = height
        a = angle
        transform = fig.ax.transData
    else:
        x, y = fig.world2pixel(xw, yw)
        pix_scale = aplpy.core.proj_plane_pixel_scales(fig._wcs)
        sx, sy = pix_scale[fig.x], pix_scale[fig.y]
        w = width / sx
        h = height / sy
        a = angle
        transform = fig.ax.transData

    # x = x - w / 2.
    # y = y - h / 2.
    #
    # patches = []
    # for i in range(len(x)):
    #     patches.append(Rectangle((x[i], y[i]), width=w[i], height=h[i],
    #                               angle=a[i]))

    xp = x - w / 2.
    yp = y - h / 2.
    radeg = np.pi / 180
    xr = (xp - x) * np.cos((angle) * radeg) - (yp - y) * np.sin(
        (angle) * radeg) + x
    yr = (xp - x) * np.sin((angle) * radeg) + (yp - y) * np.cos(
        (angle) * radeg) + y

    patches = []
    for i in range(len(xr)):
        patches.append(
            Rectangle((xr[i], yr[i]), width=w[i], height=h[i], angle=a[i]))

    # Due to bugs in matplotlib, we need to pass the patch properties
    # directly to the PatchCollection rather than use match_original.
    p = PatchCollection(patches, transform=transform, **kwargs)

    if zorder is not None:
        p.zorder = zorder
    c = fig.ax.add_collection(p)

    if layer:
        rectangle_set_name = layer
    else:
        fig._rectangle_counter += 1
        rectangle_set_name = 'rectangle_set_' + str(fig._rectangle_counter)

    fig._layers[rectangle_set_name] = c

    return fig

# ------------------------------------------------------------------------------
#  Plotting functions for image_cutouts
# ------------------------------------------------------------------------------

def open_image(filename, ra, dec, fov, image_folder_path, verbosity=0):

    """Opens an image defined by the filename with a fov of at least the
    specified size (in arcseonds).

    :param filename:
    :param ra:
    :param dec:
    :param fov:
    :param image_folder_path:
    :param verbosity:
    :return:
    """

    filenames_available = glob.glob(filename)

    file_found = False
    open_file_fov = None
    file_path = None
    if len(filenames_available) > 0:
        for filename in filenames_available:

            try:
                file_fov = int(filename.split("_")[3].split(".")[0][3:])
            except:
                file_fov = 9999999

            if fov <= file_fov:
                data, hdr = fits.getdata(filename, header=True)
                file_found = True
                file_path =filename
                open_file_fov = file_fov

    if file_found:
        if verbosity > 0:
            print("Opened {} with a fov of {} "
                  "arcseconds".format(file_path, open_file_fov))

        return data, hdr, file_path

    else:
        if verbosity > 0:
            print("File {} in folder {} not found. Target with RA {}"
                  " and Decl {}".format(filename, image_folder_path,
                                        ra, dec))
        return None, None, None


def make_mult_png_fig(ra, dec, surveys, bands,
                  fovs, apertures, square_sizes, image_folder_path, mag_list=None,
                  magerr_list=None, sn_list=None,
                  forced_mag_list=None, forced_magerr_list=None,
                  forced_sn_list=None, n_col=3,
                  n_sigma=3, color_map_name='viridis',
                  add_info_label=None, add_info_value=None, verbosity=0):
    """Create a figure to plot cutouts for one source in all specified surveys
    and bands.

    :param ra: float
        Right Ascension of the target
    :param dec: float
        Declination of the target
     :param surveys: list of strings
        List of survey names, length has to be equal to bands and fovs
    :param bands: list of strings
        List of band names, length has to be equal to surveys and fovs
    :param fovs: list of floats
        Field of view in arcseconds of image cutouts, length has be equal to
        surveys, bands and apertures.
    :param apertures: list of floats
        List of apertures in arcseconds for forced photometry calculated,
        length has to be equal to surveys, bands and fovs
    :param square_sizes: list of floats
        List of
    :param image_folder_path: string
        Path to the directory where all the images are be stored
    :param mag_list: list of floats
        List of magnitudes for each survey/band
    :param magerr_list: list of floats
         List of magnitude errors for each survey/band
    :param sn_list: list of floats
         List of S/N for each survey/band
    :param forced_mag_list: list of floats
         List of forced magnitudes for each survey/band
    :param forced_magerr_list: list of floats
        List of forced magnitude errors for each survey/band
    :param forced_sn_list: list of floats
        List of forced S/N for each survey/band
    :param n_col: int
        Number of columns
    :param n_sigma: int
        Number of sigmas for the sigma-clipping routine that creates the
        boundaries for the color map.
    :param color_map_name: string
        Name of the color map
    :param add_info_value : string
        Value for additional information added to the title of the figure
    :param add_info_label : string
        Label for additional information added to the title of the figure
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution
    :return: matplotlib.figure
        Figure with the plot.
    """

    n_images = len(surveys)

    n_row = int(math.ceil(n_images / n_col))

    fig = plt.figure(figsize=(5*n_col, 5*n_row))

    fig = _make_mult_png_axes(fig, n_row, n_col, ra, dec, surveys, bands,
                  fovs, apertures, square_sizes, image_folder_path, mag_list,
                  magerr_list, sn_list,
                  forced_mag_list, forced_magerr_list,
                  forced_sn_list, n_sigma, color_map_name, verbosity)

    coord_name = ut.coord_to_name(np.array([ra]),
                                  np.array([dec]),
                                  epoch="J")
    if add_info_label is None or add_info_value is None:
        fig.suptitle(coord_name[0])
    else:
        fig.suptitle(coord_name[0]+' '+add_info_label+'='+add_info_value)

    return fig


def _make_mult_png_axes(fig, n_row, n_col, ra, dec, surveys, bands,
                  fovs, apertures, square_sizes, image_folder_path, mag_list=None,
                  magerr_list=None, sn_list=None,
                  forced_mag_list=None, forced_magerr_list=None,
                  forced_sn_list=None,
                  n_sigma=3, color_map_name='viridis', verbosity=0):
    """ Create axes components to plot one source in all specified surveys
    and bands.

    :param fig: matplotlib.figure
        Figure
    :param n_row: int
        Number of rows
    :param n_col: int
        Number of columns
     :param ra: float
        Right Ascension of the target
    :param dec: float
        Declination of the target
     :param surveys: list of strings
        List of survey names, length has to be equal to bands and fovs
    :param bands: list of strings
        List of band names, length has to be equal to surveys and fovs
    :param fovs: list of floats
        Field of view in arcseconds of image cutouts, length has be equal to
        surveys, bands and apertures.
    :param apertures: list of floats
        List of apertures in arcseconds for forced photometry calculated,
        length has to be equal to surveys, bands and fovs
    :param square_sizes: list of floats
        List of
    :param image_folder_path: string
        Path to the directory where all the images are be stored
    :param mag_list: list of floats
        List of magnitudes for each survey/band
    :param magerr_list: list of floats
         List of magnitude errors for each survey/band
    :param sn_list: list of floats
         List of S/N for each survey/band
    :param forced_mag_list: list of floats
         List of forced magnitudes for each survey/band
    :param forced_magerr_list: list of floats
        List of forced magnitude errors for each survey/band
    :param forced_sn_list: list of floats
        List of forced S/N for each survey/band
    :param n_col: int
        Number of columns
    :param n_sigma: int
        Number of sigmas for the sigma-clipping routine that creates the
        boundaries for the color map.
    :param color_map_name: string
        Name of the color map
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution
    :return: matplotlib.figure
        Figure with the plot.
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
        coord_name = ut.coord_to_name(np.array([ra]), np.array([dec]),
                                      epoch="J")

        filename = image_folder_path + '/' + coord_name[0] + "_" + survey + "_" + \
                   band + "*.fits"

        # FROM HERE - This is available as an extra function now.

        data, hdr, file_path = open_image(filename, ra, dec, fov,
                                       image_folder_path,
                               verbosity)

        if data is not None and hdr is not None:
            file_found = True
        else:
            file_found = False

        # Old plotting routine to modify, currently it only plots images for
        # surveys and bands that it could open, no auto download implemented
        if file_found:
            wcs_img = wcs.WCS(hdr)

            pixcrd = wcs_img.wcs_world2pix(ra, dec, 0)
            positions = (np.float(pixcrd[0]), np.float(pixcrd[1]))
            overlap = True

            if verbosity >= 4:
                print("[DIAGNOSTIC] Image file shape {}".format(data.shape))

            try:
                img_stamp = Cutout2D(data, positions, size=fov * u.arcsec,
                                     wcs=wcs_img)

                if verbosity >= 4:
                    print("[DIAGNOSTIC] Cutout2D file shape {}".format(
                        img_stamp.shape))

            except:
                print("Source not in image")
                overlap = False
                img_stamp = None


            if img_stamp is not None:

                if overlap:
                    img_stamp = img_stamp.data

                hdu = fits.ImageHDU(data=img_stamp, header=hdr)

                axs = aplpy.FITSFigure(hdu, figure=fig, subplot=(n_row, n_col,
                                                                       idx + 1), north=True)

                # Check if input color map name is a color map, else use viridis
                try:
                    cm = plt.get_cmap(color_map_name)
                except ValueError:
                    print('Color map argument is not a color map. Setting '
                          'default: viridis')
                    cm = plt.get_cmap('viridis')
                    color_map_name = 'viridis'

                # Sigma-clipping of the color scale
                mean = np.mean(img_stamp[~np.isnan(img_stamp)])
                std = np.std(img_stamp[~np.isnan(img_stamp)])
                upp_lim = mean + n_sigma * std
                low_lim = mean - n_sigma * std
                axs.show_colorscale(vmin=low_lim, vmax=upp_lim,
                                    cmap=color_map_name)

                # Plot circular aperture (forced photometry flux)
                (yy, xx) = img_stamp.shape
                circx = (xx * 0.5)  # + 1
                circy = (yy * 0.5)  # + 1
                aper_pix = aperture_inpixels(aperture, hdr)
                circle = plt.Circle((circx, circy), aper_pix, color='r', fill=False,
                                    lw=1.5)
                fig.gca().add_artist(circle)

                # Plot rectangular aperture (error region)
                rect_inpixels = aperture_inpixels(size, hdr)
                square = plt.Rectangle((circx - rect_inpixels * 0.5,
                                        circy - rect_inpixels * 0.5),
                                       rect_inpixels, rect_inpixels,
                                       color='r', fill=False, lw=1.5)
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

                    fig.gca().text(0.03, 0.16, forcedlabel, color='black',
                             weight='bold', fontsize='large',
                             bbox=dict(facecolor='white', alpha=0.6),
                             transform=fig.gca().transAxes)

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

                    fig.gca().text(0.03, 0.04, maglabel, color='black',
                             weight='bold',
                             fontsize='large',
                             bbox=dict(facecolor='white', alpha=0.6),
                             transform=fig.gca().transAxes)

                fig.gca().set_title(survey + " " + band)

    return fig

# ------------------------------------------------------------------------------
#  Finding Chart plotting routine
# ------------------------------------------------------------------------------

def make_finding_charts(table, ra_column_name, dec_column_name,
                        target_column_name, survey, band,
                        aperture, fov, image_folder_path,
                        offset_table=None,
                        offset_id = 0,
                        offset_focus = False,
                        offset_ra_column_name=None,
                        offset_dec_column_name=None,
                        pos_angle_column_name=None,
                        offset_mag_column_name=None,
                        offset_id_column_name=None,
                        # offset_finding_chart=True,
                        label_position='bottom',
                        slit_width=None,
                        slit_length=None,
                        format ='pdf',
                        auto_download=False, verbosity=0):

    """Create and save finding charts plots for all targets in the input table.

    :param table: pandas.core.frame.DataFrame
        Dataframe with targets to plot finding charts for
    :param ra_column_name: string
        Right ascension column name
    :param dec_column_name: string
        Declination column name
    :param target_column_name: string
        Name of the target identifier column
    :param survey: string
        Survey name
    :param band: string
        Passband name
    :param aperture: float
        Aperture to plot in arcseconds
    :param fov: float
        Field of view in arcseconds
    :param image_folder_path: string
        Path to where the image will be stored
    :param offset_table: pandas.core.frame.DataFrame
        Pandas dataframe with offset star information for all targets
    :param offset_id: int
        Integer indicating the primary offset from the offset table
    :param offset_focus: boolean
        Boolean to indicate whether offset star will be in the center or not
    :param offset_ra_column_name: string
        Offset star dataframe right ascension column name
    :param offset_dec_column_name: string
        Offset star dataframe declination column name
    :param pos_angle_column_name: string
        Offset star dataframe position angle column name
    :param offset_mag_column_name: string
        Offset star dataframe magnitude column name
    :param offset_id_column_name: string
        Offset star dataframe identifier column name
    :param label_position: string
        String that defines the label position for the offset stars.
        Possible label positions are ["left", "right", "top", "bottom",
         "topleft"]
    :param slit_width: float
        Slit width in arcseconds.
    :param slit_length: float
        Slit length in arcseconds
    :param format: string
        A string indicating in which format the finding charts are save.
        Possible formats: 'pdf', 'png'
    :param auto_download: boolean
        Boolean to indicate whether images should be automatically downloaded.
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution.
    """

    surveys = [survey]
    bands = [band]
    fovs = [fov]

    print(offset_table)
    print(table)

    for idx in table.index:
        ra = table.loc[idx, ra_column_name]
        dec = table.loc[idx, dec_column_name]
        target_name = table.loc[idx, target_column_name]

        if offset_table is not None:
            offset_target = offset_table.query('target_name=="{}"'.format(
                            target_name))

            # Set position angle
            if len(offset_target) > 0:
                if pos_angle_column_name is not None:
                    position_angle = offset_target.loc[offset_target.index[0],
                                                   pos_angle_column_name]
                else:
                    target_coords = SkyCoord(ra=ra, dec=dec,
                                             unit=(u.deg, u.deg),
                                             frame='icrs')
                    offset_coords = SkyCoord(ra=offset_target.loc[:,
                                                offset_ra_column_name].values,
                                             dec=offset_target.loc[:,
                                                 offset_dec_column_name].values,
                                             unit=(u.deg, u.deg),
                                             frame='icrs')
                    # Calculate position angles(East of North)
                    pos_angles = offset_coords.position_angle(target_coords).to(
                        u.deg)
                    # Take position angle to offset_id star in list
                    position_angle = pos_angles[offset_id].to(u.deg).value

            else:
                position_angle = 0
                offset_target = None
        else:
            offset_target = None
            position_angle = 0

        if offset_target is not None:
            offset_target.reset_index(inplace=True, drop=True)


        if auto_download:
            if offset_focus:
                ct.get_photometry(offset_target.loc[[0]],
                                     offset_ra_column_name,
                                     offset_dec_column_name,
                                     surveys,
                                     bands,
                                     image_folder_path,
                                     fovs,
                                     # n_jobs=1,
                                     verbosity=verbosity)
            else:
                ct.get_photometry(table.loc[[idx]],
                                     ra_column_name,
                                     dec_column_name,
                                     surveys,
                                     bands,
                                     image_folder_path,
                                     fovs,
                                     # n_jobs=1,
                                     verbosity=verbosity)



        fig = make_finding_chart(ra, dec, survey, band, aperture, fov,
                                 image_folder_path,
                                 offset_df=offset_target,
                                 offset_id=offset_id,
                                 offset_focus=offset_focus,
                                 offset_ra_column_name=offset_ra_column_name,
                                 offset_dec_column_name=offset_dec_column_name,
                                 offset_mag_column_name=offset_mag_column_name,
                                 offset_id_column_name=offset_id_column_name,
                                 label_position=label_position,
                                 slit_width=slit_width,
                                 slit_length=slit_length,
                                 position_angle=position_angle,
                                 verbosity=verbosity)

        if format == 'pdf':
            fig.save('fc_{}.pdf'.format(target_name), transparent=False)
        if format == 'png':
            fig.save('fc_{}.png'.format(target_name), transparent=False)

        print('{} created'.format('fc_{}'.format(target_name)))


def make_finding_chart(ra, dec, survey, band, aperture, fov,
                       image_folder_path,
                       offset_df=None,
                       offset_id=0,
                       offset_focus=False,
                       offset_ra_column_name=None,
                       offset_dec_column_name=None,
                       offset_mag_column_name=None,
                       offset_id_column_name=None,
                       label_position='bottom',
                       slit_width=None, slit_length=None,
                       position_angle=None, verbosity=0):

    """Make the finding chart figure and return it.

    This is an internal function, but can be used to create one finding chart.

    :param ra: float
        Right ascension of the target in decimal degrees
    :param dec: float
        Declination of the target in decimal degrees
    :param survey: string
        Survey name
    :param band: string
        Passband name
    :param aperture: float
        Size of the plotted aperture in arcseconds
    :param fov: float
        Field of view in arcseconds
    :param image_folder_path: string
        Path to where the image will be stored
    :param offset_df: pandas.core.frame.DataFrame
        Pandas dataframe with offset star information
    :param offset_id: int
        Integer indicating the primary offset from the offset table
    :param offset_focus: boolean
        Boolean to indicate whether offset star will be in the center or not
    :param offset_ra_column_name: string
        Offset star dataframe right ascension column name
    :param offset_dec_column_name: string
        Offset star dataframe declination column name
    :param offset_mag_column_name: string
        Offset star dataframe magnitude column name
    :param offset_id_column_name: string
        Offset star dataframe identifier column name
    :param label_position: string
        String that defines the label position for the offset stars.
        Possible label positions are ["left", "right", "top", "bottom",
         "topleft"]
    :param slit_width: float
        Slit width in arcseconds.
    :param slit_length: float
        Slit length in arcseconds
    :param position_angle:
        Position angle for the observation.
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution.
    :return: matplotlib.figure
        Return the matplotlib figure of the finding chart.
    """



    if offset_focus:
        im_ra = offset_df.loc[offset_id, offset_ra_column_name]
        im_dec = offset_df.loc[offset_id, offset_dec_column_name]
    else:
        im_ra = ra
        im_dec = dec

    coord_name = ut.coord_to_name(np.array([im_ra]), np.array([im_dec]),
                                  epoch="J")

    filename = image_folder_path + '/' + coord_name[0] + "_" + survey + "_" + \
               band + "*.fits"

    data, hdr, file_path = open_image(filename, im_ra, im_dec,
                                      fov,
                                      image_folder_path,
                                      verbosity=verbosity)

    # Reproject data if position angle is specified
    if position_angle != 0:
        hdr['CRPIX1'] = int(hdr['NAXIS1'] / 2.)
        hdr['CRPIX2'] = int(hdr['NAXIS2'] / 2.)
        hdr['CRVAL1'] = im_ra
        hdr['CRVAL2'] = im_dec

        new_hdr = hdr.copy()

        pa_rad = np.deg2rad(position_angle)

        # TODO: Note that the rotation definition here reflects one axis
        # TODO: to make sure that it is a rotated version of north up east left
        # TODO: both 001 components have a negative sign!
        new_hdr['PC001001'] = -np.cos(pa_rad)
        new_hdr['PC001002'] = np.sin(pa_rad)
        new_hdr['PC002001'] = np.sin(pa_rad)
        new_hdr['PC002002'] = np.cos(pa_rad)

        from reproject import reproject_interp

        data, footprint = reproject_interp((data, hdr),
                                           new_hdr,
                                           shape_out=[hdr['NAXIS1'],
                                                      hdr['NAXIS2']])
        hdr = new_hdr

    if data is not None:
        # Plotting routine from here on.
        hdu = fits.PrimaryHDU(data, hdr)

        # De-rotate image along the position angle
        fig = aplpy.FITSFigure(hdu)

        if fov is not None:
            fig.recenter(im_ra, im_dec, radius=fov / 3600. * 0.5)

        try:
            zscale = ZScaleInterval()
            z1, z2 = zscale.get_limits(data)
            fig.show_grayscale(vmin=z1, vmax=z2)
        except Exception as e:
            print('Exception encountered: {}'.format(str(e)))
            fig.show_grayscale(pmin=10, pmax=99)

        fig.add_scalebar(fov/4/3600., '{:.1f} arcmin'.format(fov/4/60.),
                         color='black',
                         font='serif',
                         linewidth=4)

        if slit_length is not None and slit_width is not None:

            if position_angle == 0:
                _plot_slit(fig, im_ra, im_dec, slit_length, slit_width,
                           position_angle)
            else:
                _plot_slit(fig, im_ra, im_dec, slit_length, slit_width,
                           0)


        if offset_df is not None and offset_ra_column_name is not None and \
            offset_dec_column_name is not None and offset_mag_column_name is \
            not None and offset_id_column_name is not None:
            print("[INFO] Generating offsets for {}".format(filename))

            _plot_offset_stars(fig, ra, dec, offset_df, fov,
                               offset_id,
                               offset_ra_column_name,
                               offset_dec_column_name,
                               offset_mag_column_name,
                               offset_id_column_name,
                               label_position=label_position)

            _plot_info_box(fig, ra, dec, offset_df, offset_ra_column_name,
                           offset_dec_column_name, offset_mag_column_name)

        fig.show_circles(xw=ra, yw=dec, radius=aperture / 3600., edgecolor='red',
                         alpha=1, lw=3)

        fig.axis_labels.set_xtext('Right Ascension')
        fig.axis_labels.set_ytext('Declination')

        c = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree))
        title = 'RA= {0} ; DEC = {1}'.format(
            c.ra.to_string(precision=3, sep=":", unit=u.hour),
            c.dec.to_string(precision=3, sep=":", unit=u.degree, alwayssign=True))
        plt.title(title)

        fig.add_grid()
        fig.grid.show()

        fig.set_theme('publication')

        return fig




def _plot_slit(fig, ra, dec, slit_length, slit_width, position_angle):
    # slit_label = 'PA=${0:.2f}$deg\n \n'.format(position_angle)
    # slit_label += 'width={0:.1f}"; length={1:.1f}"'.format(
    #     slit_width, slit_length)

    fig = show_rectangles(fig, ra, dec, slit_width / 3600., slit_length / 3600.,
                        edgecolor='w', lw=1.0, angle=position_angle,
                        coords_frame='world')


    # if position_angle > 0 and position_angle < 180:
    #     angle_offset = 180
    #     dec_offset = 0
    # else:
    #     angle_offset = 0
    #     dec_offset = 0


    # fig.add_label(ra, dec + dec_offset, slit_label,
    #               rotation=position_angle + angle_offset + 90,
    #               size='large', color='w')


position_dict = {"left": [8, 0], "right": [-8, 0], "top": [0, 5],
                 "bottom": [0, -5], "topleft": [8, 5]}

def _plot_offset_stars(fig, ra, dec, offset_df, fov, offset_id,
                       ra_column_name,
                       dec_column_name,
                       mag_column_name,
                       id_column_name,
                       label_position="left"):

    # Check if star is in image

    radius = fov / 25. / 3600.

    ra_pos, dec_pos = position_dict[label_position]

    fig.show_circles(xw=offset_df.loc[offset_id, ra_column_name],
                     yw=offset_df.loc[offset_id, dec_column_name],
                     radius=radius * 0.5,
                     edgecolor='blue',
                     lw=3)

    fig.show_rectangles(offset_df.drop(offset_id)[ra_column_name],
                        offset_df.drop(offset_id)[dec_column_name],
                        radius, radius, edgecolor='blue', lw=1)

    abc_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

    for num, idx in enumerate(offset_df.index):
        ra_off = offset_df.loc[idx, ra_column_name]
        dec_off = offset_df.loc[idx, dec_column_name]

        target_coords = SkyCoord(ra=ra, dec=dec,
                                 unit=(u.deg, u.deg),
                                 frame='icrs')
        offset_coords = SkyCoord(ra=ra_off,
                                 dec=dec_off, unit=(u.deg, u.deg),
                                 frame='icrs')

        separation = offset_coords.separation(target_coords).to(u.arcsecond)

        label = '{}'.format(abc_dict[num])

        if separation.value <= fov/2.:
            if idx == offset_id:
                fig.add_label(ra_off + ra_pos * 5 / 3600. / 3.,
                              dec_off + dec_pos * 5 / 3600. / 3., label,
                              color='blue', size='x-large',
                              verticalalignment='center', family='serif')

            else:
                fig.add_label(ra_off + ra_pos * radius/5., dec_off + dec_pos *
                          radius/5., label,
                          color='blue', size='large',
                          verticalalignment='center', family='serif')



def _plot_info_box(fig, ra, dec, offset_df, ra_column_name, dec_column_name,
                       mag_column_name,):


    target_info = 'Target: RA={:.4f}, DEC={:.4f}'.format(ra, dec)

    info_list = [target_info]

    abc_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4:'E'}

    for num, idx in enumerate(offset_df.index):
        ra_off = offset_df.loc[idx, ra_column_name]
        dec_off = offset_df.loc[idx, dec_column_name]

        target_coords = SkyCoord(ra=ra, dec=dec,
                                 unit=(u.deg, u.deg),
                                 frame='icrs')
        offset_coords = SkyCoord(ra=ra_off,
                                 dec=dec_off, unit=(u.deg, u.deg),
                                 frame='icrs')
        # Calculate position angles and separations (East of North)
        pos_angles = offset_coords.position_angle(target_coords).to(u.deg)
        separations = offset_coords.separation(target_coords).to(u.arcsecond)
        dra, ddec = offset_coords.spherical_offsets_to(target_coords)

        mag = offset_df.loc[idx, mag_column_name]
        info = '{}:\t RA={:.4f}, DEC={:.4f}, {}={:.2f}, PosAngle={' \
               ':.2f}'.format(abc_dict[num],
                                                          ra_off,
                                                  dec_off, mag_column_name,
                                                                             mag, pos_angles)
        info_off = 'Sep={:.2f}, Dra={:.2f}, ' \
                   'Ddec={:.2f}'.format(separations, dra.to(
            'arcsecond'), ddec.to('arcsecond'))
        info_list.append(info)
        info_list.append(info_off)


    ax = plt.gca()
    boxdict = dict(facecolor='white', alpha=0.5, edgecolor='none')
    ax.text(.02, 0.02, "\n".join(info_list), transform=ax.transAxes,
            fontsize='small',
            bbox=boxdict)


# ------------------------------------------------------------------------------
#  Determine forced photometry for sources in cutouts.
# ------------------------------------------------------------------------------


def get_forced_photometry(table, ra_col_name, dec_col_name, surveys,
                          bands, apertures, fovs, image_folder_path,
                          auto_download=True,
                          verbosity=0):
    """Calculate forced photometry for all objects in the table Data Frame.

    In the current version of this routine forced photometry calculations for
    the following surveys and bands is available:
    survey: 'desdr1'
        bands: 'grizy'
    survey: "unwise-allwise, unwise-neo1, unwise-neo2, "unwise-neo3",
    "unwise-neo4", "unwise-neo5", "unwise-neo6"
        bands: 'w1w2w3w4

    This function takes a table object (astropy table, astropy fitstable or
    DataFrame) with specified Ra and Dec. It eiher looks for the image
    cutouts associated with each survey/band/fov entry or automatically
    downloads them, if specified. If the image cutouts are found forced
    photometry is calculated within the specified aperture.

    A note on confusing terminology in the function:
    img_name : Name of the image to be opened
        [Epoch Identifier][RA in HHMMSS.SS][DEC in DDMMSS.SS]_
                                 [SURVEY]_[PASSBAND]_fov[FIELD OF VIEW].fits
    filename : Path to the image without field of view. This variable is used
        to find all images of the source with different field of views
        [Image folder path]/[Epoch Identifier][RA in HHMMSS.SS]
            [DEC in DDMMSS.SS]_[SURVEY]_[PASSBAND]_*.fits
    file_path : Path to the image to be opened
        [Image folder path]/[Epoch Identifier][RA in HHMMSS.SS]
            [DEC in DDMMSS.SS]_[SURVEY]_[PASSBAND]_fov[FIELD OF VIEW].fits

    For each survey/band the following columns are added to the input table:
    forced_[survey]_mag_[band]
        Forced photometry magnitude for the object in the given survey/band.
        The magnitudes are all in the AB system
    forced_[survey]_flux_[band]
        Forced photometry flux for the object in the given survey/band
    forced_[survey]_sn_[band]
        Forced photometry S/N for the object in the given survey/band
    forced_[survey]_magerr_[band]
        Forced photometry magnitude error for the object in the given
        survey/band
    forced_[survey]_comment_[band]
        A comment with regard to the forced photometry calculation for each
        object in the given survey/band.
        If the forced photometry calculation is successful the comment will
        give the used apertures: 'ap_[aperture in arcseconds]'
        If the forced photometry calculation is unsuccessfull the comment will
        reflect the problem:
        'image_too_small': cutout image is too small to calculate the forced
         photometry (minimum pixel size 50)
        'image_not_available': cutout image could not be found and/or downloaded
        'crashed': bad things happened! (Check try-except clause in
        calculate_forced_aperture_photometry)

    Lists of equal length need to be supplied to surveys, bands, apertures and
    fovs.

    :param table: table object
        Input data table with at least RA and Decl. columns
    :param ra_col_name: string
        Exact string for the RA column in the table
    :param dec_col_name: string
        Exact string for the Decl. column in the table
    :param surveys: list of strings
        List of survey names, length has to be equal to bands, apertures and
        fovs
    :param bands: list of strings
        List of band names, length has to be equal to surveys, apertures and
        fovs
    :param apertures: list of floats
        List of apertures in arcseconds for forced photometry calculated,
        length has to be equal to surveys, bands and fovs
    :param fovs: list of floats
        Field of view in arcseconds of image cutouts, length has be equal to
        surveys, bands and apertures
    :param image_folder_path: string
        Path to the directory where all the images will be stored
    :param auto_download: Boolean
        Switch to enable/disable auto-downloading the cutouts images
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution
    :return: DataFrame
        Returns a DataFrame with the added columns for the forced photometry
        calculation.
    """

    # Check if table is pandas DataFrame otherwise convert to one
    table, format = ct.check_if_table_is_pandas_dataframe(table)
    # Add a column to the table specifying the object name used
    # for the image name
    table.loc[:, 'temp_object_name'] = ut.coord_to_name(table.loc[:,
                                                           ra_col_name].values,
                                                 table.loc[
                                                 :, dec_col_name].values,
                                                 epoch="J")

    for jdx, survey in enumerate(surveys):

        band = bands[jdx]
        aperture = apertures[jdx]
        fov = fovs[jdx]

        for idx in table.index:

            ra = table.loc[idx, ra_col_name]
            dec = table.loc[idx, dec_col_name]

            filename = image_folder_path + '/' + \
                       table.loc[idx, 'temp_object_name'] + "_" \
                       + survey + "_" + band + "*.fits"

            data, hdr, file_path = open_image(filename, ra, dec, fov,
                                              image_folder_path, verbosity)

            if data is not None:
                img_name = file_path.split('/')[-1]


            if data is None and auto_download is True:

                if survey in ["desdr1", "desdr2"]:
                    url = ct.get_des_deepest_image_url(ra,
                                                       dec,
                                                       data_release=survey[-3:],
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
                        "unwise-neo3, unwise-neo4, unwise-neo5,"
                        "unwise-neo6".format(survey, band))

                if url is not None:
                    img_name = table.loc[idx,'temp_object_name'] + "_" + \
                                         survey +  \
                               "_" + band + "_fov" + '{:d}'.format(fov)
                    ct.download_image(url, image_name=img_name,
                                   image_folder_path=image_folder_path,
                                   verbosity=verbosity)

                    file_path = image_folder_path + '/' + img_name + '.fits'
                    data, hdr = fits.getdata(file_path, header=True)


            file_size_sufficient = False

            if data is not None:
                # Check if file is sufficient
                file_size_sufficient = check_image_size(img_name,
                                                        file_path,
                                                        verbosity)

            if data is not None and file_size_sufficient is True:

                mag, flux, sn, err, comment = \
                    calculate_forced_aperture_photometry(file_path,
                                                         ra, dec, survey, band,
                                                         aperture,
                                                         verbosity=verbosity)
                table.loc[idx, 'forced_{}_mag_{}'.format(survey, band)] = mag
                table.loc[idx, 'forced_{}_flux_{}'.format(survey, band)] = flux
                table.loc[idx, 'forced_{}_sn_{}'.format(survey, band)] = sn
                table.loc[idx, 'forced_{}_magerr_{}'.format(survey, band)] = \
                    err
                table.loc[idx, 'forced_{}_{}_comment'.format(survey, band)] =\
                    comment

            if data is not None and file_size_sufficient is not True:

                table.loc[idx, 'forced_{}_{}_comment'.format(survey, band)] = \
                    'image_too_small'.format(aperture)

            if data is None:

                table.loc[idx, 'forced_{}_{}_comment'.format(survey, band)] = \
                    'image_not_available'.format(aperture)

    table.drop(columns='temp_object_name', inplace=True)

    table = ct.convert_table_to_format(table, format)

    return table


def get_forced_photometry_mp(table, ra_col_name, dec_col_name, surveys,
                          bands, apertures, fovs, image_folder_path, n_jobs=5,
                          auto_download=True,
                          verbosity=0):
    """Calculate forced photometry in multiprocessing mode.

    This function works analogous to get_forced_photometry only allowing to
    use multiple processor (python multiprocessing module).

    :param table: table object
        Input data table with at least RA and Decl. columns
    :param ra_col_name: string
        Exact string for the RA column in the table
    :param dec_col_name: string
        Exact string for the Decl. column in the table
    :param surveys: list of strings
        List of survey names, length has to be equal to bands, apertures and
        fovs
    :param bands: list of strings
        List of band names, length has to be equal to surveys, apertures and
        fovs
    :param apertures: list of floats
        List of apertures in arcseconds for forced photometry calculated,
        length has to be equal to surveys, bands and fovs
    :param fovs: list of floats
        Field of view in arcseconds of image cutouts, length has be equal to
        surveys,
        bands and apertures
    :param image_folder_path: string
        Path to the directory where all the images will be stored
    :param n_jobs:
         Number of cores to be used
    :param auto_download: Boolean
        Switch to enable/disable auto-downloading the cutouts images
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution
    :return: DataFrame
        Returns a DataFrame with the added columns for the forced photometry
        calculation.
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

        # # Create image names without the fov ending.
        # temp = table.temp_object_name

        mp_args = list(zip(index,
                           ra,
                           dec,
                           itertools.repeat(survey),
                           itertools.repeat(band),
                           itertools.repeat(aperture),
                           itertools.repeat(fov),
                           itertools.repeat(image_folder_path),
                           table.temp_object_name,
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
                              band, aperture, fov, image_folder_path,
                              temp_object_name,
                              auto_download=True,
                              verbosity=0):
    """Calculate forced photometry for one object at a time.

    :param index:
    :param ra: float
        Right Ascension of the target
    :param dec: float
        Declination of the target
    :param survey: string
        Survey name
    :param band: string
        Passband name
    :param aperture: float
        Aperture to calculate forced photometry in in arcseconds
    :param fov: float
        Field of view in arcseconds
    :param image_folder_path: string
        Path to where the image will be stored
    :param img_name:
        The name of the image to be opened for the forced photometry
        calculation (excluding the fov:
        [Epoch Identifier][RA in HHMMSS.SS][DEC in DDMMSS.SS]_
                                 [SURVEY]_[PASSBAND]
    :param auto_download: Boolean
        Switch to enable/disable auto-downloading the cutouts images
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution
    :return: tuple(int, float, float, float, float, string)
        Returns a tuple with the forced photometry values:
        index, mag, flux, sn, err, comment
    """

    # Create image filename to check against files in cutout folder
    filename = image_folder_path + '/' + temp_object_name + "_" + survey + "_" \
               + band + "*.fits"

    data, hdr, file_path = open_image(filename, ra, dec, fov,
                                      image_folder_path, verbosity)

    if data is not None:
        img_name = file_path.split('/')[-1]

    if data is None and auto_download is True:

        if survey in ["desdr1", "desdr2"]:
            url = ct.get_des_deepest_image_url(ra,
                                               dec,
                                               data_release=survey[-3:],
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
                "unwise-neo3, unwise-neo4, unwise-neo5,"
                "unwise-neo6".format(survey, band))

        if url is not None:
            img_name = temp_object_name + "_" + survey + \
                       "_" + band + "_fov" + '{:d}'.format(fov)
            ct.download_image(url, image_name=img_name,
                              image_folder_path=image_folder_path,
                              verbosity=verbosity)

            file_path = image_folder_path + '/' + img_name + '.fits'
            data, hdr = fits.getdata(file_path, header=True)

    file_size_sufficient = False
    if data is not None:
        # Check if file is sufficient
        file_size_sufficient = check_image_size(img_name,
                                                file_path,
                                                verbosity)

    if data is not None and file_size_sufficient is True:

        mag, flux, sn, err, comment = \
            calculate_forced_aperture_photometry(file_path,
                                                 ra, dec, survey, band,
                                                 aperture,
                                                 verbosity=verbosity)

        return index, mag, flux, sn, err, comment

    if data is not None and file_size_sufficient is not True:
        comment = 'image_too_small'.format(aperture)

        return index, np.nan, np.nan, np.nan, np.nan, comment

    if data is None:
        comment = 'image_not_available'.format(aperture)

        return index, np.nan, np.nan, np.nan, np.nan, comment


def calculate_forced_aperture_photometry(filepath, ra, dec, survey,
                                         band, aperture,
                                         verbosity=0):
    """Calculates the forced photometry for a Ra/Dec position on a given
    image file specified by filepath.

    :param filepath: string
        Path to the image on which to calculate the forced photometry.
    :param ra: float
        Right ascension of the source for which forced photometry should be
        calculated.
    :param dec: float
        Declination of the source for which forced photometry should be
        calculated.
    :param survey: string
        Survey keyword; The magnitude calculation depends on the survey
        photometry and hence this keyword sets the flux to magnitude
        conversion accordingly.
    :param aperture: float
        Aperture in arcseconds in over which the forced photometry is
        calculated.
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution
    :return: tuple(float, float, float, float, string)
        Returns a tuple with the forced photometry values:
        mag, flux, sn, err, comment
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

        survey_band = survey+'_'+band
        mags = pt.vega_to_ab(mags, survey_band)

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
    pixelscale = get_pixelscale(hdr)
    aperture /= pixelscale #pixels

    return aperture


def get_pixelscale(hdr):
    '''
    Get pixelscale from header and return in it in arcsec/pixel
    '''

    wcs_img = wcs.WCS(hdr)
    scale = np.mean(proj_plane_pixel_scales(wcs_img)) * 3600

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
        raise ValueError("Survey name not recognized: {}".format(survey))

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



