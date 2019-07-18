#!/usr/bin/env python
from __future__ import print_function, division

import os
import tarfile
import pandas as pd
import numpy as np
import gzip
import shutil
import itertools
import multiprocessing as mp

import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits

from astroquery.vizier import Vizier
from astroquery.irsa import Irsa
from astroquery.vsa import Vsa

from dl import queryClient as qc

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
import pyvo

from qso_toolbox import utils as ut

# ------------------------------------------------------------------------------
#  Supported surveys, data releases, bands
# ------------------------------------------------------------------------------

astroquery_dict = {
                    'tmass': {'service': 'irsa', 'catalog': 'fp_psc',
                              'ra': 'ra', 'dec': 'dec', 'mag_name':
                              'TMASS_J', 'mag': 'j_m', 'distance':
                              'dist', 'data_release': None},
                    'nomad': {'service': 'vizier', 'catalog': 'NOMAD',
                              'ra': 'RAJ2000', 'dec': 'DECJ2000',
                              'mag_name': 'R', 'mag': 'Rmag', 'distance':
                              'distance', 'data_release': None},
                    'vhsdr6': {'service': 'vsa', 'catalog': 'VHS',
                               'ra': 'ra', 'dec': 'dec',
                               'data_release': 'VHSDR6', 'mag_name': 'VHS_J',
                               'mag': 'jAperMag3', 'distance': 'distance'}
                  }

datalab_offset_dict = {'des_dr1.main': {'ra': 'ra', 'dec': 'dec',
                                          'mag': 'mag_auto_z',
                                          'mag_name': 'mag_auto_z'}}

# To add more surveys from the VISTA Science Archive, this dictionary can be
# expanded:
vsa_info_dict = {'vhsdr6': ('VHS','VHSDR6','tilestack')}



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


def convert_urltable_to_pandas(data, sep=',', header=0, skip_header=1,
                               skip_footer=1, linesep='\n'):
    """

    :param data:
    :param sep:
    :param header:
    :param skip_header:
    :param skip_footer:
    :param linesep:
    :return:
    """
    data_string = data.read().decode('utf-8').split(linesep)

    if data_string[0] == 'no rows found':
        return None
    else:
        df = pd.DataFrame(columns=data_string[header].split(sep))


        for dat in data_string[skip_header:-skip_footer]:

            df = df.append(pd.Series(dat.split(sep),
                                     index=data_string[0].split(sep)),
                           ignore_index=True)

        return df

# ------------------------------------------------------------------------------
#  Download catalog data / Offset star queries
# ------------------------------------------------------------------------------


def query_region_astroquery(ra, dec, radius, service, catalog,
                            data_release=None):
    """ Returns the catalog data of sources within a given radius of a defined
    position using astroquery.

    :param ra: float
        Right ascension
    :param dec: float
        Declination
    :param radius: float
        Region search radius in arcseconds
    :param service: string
        Astroquery class used to query the catalog of choice
    :param catalog: string
        Catalog to query
    :param data_release:
        If needed by astroquery the specified data release (e.g. needed for VSA)
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the returned matches
    """

    target_coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')

    if service == 'vizier':
        result = Vizier.query_region(target_coord, radius=radius * u.arcsecond,
                                     catalog=catalog, spatial='Cone')
        result = result[0]

    elif service == 'irsa':
        result = Irsa.query_region(target_coord, radius=radius * u.arcsecond,
                                   catalog=catalog, spatial='Cone')
    elif service == 'vsa':
        result = Vsa.query_region(target_coord, radius=radius * u.arcsecond,
                                   programme_id=catalog, database=data_release)
    else:
        raise KeyError('Astroquery class not recognized. Implemented classes '
                       'are: Vizier, Irsa, VSA')

    return result.to_pandas()



def get_astroquery_offset(target_name, target_ra, target_dec, radius, catalog,
                          quality_query=None, n=3, verbosity=0):
    """Return the n nearest offset stars specified by the quality criteria
    around a given target using astroquery.

    :param target_name: string
        Identifier for the target
    :param target_ra: float
        Target right ascension
    :param target_dec:
        Target Declination
    :param radius: float
        Maximum search radius in arcseconds
    :param catalog: string
        Catalog (and data release) to retrieve the offset star data from. See
        astroquery_dict for implemented catalogs.
    :param quality_query: string
        A string written in pandas query syntax to apply quality criteria on
        potential offset stars around the target.
    :param n: int
        Number of offset stars to retrieve. (Maximum: n=5)
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution.
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the retrieved offset stars for the given
        target.
    """


    service = astroquery_dict[catalog]['service']
    cat = astroquery_dict[catalog]['catalog']
    ra = astroquery_dict[catalog]['ra']
    dec = astroquery_dict[catalog]['dec']
    mag = astroquery_dict[catalog]['mag']
    mag_name = astroquery_dict[catalog]['mag_name']
    distance = astroquery_dict[catalog]['distance']
    dr = astroquery_dict[catalog]['data_release']

    df = query_region_astroquery(target_ra, target_dec, radius, service, cat,
                                 dr)

    if quality_query is not None:
        df.query(quality_query, inplace=True)

    if df.shape[0] > 0:
        # Sort DataFrame by match distance
        df.sort_values(distance, ascending=True, inplace=True)
        # Keep only the first three entries
        offset_df = df[:n]


        # Build the offset DataFrame
        offset_df.loc[:, 'target_name'] = target_name
        offset_df.loc[:, 'target_ra'] = target_ra
        offset_df.loc[:, 'target_dec'] = target_dec
        offset_df.loc[:, 'offset_ra'] = df[ra]
        offset_df.loc[:, 'offset_dec'] = df[dec]
        for jdx, idx in enumerate(offset_df.index):
            abc_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4:'E'}

            letter = abc_dict[jdx]

            offset_df.loc[idx, 'offset_name'] = target_name + '_offset_' +  \
                                                letter
            offset_df.loc[
                idx, 'offset_shortname'] = target_name + '_offset_' + letter


            offset_df.loc[:, mag_name] = df[mag]

        # GET THIS INTO A SEPARATE FUNCTION
        target_coords = SkyCoord(ra=target_ra, dec=target_dec,
                                 unit=(u.deg, u.deg),
                                 frame='icrs')
        offset_coords = SkyCoord(ra=offset_df.offset_ra.values,
                                 dec=offset_df.offset_dec, unit=(u.deg, u.deg),
                                 frame='icrs')


        # Calculate position angles and separations (East of North)
        pos_angles = offset_coords.position_angle(target_coords).to(u.deg)
        separations = offset_coords.separation(target_coords).to(u.arcsecond)
        dra, ddec = offset_coords.spherical_offsets_to(target_coords)
        # UNTIL HERE

        if verbosity > 1:
            print('Offset delta ra: {}'.format(dra))
            print('Offset delta dec: {}'.format(ddec))
            print('Offset separation: {}'.format(separations))
            print('Offset position angle: {}'.format(pos_angles))

        offset_df.loc[:, 'separation'] = separations.value
        offset_df.loc[:, 'pos_angle'] = pos_angles.value
        offset_df.loc[:, 'dra_offset'] = dra.to(u.arcsecond).value
        offset_df.loc[:, 'ddec_offset'] = ddec.to(u.arcsecond).value

        return offset_df[
            ['target_name', 'target_ra', 'target_dec', 'offset_name',
             'offset_shortname', 'offset_ra', 'offset_dec',
             mag, 'separation', 'pos_angle', 'dra_offset',
             'ddec_offset']]
    else:
        print("Offset star for {} not found.".format(target_name))
        return pd.DataFrame()



def get_offset_stars_astroquery(df, target_name_column, target_ra_column,
                     target_dec_column, radius, catalog='tmass', n=3,
                                quality_query=None, verbosity=0):
    """Get offset stars for all targets in the input DataFrame using astroquery.


    :param df: pandas.core.frame.DataFrame
        Dataframe with targets to retrieve offset stars for
    :param target_name_column: string
        Name of the target identifier column
    :param target_ra_column: string
        Right ascension column name
    :param target_dec_column: string
        Declination column name
     :param radius: float
        Maximum search radius in arcseconds
    :param catalog: string
        Catalog (and data release) to retrieve the offset star data from. See
        astroquery_dict for implemented catalogs.
    :param n: int
        Number of offset stars to retrieve. (Maximum: n=5)
    :param quality_query: string
        A string written in pandas query syntax to apply quality criteria on
        potential offset stars around the target.
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution.
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the retrieved offset stars for all targets
        in the input dataframe.
    """

    offset_df = pd.DataFrame()

    for idx in df.index:
        target_name = df.loc[idx, target_name_column]
        target_ra = df.loc[idx, target_ra_column]
        target_dec = df.loc[idx, target_dec_column]


        temp_df = get_astroquery_offset(target_name, target_ra, target_dec, radius, catalog,
                          quality_query=quality_query, n=n, verbosity=verbosity)


        offset_df = offset_df.append(temp_df, ignore_index=True)

        offset_df.to_csv('temp_offset_df.csv', index=False)

    return offset_df


def get_offset_stars_datalab(df, target_name_column, target_ra_column,
                     target_dec_column, radius, survey='des_dr1', table='main',
                             n=3, where=None, verbosity=0):
    """Get offset stars for all targets in the input DataFrame using the
    NOAO datalab.

    :param df: pandas.core.frame.DataFrame
        Dataframe with targets to retrieve offset stars for
    :param target_name_column: string
        Name of the target identifier column
    :param target_ra_column: string
        Right ascension column name
    :param target_dec_column: string
        Declination column name
     :param radius: float
        Maximum search radius in arcseconds
    :param survey: string
        Survey keyword for the datalab query.
    :param table: string
        Table keyword for the datalab query.
    :param n: int
        Number of offset stars to retrieve. (Maximum: n=5)
    :param where: string
        A string written in ADQL syntax to apply quality criteria on
        potential offset stars around the target.
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution.
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the retrieved offset stars for all targets
        in the input dataframe.
    """

    offset_df = pd.DataFrame()

    for idx in df.index:
        target_name = df.loc[idx, target_name_column]
        target_ra = df.loc[idx, target_ra_column]
        target_dec = df.loc[idx, target_dec_column]

        temp_df = get_datalab_offset(target_name, target_ra, target_dec, radius,
                                     survey, table, columns=None,
                                     where=where, n=n,
                                     verbosity=verbosity)

        offset_df = offset_df.append(temp_df, ignore_index=True)

        offset_df.to_csv('temp_offset_df.csv', index=False)

    os.remove('temp_offset_df.csv')

    return offset_df


def get_datalab_offset(target_name, target_ra, target_dec, radius, survey,
                       table, columns=None, where=None, n=3, verbosity=0):
    """Return the n nearest offset stars specified by the quality criteria
    around a given target using the NOAO datalab.

    :param target_name: string
        Identifier for the target
    :param target_ra: float
        Target right ascension
    :param target_dec:
        Target Declination
    :param radius: float
        Maximum search radius in arcseconds
    :param survey: string
        Survey keyword for the datalab query.
    :param table: string
        Table keyword for the datalab query.
    :param where: string
        A string written in ADQL syntax to apply quality criteria on
        potential offset stars around the target.
    :param n: int
        Number of offset stars to retrieve. (Maximum: n=5)
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution.
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the retrieved offset stars for the given
        target.
    """

    df = query_region_datalab(target_ra, target_dec, radius, survey=survey,
                              table=table, columns=columns, where=where,
                              verbosity=verbosity)

    catalog = survey+'.'+table
    ra = datalab_offset_dict[catalog]['ra']
    dec = datalab_offset_dict[catalog]['dec']
    mag = datalab_offset_dict[catalog]['mag']
    mag_name = datalab_offset_dict[catalog]['mag_name']

    # distance column is in arcminutes!!

    if df.shape[0] > 0:
        # Sort DataFrame by match distance
        df.sort_values('distance', ascending=True, inplace=True)
        # Keep only the first three entries
        offset_df = df[:n]

        # Build the offset DataFrame
        offset_df.loc[:, 'target_name'] = target_name
        offset_df.loc[:, 'target_ra'] = target_ra
        offset_df.loc[:, 'target_dec'] = target_dec
        offset_df.loc[:, 'offset_ra'] = df[ra]
        offset_df.loc[:, 'offset_dec'] = df[dec]
        for jdx, idx in enumerate(offset_df.index):
            abc_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

            letter = abc_dict[jdx]

            offset_df.loc[idx, 'offset_name'] = target_name + '_offset_' + \
                                                letter
            offset_df.loc[
                idx, 'offset_shortname'] = target_name + '_offset_' + letter

            offset_df.loc[:, mag_name] = df[mag]

        # GET THIS INTO A SEPARATE FUNCTION
        target_coords = SkyCoord(ra=target_ra, dec=target_dec,
                                 unit=(u.deg, u.deg),
                                 frame='icrs')
        offset_coords = SkyCoord(ra=offset_df.offset_ra.values,
                                 dec=offset_df.offset_dec, unit=(u.deg, u.deg),
                                 frame='icrs')

        # Calculate position angles and separations (East of North)
        pos_angles = offset_coords.position_angle(target_coords).to(u.deg)
        separations = offset_coords.separation(target_coords).to(u.arcsecond)
        dra, ddec = offset_coords.spherical_offsets_to(target_coords)
        # UNTIL HERE

        if verbosity > 1:
            print('Offset delta ra: {}'.format(dra))
            print('Offset delta dec: {}'.format(ddec))
            print('Offset separation: {}'.format(separations))
            print('Offset position angle: {}'.format(pos_angles))

        offset_df.loc[:, 'separation'] = separations.value
        offset_df.loc[:, 'pos_angle'] = pos_angles.value
        offset_df.loc[:, 'dra_offset'] = dra.to(u.arcsecond).value
        offset_df.loc[:, 'ddec_offset'] = ddec.to(u.arcsecond).value

        return offset_df[
            ['target_name', 'target_ra', 'target_dec', 'offset_name',
             'offset_shortname', 'offset_ra', 'offset_dec',
             mag, 'separation', 'pos_angle', 'dra_offset',
             'ddec_offset']]

    else:
        print("Offset star for {} not found.".format(target_name))

        return pd.DataFrame()


def query_region_datalab(ra, dec, radius, survey='des_dr1', table='main',
                         columns=None, where=None, verbosity=0):
    """ Returns the catalog data of sources within a given radius of a defined
    position using the NOAO datalab.

    :param ra: float
        Right ascension
    :param dec: float
        Declination
    :param radius: float
        Region search radius in arcseconds
    :param survey: string
        Survey keyword for the datalab query.
    :param table: string
        Table keyword for the datalab query.
    :param columns:
        The names of the columns that should be returned.
    :param where: string
        A string written in ADQL syntax to apply quality criteria on
        potential offset stars around the target.
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the returned matches
    """

    radius_deg = radius / 3600.

    # Build SQL query
    if columns is not None:
        sql_query = 'SELECT {} '.format(columns)
    else:
        sql_query = 'SELECT * '.format(survey, table)

    sql_query += ', q3c_dist(ra, dec, {}, {}) as distance '.format(ra, dec)

    sql_query += 'FROM {}.{} WHERE '.format(survey, table)

    sql_query += 'q3c_radial_query(ra, dec, {}, {}, {}) '.format(ra, dec,
                                                                radius_deg)
    if where is not None:
        sql_query += 'AND {}'.format(where)

    # Query DataLab and write result to temporary file
    if verbosity > 0:
        print("SQL QUERY: {}".format(sql_query))

    result = qc.query(sql=sql_query)

    f = open('temp.csv', 'w+')
    f.write(result)
    f.close()

    # Open temporary file in a dataframe and delete the temporary file
    df = pd.read_csv('temp.csv')

    os.remove('temp.csv')

    return df


def query_region_ps1(ra, dec, radius, survey='dr1', catalog='mean',
                     add_criteria=None, verbosity=0):
    """ Returns the catalog data of sources within a given radius of a defined
    position using the MAST website.

    :param ra: float
        Right ascension
    :param dec: float
        Declination
    :param radius: float
        Region search radius in arcseconds
    :param survey: string
        Survey keyword for the PanSTARRS MAST query.
    :param catalog: string
        Catalog keyword for the PanSTARRS MAST query.
    :param columns:
        The names of the columns that should be returned.
    :param  add_criteria: string
        A string with conditions to apply additional quality criteria on
        potential offset stars around the target.
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the returned matches
    """
    urlbase = 'https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/'

    if add_criteria is None:
        url = urlbase + \
              '{}/{}?ra={}&dec={}&radius={}&format=csv'.format(survey, catalog,
                                                               ra,
                                                               dec,
                                                               radius)
    else:
        url = urlbase + '{}/{}?ra={}&dec={}&radius={}&' + \
              add_criteria + 'format=csv'.format(survey, catalog, ra, dec,
                                                 radius)
    if verbosity>0:
        print('Opening {}'.format(url))

    ps1_data = urlopen(url)
    check_ok = ps1_data.msg == 'OK'

    if check_ok:
        df = pd.read_csv(url)
        return df

    else:
        raise ValueError('Could not retrieve PS1 data.')


def get_ps1_offset_star(target_name, target_ra, target_dec, radius=300,
                        catalog='mean', data_release='dr2',
                        quality_query=None, n=3, verbosity=0):
    """Return the n nearest offset stars specified by the quality criteria
    around a given target using the MAST website for PanSTARRS.

    It will always retrieve the z-band magnitude for the offset star. This is
    hardcoded. Depending on the catalog it will be the mean of stack magnitude.

    :param target_name: string
        Identifier for the target
    :param target_ra: float
        Target right ascension
    :param target_dec:
        Target Declination
    :param radius: float
        Maximum search radius in arcseconds
    :param catalog: string
        Catalog to retrieve the offset star data from. (e.g. 'mean', 'stack')
    :param data_release: string
        The specific PanSTARRS data release
    :param quality_query: string
        A string written in pandas query syntax to apply quality criteria on
        potential offset stars around the target.
    :param n: int
        Number of offset stars to retrieve. (Maximum: n=5)
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution.
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the retrieved offset stars for the given
        target.
    """

    # Convert radius in degrees
    radius_degree = radius / 3600.

    if verbosity>1:
        print('Querying PS1 Archive ({},{}) for {}'.format(catalog,
                                                           data_release,
                                                           target_name))
    # Query the PanStarrs 1 archive
    df = query_region_ps1(target_ra, target_dec, radius_degree,
                          survey=data_release,
                          catalog=catalog, add_criteria=None,
                          verbosity=verbosity)

    # Drop duplicated targets
    df.drop_duplicates(subset='objName', inplace=True)
    # Apply quality criteria query
    if quality_query is not None:
        df.query(quality_query, inplace=True)
    if df.shape[0] > 0:
        # Sort DataFrame by match distance
        df.sort_values('distance', ascending=True, inplace=True)
        # Keep only the first three entries
        offset_df = df[:n]

        # Build the offset DataFrame
        offset_df.loc[:, 'target_name'] = target_name
        offset_df.loc[:, 'target_ra'] = target_ra
        offset_df.loc[:, 'target_dec'] = target_dec
        offset_df.loc[:, 'offset_ra'] = df.raMean
        offset_df.loc[:, 'offset_dec'] = df.decMean
        for jdx, idx in enumerate(offset_df.index):
            abc_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

            letter = abc_dict[jdx]

            offset_df.loc[idx, 'offset_name'] = target_name + '_offset_' + \
                                                letter
            offset_df.loc[
                idx, 'offset_shortname'] = target_name + '_offset_' + letter

        if catalog == 'mean':
            mag = 'ps1_' + data_release + '_mean_psfmag_z'
            offset_df.loc[:, mag] = df.zMeanPSFMag
        elif catalog == 'stack':
            mag = 'ps1_' + data_release + '_stack_psfmag_z'
            offset_df.loc[:, mag] = df.zPSFMag
        else:
            raise ValueError(
                'Catalog value not understood ["mean","stack"] :{}'.format(catalog))

        target_coords = SkyCoord(ra=target_ra, dec=target_dec, unit=(u.deg, u.deg),
                                 frame='icrs')
        offset_coords = SkyCoord(ra=offset_df.offset_ra.values,
                                 dec=offset_df.offset_dec, unit=(u.deg, u.deg),
                                 frame='icrs')
        # Calculate position angles and separations (East of North)
        pos_angles = offset_coords.position_angle(target_coords).to(u.deg)
        separations = offset_coords.separation(target_coords).to(u.arcsecond)
        dra, ddec = offset_coords.spherical_offsets_to(target_coords)

        if verbosity > 1:
            print('Offset delta ra: {}'.format(dra))
            print('Offset delta dec: {}'.format(ddec))
            print('Offset separation: {}'.format(separations))
            print('Offset position angle: {}'.format(pos_angles))


        offset_df.loc[:, 'separation'] = separations.value
        offset_df.loc[:, 'pos_angle'] = pos_angles.value
        offset_df.loc[:, 'dra_offset'] = dra.to(u.arcsecond).value
        offset_df.loc[:, 'ddec_offset'] = ddec.to(u.arcsecond).value

        return offset_df[['target_name', 'target_ra', 'target_dec', 'offset_name',
                          'offset_shortname', 'offset_ra', 'offset_dec',
                          mag, 'separation', 'pos_angle', 'dra_offset',
                          'ddec_offset']]
    else:
        print("Offset star for {} not found.".format(target_name))
        return pd.DataFrame()


def get_offset_stars_ps1(df, target_name_column, target_ra_column,
                     target_dec_column, radius, data_release='dr2',
                     catalog='mean', quality_query=None, verbosity=0):
    """Get offset stars for all targets in the input DataFrame for PanSTARRS
    using the MAST website.

    Currently this runs slowly as it queries the PanSTARRS 1 archive for each
    object. But it runs!

    It will always retrieve the z-band magnitude for the offset star. This is
    hardcoded in get_ps1_offset_star(). Depending on the catalog it will be
    the mean of stack magnitude.


    :param df: pandas.core.frame.DataFrame
        Dataframe with targets to retrieve offset stars for
    :param target_name_column: string
        Name of the target identifier column
    :param target_ra_column: string
        Right ascension column name
    :param target_dec_column: string
        Declination column name
     :param radius: float
        Maximum search radius in arcseconds
       :param catalog: string
        Catalog to retrieve the offset star data from. (e.g. 'mean', 'stack')
    :param data_release: string
        The specific PanSTARRS data release
    :param n: int
        Number of offset stars to retrieve. (Maximum: n=5)
    :param quality_query: string
        A string written in pandas query syntax to apply quality criteria on
        potential offset stars around the target.
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution.
    :return: pandas.core.frame.DataFrame
        Returns the dataframe with the retrieved offset stars for all targets
        in the input dataframe.
    """
    offset_df = pd.DataFrame()

    for idx in df.index:
        target_name = df.loc[idx, target_name_column]
        target_ra = df.loc[idx, target_ra_column]
        target_dec = df.loc[idx, target_dec_column]


        temp_df = get_ps1_offset_star(target_name, target_ra, target_dec,
                                        radius=radius, catalog=catalog,
                                        data_release=data_release,
                                        quality_query=quality_query,
                                      verbosity=verbosity)


        offset_df = offset_df.append(temp_df, ignore_index=True)

        offset_df.to_csv('temp_offset_df.csv', index=False)

    return offset_df

# ------------------------------------------------------------------------------
#  Download catalog images
# ------------------------------------------------------------------------------

def get_vsa_info(survey):

    if survey in vsa_info_dict:
        return vsa_info_dict[survey]
    else:
        return None


def get_photometry(table, ra_col_name, dec_col_name, surveys, bands, image_folder_path,
                   fovs, verbosity=0):
    """Download photometric images for all objects in the given input table.

    Lists need to be supplied to specify the survey, the photometric passband
    and the field of fiew. Each entry in the survey list corresponds to one
    entry in the passband and field of view lists.

    The downloaded images will be save in the image_folder_path directory using a
    unique filename based on the target position, survey + data release,
    passband and field of view.

    Image name
    [Epoch Identifier][RA in HHMMSS.SS][DEC in DDMMSS.SS]_
                                 [SURVEY]_[PASSBAND]_fov[FIELD OF VIEW].fits

    A example for DES DR1 z-band with a field of view of 100 arcsec:
    J224029.28-000511.83_desdr1_z_fov100.fits

    The list of field of views will be accurately downloaded for desdr1. For
    the download of the unWISE image cutouts the field of views will be
    converted to number of pixels with npix = fov / 60. /4. * 100, with an
    upper limit of 256 pixels.

    IMPORTANT:
    The function will skip downloads for targets with exactly the same
    specifications (filenames) that already exist in the folder.

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
    :param image_folder_path: string
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
            ra = table.loc[idx, ra_col_name]
            dec = table.loc[idx, dec_col_name]

            if survey == "ps1" and band in ["g", "r", "i", "z", "y"]:
                img_name = table.temp_object_name[idx] + "_" + survey + "_" + \
                           band + "_fov" + '{:d}'.format(fov)

                file_path = image_folder_path + '/' + img_name + '.fits'
                file_exists = os.path.isfile(file_path)

                if file_exists is not True:
                    url_list = get_ps1_image_cutout_url(ra, dec, fov=fov,
                                                   bands=band,
                                                   verbosity=verbosity)
                    if url_list is not None:
                        url = url_list[0]
                    else:
                        url = None

                else:
                    url = None

            elif survey in ['vhsdr6'] and band in ['J', 'H', 'Ks']:

                vsa_info = get_vsa_info(survey)

                if vsa_info is not None:

                    img_name = table.temp_object_name[idx] + "_" + survey + "_" + \
                               band + "_fov" + '{:d}'.format(fov)

                    file_path = image_folder_path + '/' + img_name + '.fits'
                    file_exists = os.path.isfile(file_path)

                    if file_exists is not True:

                        url_list = get_vsa_image_url(ra, dec, fov=fov,
                                                     band=band,
                                                     vsa_info=vsa_info)
                        if url_list:
                            url = url_list[-1]
                        else:
                            url = None

                    else:
                        url = None

                else:
                    url = None
                    print('Survey {} is not in vsa_info_dict. \n Download '
                          'not possible.'.format(survey))

            elif survey == "desdr1" and band in ["g", "r", "i", "z", "Y"]:

                img_name = table.temp_object_name[idx] + "_" + survey + "_" + \
                           band + "_fov" + '{:d}'.format(fov)

                file_path = image_folder_path + '/' + img_name + '.fits'
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
                npix = int(round(fov/60./4. * 100))

                img_name = table.temp_object_name[idx] + "_" + survey + "_" + \
                           band + "_fov" + '{:d}'.format(fov)

                file_path = image_folder_path + '/' + img_name + '.fits'
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
                                 "Possible survey names include: desdr1, ps1,"
                                 "vhsdr6, "
                                 "unwise-allwise, unwise-neo1, unwise-neo2, "
                                 "unwise-neo3".format(survey, band))

            if url is not None:
                download_image(url, image_name=img_name, image_folder_path=image_folder_path,
                               verbosity=verbosity)


def get_photometry_mp(table, ra_col_name, dec_col_name, surveys, bands,
                      image_folder_path, fovs, n_jobs=2,
                      verbosity=0):
    """Download photometric images for all objects in the given input table
    using multiprocessing.

    Lists need to be supplied to specify the survey, the photometric passband
    and the field of fiew. Each entry in the survey list corresponds to one
    entry in the passband and field of view lists.

    The downloaded images will be save in the image_folder_path directory using a
    unique filename based on the target position, survey + data release,
    passband and field of view.

    Image name
    [Epoch Identifier][RA in HHMMSS.SS][DEC in DDMMSS.SS]_
                                 [SURVEY]_[PASSBAND]_fov[FIELD OF VIEW].fits

    A example for DES DR1 z-band with a field of view of 100 arcsec:
    J224029.28-000511.83_desdr1_z_fov100.fits

    The list of field of views will be accurately downloaded for desdr1. For
    the download of the unWISE image cutouts the field of views will be
    converted to number of pixels with npix = fov / 60. /4. * 100, with an
    upper limit of 256 pixels.

    IMPORTANT:
    The function will skip downloads for targets with exactly the same
    specifications (filenames) that already exist in the folder.

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
    :param image_folder_path: string
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

        vsa_info = get_vsa_info(survey)

        mp_args = list(zip(table[ra_col_name].values,
                           table[dec_col_name].values,
                           itertools.repeat(survey),
                           itertools.repeat(band),
                           itertools.repeat(fov),
                           itertools.repeat(image_folder_path),
                           table['temp_object_name'].values,
                           itertools.repeat(vsa_info),
                           itertools.repeat(verbosity)))


        # alternative idea: get all urls first and then download them.
        with mp.Pool(processes=n_jobs) as pool:
            pool.starmap(_mp_photometry_download, mp_args)



def _mp_photometry_download(ra, dec, survey, band,  fov, image_folder_path,
                            temp_object_name, vsa_info, verbosity):
    """Download one photometric image.

    This function is designed to be an internal function to be called by the
    multiprocessing module using pool.map() .


    :param ra: float
        Right Ascension of the target in decimal degrees.
    :param dec: float
        Declination of the target in decimal degrees.
    :param survey: string
        Survey name
    :param band: string
        Passband name
    :param fov: float
        Field of view in arcseconds
    :param image_folder_path: string
        Path to where the image will be stored
    :param temp_object_name:
        Temporary object name specifying the coordinates of the target
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution
    :return: None
    """

    # Adding PanSTARRS 1 download here for multiprocessing. However it would
    # be faster to implement not all objects per "band" multiprocessing,
    # but all bands per object multiprocessing for PS1.
    if survey == "ps1" and band in ["g", "r", "i", "z", "y"]:
        img_name = temp_object_name + "_" + survey + "_" + \
                   band + "_fov" + '{:d}'.format(fov)

        file_path = image_folder_path + '/' + img_name + '.fits'
        file_exists = os.path.isfile(file_path)

        if file_exists is not True:
            url_list = get_ps1_image_cutout_url(ra, dec, fov=fov,
                                                bands=band,
                                                verbosity=verbosity)
            if url_list is not None:
                url = url_list[0]
            else:
                url = None

        else:
            url = None

    elif survey in ['vhsdr6'] and band in ['J', 'H', 'Ks']:

        if vsa_info is not None:

            img_name = temp_object_name + "_" + survey + "_" + \
                       band + "_fov" + '{:d}'.format(fov)

            file_path = image_folder_path + '/' + img_name + '.fits'
            file_exists = os.path.isfile(file_path)

            if file_exists is not True:

                url_list = get_vsa_image_url(ra, dec, fov=fov,
                                             band=band,
                                             vsa_info=vsa_info)
                if url_list:
                    url = url_list[-1]
                else:
                    url = None

            else:
                url = None

        else:
            url = None
            print('Survey {} is not in vsa_info_dict. \n Download '
                  'not possible.'.format(survey))

    elif survey == "desdr1":
        img_name = temp_object_name + "_" + survey + "_" + \
                   band + "_fov" + '{:d}'.format(fov)

        file_path = image_folder_path + '/' + img_name + '.fits'
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
        npix = int(round(fov / 60. / 4. * 100))

        img_name = temp_object_name + "_" + survey + "_" + \
                   band + "_fov" + '{:d}'.format(fov)

        file_path = image_folder_path + '/' + img_name + '.fits'
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
                         "Possible survey names include: desdr1, ps1, "
                         "unwise-allwise, unwise-neo1, unwise-neo2, "
                         "unwise-neo3".format(survey, band))

    if url is not None:
        try:
            download_image(url, image_name=img_name, image_folder_path=image_folder_path,
                       verbosity=verbosity)
        except:
            # if verbosity >0:
            print('Download error')

    return 0


def download_image(url, image_name, image_folder_path, verbosity=0):
    """Download an image cutout specified by the given url.

    :param url: str
        URL to image cutout
    :param image_name: str
        Unique name of the image: "Coordinate"+"Survey"+"Band"
    :param image_folder_path: str
        Path to where the image is saved to
    :param verbosity: int
        Verbosity > 0 will print verbose statements during the execution
    :return:
    """

    # Check if download directory exists. If not, create it
    if not os.path.exists(image_folder_path):
        os.makedirs(image_folder_path)

    survey = image_name.split("_")[1]

    # Try except clause for downloading the image
    try:
        datafile = urlopen(url)

        check_ok = datafile.msg == 'OK'

        if check_ok:

            if survey in ['desdr1', 'ps1', 'vhsdr6']:

                file = datafile.read()

                output = open(image_folder_path+'/'+image_name+'.fits', 'wb')
                output.write(file)
                output.close()
                if verbosity > 0:
                    print("Download of {} to {} completed".format(image_name,
                                                                  image_folder_path))

            elif survey in ["unwise-allwise", "unwise-neo1", "unwise-neo2",
                            "unwise-neo3"]:


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

                output = open(image_folder_path + '/' + image_name + '.fits', 'wb')
                output.write(untar)
                output.close()
                if verbosity > 0:
                    print("Download of {} to {} completed".format(image_name,
                                                                  image_folder_path))

            else:
                raise ValueError("Survey name not recognized: {}. "
                                 "\n "
                                 "Possible survey names include: desdr1, ps1, "
                                 "vhsdr6, "
                                 "unwise-allwise, unwise-neo1, unwise-neo2, "
                                 "unwise-neo3".format(survey))

        else:
            if verbosity > 0:
                print("Download of {} unsuccessful".format(image_name))
                print("Tried to download from: {}".forma(url))

    except (IncompleteRead, HTTPError, AttributeError, ValueError) as err:
        print(err)
        if verbosity > 0:
            print("Download of {} unsuccessful".format(image_name))

# ------------------------------------------------------------------------------
#  Catalog specific URL construction
# ------------------------------------------------------------------------------

def get_ps1_filenames(ra, dec, bands='g'):
    """

    :param ra:
    :param dec:
    :param bands:
    :return:
    """
    url_base = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?'
    ps1_url = url_base + 'ra={}&dec={}&filters={}'.format(ra, dec, bands)

    table = Table.read(ps1_url, format='ascii')

    # Sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]

    filenames = table['filename']

    if len(filenames) > 0:
        return filenames
    else:
        print("No PS1 image is available for this position.")
        return None


def get_ps1_image_url(ra, dec, bands='g'):
    """

    :param ra:
    :param dec:
    :param bands:
    :return:
    """
    filenames = get_ps1_filenames(ra, dec, bands)

    if filenames is not None:

        url_list = []

        for filename in filenames:
            url_list.append('http://ps1images.stsci.edu{}'.format(filename))

        return url_list
    else:
        return None


def get_ps1_image_cutout_url(ra, dec, fov, bands='g', verbosity=0):
    """

    :param ra:
    :param dec:
    :param fov:
    :param bands:
    :param verbosity:
    :return:
    """


    # Convert field of view in arcsecond to pixel size (1 pixel = 0.25 arcseconds)
    size = fov * 4

    filenames = get_ps1_filenames(ra, dec, bands)

    if filenames is not None:

        url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
               "ra={ra}&dec={dec}&size={size}&format=fits").format(**locals())

        urlbase = url + "&red="
        url_list = []
        for filename in filenames:
            url_list.append(urlbase + filename)

        return url_list
    else:
        return None


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
    import time
    time.sleep(2)
    # Set the DES DR1 NOAO sia url
    def_access_url = "https://datalab.noao.edu/sia/des_dr1"
    svc = sia.SIAService(def_access_url)

    if verbosity > 0:
        print(svc)

    fov = fov / 3600.

    siaresults = None
    if isinstance(svc, sia.SIAService):
        try:
            siaresults = svc.search((ra, dec),
                                (fov / np.cos(dec * np.pi / 180), fov))
        except pyvo.dal.DALQueryError as err:
            print(err)
            siaresults = None

    if isinstance(siaresults, sia.SIAResults):


        try:
            img_table = siaresults.to_table()

        except:
            img_table = siaresults.table

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

    else:
        print('SIA Error')
        return None


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


def get_vsa_image_url(ra, dec, fov, band, vsa_info=('VHS','VHSDR6',
                                                    'tilestack')):
    programme_id, database, frame_type = vsa_info

    target_coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')

    url_list = Vsa.get_image_list(target_coord, image_width=fov * u.arcsecond,
                                    programme_id=programme_id,
                                    database=database,
                                    frame_type=frame_type,
                                    waveband=band)

    return url_list

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


def download_vhs_wget_cutouts(wget_filename, image_folder_path, survey_name='vhsdr6',
                              fov=None, verbosity=0):
    """Download the image cutouts based on the VSA 'wget' file specified.

    :param wget_filename: str
        Filepath and filename to the "wget" Vista Science Archive file
    :param image_folder_path: str
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
        file_path = image_folder_path + '/' + image_name + '.fits'
        file_exists = os.path.isfile(file_path)

        if file_exists:
            if verbosity > 0:
                print("Image of {} already exists in {}.".format(image_name,
                                                              image_folder_path))
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
                    with open(image_folder_path + '/' + image_name + '.fits',
                              'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                if verbosity > 0:
                    print("Download of {} to {} completed".format(image_name,
                                                                  image_folder_path))
            else:
                if verbosity > 0:
                    print("Download of {} unsuccessful".format(image_name))
