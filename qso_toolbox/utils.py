
import re
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

def decra_to_hms(dra):
    """Convert Right Ascension in decimal degrees to hours, minutes, seconds.

    :param dra: float
        Right Ascension in decimal degrees
    :return: integer, integer, float
        Right Ascension in hours, minutes, seconds
    """
    ra_minutes, ra_seconds = divmod(dra / 15 * 3600, 60)
    ra_hours, ra_minutes = divmod(ra_minutes, 60)

    return ra_hours, ra_minutes, ra_seconds

def decdecl_to_dms(ddecl):
    """Convert Declination in decimal degrees to degrees, minutes, seconds

    :param ddecl: float
        Declination in decimal degrees
    :return: integer, integer, float
        Declination in degrees, minutes, seconds
    """
    is_negative = ddecl < 0
    ddecl = abs(ddecl)
    decl_minutes, decl_seconds = divmod(ddecl * 3600, 60)
    decl_degrees, decl_minutes = divmod(decl_minutes, 60)
    decl_degrees[is_negative] = - decl_degrees[is_negative]

    return decl_degrees, decl_minutes, decl_seconds

def hmsra2decdeg(ra_hms, delimiter=':'):

    if isinstance(ra_hms, float) or isinstance(ra_hms, int):

        return convert_hmsra2decdeg(ra_hms, delimiter=delimiter)

    elif isinstance(ra_hms, np.ndarray):

        ra_deg = np.zeros_like(ra_hms)
        for idx, ra in enumerate(ra_hms):

            ra_deg[idx] = convert_hmsra2decdeg(ra, delimiter=delimiter)

        return ra_deg

    else:
        raise TypeError("Input type {} not understood (float, int, "
                        "np.ndarray)".format(type(ra_hms)))


def convert_hmsra2decdeg(ra_hms, delimiter=':'):

    if delimiter is None:
        ra_hours = float(ra_hms[0:2])
        ra_minutes = float(ra_hms[2:4])
        ra_seconds = float(ra_hms[4:10])
    if delimiter ==':':
        ra_hours = float(ra_hms[0:2])
        ra_minutes = float(ra_hms[3:5])
        ra_seconds = float(ra_hms[6:12])

    return (ra_hours + ra_minutes/60. + ra_seconds/3600.) * 15.


def dmsdec2decdeg(dec_dms, delimiter=':'):


    if isinstance(dec_dms, float) or isinstance(dec_dms, int):

        return convert_hmsra2decdeg(dec_dms, delimiter=delimiter)

    elif isinstance(dec_dms, np.ndarray):

        dec_deg = np.zeros_like(dec_dms)
        for idx, dec in enumerate(dec_dms):
            dec_deg[idx] = convert_dmsdec2decdeg(dec, delimiter=delimiter)

        return dec_deg

    else:
        raise TypeError("Input type {} not understood (float, int, "
                        "np.ndarray)".format(type(dec_dms)))

def convert_dmsdec2decdeg(dec_dms,delimiter=':'):

    if delimiter is None:
        dec_degrees = float(dec_dms[0:3])
        dec_minutes = float(dec_dms[3:5])
        dec_seconds = float(dec_dms[5:10])
    if delimiter is not None:
        dec_degrees = float(dec_dms.split(delimiter)[0])
        dec_minutes = float(dec_dms.split(delimiter)[1])
        dec_seconds = float(dec_dms.split(delimiter)[2])

    # print(dec_dms[0])

    if dec_dms[0] == '-':
        is_positive = False
    else:
        is_positive = True

    dec = abs(dec_degrees) + dec_minutes/60. + dec_seconds/3600.

    if is_positive is False:
        dec = -dec

    return dec

def designation_to_coord(designation, delimiter=None, prefix=None):

    if prefix is None:
        prefix = re.search(r'[a-z]+', designation, re.IGNORECASE).group(0)

    designation = designation.replace(prefix, '')

    if '+' in designation:
        ra_hms = designation.split('+')[0]
        dec_dms = '+' + designation.split('+')[1]
    elif '-' in designation:
        ra_hms = designation.split('-')[0]
        dec_dms = '-' + designation.split('-')[1]

    ra_deg = convert_hmsra2decdeg(ra_hms, delimiter=delimiter)
    dec_deg = convert_dmsdec2decdeg(dec_dms, delimiter=delimiter)

    return ra_deg, dec_deg


def coord_to_hmsdms(dra, ddecl):
    ra_hours, ra_minutes, ra_seconds = decra_to_hms(dra)
    decl_degrees, decl_minutes, decl_seconds = decdecl_to_dms(ddecl)

    coord_list = []
    for idx in range(len(dra)):
        ra = '{:02g}:{:02g}:{:05.3f}'.format(ra_hours[idx],
                                           ra_minutes[idx],
                                           ra_seconds[idx])
        dec = '{:+03g}:{:02g}:{:05.2f}'.format(decl_degrees[idx],
                                             decl_minutes[idx],
                                             decl_seconds[idx])

        coord_list.append([ra, dec])

    return coord_list


def coord_to_name(dra, dd, epoch ='J'):
    """Return an object name based on its Right Ascension and Declination.

    :param dra: float
        Right Ascension of the target in decimal degrees
    :param dd: float
        Declination of the target in decimal degrees
    :param epoch: string
        Epoch string (default: J), can also be substituted for survey
        abbreviation.
    :return: string
        String based on the targets coordings [epoch][RA in HMS][Dec in DMS]
    """

    ra_hours, ra_minutes, ra_seconds = decra_to_hms(dra)
    decl_degrees, decl_minutes, decl_seconds = decdecl_to_dms(dd)

    coord_name_list = []

    for idx in range(len(dra)):

        coord_name_list.append('{0:1}{1:02g}{2:02g}{3:05.2f}{4:+03g}{5:02g}{'
                               '6:05.2f}'.format(epoch,
                                             ra_hours[idx],
                                             ra_minutes[idx],
                                             ra_seconds[idx],
                                             decl_degrees[idx],
                                             decl_minutes[idx],
                                             decl_seconds[idx]))

    return coord_name_list


def coord_to_vhsname(dra, dd, epoch ='J'):
    """Return an object name based on its Right Ascension and Declination.

    :param dra: float
        Right Ascension of the target in decimal degrees
    :param dd: float
        Declination of the target in decimal degrees
    :param epoch: string
        Epoch string (default: J), can also be substituted for survey
        abbreviation.
    :return: string
        String based on the targets coordings [epoch][RA in HMS][Dec in DMS]
    """

    ra_hours, ra_minutes, ra_seconds = decra_to_hms(dra)
    decl_degrees, decl_minutes, decl_seconds = decdecl_to_dms(dd)

    ra_seconds_list = []

    for idx, ra_sec in enumerate(ra_seconds):
        ra_seconds_str = '{:05.2f}'.format(ra_sec)[0:2] \
                           + '{:05.2f}'.format(ra_sec)[3:]
        ra_seconds_list.append(ra_seconds_str)

    decl_seconds_list = []

    for idx, decl_sec in enumerate(decl_seconds):
        decl_sec = np.around(decl_sec, decimals=1)
        decl_seconds_str = '{:05.2f}'.format(decl_sec)[0:2] \
                           + '{:05.2f}'.format(decl_sec)[3:4]
        decl_seconds_list.append(decl_seconds_str)

    coord_name_list = []

    for idx in range(len(dra)):

        coord_name_list.append('{0:1}{1:02g}{2:02g}{3:}{4:+03g}{5:02g}{'
                               '6:}'.format(epoch,
                                             ra_hours[idx],
                                             ra_minutes[idx],
                                             ra_seconds_list[idx],
                                             decl_degrees[idx],
                                             decl_minutes[idx],
                                             decl_seconds_list[idx]))

    return coord_name_list


def get_offset_parameters(df, ra_column_name, dec_column_name,
                          ra_offset_column_name, dec_offset_column_name,
                          verbosity=0):

    df['dra'] = None
    df['ddec'] = None
    df['separation'] = None
    df['pos_angle'] = None

    for idx in df.index:

        ra = df.loc[idx, ra_column_name]
        dec = df.loc[idx, dec_column_name]
        ra_off = df.loc[idx, ra_offset_column_name]
        dec_off = df.loc[idx, dec_offset_column_name]

        dra, ddec, separation, pos_angle = calculate_offset_parameters(
            ra, dec, ra_off, dec_off, verbosity=verbosity)

        df.loc[idx, 'dra'] = dra.to(u.arcsecond).value
        df.loc[idx, 'ddec'] = ddec.to(u.arcsecond).value
        df.loc[idx, 'separation'] = separation.value
        df.loc[idx, 'pos_angle'] = pos_angle.value

    return df


def calculate_offset_parameters(ra, dec, ra_offset, dec_offset, verbosity=0):

    target_coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg),
                             frame='icrs')
    offset_coords = SkyCoord(ra=ra_offset,
                             dec=dec_offset, unit=(u.deg, u.deg),
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

    return dra, ddec, separations, pos_angles