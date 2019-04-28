

def decra_to_hms(dra):
    """Convert Right Ascension in deciaml degrees to hours, minutes, seconds.

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
    decl_degrees, dec_minutes = divmod(decl_minutes, 60)
    decl_degrees[~is_negative] = - decl_degrees[~is_negative]

    return decl_degrees, decl_minutes, decl_seconds

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
