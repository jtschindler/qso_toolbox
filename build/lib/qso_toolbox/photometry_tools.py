
import numpy as np


def flux_to_ab(flux, unit='Jy'):

    if unit == "Janksy" or unit == "Jy":
        return -5./2.*np.log10(flux/3631.)  # AB magnitude
    else:
        raise NotImplementedError


def ab_to_flux(ab_mag, unit='Jy'):

    if unit == "Janksy" or unit == "Jy":
        return np.power(10, -0.4*ab_mag)*3631.  # flux in Jy
    else:
        raise NotImplementedError


def ab_err_to_flux_err(ab_mag, ab_magerr, unit='Jy'):

    if unit == "Janksy" or unit == "Jy":
        return abs(-0.4*np.log(10) *
                   ab_magerr * np.power(10, -0.4*ab_mag) * 3631)
    else:
        raise NotImplementedError


def asinh_mag_to_flux(mag, band, unit='Jy'):

    if band == 'SDSS_u':
        b = 1.4e-10
    elif band == 'SDSS_g':
        b = 0.9e-10
    elif band == 'SDSS_r':
        b = 1.2e-10
    elif band == 'SDSS_i':
        b = 1.8e-10
    elif band == 'SDSS_z':
        b = 7.4e-10
    else:
        print ('Unrecognized photometric band')
        print ('Conversion unsuccessful!')
        return -1

    if unit == "Janksy" or unit == "Jy":
        f_f_0 = np.sinh((-mag) / (2.5/np.log(10)) - np.log(b))*2.*b
        return 3631 * f_f_0
    else:
        raise NotImplementedError


def asinh_mag_err_to_flux_err(mag, mag_err, band, unit='Jy'):

    if band == 'SDSS_u':
        b = 1.4e-10
    elif band == 'SDSS_g':
        b = 0.9e-10
    elif band == 'SDSS_r':
        b = 1.2e-10
    elif band == 'SDSS_i':
        b = 1.8e-10
    elif band == 'SDSS_z':
        b = 7.4e-10
    else:
        print('Unrecognized photometric band')
        print('Conversion unsuccessful!')
        return -1

    if unit == "Janksy" or unit == "Jy":
        return abs(2*b*np.cosh((-mag) / (2.5/np.log(10))
                   - np.log(b)) * ((-mag_err) / (2.5/np.log(10)))*3631)
    else:
        raise NotImplementedError


def deredden(to_deredden, band, ext, ext_band, input_type="AB"):
    """

    extinction at a given band A_i
    extinction coefficient E(B-V)
    relative to absolute extinction coefficient R_i

    R_i = A_i / E(B-V)

    A_i = E(B-V) * R_i


    :param to_deredden:
    :param band:
    :param ext:
    :param ext_band:
    :param input_type:
    :return:
    """
    ext_deltamag_dict = \
        {'A_V': 3.1,
         'SDSS_u': 4.239,
         'SDSS_g': 3.303,
         'SDSS_r': 2.285,
         'SDSS_i': 1.698,
         'SDSS_z': 1.263,
         'TMASS_j': 0.709,
         'TMASS_h': 0.449,
         'TMASS_k': 0.302,
         'WISE_w1': 0.189,
         'WISE_w2': 0.146,
         'WISE_w3': 0.0,
         'WISE_w4': 0.0,
         'UNWISE_w1': 0.189,
         'UNWISE_w2': 0.146,
         'PS_g': 3.172,
         'PS_r': 2.271,
         'PS_i': 1.682,
         'PS_z': 1.322,
         'PS_y': 1.087,
         'VHS_z': 1.395,
         'VHS_y': 1.017,
         'VHS_j': 0.705,
         'VHS_h': 0.441,
         'VHS_k': 0.308
         }

    extinction = ext/ext_deltamag_dict[ext_band]*ext_deltamag_dict[band]

    if input_type == "AB":
        return to_deredden - extinction
    if input_type == "Vega":
        return to_deredden - extinction
    if input_type == "flux":
        return to_deredden / np.power(10, -0.4*extinction)
    if input_type == "asinh":
        raise NotImplementedError


def vega_to_ab(mag, band_name, output_flux=False, asinh_mag=False, flux_unit='Jy'):
    mag_corr_dict = {'SDSS_u': -0.04,
                     'SDSS_g': 0.0,
                     'SDSS_r': 0.0,
                     'SDSS_i': 0.0,
                     'SDSS_z': 0.02,
                     'TMASS_j': 0.894,
                     'TMASS_h': 1.374,
                     'TMASS_k': 1.84,
                     'WISE_w1': 2.699,
                     'WISE_w2': 3.339,
                     'WISE_w3': 5.174,
                     'WISE_w4': 6.62,
                     'UNWISE_w1': 2.699,
                     'UNWISE_w2': 3.339,
                     'PS_g': 0.0,
                     'PS_r': 0.0,
                     'PS_i': 0.0,
                     'PS_z': 0.0,
                     'PS_y': 0.0,
                     'VHS_Z': 0.502,
                     'VHS_Y': 0.600,
                     'VHS_J': 0.916,
                     'VHS_H': 1.366,
                     'VHS_K': 1.827,
                     'unwise-neo3_w1': 2.699,
                     'unwise-neo3_w2': 3.339,
                     'desdr1_g': 0.0,
                     'desdr1_r': 0.0,
                     'desdr1_i': 0.0,
                     'desdr1_z': 0.0,
                     'desdr1_Y': 0.0
                     }

    if output_flux:
        if asinh_mag:
            # First, conversion of asinh magnitudes to fluxes
            flux = asinh_mag_to_flux(mag, band_name, unit=flux_unit)
            # Second, calculation of the flux correction in the AB system
            ab_flux_correction = np.power(10, -0.4*mag_corr_dict[band_name])
            # Return of the product of flux and correction
            return flux*ab_flux_correction
        else:
            ab_mag = mag + mag_corr_dict[band_name]
            return ab_to_flux(ab_mag, unit=flux_unit)

    else:
        if asinh_mag:
            print ("This operation is not supported.")
        else:
            return mag+mag_corr_dict[band_name]


# ------------------------------------------------------------------------------
#  Functions to integrate and generalize
# ------------------------------------------------------------------------------


def flux_to_magnitude(flux, survey):
    """

    :param flux:
    :param survey:
    :return:
    """
    if survey == "desdr1":
        zpt = 30.
    else:
        raise ValueError("Survey name not recgonized: {}".format(survey))

    return -2.5 * np.log10(flux) + zpt


def mag_err(noise_flux_ratio, verbose=True):
    '''
    Calculates the magnitude error from the input noise_flux_ratio
    which is basically the inverse of the Signal-to-Noise ratio
    '''
    err = (2.5 / np.log(10)) * noise_flux_ratio
    if verbose:
        print(err)
    return err

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

    return mag,err