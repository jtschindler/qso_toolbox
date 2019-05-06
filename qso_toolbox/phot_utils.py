
import numpy as np




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
