import os
import pandas as pd
from qso_toolbox import utils as ut
from qso_toolbox import catalog_tools as ct

targets = pd.read_csv('/Users/schindler/Observations/LBT/MODS/190607-190615/lukas_efficiency_candidates.csv')

offsets = pd.read_csv('')

# query = 'rMeanPSFMag - rMeanApMag < 0.05 and 10 < zMeanPSFMag < 18'
# offsets = ct.get_offset_stars(targets, 'name', 'ps_ra', 'ps_dec', radius=300,
#                               quality_query=query)
#
# offsets.to_csv('lukas_offsets.csv', index=False)

os.system('modsProject -p LBTB PS1-QSO-LBTMODS')

os.chdir('./PS1-QSO-LBTMODS')

# Create observation and acquisition scripts
coord_list = ut.coord_to_hmsdms(targets['ps_ra'], targets['ps_dec'])

for idx in targets.index:
    target_name = targets.loc[idx,'name']
    target_mag = targets.loc[idx, 'zmag_AB']
    target_priority = targets.loc[idx, 'priority']
    # pos_angle =

    if target_mag <= 20:
        exp_time = 900
    else:
        exp_time = 1200

    make_obs_string = "mkMODSObs -o {} -m  red grating -s LS5x60x1.2 -l 1.2 " \
                      "-rfilter GG495 -e {} -n 1 {}_pr{}".format(target_name,
                                                                 exp_time,
                                                                 target_name,
                                                                 target_priority)
    print(make_obs_string)
    os.system(make_obs_string)

    target_ra_hms = coord_list[idx][0]
    target_dec_dms = coord_list[idx][1]

    make_acq_string = "mkMODSAcq -o {} -c '{} {}' -g '{} {}' -p {} -m " \
                      "longslit -a Red -f z_sdss -s LS5x60x1.2 -l 1.2 {}_pr{}".format(target_name,
                                                 target_ra_hms,
                                                 target_dec_dms,
                                                 target_ra_hms,
                                                 target_dec_dms,
                                                 pos_angle,
                                                 target_name,
                                                 target_priority)
    print(make_acq_string)
    os.system(make_acq_string)

# Create the blind offset acquistion scripts
for idx in targets.index:
    target_name = targets.loc[idx,'name']
    target_priority = targets.loc[idx, 'priority']

    acq_filename = '{}_pr{}.acq'.format(target_name, target_priority)

    blind_acq_filename = '{}_pr{}_blind.acq'.format(target_name,
                                                    target_priority)

    target_offsets = offsets.query('target_name=="{}"'.format(target_name))

    if target_offsets.shape[0] > 0 :

        # Take first offset
        dra = target_offsets.loc[target_offsets.index[0], 'dra_offset']
        ddec = target_offsets.loc[target_offsets.index[0], 'ddec_offset']

        file = open('./{}'.format(acq_filename), 'r')
        file_lines = file.readlines()[:-2]
        file.close()
        new_acq_file = open('./{}'.format(blind_acq_filename), 'w')

        for line in file_lines:
            new_acq_file.write(line)

        new_acq_file.write("  PAUSE\n")
        new_acq_file.write("  syncoffset\n")
        new_acq_file.write("  PAUSE\n")
        new_acq_file.write("  OFFSET {} {} rel\n".format(dra, ddec))
        new_acq_file.write("  UPDATEPOINTING\n")
        new_acq_file.write("  SlitGO\n")
        new_acq_file.write("  PAUSE\n")
        new_acq_file.write("\n")
        new_acq_file.write("end\n")

        new_acq_file.close()
