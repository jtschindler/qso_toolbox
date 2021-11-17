import os
import pandas as pd
from qso_toolbox import utils as ut

targets = pd.read_csv('/Users/schindler/Observations/LBT/MODS/190607-190615/lukas_efficiency_candidates.csv')
offsets = pd.read_csv('lukas_offsets.csv')

os.system('modsProject -p LBTB PS1-QSO-LBTMODS')

os.chdir('./PS1-QSO-LBTMODS')

# Create observation and acquisition scripts
for idx in targets.index:
    target_name = targets.loc[idx,'name']
    target_mag = targets.loc[idx, 'zmag_AB']
    target_priority = targets.loc[idx, 'priority']

    if target_mag <= 20:
        exp_time = 900
    else:
        exp_time = 1200

    os.system('mkMODSObs -o {} -m  red grating -s LS5x60x1.2 -l 1.2 -rfilter '
              'GG495 -e {} -n 1 {}_pr{}'.format(target_name, exp_time,
                                                target_name, target_priority))
    # target_ra_hhmmss = ut.

    os.system("mkMODSAcq -o PSO000p41 -c '00:01:28.396 +41:00:42.54' -g '00:01:28.396 +41:00:42.54' -p +0. -m longslit -a Red -f z_sdss -s LS5x60x1.2 -l 1.2 PSO000p41_pr1")
