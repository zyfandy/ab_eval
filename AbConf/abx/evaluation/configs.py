#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
'''
Four parts:
1. basic variables
2. benchmark definitions and configs for data processing
3. definitions for antibody numbering system
4. optional dependencies for pipelines
'''

# 1. basic variables
PROJ_DIR = os.path.split(__file__)[0]
RENUMBER = os.path.join(PROJ_DIR, 'common', 'renumber.py')
# DockQ 
# IMPORTANT: change it to your path to DockQ project)
DOCKQ_DIR = '/home/data1/cjm/project/DockQ/build/lib.linux-x86_64-cpython-311/DockQ'
# cache directory
CACHE_DIR = os.path.join(PROJ_DIR, '__cache__')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


# 2. configs related to data process
AG_TYPES = ['protein', 'peptide']
RAbD_PDB = ['1a14', '1a2y', '1fe8', '1ic7', '1iqd', '1n8z', '1ncb', '1osp', '1uj3', '1w72', '2adf', '2b2x', '2cmr', '2dd8', '2ghw', '2vxt', '2xqy', '2xwt', '2ypv', '3bn9', '3cx5', '3ffd', '3h3b', '3hi6', '3k2u', '3l95', '3mxw', '3nid', '3o2d', '3rkd', '3s35', '3uzq', '3w9e', '4cmh', '4dtg', '4dvr', '4etq', '4ffv', '4fqj', '4g6j', '4g6m', '4h8w', '4ki5', '4lvn', '4ot1', '4qci', '4xnq', '4ydk', '5b8c', '5bv7', '5d93', '5d96', '5en2', '5f9o', '5ggs', '5hi4', '5j13', '5l6y', '5mes', '5nuz']
Diffab_PDB=['5xku_C_B_A', '7chf_A_B_R', '7chf_H_L_R', '7che_H_L_R', '5tlk_B_A_X', '5tlj_D_C_X', '5tlk_F_E_Y', '5w9h_H_I_G', '5tlj_B_A_X', '5tl5_H_L_A', '7bwj_H_L_E', '7d6i_B_C_A', '8ds5_C_B_A', '5w9h_B_C_A', '7chb_H_L_R', '5w9h_E_F_D', '7che_A_B_R', '5tlk_H_G_Y', '5tlk_D_C_X']
#Diffab_PDB=['5xku', '7chf', '7che', '5tlk', '5tlj', '5tl5', '7bwj', '7d6i', '8ds5', '7chb', '5w9h']

CONTACT_DIST = 6.6  # 6.6 A between one pair of atoms means the two residues are interacting
NUM_INTERFACE_RESIDUES = 48

# 3. antibody numbering, [start, end] of residue id, both start & end are included
# 3.1 IMGT numbering definition
class IMGT:
    # heavy chain
    HFR1 = (1, 26)
    HFR2 = (39, 55)
    HFR3 = (66, 104)
    HFR4 = (118, 129)

    H1 = (27, 38)
    H2 = (56, 65)
    H3 = (105, 117)

    # light chain
    LFR1 = (1, 26)
    LFR2 = (39, 55)
    LFR3 = (66, 104)
    LFR4 = (118, 129)

    L1 = (27, 38)
    L2 = (56, 65)
    L3 = (105, 117)

    Hconserve = {
        23: ['CYS'],
        41: ['TRP'],
        104: ['CYS']
    }

    Lconserve = {
        23: ['CYS'],
        41: ['TRP'],
        104: ['CYS']
    }

    @classmethod
    def renumber(cls, pdb, out_pdb):
        code = os.system(f'python {RENUMBER} {pdb} {out_pdb} imgt 0')
        return code

# 3.2 Chothia numbering definition
class Chothia:
    # heavy chain
    HFR1 = (1, 25)
    HFR2 = (33, 51)
    HFR3 = (57, 94)
    HFR4 = (103, 113)

    H1 = (26, 32)
    H2 = (52, 56)
    H3 = (95, 102)

    # light chain
    LFR1 = (1, 23)
    LFR2 = (35, 49)
    LFR3 = (57, 88)
    LFR4 = (98, 107)

    L1 = (24, 34)
    L2 = (50, 56)
    L3 = (89, 97)

    Hconserve = {
        92: ['CYS']
    }

    Lconserve = {
        88: ['CYS']
    }

    @classmethod
    def renumber(cls, pdb, out_pdb):
        code = os.system(f'python {RENUMBER} {pdb} {out_pdb} chothia 0')
        return code
