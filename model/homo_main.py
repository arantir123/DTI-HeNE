# coding=UTF-8
'''
*************************************************************************
Copyright (c) 2017, Rawan Olayan

>>> SOURCE LICENSE >>>
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation (www.fsf.org); either version 2 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

A copy of the GNU General Public License is available at
http://www.fsf.org/licensing/licenses

>>> END OF LICENSE >>>
*************************************************************************
'''
import sys, random, argparse
from homo_graph_utils import *
from homo_SNF import *
import numpy as np
import os


def split_unkown_interactions(DT, folds=10):
    row, col = DT.shape
    negative = []
    for i in range(row):
        for j in range(col):
            if DT[i][j] == 0:
                negative.append((i, j))

    random.shuffle(negative)

    testing = np.array_split(np.array(negative), folds)
    return [list(fold) for fold in testing]


def get_similarities(sim_file, dMap, target_type):
    sim = []
    base_path='..\data\{}_file\{}_homogeneous'.format(target_type,target_type)
    for line in open(sim_file).readlines():
        edge_list = get_edge_list(os.path.join(base_path,line.strip()))
        sim.append(make_sim_matrix(edge_list, dMap))
    return sim


def run_DDR(R_file, D_sim_file, T_sim_file, K_SNF, T_SNF,target_type):
    # read interaction and similarity files
    (D, T, DT_signature, aAllPossiblePairs, dDs, dTs, diDs, diTs) = get_All_D_T_thier_Labels_Signatures(R_file)
    R = get_edge_list(R_file)
    DT = get_adj_matrix_from_relation(R, dDs, dTs)
    # print D_sim_file
    D_sim = get_similarities(D_sim_file, dDs, target_type)
    T_sim = get_similarities(T_sim_file, dTs, target_type)

    row, col = DT.shape

    # ---------------------- Start DDR functionality ---------------------------------

    labels = mat2vec(DT)
    #test_idx = []

    #folds_features = []
    #for fold in split_unkown_interactions(DT, no_of_splits):

    # -------- infer zero intactions for Drugs and targets---------------
    DT_impute_D = impute_zeros(DT, D_sim[0])
    DT_impute_T = impute_zeros(np.transpose(DT), T_sim[0])

    # -------- construct GIP similarity drugs and targegs ----------------

    GIP_D = Get_GIP_profile(np.transpose(DT_impute_D), "d")
    GIP_T = Get_GIP_profile(DT_impute_T, "t")

    # -------- Perform SNF ----------------------------------------------

    WD = []
    WT = []

    for s in D_sim:
        WD.append(s)
    WD.append(GIP_D)

    for s in T_sim:
        WT.append(s)
    WT.append(GIP_T)

    D_SNF = SNF(WD, K_SNF, T_SNF)
    t_SNF = SNF(WT, K_SNF, T_SNF)


    return D_SNF,t_SNF,dDs, dTs

def run_DDR(R_file, D_sim_file, T_sim_file, K_SNF, T_SNF,target_type):
    # read interaction and similarity files
    (D, T, DT_signature, aAllPossiblePairs, dDs, dTs, diDs, diTs) = get_All_D_T_thier_Labels_Signatures(R_file)
    R = get_edge_list(R_file)
    DT = get_adj_matrix_from_relation(R, dDs, dTs)
    # print D_sim_file
    D_sim = get_similarities(D_sim_file, dDs, target_type)
    T_sim = get_similarities(T_sim_file, dTs, target_type)

    row, col = DT.shape

    # ---------------------- Start DDR functionality ---------------------------------

    labels = mat2vec(DT)
    #test_idx = []

    #folds_features = []
    #for fold in split_unkown_interactions(DT, no_of_splits):

    # -------- infer zero intactions for Drugs and targets---------------
    DT_impute_D = impute_zeros(DT, D_sim[0])
    DT_impute_T = impute_zeros(np.transpose(DT), T_sim[0])

    # -------- construct GIP similarity drugs and targegs ----------------

    GIP_D = Get_GIP_profile(np.transpose(DT_impute_D), "d")
    GIP_T = Get_GIP_profile(DT_impute_T, "t")

    # -------- Perform SNF ----------------------------------------------

    WD = []
    WT = []

    for s in D_sim:
        WD.append(s)
    WD.append(GIP_D)

    for s in T_sim:
        WT.append(s)
    WT.append(GIP_T)

    D_SNF = SNF(WD, K_SNF, T_SNF)
    t_SNF = SNF(WT, K_SNF, T_SNF)

    # --------- Get neigborhood for drugs and target --------------------

    DS_D = FindDominantSet(D_SNF, 5)
    DS_T = FindDominantSet(t_SNF, 5)

    np.fill_diagonal(DS_D, 0)
    np.fill_diagonal(DS_T, 0)

    return DT,DS_D, DS_T, dDs, dTs





