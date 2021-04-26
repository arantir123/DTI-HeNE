#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'YY'

from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import sys
import numpy as np
import random
import os
import copy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, auc, precision_recall_fscore_support
from bipart_main import BiNE_training
import homo_main


def run_DTI_pred(args):
    BiNE_training(args)
    R_file = '..\data\{}_file\{}_interactions\{}_admat_dgc_mat_2_line.txt'.format(args.target_type, args.target_type,
                                                                                  args.target_type)
    # "Name of the file containg drug similarties (homogeneous) file names"
    D_sim_file = '..\data\{}_file\{}_homogeneous\{}_D_SimLine_files.txt'.format(args.target_type, args.target_type,
                                                                                args.target_type)
    # "Name of the file containg target similarties (homogeneous) file names"
    T_sim_file = '..\data\{}_file\{}_homogeneous\{}_T_SimLine_files.txt'.format(args.target_type, args.target_type,
                                                                                args.target_type)
    # "Number of neighbors similarity fusion. Default: 3"
    K_SNF = args.K_SNF
    # "Number of iteration for similarity fusion. Default: 10"
    T_SNF = args.T_SNF
    print('================== Start to integrate homogeneous drug and target matrices ====================')
    D_T_, DS_D, DS_T, dDs, dTs = homo_main.run_DDR(R_file, D_sim_file, T_sim_file, K_SNF, T_SNF, args.target_type)
    print('============== Start to predict the probability score of each drug-target pair ================')
    DTI_pred = drug_target_prediction(args=args, D_T_=D_T_, DS_D=DS_D, DS_T=DS_T, dDs=dDs, dTs=dTs, threshold=0.001)

    return DTI_pred


def drug_target_prediction(args, D_T_, DS_D, DS_T, dDs, dTs, threshold=0.001):
    DT = pd.read_csv(args.train_data, sep='\t', header=None)
    vector_u = args.vectors_u
    vector_v = args.vectors_v
    row, col = DT.shape[0], DT.shape[1]
    D, T = {}, {}
    D_name, T_name = set(), set()
    D_total, T_total = {}, {}
    for i in range(row):
        if (DT[0].iloc[i] not in D.keys()):
            D[DT[0].iloc[i]] = []
        D[DT[0].iloc[i]].append(DT[1].iloc[i])
        D_name.update([DT[0].iloc[i]])
    for i in range(row):
        if (DT[1].iloc[i] not in T.keys()):
            T[DT[1].iloc[i]] = []
        T[DT[1].iloc[i]].append(DT[0].iloc[i])
        T_name.update([DT[1].iloc[i]])
    total_DTI = len(D_name) * len(T_name)
    print('total number of drug-target pairs in the DTIs space:', total_DTI)
    for i in list(D_name):
        if (i not in D_total):
            D_total[i] = []
        for j in list(T_name):
            D_total[i].append(j)
    for i in list(T_name):
        if (i not in T_total):
            T_total[i] = []
        for j in list(D_name):
            T_total[i].append(j)
    if (args.type == 'SP'):
        count = 0
        for i in D_name:
            temp = D_total[i]
            for j in D[i]:
                temp.remove(j)
                count += 1
    AUPR_total, AUC_total = [], []

    for i in range(int(args.cycle)):
        if (args.type == 'SP'):
            print('=============== Processing the SP task ==================')
            pos_neg_vector = fold_validation_data_create_for_sp(vector_u=vector_u, vector_v=vector_v,
                                                                vertice_total=D_total, D=D, D_T_=D_T_, DS_D=DS_D,
                                                                DS_T=DS_T, dDs=dDs, dTs=dTs, fold_nums=args.fold,
                                                                concat_type=args.concat)
            all_scores, all_AUPR, all_AUC, all_pair_name = fold_validation_data_split(pos_neg_vector=pos_neg_vector,
                                                                                      trees=args.trees, c=args.c,
                                                                                      fold_nums=args.fold)
            potential_dti = []
            fold_nums = int(args.fold)
            threshold = float(threshold)
            for temp_ in range(fold_nums):
                cal = dict(zip(all_pair_name[temp_], all_scores[temp_]))
                cal_ = []
                for i in cal:
                    if (cal[i] > threshold):
                        cal_.append(i)
                for i in D:
                    for j in D[i]:
                        if (('u' + i, 'i' + j) in cal_):
                            cal_.remove(('u' + i, 'i' + j))
                for i in cal_:
                    potential_dti.append([(i[0][2:], i[1][2:]), cal[i]])
            potential_dti.sort()
            matrix_ = pd.DataFrame(potential_dti)
            if not (matrix_.empty):
                matrix_ = matrix_.sort_values(1, ascending=False)
                DTI_pred = np.array(matrix_)[:, :1][:30]
                print('===== Predicted DTIs with top 30 probability scores =====')
                print(DTI_pred)
            else:
                DTI_pred = np.array(0)
                print('================= No DTIs are predicted =================')

        AUPR_total.append(np.mean(all_AUPR))
        AUC_total.append((np.mean(all_AUC)))
    return DTI_pred


def D_part_calculation(row_label, col_label, drug, target, DS_D, D_T_, concat_type=1):
    row1, col1 = DS_D.shape
    row2, col2 = D_T_.shape
    sim = []
    accum = 0
    for k in range(row2):
        # concat parameter note:
        # scheme 1: multply
        # scheme 2: concatenate
        # scheme 3: addiction
        # nr SP task: concat2
        # gpcr SP task: concat1
        # ic SP task: concat2/concat3
        # e SP task: concat2/concat3
        if (k != row_label):
            # mul=DS_D[row_label][k]*D_T_[k][col_label]
            if (concat_type == 1):
                # scheme 1 (multply)
                temp1 = np.dot(np.dot(drug, DS_D[row_label][k]), D_T_[k][col_label])
                temp1 = np.multiply(temp1, target)
            if (concat_type == 2):
                # scheme 2 (concatenate)
                part1 = np.dot(drug, DS_D[row_label][k])
                part2 = np.dot(target, D_T_[k][col_label])
                temp1 = np.concatenate((part1, part2))
            if (concat_type == 3):
                # scheme 3 (addiction)
                part1 = np.dot(drug, DS_D[row_label][k])
                part2 = np.dot(target, D_T_[k][col_label])
                temp1 = np.array(part1 + part2)
            accum += temp1
    # if(sim!=[]):
    #     max_value=max(sim)
    #     temp2=np.dot(drug,max_value)
    #     maxm=np.multiply(temp2,target)
    return accum


def T_part_calculation(row_label, col_label, drug, target, DS_T, D_T_, concat_type=1):
    row1, col1 = D_T_.shape
    row2, col2 = DS_T.shape
    sim = []
    accum = 0
    for k in range(row2):
        if (k != col_label):
            # mul=D_T_[row_label][k]*DS_T[k][col_label]
            if (concat_type == 1):
                # scheme 1 (multply)
                temp1 = np.dot(np.dot(drug, D_T_[row_label][k]), DS_T[k][col_label])
                temp1 = np.multiply(temp1, target)
            if (concat_type == 2):
                # scheme 2 (concatenate)
                part1 = np.dot(drug, D_T_[row_label][k])
                part2 = np.dot(target, DS_T[k][col_label])
                temp1 = np.concatenate((part1, part2))
            if (concat_type == 3):
                # scheme 3 (addiction)
                part1 = np.dot(drug, D_T_[row_label][k])
                part2 = np.dot(target, DS_T[k][col_label])
                temp1 = np.array(part1 + part2)
            accum += temp1
            # sim.append(mul)
    # if(sim!=[]):
    #     max_value=max(sim)
    #     temp2=np.dot(drug,max_value)
    #     maxm=np.multiply(temp2,target)
    return accum


def fold_validation_data_split(pos_neg_vector, trees, c, fold_nums=10):
    all_scores, all_AUPR, all_AUC, all_pair_name = [], [], [], []
    fold_nums = int(fold_nums)
    counter=0
    for fold_num in range(fold_nums):
        counter+=1
        index = [fold_num]
        # index_remain=set(list_)-set(index)
        temp_test = pos_neg_vector[fold_num]
        temp_train = copy.deepcopy(pos_neg_vector)
        temp_train.pop(fold_num)
        random.shuffle(temp_test)
        random.shuffle(temp_train)
        X_train, Y_train, train_pair_name = [], [], []
        # print(len(temp_train),len(temp_test))
        # len(temp_train)=9,len(temp_test)=32182
        for one_fold in temp_train:
            for i in one_fold:
                X_train.append(i[0])
                Y_train.append(i[1])
                train_pair_name.append((i[2], i[3]))
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_test, Y_test, test_pair_name = [], [], []
        for i in temp_test:
            # print(i[1])
            # if(i[1]==0):
            X_test.append(i[0])
            Y_test.append(i[1])
            test_pair_name.append((i[2], i[3]))
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        print('Start the {}th fold novel DTI prediction'.format(counter))
        scores_testing, AUPR, AUC, test_pair_name = run_DTIpred_one_fold(X_train, Y_train, X_test, Y_test,
                                                                         test_pair_name, trees=trees, c=c,
                                                                         )
        all_scores.append(scores_testing)
        # print('Fold{} AUPR:{:.5f},AUC:{:.5f}'.format(fold_num, AUPR, AUC))
        all_AUPR.append(AUPR)
        all_AUC.append(AUC)
        all_pair_name.append(test_pair_name)

    mean_AUPR = np.mean(all_AUPR)
    mean_AUC = np.mean(all_AUC)
    # print('mean_AUPR:{:.5f},mean_AUC:{:.5f}'.format(mean_AUPR, mean_AUC))
    return all_scores, all_AUPR, all_AUC, all_pair_name


def run_DTIpred_one_fold(X_train, Y_train, X_test, Y_test, test_pair_name, trees, c):
    max_abs_scaler = MaxAbsScaler()
    X_train_maxabs_fit = max_abs_scaler.fit(X_train)
    X_train_maxabs_transform = max_abs_scaler.transform(X_train)
    X_test_maxabs_transform = max_abs_scaler.transform(X_test)

    # model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
    #                          algorithm="SAMME",
    #                          n_estimators=kernel, learning_rate=C)
    # model = SVM.SVC(kernel=kernel,C=C, probability=True, class_weight='balanced',random_state=1357)
    model = RandomForestClassifier(n_estimators=trees, n_jobs=6, criterion=c, class_weight="balanced",
                                   random_state=1357)
    model.fit(X_train_maxabs_transform, Y_train)

    test_prob = model.predict_proba(X_test_maxabs_transform)[:, 1]
    precision, recall, _ = precision_recall_curve(Y_test, test_prob)

    AUPR = auc(recall, precision)
    AUC = roc_auc_score(Y_test, test_prob)

    return test_prob, AUPR, AUC, test_pair_name


def fold_validation_data_create_for_sp(vector_u, vector_v, vertice_total, D, D_T_, DS_D, DS_T, dDs, dTs, fold_nums=10,
                                       concat_type=1):
    vector_u = pd.read_csv(vector_u, sep=' ', index_col=0, header=None)
    vector_v = pd.read_csv(vector_v, sep=' ', index_col=0, header=None)
    negative = []

    for j in vertice_total:
        for k in vertice_total[j]:
            negative.append((j, k))
    random.shuffle(negative)

    testing = np.array_split(negative, int(fold_nums))
    testing = [list(fold) for fold in testing]

    pos_neg = []
    for j in testing:
        temp = []
        for k in j:
            temp.append(tuple(list(k) + [0]))

        for l in D:
            for m in D[l]:
                temp.append((l, m, 1))

        random.shuffle(temp)
        pos_neg.append(temp)

    pos_neg_vector = []
    counter = 0
    for i in pos_neg:
        counter += 1
        print('Start the {}th fold embedding generation'.format(counter))
        temp = []
        for j in i:
            drug = np.array(vector_u.loc[j[0]][:-1])
            target = np.array(vector_v.loc[j[1]][:-1])
            row_label = dDs[j[0][1:]]
            col_label = dTs[j[1][1:]]

            Dpart_avg = D_part_calculation(row_label=row_label, col_label=col_label, drug=drug, target=target,
                                           DS_D=DS_D,
                                           D_T_=D_T_,
                                           concat_type=concat_type)
            Tpart_avg = T_part_calculation(row_label=row_label, col_label=col_label, drug=drug, target=target,
                                           DS_T=DS_T,
                                           D_T_=D_T_,
                                           concat_type=concat_type)
            x = np.concatenate((np.array(Dpart_avg), np.array(Tpart_avg)))
            y = j[2]
            u_name = 'u' + j[0]
            v_name = 'i' + j[1]
            temp.append([x, y, u_name, v_name])
        pos_neg_vector.append(temp)

    return pos_neg_vector


def main():
    parser = ArgumentParser("hetero_DTI_pred",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--train-data',
                        default=r'../rating_train.dat',
                        help='Input bipartite DTI file.')

    parser.add_argument('--model-name', default='default',
                        help='name of model.')

    parser.add_argument('--vectors-u',
                        default=r'../vectors_u.dat',
                        help="file of embedding vectors of drugs")

    parser.add_argument('--vectors-v',
                        default=r'../vectors_v.dat',
                        help="file of embedding vectors of targets")

    parser.add_argument('--ws', default=5, type=int,
                        help='window size.')

    parser.add_argument('--ns', default=4, type=int,
                        help='number of negative samples.')

    parser.add_argument('--d', default=128, type=int,
                        help='embedding size.')

    parser.add_argument('--maxT', default=32, type=int,
                        help='maximal walks per vertex.')

    parser.add_argument('--minT', default=1, type=int,
                        help='minimal walks per vertex.')

    parser.add_argument('--p', default=0.15, type=float,
                        help='walk stopping probability.')

    parser.add_argument('--alpha', default=0.01, type=float,
                        help='trade-off parameter alpha.')

    parser.add_argument('--beta', default=0.01, type=float,
                        help='trade-off parameter beta.')

    parser.add_argument('--gamma', default=0.1, type=float,
                        help='trade-off parameter gamma.')

    parser.add_argument('--lam', default=0.01, type=float,
                        help='learning rate lambda.')

    parser.add_argument('--max-iter', default=50, type=int,
                        help='maximal number of iterations.')

    parser.add_argument('--large', default=0, type=int,
                        help='for large bipartite, 1 do not generate homogeneous graph file; 2 do not generate homogeneous graph')

    parser.add_argument('--mode', default='hits', type=str,
                        help='metrics of centrality')

    parser.add_argument('--type', default='SP', type=str,
                        help='types of DTI prediction')

    parser.add_argument('--fold', default='10', type=str,
                        help='number of CV folds')

    parser.add_argument('--cycle', default='1', type=str,
                        help='number of excecutions of 10-fold CV')

    parser.add_argument('--trees', default=1, type=int,
                        help='number of trees')

    parser.add_argument('--c', default='gini', type=str,
                        help='parameter of split')

    parser.add_argument('--picture', default=0, type=int,
                        help='whether or not ploting the loss curve')

    parser.add_argument('--target_type', default='e', type=str,
                        help='type of the target proteins')

    parser.add_argument('--restart', default=0.7, type=float,
                        help='restart probability of truncated random walks')

    parser.add_argument('--K_SNF', default=3, type=int,
                        help='number of neighbors similarity fusion')

    parser.add_argument('--T_SNF', default=10, type=int,
                        help='number of iteration for similarity fusion')

    parser.add_argument('--concat', default=1, type=int,
                        help='scheme of the drug-target embedding generation')

    # mission: e/gpcr/ic/nr
    mission = 'nr'
    base_root = '..\data\{}_file'.format(mission)
    args = parser.parse_args(['--train-data', os.path.join(base_root, '{}_bipartite_DTI.dat'.format(mission)),
                              '--lam', '0.1',
                              '--max-iter', '100',
                              '--model-name', 'drug_target_{}'.format(mission),
                              '--large', '2',
                              '--gamma', '0.1',
                              '--vectors-u', os.path.join(base_root, '{}_vector_u.dat'.format(mission)),
                              '--vectors-v', os.path.join(base_root, '{}_vector_v.dat'.format(mission)),
                              '--ns', '4',
                              '--ws', '5',
                              '--p', '0.15',
                              '--beta', '0.1',
                              '--type', 'SP',
                              '--fold', '10',
                              '--cycle', '1',
                              '--trees', '100',
                              '--picture', '0',
                              '--restart', '0.7',
                              '--target_type', '{}'.format(mission),
                              '--d', '128',
                              '--concat', '1'
                              ])

    run_DTI_pred(args)

if __name__ == "__main__":
    sys.exit(main())
