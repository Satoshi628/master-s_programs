#coding: utf-8
#----- 標準ライブラリ -----#
import math
import os
#----- 専用ライブラリ -----#
import h5py
import numpy as np
import matplotlib.pyplot as plt

#----- 自作ライブラリ -----#

############## main ##############
if __name__ == '__main__':
    
    ours_path = "/mnt/kamiya/code/my_exp/frame_interpolation_track/outputs/GFP_new/0/result/track.txt"
    # ours_path = "/mnt/kamiya/code/my_exp/frame_interpolation_track/outputs/RECEPTOR/0/result/track.txt"
    ours_track = np.loadtxt(ours_path, delimiter=",")
    _, ours_length = np.unique(ours_track[:,1], return_counts=True)

    MPM_path = "/mnt/kamiya/code/my_exp/frame_interpolation_track/track_MPM.npy"
    # MPM_path = "/mnt/kamiya/code/my_exp/My_MPM2/outputs/RECEPTOR/0/result/track.npy"
    MPM_track = np.load(MPM_path)
    MPM_track = MPM_track[MPM_track[:,2] != -2]
    _, MPM_length = np.unique(MPM_track[:,1], return_counts=True)


    PTGT_path = "/mnt/kamiya/code/my_exp/frame_interpolation_track/track_PTGT.npy"
    # PTGT_path = "/mnt/kamiya/code/my_exp/TATR3D/outputs/RECEPTOR/0/track.npy"
    PTGT_track = np.load(PTGT_path)
    PTGT_track = PTGT_track[PTGT_track[:,2] != -2]
    _, PTGT_length = np.unique(PTGT_track[:,1], return_counts=True)

    print(PTGT_track.shape)
    
    GT_path = "/mnt/kamiya/dataset/OIST2/GFP_v9_Mid/10/det.txt"
    GT_track = np.loadtxt(GT_path, delimiter=",")
    # GT_track = h5py.File("/mnt/kamiya/dataset/PTC/RECEPTOR/Cell_Point_Annotation.hdf5", mode='r')["Sample5"][...]
    _, GT_length = np.unique(GT_track[:,1], return_counts=True)

    
    bins = 20
    fig, ax = plt.subplots()
    ax.hist(ours_length, bins=bins, color="blue", label="ours", alpha=0.3, density=True)
    ax.hist(GT_length, bins=bins, color="red", label="GT",alpha=0.3, density=True)
    # ax.legend()
    # ax.set_xlim(0, 300)
    # ax.set_ylim(0, 0.05)
    # plt.savefig("hist_ours", bbox_inches="tight")
    # plt.close()

    # fig, ax = plt.subplots()
    ax.hist(MPM_length, bins=bins, color="green", label="MPM", alpha=0.3, density=True)
    # ax.hist(GT_length, bins=bins, color="red", label="GT",alpha=0.3, density=True)
    # ax.legend()
    # ax.set_xlim(0, 300)
    # ax.set_ylim(0, 0.05)
    # plt.savefig("hist_MPM", bbox_inches="tight")
    # plt.close()

    # fig, ax = plt.subplots()
    ax.hist(PTGT_length, bins=bins, color="orange", label="PTGT", alpha=0.3, density=True)
    # ax.hist(GT_length, bins=bins, color="red", label="GT",alpha=0.3, density=True)
    ax.legend()
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 0.05)
    plt.savefig("hist", bbox_inches="tight")
    plt.close()