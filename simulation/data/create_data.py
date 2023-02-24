"""
Author: Catia Fortunato
Create pytorch dataset from experimental data.

Create dataset (dict) for movement training.
The output dimensions are uncorrelated in this implementation.
The output has time components.
The target and the go cue vary from trial to trial.
Cue is based on preparatory activity from monkey trials. 

Structure (keys):
---------
    - target_id: ntrials x 1
    - target_output: ntrials x time x noutputdimensions
    - params
    - stimulus ntrials x time x ninput dimensions
"""

import pandas as pd
import numpy as np
import random

import os
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

from pyaldata import *

def add_padding(a):
    b=np.zeros((400,2))
    b[:,0] = np.pad(a[:,0],(210,0), mode='constant', constant_values = (a[0,0]))
    b[:,1] = np.pad(a[:,1],(210,0), mode='constant', constant_values = (a[0,1]))
    return b

def remove_offset(a):
    a[210:310,0] = a[0,0]
    a[210:310,1] = a[0,1]

    return a

def get_target_id(trial):
    return int(np.round((trial.target_direction + np.pi) / (0.25*np.pi))) - 1

def create_inputs(prep_act, go_onset, cue_onset, hold):
    input = np.zeros((400,15))

    input[:int(go_onset),0] = hold
    input[int(cue_onset):,1:] = prep_act

    return input


data_dir = '/home/cf620/Downloads/Chewie/'
file_name = 'Chewie_CO_CS_2016-10-14.mat'

df = mat2dataframe(data_dir+file_name, shift_idx_fields=True)

td = select_trials(df, "result == 'R'")
td = select_trials(td, "epoch == 'BL'")
td = add_firing_rates(td, 'smooth')

move_td = restrict_to_interval(td, "idx_movement_on", rel_start=-100, rel_end=89)

move_td = transform_signal(move_td,'pos','center')

move_td['target_output'] = [add_padding(move_td.loc[i, 'pos']) for i in move_td.index]

move_td['target_output'] = [remove_offset(move_td.loc[i, 'target_output']) for i in move_td.index]

prep_td = move_td

prep_td =combine_time_bins(prep_td,3)

prep_td = restrict_to_interval(prep_td, "idx_movement_on", rel_start=-7, rel_end=14)

from sklearn.decomposition import PCA, FactorAnalysis

pca_dims = 14

prep_td = dim_reduce(prep_td, PCA(pca_dims), "M1_rates", "PMd_pca")


prep_td["target_id"] = prep_td.apply(get_target_id, axis=1)

prep_td=transform_signal(prep_td,'PMd_pca','center_normalize')

prep_td['stimulus'] = [np.mean(prep_td.loc[i,'PMd_pca'], axis=0) for i in prep_td.index]

prep_td['idx_go_cue'] = [move_td.loc[i,'idx_go_cue']+160 for i in prep_td.index]

prep_td['idx_movement_on'] = [move_td.loc[i,'idx_movement_on']+210 for i in prep_td.index]

prep_td = prep_td[prep_td['idx_go_cue'].notna()]
print(len(prep_td))
prep_td['input'] = [create_inputs(prep_td.loc[i,'stimulus'],prep_td.loc[i,'idx_movement_on'],prep_td.loc[i,'idx_go_cue'],1) for i in prep_td.index]

print(len(prep_td))

ntargets = len(np.unique(prep_td['target_id']))
tsteps = 400
ntrials = len(prep_td)
output_dim = 2
input_dim = 15
dt = 0.01 #sec

params =  {'ntargets':ntargets,
          'tsteps':tsteps,
          'ntrials':ntrials,
          'output_dim':output_dim,
          'input_dim':input_dim,
          'dt':dt, 
          'use_velocities':False}


stimulus = np.zeros((ntrials, tsteps,input_dim))
target = np.zeros((ntrials,tsteps, output_dim))
go_onset = np.zeros(ntrials)
cue_onset = np.zeros(ntrials)
target_id = np.zeros(ntrials)

k=0
for i in prep_td.index:
    stimulus[k,:,:] = prep_td.loc[i,'input']
    target[k,:,:] = move_td.loc[i,'target_output']
    go_onset[k] = prep_td.loc[i,'idx_movement_on']
    cue_onset[k] = prep_td.loc[i,'idx_go_cue']
    target_id[k] = prep_td.loc[i, 'target_id']
    k+=1

test_idx = random.sample(range(ntrials), 200)

train_idx = np.setdiff1d(range(ntrials), test_idx)

test_set = {'params':params,
       'target':target[test_idx],
       'stimulus':stimulus[test_idx],
       'go_onset':go_onset[test_idx],
       'cue_onset':cue_onset[test_idx],
       'target_id':target_id[test_idx],
       'target_param':target_id[test_idx],
       }

dic = {'params':params,
       'target':target[train_idx],
       'stimulus':stimulus[train_idx],
       'go_onset':go_onset[train_idx],
       'cue_onset':cue_onset[train_idx],
       'target_id':target_id[train_idx],
       'target_param':target_id[train_idx],
       'test_set1':test_set,
       }

savdir = '/home/cf620/git/base_centerout_rnn/simulation/data/'
savname='Chewie_CO_CS_2016-10-14_position_30ms'

np.save(savdir+savname,dic)
