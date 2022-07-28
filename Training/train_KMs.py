# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from dqn.agent import CarRacingDQN
import os
import tensorflow as tf
import gym
import _thread
import re
import sys

import h5py

from matplotlib import pyplot as plt

import numpy as np

import pickle as pkl

from collections import Counter
from sklearn.utils import resample

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import itertools

import glob
import time as time

track_type = 'Cycle'

if track_type == 'Square':
    checkpoint_path = 'data/square_20210608'
elif track_type == 'Ellipse':
    checkpoint_path = 'data/ellipse_20210608'
elif track_type == 'ZigZag':
    checkpoint_path = 'data/zigzag_20210608'
elif track_type == 'Cycle':
    checkpoint_path = '/home/dahlin/Desktop/Car Racing/DQN/data/checkpoint_20210225_num_frame_stack_1_frame_skip_3/Checkpoints'


# %%
if track_type != 'Combine':
    #rollout_files = sorted(glob.glob("data/checkpoint02/dqn*.h5"),key=os.path.getmtime)
    rollout_files = sorted(glob.glob(os.path.join(checkpoint_path,'dqn*.h5')),key=os.path.getmtime)
    print('Loading '+rollout_files[-1])
    f = h5py.File(rollout_files[-1],'r')
    actions_rollout = f['actions_rollout'][:]
    frames_rollout = f['frames_rollout'][:]
    f.close()
else:
    actions_rollout = []
    frames_rollout = []
    for checkpoint_path in ['data/square_20210608','data/ellipse_20210608','data/zigzag_20210608']:
        rollout_files = sorted(glob.glob(os.path.join(checkpoint_path,'dqn*.h5')),key=os.path.getmtime)
        print('Loading '+rollout_files[-1])
        f = h5py.File(rollout_files[-1],'r')
        actions_rollout.extend(f['actions_rollout'][:])
        actions_rollout.extend(f['actions_rollout'][:])
        frames_rollout.extend(f['frames_rollout'][:])
        frames_rollout.extend(f['frames_rollout'][:])
        actions_rollout.extend(f['actions_rollout'][:])
        actions_rollout.extend(f['actions_rollout'][:])
        frames_rollout.extend(f['frames_rollout'][:])
        frames_rollout.extend(f['frames_rollout'][:])
        f.close()
    actions_rollout = np.array(actions_rollout)
    frames_rollout = np.array(frames_rollout)

print(actions_rollout.shape)
print(frames_rollout.shape)


# %%
print('actions rollout shape: '+str(actions_rollout.shape))
print('frames rollout shape: '+str(frames_rollout.shape))

hist,bin_edges = np.histogram(actions_rollout,bins=[0, 1, 2, 3,4,5,6,7,8,9,10,11,12])
plt.bar(bin_edges[:-1], hist, width = 0.5, color='#0504aa',alpha=0.7)

actions_rollout,frames_rollout = resample(actions_rollout,frames_rollout,replace=False)
#frames_rollout = resample(frames_rollout,replace=False)

print('actions rollout shape: '+str(actions_rollout.shape))
print('frames rollout shape: '+str(frames_rollout.shape))

# hold out training samples for validation
#valid_frac = 0.2
valid_frac = 0.1
num_valid = int(np.round(len(actions_rollout)*valid_frac))
num_train = int(np.round(len(actions_rollout)*(1-valid_frac)))
frames_valid, actions_valid = frames_rollout[-num_valid:], actions_rollout[-num_valid:]
frames_train, actions_train = frames_rollout[:-num_valid], actions_rollout[:-num_valid]
print(frames_valid.shape,actions_valid.shape,frames_train.shape,actions_train.shape)


# %%
n_classes = 12

# convert labels to 1-hot vectors
#actions_train = tf.keras.utils.to_categorical(actions_train, n_classes)
#actions_valid = tf.keras.utils.to_categorical(actions_valid, n_classes)

## normalize inputs and cast to float
#frames_train = (frames_train / np.max(frames_train)).astype(np.float32)
#frames_valid = (frames_valid / np.max(frames_valid)).astype(np.float32)

frames_train_flat = frames_train.reshape((frames_train.shape[0], -1))
frames_valid_flat = frames_valid.reshape((frames_valid.shape[0], -1))

scaler = StandardScaler()
frames_train_scaled = scaler.fit_transform(frames_train_flat)


# %%
if track_type != 'Combine':
    savepath = checkpoint_path+'/Students/KM/KM_frames_'+str(num_train)+'_train_points'
else:
    savepath = 'data/Combine/Students/KM/KM_frames_'+str(num_train)+'_train_points'
if not os.path.exists(savepath):
    os.makedirs(savepath)
with open(os.path.join(savepath,'scaler.pkl'),'wb') as f:
    pkl.dump(scaler,f)

C_range = [0.1, 1, 10]
gamma_range = [0.1, 1, 10]

C_gamma_grid = itertools.product(C_range,gamma_range)
C_gamma_list = [(C,gamma) for (C,gamma) in itertools.product(C_range,gamma_range)]

#clfs = []
train_times = []
val_times = []
val_scores = []

for i,(C,gamma) in enumerate(C_gamma_grid):
    print('-----------------------------------')
    print('C = '+str(C)+' Gamma = '+str(gamma))
    start = time.time()
    
    clf = SVC(C=C,gamma=gamma, class_weight='balanced',cache_size=8000)
    clf.fit(frames_train_scaled, actions_train)
    end = time.time()
    train_time = end-start
    train_times.append(train_time)
    print('Training time: '+str(train_time)+' s')
    start = time.time()
    val_score = clf.score(scaler.transform(frames_valid_flat),actions_valid)
    val_scores.append(val_score)
    print('Val score: '+str(val_score))
    end = time.time()
    val_times.append(end-start)
    print('Scoring time: '+str(end-start)+' s')
    print('-----------------------------------')

    with open(os.path.join(savepath,'model_'+str(i)+'.pkl'),'wb') as f:
        pkl.dump(clf,f) 


# %%
print('Total training time: '+str(np.sum(train_times)))
print('Total val time: '+str(np.sum(val_times)))

best = np.argmax(val_scores)
print('Best val parameters: C: '+str(C_gamma_list[best][0])+' gamma: '+str(C_gamma_list[best][1]),' Val score: ',val_scores[best])

with open(os.path.join(savepath,track_type+'_performance_data.pkl'),'wb') as f:
    pkl.dump([train_times,val_times,val_scores,C_gamma_list],f)


# %%
results_file = os.path.join(savepath,track_type+'_performance_data.pkl')
#if type(experiments)==str:
#    results = os.path.join(experiments,'performance.pkl')
with open(results_file,'rb') as f:
    [train_times,val_times,val_scores,C_gamma_list] = pkl.load(f)

sort_by_val_score = True

if sort_by_val_score:
    val_scores_sort = np.sort(val_scores)
    sort_args = np.argsort(val_scores)
    C_gammas_sorted = [C_gamma_list[i] for i in sort_args]
    C_gamma_str = [str(C_gamma) for C_gamma in C_gammas_sorted]
    plt.figure(dpi=200)
    plt.stem(val_scores_sort)
else:
    plt.figure(dpi=200)
    plt.stem(val_scores)
    C_gamma_str = [str(C_gamma) for C_gamma in C_gamma_list]

plt.xticks(ticks=np.arange(len(C_gamma_str)),labels=C_gamma_str,rotation='45')
plt.grid()
plt.xlabel('(C, gamma)')
plt.ylabel('Validation Accuracy')
plt.ylim((0,1))
if track_type == 'Square':
    plt.title('Square Track KMs')
elif track_type == 'Ellipse':
    plt.title('Elliptical Track KMs')
elif track_type == 'ZigZag':
    plt.title('ZigZag Track KMs')
elif track_type == 'Cycle':
    plt.title('Cycled Tracks KMs')
plt.tight_layout()
plt.savefig(os.path.join(savepath,track_type+'_accuracy_plot.jpg'))


# %%
#savepath = 'hard_trees_features'+str(num_data_points)
if track_type != 'Combine':
    savepath = checkpoint_path+'/Students/HDT/hard_trees_frames_'+str(num_train)+'_train_points'
else:
    savepath = 'data/Combine/Students/HDT/hard_trees_frames_'+str(num_train)+'_train_points'

if not os.path.exists(savepath):
    os.makedirs(savepath)

train_accs = []
val_accs = []

for depth in range(2,21):

    print('Tree Depth: '+str(depth))

    student = DecisionTreeClassifier(max_depth=depth,class_weight='balanced')
    student.fit(frames_train_flat,actions_train)

    train_acc=student.score(frames_train_flat,actions_train)
    val_acc=student.score(frames_valid_flat,actions_valid)

    print('Train accuracy: '+str(train_acc))
    print('Valid accuracy: '+str(val_acc))

    train_accs.append(train_acc)
    val_accs.append(val_acc)

    list_pickle = open(savepath+'/'+'hard_tree_depth_'+str(depth)+'_balanced.pkl','wb')
    pickle.dump(student,list_pickle)
    list_pickle.close()

with open(os.path.join(savepath,'accuracy_data.pkl'),'wb') as f:
    pickle.dump([train_accs,val_accs],f)


# %%
plt.figure(dpi=800)
tree_depths = np.array(range(2,21))
plt.plot(tree_depths,np.array(train_accs)*100,label='Training Set Accuracy')
plt.plot(tree_depths,np.array(val_accs)*100,label='Validation Set Accuracy')
#plt.plot(tree_depths,np.array(train_accs),label='Training Set Accuracy')
#plt.plot(tree_depths,np.array(val_accs),label='Validation Set Accuracy')
plt.xticks(tree_depths)
plt.xlabel('Tree Depth')
plt.ylabel('% Accuracy')
plt.xlim((2,20))
plt.ylim((0,100))
plt.legend()
plt.grid()
if track_type == 'Square':
    plt.title('Square Track HDTs')
elif track_type == 'Ellipse':
    plt.title('Elliptical Track HDTs')
elif track_type == 'ZigZag':
    plt.title('ZigZag Track HDTs')
elif track_type == 'Combine':
    plt.title('Combined Tracks HDTs')
fig_savepath = os.path.join(savepath,track_type+'_accuracy_plot.jpg')
plt.savefig(fig_savepath)


# %%



