import os
import sys
import numpy as np
import random
import pickle
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from layers.graph import Graph

from data_process import generate_data

import time


class Feeder(torch.utils.data.Dataset):
	""" Feeder for skeleton-based action recognition
	Arguments:
		data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
	"""

	def __init__(self, data_path, graph_args={}, train_val_test='train'):
		'''
		train_val_test: (train, val, test)
		'''
		self.data_path = data_path
		self.path_list = sorted(glob.glob(os.path.join(self.data_path,'*.txt')))
		self.all_feature=[]
		self.all_adjacency=[]
		self.all_mean_xy=[]
		self.it = 0
		#self.load_data()
		#total_num = len(self.all_feature)
		# equally choose validation set
		self.feature_num=0
		self.prev=0
		#train_id_list = list(np.linspace(0, total_num-1, int(total_num*0.8)).astype(int))
		#val_id_list = list(set(list(range(total_num))) - set(train_id_list))

		# # last 20% data as validation set

		self.train_val_test = train_val_test
		"""
		if train_val_test.lower() == 'train':
			self.all_feature = self.all_feature[train_id_list]
			self.all_adjacency = self.all_adjacency[train_id_list]
			self.all_mean_xy = self.all_mean_xy[train_id_list]
		elif train_val_test.lower() == 'val':
			self.all_feature = self.all_feature[val_id_list]
			self.all_adjacency = self.all_adjacency[val_id_list]
			self.all_mean_xy = self.all_mean_xy[val_id_list]
		"""
		self.graph = Graph(**graph_args) #num_node = 120,max_hop = 1

	#def load_data(self,path):
	#	all_feature, all_adjacency, all_mean_xy = generate_data(path)
		#with open(self.data_path, 'rb') as reader:
			# Training (N, C, T, V)=(5010, 11, 12, 120), (5010, 120, 120), (5010, 2)
		#	[self.all_feature, self.all_adjacency, self.all_mean_xy]= pickle.load(reader)
			

	def __len__(self):
		return 24*(len(self.path_list))

	def __getitem__(self, idx):
		# C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]
		#if(idx>=self.feature_num):
		try:
			now_feature = self.all_feature[idx-self.prev].copy()
		except:
			path = self.path_list[self.it]
			self.it = self.it+1
			#self.all_feature, self.all_adjacency, self.all_mean_xy = generate_data(path)
			while True:
				try:
					self.all_feature =[]
					self.all_feature, self.all_adjacency, self.all_mean_xy = generate_data(path)
					if(len(self.all_feature)>0):
						break
				except:
					self.it = self.it+1
					path = self.path_list[self.it]
			self.prev = self.feature_num
			self.feature_num = self.feature_num+len(self.all_feature)	
			now_feature = self.all_feature[idx-self.prev].copy()
		 # (C, T, V) = (11, 12, 120)
		now_mean_xy = self.all_mean_xy[idx-self.prev].copy() # (2,) = (x, y) 

		if self.train_val_test.lower() == 'train' and np.random.random()>0.5:
			angle = 2 * np.pi * np.random.random()
			sin_angle = np.sin(angle)
			cos_angle = np.cos(angle)

			angle_mat = np.array(
				[[cos_angle, -sin_angle],
				[sin_angle, cos_angle]])

			xy = now_feature[3:5, :, :]
			num_xy = np.sum(xy.sum(axis=0).sum(axis=0) != 0) # get the number of valid data

			# angle_mat: (2, 2), xy: (2, 12, 120)
			out_xy = np.einsum('ab,btv->atv', angle_mat, xy)
			now_mean_xy = np.matmul(angle_mat, now_mean_xy)
			xy[:,:,:num_xy] = out_xy[:,:,:num_xy]

			now_feature[3:5, :, :] = xy

		now_adjacency = self.graph.get_adjacency(self.all_adjacency[idx-self.prev])
		now_A = self.graph.normalize_adjacency(now_adjacency)
		
		return now_feature, now_A, now_mean_xy

