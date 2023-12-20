'''
DataLoader for training
'''

import glob, numpy, os, random, soundfile, torch
from scipy import signal
import time

class train_loader_v2(object):
	def __init__(self, train_list, train_path, num_frames, use_full=False, **kwargs):
		print(time.strftime("%m-%d %H:%M:%S") + " use_full[{}]".format(use_full))
		self.train_path = train_path
		self.num_frames = num_frames
		self.use_full = use_full
		# Load data & labels
		self.data_list  = []
		self.data_label = []
		lines = open(train_list).read().splitlines()
		dictkeys = list(set([x.split()[0] for x in lines]))
		dictkeys.sort()
		dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
		for index, line in enumerate(lines):
			speaker_label = dictkeys[line.split()[0]]
			file_name     = os.path.join(train_path, line.split()[1])
			self.data_label.append(speaker_label)
			self.data_list.append(file_name)

	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		audio, sr = soundfile.read(self.data_list[index])	
		if not self.use_full:
			length = self.num_frames * 160 + 240
			if audio.shape[0] <= length:
				shortage = length - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			start_frame = int(audio.shape[0]/2 - length/2) #0
			audio = audio[start_frame:start_frame + length]
		audio = numpy.stack([audio],axis=0)
		return torch.FloatTensor(audio[0]), self.data_label[index]

	def __len__(self):
		return len(self.data_list)
