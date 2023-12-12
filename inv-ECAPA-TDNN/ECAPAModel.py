'''
This part is used to train the speaker model and evaluate the performances
'''

import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN

class ECAPAModel(nn.Module):
	def __init__(self, lr, lr_decay, C , n_class, m, s, test_step, **kwargs):
		super(ECAPAModel, self).__init__()
		## ECAPA-TDNN
		self.speaker_encoder = ECAPA_TDNN(C = C).cuda()
		## Classifier
		self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).cuda()
		self.n_class = n_class

		self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

	def train_network(self, epoch, loader):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		index, top1, loss = 0, 0, 0
		lr = self.optim.param_groups[0]['lr']
		for num, (data, labels) in enumerate(loader, start = 1):
			self.zero_grad()
			labels            = torch.LongTensor(labels).cuda()
			logmel_data       = self.speaker_encoder.logmel(data.cuda(), aug = True)
			speaker_embedding = self.speaker_encoder.forward(logmel_data)
			nloss, prec       = self.speaker_loss.forward(speaker_embedding, labels)			
			nloss.backward()
			self.optim.step()
			index += len(labels)
			top1 += prec
			loss += nloss.detach().cpu().numpy()
			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
			" Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
			sys.stderr.flush()
		sys.stdout.write("\n")
		return loss/num, lr, top1/index*len(labels)

	def cohorts(self, loader, use_full):
		self.eval()
		import numpy as np

		if not use_full:
			c = np.zeros((self.n_class, 192))
			c_num = np.zeros((self.n_class))
	
			for num, (data, labels) in enumerate(loader, start = 1):
				with torch.no_grad():
					labels            = torch.LongTensor(labels).cuda()
					logmel_data       = self.speaker_encoder.logmel(data.cuda(), aug = False)
					speaker_embedding = self.speaker_encoder.forward(logmel_data)
					embeddings = F.normalize(speaker_embedding, p=2, dim=1)
					#embeddings = speaker_embedding

				for embedding, label in zip(embeddings, labels):
					c[label] += embedding.cpu().numpy()
					c_num[label] += 1

				sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
				" Done: %.2f%% \r"    %(100 * (num / loader.__len__())))
				sys.stderr.flush()
			sys.stdout.write("\n")

			c /= c_num.reshape((self.n_class, 1))
			return c

		c = np.zeros((self.n_class, 6, 192))
		c_num = np.zeros((self.n_class))
			
		for num, (data, labels) in enumerate(loader, start = 1):
			# Full utterance
			data_1 = data.cuda()

			# Spliited utterance matrix
			audio = np.squeeze(data.detach().cpu().numpy(), 0)
			max_audio = 300 * 160 + 240
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			feats = []
			startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
			for asf in startframe:
				feats.append(audio[int(asf):int(asf)+max_audio])
			feats = numpy.stack(feats, axis = 0).astype(numpy.float)
			data_2 = torch.FloatTensor(feats).cuda()

			with torch.no_grad():
				logmel_data_1 = self.speaker_encoder.logmel(data_1, aug = False)
				embedding_1 = self.speaker_encoder.forward(logmel_data_1)
				embedding_1s = F.normalize(embedding_1, p=2, dim=1)
				logmel_data_2 = self.speaker_encoder.logmel(data_2, aug = False)
				embedding_2 = self.speaker_encoder.forward(logmel_data_2)
				embedding_2s = F.normalize(embedding_2, p=2, dim=1)

			embedding_2s = embedding_2s.unsqueeze(0)
			for embedding_1, embedding_2, label in zip(embedding_1s, embedding_2s, labels):
				c[label][0] += embedding_1.cpu().numpy()
				c[label][1:] += embedding_2.cpu().numpy()
				c_num[label] += 1

			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" Done: %.2f%% \r"    %(100 * (num / loader.__len__())))
			sys.stderr.flush()
		sys.stdout.write("\n")

		c /= c_num.reshape((self.n_class, 1, 1))
		return c

	def eval_network(self, eval_list, eval_path, 
		as_norm=False, norm_v2=False, cohort_path=None, num_cohort=1000, eps=1e-11):
		self.eval()
		files = []
		embeddings = {}
		lines = open(eval_list).read().splitlines()
		for line in lines:
			files.append(line.split()[1])
			files.append(line.split()[2])
		setfiles = list(set(files))
		setfiles.sort()

		if as_norm:
			import numpy as np
			cohort = np.load(cohort_path)
			cohort = torch.from_numpy(cohort).float().cuda()
			cohort = F.normalize(cohort, p=2, dim=-1)
			cohort_1 = cohort[:,0,:]
			cohort_2 = cohort[:,1:,:]

		for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
			audio, _  = soundfile.read(os.path.join(eval_path, file))
			# Full utterance
			data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()

			# Spliited utterance matrix
			max_audio = 300 * 160 + 240
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			feats = []
			startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
			for asf in startframe:
				feats.append(audio[int(asf):int(asf)+max_audio])
			feats = numpy.stack(feats, axis = 0).astype(numpy.float)
			data_2 = torch.FloatTensor(feats).cuda()
			# Speaker embeddings
			with torch.no_grad():
				logmel_data_1 = self.speaker_encoder.logmel(data_1, aug = False)
				embedding_1 = self.speaker_encoder.forward(logmel_data_1)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				logmel_data_2 = self.speaker_encoder.logmel(data_2, aug = False)
				embedding_2 = self.speaker_encoder.forward(logmel_data_2)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)
			embeddings[file] = [embedding_1, embedding_2]

		scores, labels  = [], []
		scores_dict = {}; cohorts = {}; means = {}; stds = {}

		for line in lines:		
			enroll = line.split()[1]
			target = line.split()[2]

			embedding_11, embedding_12 = embeddings[enroll]
			embedding_21, embedding_22 = embeddings[target]
			# Compute the scores
			score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
			score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
			'''
			score_11 = torch.mean(torch.matmul(embedding_11, embedding_21.T))
			score_12 = torch.mean(torch.matmul(embedding_11, embedding_22.T))
			score_1 = (score_11 + score_12) / 2
			score_21 = torch.mean(torch.matmul(embedding_21, embedding_11.T))
			score_22 = torch.mean(torch.matmul(embedding_21, embedding_12.T))
			score_2 = (score_21 + score_22) / 2
			'''
			score = (score_1 + score_2) / 2
			'''
			score = torch.mean(torch.matmul(
				torch.cat((embedding_11, embedding_12), 0),
				torch.cat((embedding_21, embedding_22), 0).T
			))
			'''
			score = score.detach().cpu().numpy()

			if not as_norm:
				scores.append(score)
				labels.append(int(line.split()[0]))

			else:
				'''
				cohort_score11 = torch.matmul(embedding_11, cohort.T).squeeze()
				cohort_score12 = torch.matmul(embedding_12, cohort.T)
				cohort_score12 = torch.mean(cohort_score12, 0)
				'''
				cohort_score11 = torch.matmul(embedding_11, cohort_1.T).squeeze()
				cohort_score12 = torch.mean(torch.matmul(
				    embedding_12.unsqueeze(0), cohort_2.transpose(1,2)), [1,2])
				cohort_score1 = (cohort_score11 + cohort_score12) / 2
				'''
				cohort_score1 = torch.mean(torch.matmul(
					torch.cat((embedding_11, embedding_12), 0).unsqueeze(0),
					torch.cat((cohort_1.unsqueeze(1), cohort_2), 1).transpose(1,2)
				), [1, 2])
				'''

				idx = torch.argsort(cohort_score1)[-num_cohort:]
				if not norm_v2:
					cohort_score1 = cohort_score1[idx].detach().cpu().numpy()
				else:
					'''
					cohort_score11 = torch.matmul(embedding_21, cohort.T).squeeze()
					cohort_score12 = torch.matmul(embedding_22, cohort.T)
					cohort_score12 = torch.mean(cohort_score12, 0)
					'''
					cohort_score11 = torch.matmul(embedding_21, cohort_1.T).squeeze()
					cohort_score12 = torch.mean(torch.matmul(
					    embedding_22.unsqueeze(0), cohort_2.transpose(1,2)), [1,2])
					cohort_score1 = (cohort_score11 + cohort_score12) / 2
					'''
					cohort_score1 = torch.mean(torch.matmul(
						torch.cat((embedding_21, embedding_22), 0).unsqueeze(0),
						torch.cat((cohort_1.unsqueeze(1), cohort_2), 1).transpose(1,2)
					), [1, 2])
					'''
					cohort_score1 = cohort_score1[idx].detach().cpu().numpy()

				cohort_mean = np.mean(cohort_score1)
				cohort_std = np.std(cohort_score1, ddof=0)
				cohort_score1 = (score - cohort_mean) / (eps + cohort_std)
				#cohort_score1 = (score) / (eps + cohort_std)

				'''
				cohort_score21 = torch.matmul(embedding_21, cohort.T).squeeze()
				cohort_score22 = torch.matmul(embedding_22, cohort.T)
				cohort_score22 = torch.mean(cohort_score22, 0)
				'''
				cohort_score21 = torch.matmul(embedding_21, cohort_1.T).squeeze()
				cohort_score22 = torch.mean(torch.matmul(
				    embedding_22.unsqueeze(0), cohort_2.transpose(1,2)), [1,2])
				cohort_score2 = (cohort_score21 + cohort_score22) / 2
				'''
				cohort_score2 = torch.mean(torch.matmul(
					torch.cat((embedding_21, embedding_22), 0).unsqueeze(0),
					torch.cat((cohort_1.unsqueeze(1), cohort_2), 1).transpose(1,2)
				), [1, 2])
				'''

				idx = torch.argsort(cohort_score2)[-num_cohort:]
				if not norm_v2:
					cohort_score2 = cohort_score2[idx].detach().cpu().numpy()
				else:
					'''
					cohort_score21 = torch.matmul(embedding_11, cohort.T).squeeze()
					cohort_score22 = torch.matmul(embedding_12, cohort.T)
					cohort_score22 = torch.mean(cohort_score22, 0)
					'''
					cohort_score21 = torch.matmul(embedding_11, cohort_1.T).squeeze()
					cohort_score22 = torch.mean(torch.matmul(
					    embedding_12.unsqueeze(0), cohort_2.transpose(1,2)), [1,2])
					cohort_score2 = (cohort_score21 + cohort_score22) / 2
					'''
					cohort_score2 = torch.mean(torch.matmul(
						torch.cat((embedding_11, embedding_12), 0).unsqueeze(0),
						torch.cat((cohort_1.unsqueeze(1), cohort_2), 1).transpose(1,2)
					), [1, 2])
					'''
					cohort_score2 = cohort_score2[idx].detach().cpu().numpy()

				cohort_mean = np.mean(cohort_score2)
				cohort_std = np.std(cohort_score2, ddof=0)
				cohort_score2 = (score - cohort_mean) / (eps + cohort_std)
				#cohort_score2 = (score) / (eps + cohort_std)

				score = (cohort_score1 + cohort_score2) / 2.
				scores.append(score)
				labels.append(int(line.split()[0]))
			
		# Coumpute EER and minDCF
		EER = tuneThresholdfromScore(scores, labels, [])[1]
		fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, 1, 1)

		return EER, minDCF

	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("module.", "")
				if name not in self_state:
					print("%s is not in the model."%origname)
					continue
			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)
