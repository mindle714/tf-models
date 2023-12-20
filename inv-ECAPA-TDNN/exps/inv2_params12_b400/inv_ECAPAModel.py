'''
This part is used to train the speaker model and evaluate the performances
'''

import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN
import time

def global_norm(params):
    return torch.sqrt(sum([(torch.nan_to_num(e)**2).sum() for e in params]))

def clip_by_global_norm(grads, clip):
    gnorm = global_norm(grads)
    scale = clip * torch.min(1. / gnorm, torch.tensor(1. / clip))
    return [scale * e for e in grads]

class ECAPAModel(nn.Module):
	def __init__(self, lr, lr_decay, C , n_class, m, s, test_step, **kwargs):
		super(ECAPAModel, self).__init__()
		## ECAPA-TDNN
		self.speaker_encoder = ECAPA_TDNN(C = C).cuda()
		## Classifier
		self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).cuda()

		self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

	def train_network(self, epoch, loader, factor, mask_ratio):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		index, top1, loss = 0, 0, 0
		lr = self.optim.param_groups[0]['lr']

		batch_size = loader.batch_size
		mask_sampler = torch.distributions.categorical.Categorical(
			torch.tensor([1.-mask_ratio, mask_ratio]))
		pre_mask = mask_sampler.sample([batch_size, batch_size, 1, 1]).float().cuda()

		def inv_step(mask_idx, labels, logmel_data):
			#start = time.time()
			self.zero_grad()
			_logmel_data = logmel_data.clone().detach().cuda().requires_grad_(True)
			speaker_embedding = self.speaker_encoder.forward(_logmel_data)
			nloss, _          = self.speaker_loss.forward(speaker_embedding, labels)			
			#end = time.time()
			#print("inv_step(forward)[{}]".format(end - start))
			#start_back = time.time()

			params = list(self.speaker_encoder.parameters())
			assert len(params) == 138
			params = params[-12:]

			grads = torch.autograd.grad(nloss, params, create_graph=True)
			grads = clip_by_global_norm(grads, 5.)
			gnorm = sum([torch.sqrt((e**2).sum()) for e in grads])
			#end = time.time()
			#print("inv_step(backward)[{}]".format(end - start_back))
			#start_back2 = time.time()

			grad_noisy = torch.nan_to_num(torch.autograd.grad(gnorm, _logmel_data)[0])
			#end = time.time()
			#print("inv_step(backward2)[{}]".format(end - start_back2))
			#start_rem = time.time()

			delta = -grad_noisy * factor
			#delta = torch.max(
			#	torch.min(_logmel_data + delta, _logmel_data.max(-1, keepdim=True)[0]),
			#	_logmel_data.min(-1, keepdim=True)[0]) - _logmel_data
			delta = torch.clamp(delta, -0.1, 0.1)
			#print("inv_step(rem1)[{}]".format(time.time() - start_rem))
			#start_rem1 = time.time()

			mask = pre_mask[mask_idx % batch_size]
			#print("inv_step(rem2)[{}]".format(time.time() - start_rem1))

			self.zero_grad()
			for param in self.speaker_encoder.parameters(): param.grad = None
			#end = time.time()
			#print("inv_step(rem)[{}]".format(end - start_rem))
			#print("inv_step[{}]".format(end - start))
			return mask * delta.detach()

		for num, (data, labels) in enumerate(loader, start = 1):
			labels            = torch.LongTensor(labels).cuda()
			logmel_data       = self.speaker_encoder.logmel(data.cuda(), aug = True)
			delta = inv_step(num, labels, logmel_data)
			logmel_data = logmel_data + delta

			#start = time.time()
			self.zero_grad()
			speaker_embedding = self.speaker_encoder.forward(logmel_data)
			nloss, prec       = self.speaker_loss.forward(speaker_embedding, labels)			
			nloss.backward()
			self.optim.step()
			index += len(labels)
			top1 += prec
			loss += nloss.detach().cpu().numpy()
			#end = time.time()
			#print("step[{}]".format(end - start))
			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
			" Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
			sys.stderr.flush()
		sys.stdout.write("\n")
		return loss/num, lr, top1/index*len(labels)

	def eval_network(self, eval_list, eval_path):
		self.eval()
		files = []
		embeddings = {}
		lines = open(eval_list).read().splitlines()
		for line in lines:
			files.append(line.split()[1])
			files.append(line.split()[2])
		setfiles = list(set(files))
		setfiles.sort()

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

		for line in lines:			
			embedding_11, embedding_12 = embeddings[line.split()[1]]
			embedding_21, embedding_22 = embeddings[line.split()[2]]
			# Compute the scores
			score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
			score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
			score = (score_1 + score_2) / 2
			score = score.detach().cpu().numpy()
			scores.append(score)
			labels.append(int(line.split()[0]))
			
		# Coumpute EER and minDCF
		EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
		fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

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
