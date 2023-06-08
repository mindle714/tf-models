# /home/hejung/venv3-s3prl/lib/python3.10/site-packages/transformers/models/wav2vec2/modeling_wav2vec2.py 

import torch
torch.manual_seed(1234)
torch.use_deterministic_algorithms(True)

import random
random.seed(1234)

import numpy as np
np.random.seed(1234)

from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
#from datasets import load_dataset

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForPreTraining.from_pretrained('/home/hejung/wav2vec2-base/base_wo_dropout')
print(model)

import soundfile
pcm, _ = soundfile.read("01d22d03_nohash_0.wav")
pcm2 = pcm / 2.
#pcms = np.concatenate([
#    pcm.reshape([1, -1]), pcm2.reshape([1, -1])], 0).astype(np.float32)
pcms = np.concatenate([
    pcm.reshape([1, -1]), pcm2.reshape([1, -1]),
    pcm.reshape([1, -1]), pcm2.reshape([1, -1]),
    pcm.reshape([1, -1]), pcm2.reshape([1, -1]),
    pcm.reshape([1, -1]), pcm2.reshape([1, -1]),
    ], 0).astype(np.float32)

#input_values = feature_extractor(pcm, return_tensors="pt").input_values
input_values = torch.from_numpy(pcms)

batch_size, raw_sequence_length = input_values.shape
sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length).item()
mask_time_indices = _compute_mask_indices(
        shape=(batch_size, sequence_length), mask_prob=0.2, mask_length=2
)

sampled_negative_indices = _sample_negative_indices(
	features_shape=(batch_size, sequence_length),
	num_negatives=model.config.num_negatives,
	mask_time_indices=mask_time_indices,
)
mask_time_indices = torch.tensor(data=mask_time_indices, device=input_values.device, dtype=torch.long)
sampled_negative_indices = torch.tensor(
	data=sampled_negative_indices, device=input_values.device, dtype=torch.long
)

model = model.train()
outputs = model(input_values,
    mask_time_indices=mask_time_indices,
    sampled_negative_indices=sampled_negative_indices,
    output_hidden_states=True)
print(outputs)

# compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)
cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)
#print(cosine_sim)
#print(cosine_sim[mask_time_indices.to(torch.bool)].mean())

import matplotlib.pyplot as plt
gumbels = (
    -torch.empty([784, 320]).exponential_().log()
).cpu().numpy()
plt.hist(gumbels)
plt.savefig('gumbels_torch.png')
