import torch
m = torch.load("/home/hejung/transformers/examples/pytorch/audio-classification/unispeech-sat-base-ft-keyword-spotting/checkpoint-31930/pytorch_model.bin")
print(m.keys())
