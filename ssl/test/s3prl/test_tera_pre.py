from tera_builder import PretrainedTransformerWithHead

options = {
  'ckpt_file' : '/home/hejung/tera_960hr/states-1000000.ckpt',
  'load_pretrain' : 'True',
  'no_grad' : 'True',
  'dropout' : 0,
  'spec_aug' : 'False',
  'spec_aug_prev' : 'False',
  'output_hidden_states' : 'False',
  'select_layer' : -1,
  'permute_input' : 'False'
}
m = PretrainedTransformerWithHead(options, -1)
print(m)

import soundfile
pcm, _ = soundfile.read("/home/hejung/s3prl/s3prl/downstream/speech_commands/dummy_data/train/yes/01d22d03_nohash_0.wav")

import torch
import numpy as np
_in = torch.from_numpy(pcm.astype(np.float32))
_tmp = m([_in])
print(_tmp)

'''
(venv3.7-tera) hejung@speech:~/tf-models/ssl/test/s3prl$ python3 -m pdb test_tera_pre.py
> /home/hejung/tf-models/ssl/test/s3prl/test_tera_pre.py(1)<module>()
-> from tera_builder import PretrainedTransformerWithHead
(Pdb) b tera_builder.py:322
(Pdb) c
> /home/hejung/tf-models/ssl/test/s3prl/tera_builder.py(322)forward()
-> x, _ = self.SpecHead(x)
(Pdb) p x
tensor([[[-0.1140, -0.1691,  0.1227,  ...,  0.0281, -0.1509, -0.2364],
         [-0.0944, -0.2462,  0.1117,  ...,  0.0132, -0.0905, -0.2904],
         [-0.0614, -0.1415,  0.1611,  ...,  0.0076, -0.1719, -0.1910],
         ...,
         [-0.2137, -0.1497,  0.0100,  ..., -0.0065, -0.3174, -0.3091],
         [-0.1791, -0.1587,  0.0586,  ..., -0.0420,  0.1360, -0.2837],
         [-0.2914, -0.0790,  0.0260,  ..., -0.0436,  0.4615, -0.1525]]])
(Pdb) p self.SpecHead(x)[0].shape
torch.Size([1, 101, 80])
(Pdb) p self.SpecHead(x)[0]
tensor([[[-0.4149, -0.4064, -0.5600,  ..., -0.4582, -0.1764, -0.2049],
         [-0.6190, -0.6124, -0.9025,  ..., -0.6562, -0.3437, -0.1982],
         [-0.7483, -0.7424, -1.0447,  ..., -0.6033, -0.5067, -0.5515],
         ...,
         [-0.5010, -0.4963, -1.0103,  ..., -0.7722, -0.7143, -0.4726],
         [-1.1527, -1.1453, -0.8304,  ..., -0.6842, -0.6542, -0.4473],
         [-0.0191, -0.0129, -0.3425,  ..., -0.6437, -0.5788, -0.4879]]])

(venv3.7-tera) hejung@speech:~/tf-models/ssl/test$ python3 test_tera.py
(<tf.Tensor: shape=(1, 101, 768), dtype=float32, numpy=
array([[[-0.11399778, -0.16911498,  0.12274108, ...,  0.02805673,
         -0.15087901, -0.2363666 ],
        [-0.09442227, -0.2462019 ,  0.11173932, ...,  0.01314869,
         -0.0904882 , -0.29043987],
        [-0.06143013, -0.14151919,  0.1611378 , ...,  0.00757908,
         -0.17190604, -0.190993  ],
        ...,
        [-0.21367435, -0.14971769,  0.00995845, ..., -0.00646792,
         -0.31736594, -0.3090821 ],
        [-0.179086  , -0.15870209,  0.05857538, ..., -0.04198653,
          0.13603775, -0.28368437],
        [-0.2914045 , -0.07896738,  0.02597807, ..., -0.04358101,
          0.46148384, -0.15250033]]], dtype=float32)>, <tf.Tensor: shape=(1, 101, 80), dtype=float32, numpy=
array([[[-0.41489163, -0.40636247, -0.55996865, ..., -0.45821407,
         -0.17639816, -0.2049139 ],
        [-0.61896855, -0.6124079 , -0.90246004, ..., -0.65619683,
         -0.3437074 , -0.19820279],
        [-0.74833226, -0.74244666, -1.0447522 , ..., -0.6032849 ,
         -0.5067328 , -0.5515159 ],
        ...,
        [-0.50096923, -0.4962963 , -1.010321  , ..., -0.7721979 ,
         -0.7142946 , -0.47260952],
        [-1.1527355 , -1.1453153 , -0.83045083, ..., -0.6841597 ,
         -0.65417314, -0.44723746],
        [-0.01911091, -0.0129355 , -0.34246156, ..., -0.64367944,
         -0.5787387 , -0.4879039 ]]], dtype=float32)>)
'''
