import numpy as np

def polar(r, theta):
  return r * (np.cos(theta) + np.sin(theta) * 1j)

def diag(f):
  k1 = np.concatenate([f, np.zeros_like(f)], -1)
  k2 = np.concatenate([np.zeros_like(f), f], -1)
  return np.concatenate([k1, k2], 0)

def get_perm(n):
  return np.eye(n)[np.array([2*i for i in range(n//2)] + [2*i+1 for i in range(n//2)])]

def get_f(N):
  w = np.zeros([N, N], dtype='complex')
  for n in range(N):
    for k in range(N):
      w[n][k] = polar(1., -2.*np.pi*k*n/N)
  return w
  
def get_w(_n):
  i1 = np.concatenate([np.eye(_n), np.diag([polar(1., -2.*np.pi*e/(_n*2)) for e in range(_n)])], -1)
  i2 = np.concatenate([np.eye(_n), -np.diag([polar(1., -2.*np.pi*e/(_n*2)) for e in range(_n)])], -1)
  return np.concatenate([i1, i2], 0)
  
def matmul(es):
  assert len(es) >= 2
  ret = np.matmul(es[0], es[1])
  for e in es[2:]:
    ret = np.matmul(ret, e)
  return ret
  
def decompose_fft(N, transpose=False):
  def get_w(N):
    i1 = np.concatenate([np.eye(N), np.diag([polar(1., -2.*np.pi*e/(N*2)) for e in range(N)])], -1)
    i2 = np.concatenate([np.eye(N), -np.diag([polar(1., -2.*np.pi*e/(N*2)) for e in range(N)])], -1)
    return np.concatenate([i1, i2], 0)
	
  def get_perm(n):
    return np.eye(n)[np.array([2*i for i in range(n//2)] + [2*i+1 for i in range(n//2)])]
	
  wms = []; pns = []
  for i in range(int(np.log2(N))):
    wm = get_w(N//np.power(2, i+1))
    for j in range(i):
      wm = diag(wm)
    if transpose: wm = wm.T
    wms.append(wm)
	
    pn = get_perm(N//np.power(2, i))
    for j in range(i):
      pn = diag(pn)
    if transpose: pn = pn.T
    pns.append(pn)
	
  if transpose:
    return wms[::-1], matmul(pns)
  return wms, matmul(pns[::-1])
