import numpy as np
import matplotlib.pyplot as plt

es0 = []; es1 = []; es2 = []; es3 = []
with open("tera_dbg5.sim", "r") as f:
  for line in f:
    e = line.split()
    if len(e) != 4: break
    assert len(e) == 4
    print(e[0], e[1], e[2], e[3])
    es0.append(float(e[0]))
    es1.append(float(e[1]))
    es2.append(float(e[2]))
    es3.append(float(e[3]))

#fig = plt.figure(figsize=[6.4,4.8])
fig = plt.figure(figsize=[6.4,4.])
#fig = plt.figure(figsize=[6.4,2.4])
#ax = fig.add_subplot(2, 1, 1)
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.arange(len(es0))*10*20, es0, label='TERA-pretrained')
#        linestyle='solid')
ax.plot(np.arange(len(es1))*10*20, es1, label='TERA-normed (layer 1)')
#        linestyle='dashed')
ax.plot(np.arange(len(es2))*10*20, es2, label='TERA-normed (layer 2)')
#        linestyle='dotted')
ax.plot(np.arange(len(es3))*10*20, es3, label='TERA-normed (layer 3)')
#        linestyle='dashdot')
#ax.set_yscale('log')
ax.set_ylim([0.13, 0.3])
ax.set_xlabel('Training steps')
ax.set_ylabel('Cosine similarity')
ax.legend()

'''
import matplotlib.pyplot as plt

es0 = []; es1 = []
with open("tera_dbg5_v2.sim", "r") as f:
  for line in f:
    e = line.split()
    if len(e) != 2: break
    assert len(e) == 2
    print(e[0], e[1])
    es0.append(float(e[0]))
    es1.append(float(e[1]))

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(np.arange(len(es0))*10*20, es0, label='without norm')
ax2.plot(np.arange(len(es1))*10*20, es1, label='with norm')
#ax2.set_yscale('log')
ax2.set_ylim([17.0, 18.5])
ax2.set_xlabel('Training steps')
ax2.set_ylabel('Distance norm')
ax2.legend()
fig.tight_layout()
'''
plt.savefig("viz_sim.png")
