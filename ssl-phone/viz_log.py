import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--log", nargs='+', required=True, default=[])
parser.add_argument("--label", nargs='+', required=False, default=[])
parser.add_argument("--style", nargs='+', required=False, default=[])
parser.add_argument("--color", nargs='+', required=False, default=[])
parser.add_argument("--momentum", type=float, required=False, default=0.0)
args = parser.parse_args()

import os
import numpy as np
import re

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches

from matplotlib import font_manager

font_dirs = ['.']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# set font
plt.rcParams['font.family'] = 'Latin Modern Roman'
plt.rcParams["mathtext.fontset"] = 'cm'

#fig = plt.figure(figsize=[19.2, 9.6])
fig = plt.figure(figsize=[6.4, 3.2])
#ax_2d = fig.add_subplot(1, 2, 1)
#ax_2d.xaxis.grid(color='gray', linestyle='dashed')
#ax_2d.yaxis.grid(color='gray', linestyle='dashed')
#ax_2d.set_box_aspect(1)

#ax_2d_2 = fig.add_subplot(1, 3, 2)
#ax_2d_2.xaxis.grid(color='gray', linestyle='dashed')
#ax_2d_2.yaxis.grid(color='gray', linestyle='dashed')

ax_2d_3 = fig.add_subplot(1, 1, 1)
ax_2d_3.xaxis.grid(color='gray', linestyle='dashed')
ax_2d_3.yaxis.grid(color='gray', linestyle='dashed')
#ax_2d_3.set_box_aspect(1)

logs_list = []; logdir_list = []
for logfile in args.log:
    logdir = os.path.dirname(logfile)
    logs = [e.strip() for e in open(logfile, "r").readlines()]
    logs = logs[5:]

    logdir_list.append(logdir)
    logs_list.append(logs)

for lidx, (logs, logdir) in enumerate(zip(logs_list, logdir_list)):
    accs = []
    for log in logs:
        pat = ".*per\[([0-9.]*)\].*"
        log_re = re.search(pat, log)
        try: per = float(log_re.groups(0)[0])
        except: continue
        accs.append(per)
        
    _accs = []
    momentum = args.momentum
    
    ema = 1.
    for _idx, _acc in enumerate(accs):
        ema = ema * momentum + _acc * (1 - momentum)
        _accs.append(ema)
    accs = _accs
    
    slosses = []
    for log in logs:
        pat = ".*sloss\[([0-9.e+-]*)\].*"
        log_re = re.search(pat, log)
        try: sloss = float(log_re.groups(0)[0])
        except: continue
        slosses.append(sloss)
    
    wdiffs = []
    for log in logs:
        pat = ".*wdiff\[([0-9.e+-]*)\].*"
        log_re = re.search(pat, log)
        try: wdiff = float(log_re.groups(0)[0])
        except: continue
        wdiffs.append(wdiff)

    label = os.path.basename(logdir)
    if lidx < len(args.label):
        label = args.label[lidx]

    linestyle = 'solid'
    if lidx < len(args.style):
        linestyle = args.style[lidx]

    color = None
    if lidx < len(args.color) and args.color[lidx] != "None":
        color = args.color[lidx]

    '''
    ax_2d.plot(np.arange(len(accs)), accs,
            linestyle=linestyle,
            color=color,
            label=label, markersize=4)
#    ax_2d.set_xticklabels(["{:.1f}k".format(e * 0.1) for e in range(len(accs))])
    if lidx == 0:
        ticklabels = ax_2d.get_xticklabels()
        labels = ["{:.1f}k".format(int(re.sub(u"\u2212", "-", item.get_text())) * 0.1).replace(".0", "") for item in ticklabels]
        ax_2d.set_xticklabels(labels)
        ax_2d.set_yscale('log')
        ax_2d.xaxis.grid(color='gray', linestyle='dashed')
        ax_2d.yaxis.grid(color='gray', linestyle='dashed')
        ax_2d.set_ylabel("Error Rate")
        ax_2d.set_xlabel("Iterations")
    ax_2d.legend(loc='upper right')
    '''
    
    '''
    ax_2d_2.plot(np.arange(len(slosses)), slosses,
            linestyle=linestyle,
            color=color,
            label=label, markersize=4)
    if lidx == 0:
        ticklabels = ax_2d_2.get_xticklabels()
        labels = ["{:.1f}k".format(int(re.sub(u"\u2212", "-", item.get_text())) * 0.1).replace(".0", "") for item in ticklabels]
        ax_2d_2.set_xticklabels(labels)
#        ax_2d_2.set_yscale('log')
        ax_2d_2.xaxis.grid(color='gray', linestyle='dashed')
        ax_2d_2.yaxis.grid(color='gray', linestyle='dashed')
        ax_2d_2.set_ylabel("SSL loss ($\\mathcal{L}_{tera}$)")
        ax_2d_2.set_xlabel("Iterations")
    ax_2d_2.legend(loc='center right')
    '''
    
    ax_2d_3.plot(np.arange(len(wdiffs)), wdiffs,
            linestyle=linestyle,
            color=color,
            label=label, markersize=4)
    if lidx == 0:
        ticklabels = ax_2d_3.get_xticklabels()
        labels = ["{:.1f}k".format(int(re.sub(u"\u2212", "-", item.get_text())) * 0.1).replace(".0", "") for item in ticklabels]
        ax_2d_3.set_xticklabels(labels)
#        ax_2d_3.set_yscale('log')
        ax_2d_3.xaxis.grid(color='gray', linestyle='dashed')
        ax_2d_3.yaxis.grid(color='gray', linestyle='dashed')
        ax_2d_3.set_ylabel("$||\\theta^*-\\theta||_2$")
        ax_2d_3.set_xlabel("Iterations")
    ax_2d_3.legend(loc='lower right')

plt.tight_layout()
plt.savefig('fig5.png')
