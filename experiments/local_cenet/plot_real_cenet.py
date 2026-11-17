from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

root=Path(__file__).resolve().parent
out=root/'results/real06'
data=np.load(out/'000000_preview.npz')
cfg=yaml.safe_load((root/'source/CENet/CENET/data_cfg.yaml').read_text())
p=data['points']; labels=data['labels']
colors=np.array([cfg['color_map'][int(i)][::-1] for i in labels])/255
sel=(np.abs(p[:,0])<40)&(np.abs(p[:,1])<30)&(p[:,2]<4)&(p[:,2]>-4)
fig,axes=plt.subplots(1,2,figsize=(15,7),facecolor='white')
axes[0].scatter(p[sel,0],p[sel,1],c=p[sel,2],s=.3,cmap='viridis',vmin=-2,vmax=3)
axes[1].scatter(p[sel,0],p[sel,1],c=colors[sel],s=.3)
for a,title in zip(axes,['Input LiDAR: height color','CENet prediction: semantic color']):
    a.set_title(title);a.set_aspect('equal');a.set_xlabel('x (m)');a.set_ylabel('y (m)');a.set_facecolor('#101820')
ids,counts=np.unique(labels[sel],return_counts=True)
top=ids[np.argsort(counts)[::-1]][:8]
axes[1].legend(handles=[Patch(color=np.array(cfg['color_map'][int(i)][::-1])/255,label=cfg['labels'][int(i)]) for i in top],loc='upper right',fontsize=8)
fig.suptitle('KITTI 06 / frame 000000: real-data deployment check',fontsize=17)
fig.text(.5,.025,'FP16 | direct unprojection, no KNN | predicted classes, not manual ground truth',ha='center',fontsize=11)
fig.subplots_adjust(bottom=.13,top=.88,wspace=.18)
fig.savefig(out/'preview.png',dpi=150)
print(out/'preview.png')
