"""ScopeAD main figure from Outline_v18, Chapter 3; CQ omitted by request.

Display maps reconstruct linear0125 from cached bag-averaged normalized
layer maps. Linear fusion and bilinear resizing commute with bag averaging.
These are display-resolution illustrations, not a new inference/evaluation.
The source maps were stored in FP16, so reconstruction includes rounding.
Run with .venv/bin/python PaperWorks/figures/draw_scarf_overview.py.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
CACHE = HERE / 'cache_rsg4_mechanism'
W, H = 14, 8
SCALES = (448, 672, 896, 1120)
SHALLOW_WEIGHT = 0.125
INK, GRAY = '#18344C', '#627284'
BLUE, ORANGE, GREEN = '#3476AC', '#C67A23', '#318574'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                     'mathtext.fontset':'dejavusans', 'svg.fonttype':'none',
                     'pdf.fonttype':42})

class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(W,H), facecolor='white')
        self.ax = self.fig.add_axes([0,0,1,1], xlim=(0,W), ylim=(0,H))
        self.ax.axis('off')
    def text(self,x,y,s,size=9,color=INK,ha='center',weight='normal'):
        self.ax.text(x,y,s,fontsize=size,color=color,ha=ha,va='center',fontweight=weight)
    def box(self,x,y,w,h,fill='#FFFFFF',edge='#CDD8E2',dash=False):
        self.ax.add_patch(FancyBboxPatch((x,y),w,h,
            boxstyle='round,pad=0,rounding_size=0.08',facecolor=fill,
            edgecolor=edge,linewidth=1,linestyle='--' if dash else '-',zorder=0))
    def arrow(self,p,q,color=GRAY,dash=False):
        self.ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=10,
            color=color,linewidth=1.1,linestyle='--' if dash else '-',
            shrinkA=0,shrinkB=0,zorder=2))
    def route(self,points,color=GRAY,dash=False):
        for p,q in zip(points[:-2],points[1:-1]):
            self.ax.plot([p[0],q[0]],[p[1],q[1]],color=color,lw=1.1,
                         linestyle='--' if dash else '-',zorder=2)
        self.arrow(points[-2],points[-1],color,dash)
    def image(self,x,y,w,h,a,cmap=None,edge=BLUE):
        ax=self.fig.add_axes([x/W,y/H,w/W,h/H])
        ax.imshow(a,cmap=cmap,vmin=0 if cmap else None,vmax=2 if cmap else None)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(edge); sp.set_linewidth(1)
    def block(self,x,y,w,h,title,lines,fill='#FFFFFF',edge=BLUE):
        self.box(x,y,w,h,fill,edge)
        self.text(x+w/2,y+h-0.20,title,10,edge,weight='bold')
        for j,line in enumerate(lines):
            self.text(x+w/2,y+h-0.43-j*0.21,line,8.7)


def linear_display_maps(cache):
    """Reconstruct linear maps without reading cached gated scores."""
    scale_maps = {}
    for r in SCALES:
        z5, z11, z23 = (np.asarray(cache[f'r{r}_normalized_l{l}'], dtype=np.float32)
                       for l in (5, 11, 23))
        scale_maps[r] = (z11 + z23) / 2 + SHALLOW_WEIGHT * z5
    return scale_maps, np.mean(list(scale_maps.values()), axis=0)


def draw():
    d=np.load(CACHE/'overview_walnuts__defect.npz')
    s=np.load(CACHE/'overview_walnuts__support.npz')
    scale_maps, final_map = linear_display_maps(d)
    c=Canvas()
    c.text(.25,7.77,'ScopeAD  |  Observation Expansion and Support-Calibrated Fusion',15,ha='left',weight='bold')
    c.text(13.7,7.77,'Frozen features · normal support only',9,GRAY,ha='right')

    # A: normal support fitting, performed once.
    c.box(.2,5.94,13.6,1.57,'#F1F7F4','#BED6CD')
    c.text(.4,7.29,'A  Support fitting',12,GREEN,ha='left',weight='bold')
    c.text(3.05,7.29,'Offline, once per support set',9,GRAY,ha='left')
    for j in range(3):
        c.image(.43+j*.17,6.34+j*.06,.58,.58,s['thumbs'][j],edge=GREEN)
    c.text(1.12,6.14,r'$N$ normal images' + '\n' + '(16-shot shown)',8.5)
    c.arrow((1.62,6.65),(1.93,6.65),GREEN)
    c.block(1.95,6.16,2.23,.92,'Multi-scale features',
        ['Frozen DINOv2 ViT-L/14','4 scales × layers {5, 11, 23}'],'#E9F1F8')
    c.arrow((4.19,6.65),(4.48,6.65),GREEN)
    c.block(4.5,6.16,2.95,.92,'Memory bagging',
        ['4 bags; 256 tokens / image / bag','Shared token positions across layers'],'#E9F1F8')
    c.arrow((7.46,6.65),(7.72,6.65),GREEN)
    c.block(7.74,6.16,2.54,.92,r'48 banks $B_{l,r,b}$',
        ['3 layers × 4 scales × 4 bags','4096 tokens / bank at 16-shot'],'#E8F4EF',GREEN)
    c.arrow((10.29,6.65),(10.57,6.65),GREEN)
    c.block(10.59,6.10,3.0,.98,'Leave-source-out 1-NN',
        [r'Exclude source image $i$ → $D_m^{\mathrm{LSO}}$',
         r'$c_m=Q_{0.95}(D_m^{\mathrm{LSO}})$  [48]',
         'Support-referenced distance units'],
        '#E8F4EF',GREEN)

    # B: explicit multi-scale features -> NN -> calibrated score maps.
    c.box(.2,3.80,13.6,1.84,'#F5F8FB','#CDD8E2')
    c.text(.4,5.43,'B  Query inference',12,BLUE,ha='left',weight='bold')
    c.text(3.05,5.43,'Observation expansion: 4 resolutions × 3 representation layers',9,GRAY,ha='left')
    c.image(.43,4.34,.76,.64,d['rgb'])
    c.text(.81,4.15,r'Query $x^q$',9,weight='bold')
    c.arrow((1.23,4.68),(1.5,4.68))
    for j,r in enumerate([448,672,896,1120]):
        yy=5.02-j*.28
        c.box(1.54,yy-.10,1.55,.23,'#E9F1F8',BLUE)
        c.text(2.315,yy,f'{r} px  →  {r//14} tokens*',8)
    c.text(2.32,3.97,'*along the short side',7.8,GRAY)
    c.arrow((3.13,4.68),(3.4,4.68))
    c.block(3.42,4.17,1.97,1.04,'Frozen DINOv2',
        ['ViT-L/14 · patch 14','Read blocks 5 / 11 / 23',r'L2-normalize $f_{l,r}(p)$'],'#E9F1F8')
    c.arrow((5.42,4.68),(5.69,4.68))
    c.box(5.71,4.17,1.63,1.04,'#FFFFFF',BLUE)
    c.text(6.525,5.01,'Feature tensors',10,BLUE,weight='bold')
    for j,(l,col) in enumerate([(5,ORANGE),(11,BLUE),(23,BLUE)]):
        yy=4.79-j*.24
        for k in range(4):
            c.ax.add_patch(Rectangle((5.88+k*.19,yy-.07),.14,.14,facecolor='#E8EEF3',edgecolor=col,lw=.8))
        c.text(6.83,yy,rf'$f_{{{l},r}}$',8,col)
    c.text(6.525,4.03,'4 scales × 3 layers',8,GRAY)
    c.arrow((7.37,4.68),(7.63,4.68))
    c.block(7.65,4.17,2.03,1.04,'Multi-scale 1-NN',
        [r'$d_m(p)=\min_{u\in B_m}[1-f_{l,r}(p)^\top u]$',
         'Cosine distance', '48 component maps'], '#E9F1F8')
    c.arrow((9.71,4.68),(9.98,4.68))
    c.block(10,4.17,1.75,1.04,'Scale calibration',
        [r'$z_m(p)=d_m(p)/c_m$', 'Axis 1: common units',r'Support $Q_{0.95}$ → 1'], '#E8F4EF',GREEN)
    c.arrow((11.78,4.68),(12.05,4.68))
    c.block(12.07,4.17,1.51,1.04,'Aligned maps',
        [r'$z_{5,r,b}$',r'$z_{11,r,b}$, $z_{23,r,b}$','for each (r, b)'],'#FFFFFF')
    c.route([(8.7,6.16),(8.7,5.23)],GREEN,True)
    c.text(8.85,5.79,r'$B_m$',9,GREEN,ha='left')
    c.route([(11.05,6.10),(11.05,5.23)],GREEN,True)
    c.text(11.18,5.79,r'$c_m$',9,GREEN,ha='left')
    c.route([(12.82,4.17),(12.82,3.62),(.65,3.62),(.65,2.48)],BLUE)
    c.text(6.9,3.70,'Normalized layer maps → fusion independently for each resolution r and bag b',8.7,BLUE)

    # C: fuse calibrated maps with a fixed shallow contribution.
    c.box(.2,.40,7.52,2.98,'#F7F9FC','#CDD8E2')
    c.text(.8,3.16,'C  Contribution control',12,BLUE,ha='left',weight='bold')
    c.text(7.49,3.16,'Axis 2 · each (r, b)',9,GRAY,ha='right')
    c.block(.88,2.0,1.97,.81,'Shallow · layer 5',
        [r'$z_{5,r,b}$'],'#FFF1E0',ORANGE)
    c.block(.88,.89,1.97,.87,'Deep · layers 11, 23',
        [r'$z_{11,r,b}$, $z_{23,r,b}$'],'#E9F1F8')
    c.route([(.65,2.48),(.65,1.34),(.86,1.34)],BLUE)
    c.arrow((.65,2.48),(.86,2.48),ORANGE)
    c.arrow((2.89,2.42),(3.17,2.42),ORANGE)
    c.block(3.19,1.96,2.44,.89,'Fixed shallow weight',
        [r'$\lambda z_{5,r,b}=0.125\,z_{5,r,b}$',
         'Weighted calibrated shallow score'],'#FFF1E0',ORANGE)
    c.text(4.415,2.99,r'$\lambda=0.125$ · fixed after development selection',8.2,ORANGE)
    c.arrow((2.89,1.34),(3.17,1.34),BLUE)
    c.block(3.19,.91,2.44,.87,'Deep anchor · mean',
        [r'$z_{d,r,b}=(z_{11,r,b}+z_{23,r,b})/2$'],'#E9F1F8')
    c.route([(5.66,2.42),(6.08,2.42),(6.08,1.88)],ORANGE)
    c.route([(5.66,1.34),(6.08,1.34),(6.08,1.68)],BLUE)
    c.ax.add_patch(plt.Circle((6.08,1.78),.13,facecolor='white',edgecolor=INK,lw=1.2))
    c.text(6.08,1.78,'+',12,weight='bold')
    c.arrow((6.23,1.78),(6.49,1.78),BLUE)
    c.box(6.51,1.30,.98,.96,'#E8F4EF',GREEN)
    c.text(7,2.03,'Fused map',9,GREEN,weight='bold')
    c.text(7,1.79,r'$S_{r,b}$',12)
    c.text(7,1.53,r'$z_d+\lambda z_5$',10)
    c.text(3.94,.62,r'$S_{r,b}=(z_{11,r,b}+z_{23,r,b})/2+0.125\,z_{5,r,b}$',11,ORANGE)

    # D: separate bag average, back-projection, and scale mean.
    c.box(7.92,.40,5.88,2.98,'#F5F8FB','#CDD8E2')
    c.text(8.12,3.16,'D  Bag and spatial aggregation',12,BLUE,ha='left',weight='bold')
    c.arrow((7.50,1.78),(8.10,1.78),BLUE)
    c.block(8.12,1.29,1.58,1.26,'4-bag mean',
        [r'$S_r=\frac{1}{4}\sum_{b=1}^{4}S_{r,b}$','',''],'#E9F1F8')
    c.text(8.91,1.69,'Same token grid',8.7)
    c.text(8.91,1.45,'at each scale',8.7)
    c.arrow((9.73,1.91),(9.98,1.91),BLUE)
    c.box(10,1.29,1.75,1.26,'#FFFFFF',BLUE)
    c.text(10.875,2.36,'Bilinear ↑ (H, W)',9,BLUE,weight='bold')
    for j,r in enumerate([448,672,896,1120]):
        x=10.13+(j%2)*.80; y=1.53+(1-j//2)*.34
        a=scale_maps[r]
        c.image(x,y,.43,.31,a,'inferno')
        c.text(x+.59,y+.15,str(r),7.5)
    c.arrow((11.78,1.91),(12.03,1.91),BLUE)
    c.box(12.05,1.01,1.51,1.69,'#E8F4EF',GREEN)
    c.text(12.805,2.49,'4-scale mean',10,GREEN,weight='bold')
    c.image(12.31,1.50,.99,.83,final_map,'inferno',GREEN)
    c.text(12.805,1.27,r'Final $A(x^q)$',10,weight='bold')
    c.text(10.86,.91,r'$A(x^q)=\frac{1}{4}\sum_{r\in\mathcal{R}}\mathrm{Resize}_{H,W}[S_r]$',9)
    c.text(10.86,.62,r'Image score $s(x^q)$: top-1% pixel mean',9,weight='bold')
    c.text(.3,.19,'Solid arrows: query computation     Dashed green arrows: fitted support references',8,GRAY,ha='left')
    c.text(13.7,.19,'Heatmaps: linear0125 reconstruction · display range [0, 2]',8,GRAY,ha='right')
    return c

def main():
    c=draw()
    for stem in ['fig_scopead_overview','fig_scarf_overview','fig_scarf_overview_detailed']:
        for ext in ['png','pdf','svg']:
            path=HERE/f'{stem}.{ext}'
            c.fig.savefig(path,dpi=240 if ext=='png' else None)
        print(HERE/f'{stem}.png')
    metadata = {
        'outline': '../outlines/Outline_v18.md', 'method': 'ScopeAD',
        'fusion': 'deep_mean + 0.125 * normalized_layer5',
        'scales': list(SCALES), 'layers': [5, 11, 23], 'bags': 4,
        'calibration_quantile': 0.95, 'image_score': 'top-1% pixel mean', 'cq': False,
        'display_map_source': 'cache_rsg4_mechanism/overview_walnuts__defect.npz',
        'display_map_keys': [f'r{r}_normalized_l{l}' for r in SCALES for l in (5,11,23)],
        'display_map_status': 'reconstructed from FP16 cached bag-averaged layer maps; not a new evaluation',
    }
    (HERE / 'fig_scopead_overview.meta.json').write_text(json.dumps(metadata, indent=2) + chr(10))
    plt.close(c.fig)

if __name__=='__main__':
    main()
