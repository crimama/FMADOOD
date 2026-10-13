"""Print-readable CVPR-width ScopeAD overview with editable SVG export."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE=Path(__file__).resolve().parent
CACHE=HERE/"cache_rsg4_mechanism"
OUT=HERE/"fig_scopead_overview_cvpr"
SCALES=(448,672,896,1120); LAYERS=(5,11,23); RES=672
W,H=7.0,3.55
NAVY="#202124"; BLUE="#356B8C"; TEAL="#356B8C"; AMBER="#A65E2E"
MUTED="#62666A"; PALE="#FFFFFF"; LINE="#9A9DA0"; WHITE="#FFFFFF"
GT="#35D8F2"; ZMAX=2.0
plt.rcParams.update({"font.family":"DejaVu Sans","mathtext.fontset":"dejavuserif",
 "font.size":7.2,"svg.fonttype":"none","pdf.fonttype":42,"ps.fonttype":42})

class Canvas:
 def __init__(self):
  self.fig=plt.figure(figsize=(W,H),facecolor="white")
  self.ax=self.fig.add_axes([0,0,1,1]); self.ax.set(xlim=(0,W),ylim=(0,H)); self.ax.axis("off")
 def card(self,x,y,w,h,fill=WHITE,edge=LINE,lw=.8,r=.06):
  # Kept as a layout hook; the publication figure uses whitespace and rules,
  # rather than enclosing every operation in a UI-like card.
  return None
 def text(self,x,y,s,size=7.2,color=NAVY,weight="normal",ha="center",va="center",**kw):
  self.ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va=va,**kw)
 def arrow(self,a,b,color=MUTED,lw=1):
  self.ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=8,linewidth=lw,color=color,shrinkA=0,shrinkB=0,zorder=4))
 def image(self,x,y,w,h,a,edge=LINE,cmap=None):
  ax=self.fig.add_axes([x/W,y/H,w/W,h/H]); ax.set_xticks([]); ax.set_yticks([])
  if cmap: ax.imshow(np.asarray(a,dtype=np.float32),cmap=cmap,vmin=0,vmax=ZMAX,interpolation="bilinear")
  else: ax.imshow(a,interpolation="bilinear")
  for sp in ax.spines.values(): sp.set_color(edge); sp.set_linewidth(.8)
  return ax

def main():
 d=np.load(CACHE/"overview_walnuts__defect.npz")
 s=np.load(CACHE/"overview_walnuts__support.npz")
 meta=json.loads((CACHE/"overview_walnuts__meta.json").read_text())
 rgb=d["rgb"]; aspect=rgb.shape[0]/rgb.shape[1]
 layer={l:np.asarray(d[f"r{RES}_normalized_l{l}"],np.float32) for l in LAYERS}
 fused=(layer[11]+layer[23])/2+.125*layer[5]
 scale={r:((np.asarray(d[f"r{r}_normalized_l11"],np.float32)+np.asarray(d[f"r{r}_normalized_l23"],np.float32))/2+.125*np.asarray(d[f"r{r}_normalized_l5"],np.float32)) for r in SCALES}
 final=np.mean(list(scale.values()),axis=0)
 c=Canvas()
 # A. Support fitting, fitted once from normal support only.
 c.card(.06,2.67,6.88,.82,PALE,LINE)
 c.text(.20,3.36,"A",10,TEAL,"bold",ha="left"); c.text(.43,3.36,"SUPPORT FITTING",8.4,TEAL,"bold",ha="left")
 c.text(6.78,3.36,"once · normal support only · frozen backbone",6.8,MUTED,ha="right")
 cards=[(.18,.96),(1.48,1.38),(3.13,1.48),(4.93,1.88)]
 for x,w in cards: c.card(x,2.72,w,.53,WHITE,LINE,.7,.05)
 for i,im in enumerate(s["thumbs"][:3]): c.image(.24+i*.10,2.96+i*.008,.25,.22,im,WHITE)
 c.text(.66,3.06,"N normal",6.3,NAVY,"bold",ha="left")
 c.text(.66,2.90,"support images",5.9,MUTED,ha="left")
 c.text(2.17,3.11,"Frozen DINOv2 ViT-L/14",7.2,BLUE,"bold")
 c.text(2.17,2.98,"layers 5 / 11 / 23 · 4 scales",6.1)
 c.text(2.17,2.85,"448 / 672 / 896 / 1120 px",5.6,MUTED)
 c.text(2.17,2.75,"4 bags × 256 tokens / image",5.5,MUTED)
 c.text(3.87,3.11,"48 observation banks",7.2,TEAL,"bold")
 c.text(3.87,2.96,"m = (layer, scale, bag)",6.4)
 c.text(3.87,2.82,"leave-source-out 1-NN distances",6.0,MUTED)
 c.text(5.87,3.11,"Scale calibration",7.2,TEAL,"bold")
 c.text(5.87,2.96,r"cₘ = Q₀.₉₅(Dₘᴸˢᴼ)",7.3)
 c.text(5.87,2.82,"fixed banks and calibration",6.0,MUTED)
 for a,b in [((1.17,3.02),(1.42,3.02)),((2.90,3.02),(3.07,3.02)),((4.66,3.02),(4.87,3.02))]: c.arrow(a,b)

 # B. Query inference, organized as matching, fusion and aggregation.
 c.card(.06,.08,6.88,2.45,PALE,LINE)
 c.text(.20,2.40,"B",10,BLUE,"bold",ha="left"); c.text(.43,2.40,"QUERY INFERENCE",8.4,BLUE,"bold",ha="left")
 c.text(6.78,2.40,"per image · fixed support state · no test-time update",6.8,MUTED,ha="right")
 # Query input.
 c.card(.18,.30,1.15,1.82,WHITE,LINE,.8,.06)
 ih=.98*aspect; c.image(.265,1.11,.98,ih,rgb,NAVY)
 c.text(.755,2.02,"query xᵠ",7.0,NAVY,"bold"); c.text(.755,.91,"Frozen features",7.0,BLUE,"bold")
 c.text(.755,.74,"4 scales × 3 layers",6.2); c.text(.755,.57,"4 bags per scale",6.2)
 c.text(.755,.39,"48 calibrated maps",6.0,MUTED)
 # Multi-scale 1-NN.
 c.card(1.58,.30,1.31,1.82,"#EDF5F9","#91B5C9",.9,.06)
 c.text(2.235,1.91,"MULTI-SCALE 1-NN",7.2,BLUE,"bold")
 c.text(2.235,1.71,"per-bank nearest cosine distance",5.1)
 c.text(2.235,1.56,"minᵤ∈ᴮₘ [1 − fₘ(p)ᵀu]",6.1)
 for j,(lab,col) in enumerate([("layer 5",AMBER),("layer 11",BLUE),("layer 23",BLUE)]):
  y=1.31-j*.25; c.ax.add_patch(Rectangle((1.78,y),.51,.17,facecolor=WHITE,edgecolor=col,linewidth=.8))
  c.text(2.035,y+.085,lab,5.7,col,"bold"); c.text(2.36,y+.085,"× 4 r × 4 bags",5.5,MUTED,ha="left")
 c.text(2.235,.48,"zₘ(p) = dₘ(p) / cₘ",8.0,TEAL,"bold")
 c.arrow((1.34,1.19),(1.52,1.19),BLUE)
 # Support-calibrated layer fusion.
 c.card(3.13,.30,1.70,1.82,WHITE,"#D6A361",.9,.06)
 c.text(3.98,1.91,"CALIBRATED LAYER FUSION",6.8,AMBER,"bold")
 mw,mh=.38,.38*aspect
 for l,y,col in [(23,1.28,BLUE),(11,.83,BLUE),(5,.39,AMBER)]:
  c.image(3.46,y,mw,mh,layer[l],col,"inferno"); c.text(3.34,y+mh/2,f"z{l}",6.3,col,"bold")
 c.text(4.22,1.43,"deep mean",6.3,BLUE,"bold")
 c.text(4.22,1.27,"(z₁₁ + z₂₃) / 2",6.3,BLUE)
 c.text(4.22,.99,"+ 0.125 z₅",7.0,AMBER,"bold")
 c.text(3.96,.67,"S = z_d + 0.125 z₅",5.7,NAVY,"bold",ha="left")
 c.text(3.96,.45,"fixed shallow contribution",5.3,MUTED,ha="left")
 c.arrow((2.91,1.19),(3.07,1.19),BLUE)
 # Bag and scale aggregation.
 c.card(5.08,.30,1.68,1.82,WHITE,LINE,.8,.06)
 c.text(5.92,1.91,"MAP AGGREGATION",7.0,NAVY,"bold")
 c.text(5.92,1.70,"bag mean → resize → scale mean",5.7,MUTED)
 smw,smh=.29,.29*aspect
 for j,r in enumerate(SCALES):
  x=5.22+(j%2)*.39; y=1.22-(j//2)*.40
  c.image(x,y,smw,smh,scale[r],LINE,"inferno"); c.text(x+smw/2,y-.05,str(r),5.2,MUTED)
 c.image(6.00,.78,.55,.55*aspect,final,NAVY,"inferno").contour(d["gt"],levels=[.5],colors=[GT],linewidths=.8)
 c.text(6.28,.71,"final A(xᵠ)",6.3,NAVY,"bold")
 c.text(5.92,.52,"image score: top-1% mean",5.7,NAVY,"bold")
 c.text(5.92,.37,"cyan contour: display-only GT",5.2,MUTED)
 c.arrow((4.85,1.19),(5.02,1.19),NAVY)
 # Fitted support state is reused by query matching and normalization.
 c.ax.plot([5.87,5.87,1.43,1.43],[2.82,2.59,2.59,2.17],color=TEAL,lw=.8,ls=(0,(3,2)),zorder=2)
 c.arrow((1.43,2.17),(1.43,1.99),TEAL,.8)
 for ext in ("svg","pdf","png"):
  c.fig.savefig(OUT.with_suffix("."+ext),dpi=600 if ext=="png" else None,facecolor="white",pad_inches=0)
 OUT.with_suffix(".meta.json").write_text(json.dumps({
  "method":"ScopeAD","outline":"../outlines/Outline_v18.md","style":"restrained academic method schematic, print-first",
  "figma_file":"pending refreshed academic figure",
  "layout":"two-stage support fitting / query inference; square-corner modules, white ground, restrained ink and two accents",
  "width_in":W,"height_in":H,"layers":list(LAYERS),"scales":list(SCALES),"bags":4,
  "tokens_per_image_per_bag":256,"banks":48,"calibration_q":.95,
  "fusion":"(z11 + z23)/2 + 0.125*z5","cq":False,
  "display_heatmaps":"cached bag-averaged maps; no new inference","ground_truth":"contour only"
 },indent=2)+"\n")
 plt.close(c.fig); print(OUT.with_suffix(".svg"))

if __name__=="__main__": main()
