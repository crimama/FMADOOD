"""ScopeAD v18 in the recovered initial SCARF two-band visual style.

Run: .venv/bin/python PaperWorks/figures/draw_scopead_overview_initial_style.py
Uses cached support distributions and FP16 bag-averaged, display-resolution
normalized layer maps. Reconstructs linear0125 without gated/CQ cache maps.
Not a new evaluation. Ground truth is used only as a display contour.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache_rsg4_mechanism"
OUT = HERE / "fig_scopead_overview_initial_style"
NAME = "overview_walnuts"
RES = 672  # resolution whose per-component maps are shown

W, H = 7.0, 3.15  # inches; CVPR text width is 6.875 in
Z_MAX = 2.0       # shared colour limit for every z-unit map
SCALES = (448, 672, 896, 1120)
SHALLOW_WEIGHT = 0.125

NAVY = "#16324F"
MUTED = "#5A6B7D"
DEEP = "#2F6DA3"
DEEP_BG = "#E3EEF7"
SHALLOW = "#D07A1F"
SHALLOW_BG = "#FDF0DF"
SUPPORT = "#3E8E7E"
SUPPORT_BG = "#E6F3EF"
PANEL_BG = "#F7F9FB"
GT = "#38E1FF"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    "font.size": 5.6, "svg.fonttype": "none", "pdf.fonttype": 42,
})


def load():
    defect = np.load(CACHE / f"{NAME}__defect.npz")
    support = np.load(CACHE / f"{NAME}__support.npz")
    meta = json.loads((CACHE / f"{NAME}__meta.json").read_text())
    return defect, support, meta


class Canvas:
    """Figure-inch coordinates for boxes, arrows, text and image insets."""

    def __init__(self):
        self.fig = plt.figure(figsize=(W, H))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.axis("off")

    def box(self, x, y, w, h, fill, edge, lw=0.7, radius=0.05, ls="-", z=0):
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill, edgecolor=edge, linewidth=lw, linestyle=ls, zorder=z))

    def text(self, x, y, s, size=5.6, color=NAVY, weight="normal", ha="center",
             va="center", **kw):
        self.ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                     ha=ha, va=va, **kw)

    def arrow(self, p, q, color=MUTED, lw=0.8, ls="-", rad=0.0, head=6):
        self.ax.add_patch(FancyArrowPatch(
            p, q, arrowstyle="-|>", mutation_scale=head, linewidth=lw, color=color,
            linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
            zorder=3))

    def inset(self, x, y, w, h):
        ax = self.fig.add_axes([x / W, y / H, w / W, h / H])
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def image(self, x, y, w, h, array, edge, cmap=None, vmax=Z_MAX, lw=1.0):
        ax = self.inset(x, y, w, h)
        if cmap is None:
            ax.imshow(array, interpolation="bilinear")
        else:
            ax.imshow(np.asarray(array, dtype=np.float32), cmap=cmap, vmin=0.0,
                      vmax=vmax, interpolation="bilinear")
        for spine in ax.spines.values():
            spine.set_edgecolor(edge)
            spine.set_linewidth(lw)
        return ax


def linear_display_maps(d):
    """Linear fusion commutes with bag means and bilinear display resizing."""
    maps = {}
    for r in SCALES:
        z5, z11, z23 = (np.asarray(d[f"r{r}_normalized_l{l}"], np.float32)
                       for l in (5, 11, 23))
        maps[r] = (z11 + z23) / 2 + SHALLOW_WEIGHT * z5
    return maps, np.mean(list(maps.values()), axis=0)


def draw_support_band(c, s, meta):
    c.box(.05, 2.06, 6.90, 1.04, PANEL_BG, "#D5DEE7", lw=.6, radius=.08, z=-2)
    c.text(.13, 2.99, "A   Support fitting", size=6.8, weight="bold", ha="left", color=SUPPORT)
    c.text(1.17, 2.99, "once, normal images only, frozen backbone", size=5.4, ha="left", color=MUTED)
    for i, thumb in enumerate(s["thumbs"][::-1]):
        k = 3-i
        c.image(.16+.07*k, 2.17+.045*k, .52, .52, thumb, "white", lw=.8)
    c.text(.55, 2.13, f"N = {len(meta['support'])} normal supports", size=5.2, color=MUTED)
    c.arrow((.99,2.48),(1.12,2.48))
    c.box(1.13,2.18,1.32,.68,DEEP_BG,DEEP)
    c.text(1.79,2.75,"Frozen DINOv2 ViT-L/14",size=5.8,weight="bold")
    c.text(1.79,2.60,r"layers $l\in\{5,11,23\}$")
    c.text(1.79,2.46,"448 / 672 / 896 / 1120 px",size=5.2)
    c.text(1.79,2.32,"4 bags × 256 tokens / image",size=5.2)
    c.text(1.79,2.22,r"48 banks $B_m$, $m=(l,r,b)$",size=5.0,color=SUPPORT)
    c.arrow((2.46,2.52),(2.57,2.52))
    c.box(2.58,2.18,1.23,.68,SUPPORT_BG,SUPPORT)
    c.text(3.195,2.75,"Leave-source-out 1-NN",size=5.7,weight="bold")
    for j in range(4):
        x=2.74+.20*j
        c.ax.add_patch(Rectangle((x,2.42+(.09 if j==0 else 0)),.15,.15,
            facecolor="white" if j==0 else "#CFE4DD",edgecolor=SHALLOW if j==0 else SUPPORT,
            linewidth=.7,linestyle="--" if j==0 else "-",zorder=2))
    c.arrow((2.90,2.6),(3.07,2.53),color=SHALLOW,lw=.6,rad=-.3,head=4)
    c.text(2.81,2.36,r"hold $x_i$",size=5.0,color=SHALLOW)
    c.text(3.35,2.36,"match to rest",size=5.0,color=SUPPORT)
    c.text(3.195,2.24,r"normal distances $D_m^{\mathrm{LSO}}$",size=5.2)
    c.arrow((3.82,2.52),(3.94,2.52))
    ax=c.inset(4.03,2.31,1.46,.49)
    bins=np.linspace(0,2,61)
    for l,col in ((23,DEEP),(5,SHALLOW)):
        values=np.asarray(s[f"lso_l{l}"],np.float32)/float(s[f"lso_u_l{l}"])
        ax.hist(values,bins=bins,density=True,histtype="stepfilled",color=col,alpha=.28,lw=0)
        ax.hist(values,bins=bins,density=True,histtype="step",color=col,lw=.7,label=f"layer {l}")
    ax.axvline(1,color=NAVY,lw=.8)
    ax.set_xlim(0,2); ax.set_yticks([]); ax.set_xticks([0,1,2])
    ax.tick_params(axis="x",labelsize=4.8,length=1.5,pad=1,colors=NAVY)
    for side in ("top","right","left"): ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(.5); ax.patch.set_alpha(0)
    ax.legend(fontsize=4.6,frameon=False,loc="upper right",handlelength=1,borderaxespad=.1)
    ax.text(.96,.98,r"$Q_{0.95}\to1$",transform=ax.get_xaxis_transform(),fontsize=4.8,
            color=NAVY,ha="right",va="top")
    c.text(4.76,2.885,r"support $D_m^{\mathrm{LSO}}/c_m$ ($r$=672)",size=5.4,weight="bold")
    c.arrow((5.50,2.52),(5.65,2.52),color=SUPPORT)
    c.box(5.68,2.18,1.20,.68,SUPPORT_BG,SUPPORT)
    c.text(6.28,2.75,"Scale calibration",size=5.8,weight="bold",color=SUPPORT)
    c.text(6.28,2.57,r"$c_m=Q_{0.95}(D_m^{\mathrm{LSO}})$",size=5.9)
    c.text(6.28,2.41,"one distance unit per bank",size=5.2)
    c.text(6.28,2.28,r"fixed state: $\{B_m,c_m\}_{m=1}^{48}$",size=5.1,color=SUPPORT)


def draw_query_band(c,d,scale_maps,final_map):
    c.box(.05,.05,6.90,1.95,PANEL_BG,"#D5DEE7",lw=.6,radius=.08,z=-2)
    c.text(.13,1.89,"B   Query inference",size=6.8,weight="bold",ha="left",color=DEEP)
    c.text(1.30,1.89,"per image, fixed support state, no test-time update",size=5.4,ha="left",color=MUTED)
    aspect=d["rgb"].shape[0]/d["rgb"].shape[1]
    c.image(.14,.99,.88,.88*aspect,d["rgb"],NAVY,lw=.6)
    c.text(.58,1.78,r"query $x^q$",size=5.2,color=MUTED)
    c.arrow((.58,.98),(.58,.86),color=NAVY)
    c.box(.14,.14,.88,.71,"white",DEEP)
    c.text(.58,.76,"Multi-scale 1-NN",size=5.6,weight="bold",color=DEEP)
    c.text(.58,.64,"4 fixed scales · 3 layers",size=4.8)
    c.text(.58,.53,"Frozen DINOv2 features",size=4.8)
    c.text(.58,.42,r"$d_m=\min_{u\in B_m}(1-f^\top u)$",size=4.7)
    c.text(.58,.30,r"$z_m=d_m/c_m$",size=6.2,weight="bold",color=SUPPORT)
    c.text(.58,.20,"48 calibrated score maps",size=4.7,color=MUTED)
    c.arrow((1.03,.49),(1.12,.49),color=NAVY)
    # Bank and calibration state are both consumed by matching/normalization.
    c.ax.plot([6.28,6.28,.09,.09],[2.18,2.025,2.025,.62],
              color=SUPPORT,lw=.65,ls=(0,(3,2)),zorder=3)
    c.arrow((.09,.62),(.14,.62),color=SUPPORT,lw=.65,ls=(0,(3,2)),head=5)
    draw_fusion(c,d,scale_maps[RES])
    draw_aggregation(c,d,scale_maps,final_map)


def draw_fusion(c,d,fused):
    c.box(1.13,.12,3.49,1.66,"white",SHALLOW,lw=.9,radius=.06,z=-1)
    c.text(2.875,1.70,"C   Support-calibrated layer fusion",size=6.0,weight="bold",color=SHALLOW)
    c.text(2.875,1.585,r"per $(r,b)$: deep mean + $0.125\,z_5$; shown at $r$=672, bag mean",
           size=4.9,color=MUTED)
    mw=.52; mh=mw*d["rgb"].shape[0]/d["rgb"].shape[1]
    z={l:np.asarray(d[f"r{RES}_normalized_l{l}"],np.float32) for l in (5,11,23)}
    cx=1.30; rows={23:1.06,11:.60,5:.16}
    for l,y in rows.items():
        color=SHALLOW if l==5 else DEEP
        c.image(cx,y,mw,mh,z[l],color,cmap="inferno")
        c.text(cx-.085,y+mh/2,rf"$z_{{{l}}}$",size=6.2,color=color,weight="bold")
    dx=2.10; dy=.91
    c.arrow((cx+mw+.02,rows[23]+mh/2),(dx-.02,dy+mh*.75),color=DEEP)
    c.arrow((cx+mw+.02,rows[11]+mh/2),(dx-.02,dy+mh*.25),color=DEEP)
    c.image(dx,dy,mw,mh,(z[11]+z[23])/2,DEEP,cmap="inferno",lw=1.2)
    c.text(dx+mw/2,dy+mh+.07,r"deep $z_d=\frac{z_{11}+z_{23}}{2}$",size=5.4,color=DEEP)
    c.text(dx+mw/2,dy-.065,"unit deep weight",size=5.0,color=DEEP)
    gx,gy,gw,gh=2.06,.20,.62,.36
    c.arrow((cx+mw+.02,rows[5]+mh/2),(gx-.03,gy+gh/2),color=SHALLOW)
    ax=c.inset(gx,gy,gw,gh)
    xs=np.linspace(0,2,100)
    ax.plot(xs,SHALLOW_WEIGHT*xs,color=SHALLOW,lw=1.1)
    ax.set_xlim(0,2); ax.set_ylim(0,.3)
    ax.set_xticks([0,1,2]); ax.set_yticks([0,.25])
    ax.tick_params(labelsize=4.2,length=1.5,pad=1,colors=NAVY)
    for side in ("top","right"): ax.spines[side].set_visible(False)
    for side in ("bottom","left"): ax.spines[side].set_linewidth(.5)
    ax.patch.set_alpha(0)
    c.text(gx+gw/2,gy+gh+.07,r"fixed weight $\lambda=0.125$",size=5.1,color=SHALLOW,weight="bold")
    ex=2.95
    c.arrow((gx+gw+.03,gy+gh/2),(ex-.02,rows[5]+mh/2),color=SHALLOW)
    c.image(ex,rows[5],mw,mh,SHALLOW_WEIGHT*z[5],SHALLOW,cmap="inferno",lw=1.2)
    c.text(ex+mw/2,rows[5]+mh+.07,r"shallow $0.125\,z_5$",size=5.4,color=SHALLOW)
    px,py=3.72,1.00
    c.arrow((dx+mw+.02,dy+mh/2),(px-.065,py+.03),color=DEEP)
    c.arrow((ex+mw+.02,rows[5]+mh/2),(px-.05,py-.045),color=SHALLOW)
    c.ax.add_patch(plt.Circle((px,py),.055,facecolor="white",edgecolor=NAVY,linewidth=.8,zorder=4))
    c.text(px,py,"+",size=8,weight="bold",zorder=5)
    sx=3.88; sy=py-mh/2
    c.arrow((px+.06,py),(sx-.02,py),color=NAVY)
    c.image(sx,sy,mw,mh,fused,NAVY,cmap="inferno",lw=1.2)
    c.text(sx+mw/2,sy+mh+.08,r"$S_{r,b}=z_d+0.125\,z_5$",size=5.4,weight="bold")
    c.text(sx+mw/2,sy-.10,"layer score fusion",size=4.9,color=MUTED)
    c.text(3.83,.25,"Same colour range\nfor all score maps",size=4.8,color=MUTED,linespacing=1.2)


def draw_aggregation(c,d,scale_maps,final_map):
    c.text(5.77,1.62,"D   Map aggregation",size=5.8,weight="bold")
    c.text(5.77,1.49,r"4-bag mean $\to$ bilinear $\uparrow$ $\to$ 4-scale mean",size=4.9,color=MUTED)
    c.arrow((4.43,1.00),(4.78,1.00),color=NAVY)
    mw=.33; mh=mw*d["rgb"].shape[0]/d["rgb"].shape[1]
    for j,r in enumerate(SCALES):
        x=4.82+(j%2)*.40; y=.71+(1-j//2)*.39
        c.image(x,y,mw,mh,scale_maps[r],NAVY,cmap="inferno",lw=.65)
        c.text(x+mw/2,y-.055,str(r),size=4.5,color=MUTED)
    c.text(5.19,.55,r"$S_r=\frac{1}{4}\sum_b S_{r,b}$",size=5.5)
    fx,fy,fw=5.90,.73,.86; fh=fw*d["rgb"].shape[0]/d["rgb"].shape[1]
    c.arrow((5.56,1.04),(fx-.02,1.04),color=NAVY)
    c.text(5.72,1.15,"mean",size=4.8,color=MUTED)
    ax=c.image(fx,fy,fw,fh,final_map,NAVY,cmap="inferno")
    ax.contour(d["gt"],levels=[.5],colors=[GT],linewidths=.6)
    c.text(fx+fw/2,fy-.08,r"final map $A(x^q)$",size=5.8,weight="bold")
    c.text(fx+fw/2,fy-.18,"cyan: GT (display only)",size=4.7,color=MUTED)
    c.text(5.77,.39,r"image score $s(x^q)$ = top-1% pixel mean",size=5.1,weight="bold")
    cax=c.inset(4.86,.24,1.91,.04)
    cax.imshow(np.linspace(0,1,256)[None],cmap="inferno",aspect="auto")
    for spine in cax.spines.values(): spine.set_linewidth(.3)
    c.text(4.86,.17,"0",size=4.6,ha="left")
    c.text(6.77,.17,f"{Z_MAX:g}",size=4.6,ha="right")
    c.text(5.82,.17,"z units; support q95 → 1 before fusion",size=4.5,color=MUTED)


def main():
    d,s,meta=load()
    scale_maps,final_map=linear_display_maps(d)
    c=Canvas()
    draw_support_band(c,s,meta)
    draw_query_band(c,d,scale_maps,final_map)
    for ext in ("pdf","svg","png"):
        exported=OUT.with_suffix(f".{ext}")
        c.fig.savefig(exported,dpi=400 if ext=="png" else None)
    metadata={
        "outline":"../outlines/Outline_v18.md", "method":"ScopeAD",
        "style_reference":"main_figure_history/07_scarf_initial_20261001/draft_06",
        "generator":Path(__file__).name, "layout":"two bands: support fitting / query inference",
        "fusion":"deep_mean + 0.125 * normalized_layer5", "scales":list(SCALES),
        "layers":[5,11,23], "bags":4, "tokens_per_image_per_bag":256,
        "calibration_quantile":.95, "image_score":"top-1% pixel mean", "cq":False,
        "display_map_source":"cache_rsg4_mechanism/overview_walnuts__defect.npz",
        "display_map_keys":[f"r{r}_normalized_l{l}" for r in SCALES for l in (5,11,23)],
        "display_map_status":"linear0125 reconstructed from FP16 cached bag-averaged, resized layer maps; not a new evaluation",
        "support_histogram_source":"overview_walnuts__support.npz: lso_l5, lso_l23 / lso_u_l5, lso_u_l23",
        "ground_truth_use":"final display contour only",
    }
    OUT.with_suffix(".meta.json").write_text(json.dumps(metadata,indent=2)+"\n")
    plt.close(c.fig)
    print(OUT)


if __name__ == "__main__":
    main()
