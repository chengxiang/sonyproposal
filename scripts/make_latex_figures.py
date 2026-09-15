#!/usr/bin/env python3
"""Compact, print-ready versions of the proposal's four existing figures.

Run from any directory. All PDFs are exactly 7 inches wide, with labels of
at least 10 points. Schematics remain illustrative; quantitative values and
image crops are copied unchanged from the documented original figures.
Detailed experimental qualifications belong in the proposal captions.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.image as mpimg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "latex" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
QA = ROOT.parent / "latex_figure_qa"
QA.mkdir(exist_ok=True)
INK, MUTED = "#203342", "#536575"
BLUE, TEAL, ORANGE = "#2865A6", "#167E83", "#C8782F"
GRAY, RULE = "#75879A", "#C9D3DC"
PB, PT, PO = "#EEF4FA", "#EDF7F5", "#FCF4EB"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "text.color": INK,
                     "axes.labelcolor": INK, "axes.edgecolor": RULE})

def canvas(height):
    fig = plt.figure(figsize=(7, height), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return fig, ax

def text(ax, x, y, s, size=10, weight="normal", color=INK,
         ha="left", va="center", **kwargs):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
                   ha=ha, va=va, linespacing=1.2, **kwargs)

def box(ax, x, y, w, h, color="white", edge=RULE):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=.012",
                 facecolor=color, edgecolor=edge, linewidth=.8))

def arrow(ax, start, end, color=MUTED, both=False):
    ax.add_patch(FancyArrowPatch(start, end,
                 arrowstyle="<->" if both else "-|>",
                 mutation_scale=9, linewidth=1.2, color=color,
                 shrinkA=0, shrinkB=0))

def vector(ax, x, y, w, h, vals, color):
    rgb = np.array(matplotlib.colors.to_rgb(color))
    cell = w / len(vals)
    for k, val in enumerate(vals):
        ax.add_patch(Rectangle((x+k*cell, y), cell*.93, h,
                    facecolor=(1-val)*np.ones(3)+val*rgb,
                    edgecolor="white", linewidth=.3))

def save(fig, basename):
    # Do not use bbox_inches="tight": physical text sizes assume 7-inch width.
    fig.savefig(OUT / (basename+".pdf"), facecolor="white")
    fig.savefig(QA / (basename+".png"), dpi=180, facecolor="white")
    plt.close(fig)

def overview():
    from make_conceptual_figures import build_overview
    save(build_overview(), "fig1_mechanism_overview")


def recovery():
    from make_conceptual_figures import build_recovery_schedule
    save(build_recovery_schedule(), "fig2_recovery_schedule")


def reuse():
    fig, ax = canvas(2.6)
    xs=[.023,.195,.367]
    w=.127; h=w*7/2.6
    ys=[.470,.025]
    for x,label in zip(xs,["Source","Source QK +\nadapted VO/MLP","Jointly\nadapted"]):
        text(ax,x+w/2,.932,label,10,"bold",ha="center")
    crops=[(148,120,384,356),(1602,120,1839,356),(1845,120,2081,356)]
    for filename,label,y,col in zip(["source_transfer_aahq.png","source_transfer_stl10.png"],
                                    ["CelebA → AAHQ","CelebA → STL-10"],ys,[TEAL,ORANGE]):
        text(ax,.023,y+h+.014,label,10,"bold",col,va="bottom")
        source=mpimg.imread(ROOT/"figures"/filename)
        assert source.shape[:2]==(1174,2099)
        for x,(l,t,r,b) in zip(xs,crops):
            iax=fig.add_axes([x,y,w,h])
            iax.imshow(source[t:b,l:r],interpolation="nearest",aspect="auto")
            iax.axis("off")
    text(ax,.620,.954,"Recovery with source QK",10.5,"bold")
    text(ax,.620,.880,"Joint endpoint = 1",10,color=MUTED)
    gax=fig.add_axes([.686,.224,.256,.550])
    values=[.815,.143]
    gax.barh([1,0],values,height=.35,color=[TEAL,ORANGE],zorder=3)
    gax.set(xlim=(0,1.08),ylim=(-.5,1.5),yticks=[1,0],xticks=[0,.5,1])
    gax.set_yticklabels(["AAHQ","STL-10"])
    gax.set_xticklabels(["0","0.5","1"])
    gax.tick_params(axis="both",length=0,labelsize=10,pad=4)
    gax.spines["left"].set_visible(False)
    gax.grid(axis="x",color="#e3e8ec",lw=.7)
    for y,value,col in zip([1,0],values,[TEAL,ORANGE]):
        gax.text(value+.020,y,f"{value:.3f}",fontsize=10.5,fontweight="bold",color=col,va="center")
    text(ax,.814,.064,"Normalized DINOv2\nrecovery",10,color=MUTED,ha="center")
    save(fig,"fig3_component_reuse")

def feasibility():
    data=json.loads((ROOT/"figures"/"feasibility_data.json").read_text())
    fig, axes=plt.subplots(1,3,figsize=(7,2.65))
    fig.subplots_adjust(left=.075,right=.995,top=.74,bottom=.225,wspace=.65)
    vals=[[data["elf"]["synchronous_accuracy_percent"],data["elf"]["asynchronous_accuracy_percent"]],
          [data["schedule"]["sfd_xl_epochs"],data["schedule"]["lwd_epochs"]],
          [data["sharing"]["baseline"]["fid_50k"],data["sharing"]["shared"]["fid_50k"]]]
    for k,(ax,values,title,sub,ticks,ylabel,limit) in enumerate(zip(axes,vals,
       ["A. Qwen states","B. Learned schedules","C. Shared MLPs"],
       ["Same ELF checkpoint","Comparable FID (~1.05)","About 10.2M weights"],
       [["Sync.","Async."],["SFD-XL\nFID 1.06","LWD\nFID 1.05"],["DiT","Shared\nDiT-MLP"]],
       ["GSM8K accuracy (%)","Training epochs","FID-50k"], [75,1000,22])):
        pos=ax.get_position()
        # Titles aligned above their panels; no oversized figure headline.
        fig.text(pos.x0+pos.width/2,.933,title,fontsize=10.5,fontweight="bold",ha="center")
        fig.text(pos.x0+pos.width/2,.856,sub,fontsize=10,color=MUTED,ha="center")
        bars=ax.bar([0,1],values,width=.59,color=["#7497BD",TEAL])
        ax.set_xticks([0,1],ticks,fontsize=10)
        ax.set_ylim(0,limit)
        ax.set_ylabel(ylabel,fontsize=10,labelpad=3)
        ax.tick_params(axis="both",labelsize=10,length=2,pad=3)
        ax.grid(axis="y",color="#e6ebef",lw=.6)
        ax.set_axisbelow(True)
        for b,value in zip(bars,values):
            label=f"{value:.2f}" if k==0 else f"{value:.0f}" if k==1 else f"{value:.3f}"
            ax.text(b.get_x()+b.get_width()/2,value+limit*.025,label,ha="center",fontsize=10.5,fontweight="bold")
    save(fig,"fig4_existing_feasibility")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", type=int, choices=[1, 2, 3, 4],
                        default=[1, 2, 3, 4], help="Figure numbers to rebuild")
    selected = parser.parse_args().figures
    makers = {1: overview, 2: recovery, 3: reuse, 4: feasibility}
    for number in selected:
        makers[number]()
    print(f"Created 7-inch-wide PDF figures {selected}; all labels are at least 10 pt.")
