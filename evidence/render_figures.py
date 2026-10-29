"""Render public numerical evidence; no network access or model execution."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'evidence/current-decision'
FIG=ROOT/'docs/figures'
LABELS={'online':'Recent observations only','hybrid_v1':'Hybrid + targeted recall','cached_hybrid':'Same hybrid, shared calculation','utility':'Utility-based selection','optimized':'Memory + actuator response','fresh_current_response':'Current calibration, no memory'}

def main():
    s=json.loads((DATA/'summary.json').read_text());rows=json.loads((DATA/'episodes.json').read_text());flow=json.loads((DATA/'memory-flow.json').read_text())
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(11.5,5.2))
    values=[s['arms'][a]['mean_rms_uT'] for a in LABELS]
    ax.barh(range(6),values,color=['#bdbdbd','#aaa','#999','#777','#111','#555'])
    ax.set_yticks(range(6),list(LABELS.values()));ax.invert_yaxis();ax.set_xlim(0,.16);ax.set_xlabel('Mean residual RMS (simulated µT; lower is better)')
    for i,v in enumerate(values):ax.text(v+.001,i,f'{v:.6f}',va='center',fontsize=9)
    ax.set_title('Four sequential scenarios · Six controllers · Real Basira / TSST / ODVL calls',pad=16)
    fig.text(.99,.01,'Memory + response: −1.22% vs recent observations; −0.60% vs fresh calibration. Descriptive, not general superiority.',ha='right',fontsize=8,color='#555')
    fig.tight_layout(rect=(0,.035,1,1))
    for ext in ('png','svg'):fig.savefig(FIG/f'results-overview.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(11.5,7.8),gridspec_kw={'height_ratios':[1.1,1.]})
    x=np.arange(4)
    for j,(a,color) in enumerate([('online','#bbb'),('utility','#888'),('optimized','#111'),('fresh_current_response','#555')]):
        v=[next(r['metrics']['rms_uT'] for r in rows if r['arm']==a and r['episode_index']==i) for i in range(44,48)]
        axs[0].bar(x+(j-1.5)*.18,v,width=.17,color=color,label=LABELS[a])
    axs[0].set_xticks(x,['45 · Device A','46 · Device B','47 · A changed','48 · Changed A returns']);axs[0].set_ylabel('Residual RMS (simulated µT)');axs[0].legend(frameon=False,ncol=2,fontsize=8)
    scale=[r['calibration_scale'] for r in flow]
    axs[1].axhline(1,color='#aaa',ls='--');axs[1].plot(x,scale,'ko-',lw=2);axs[1].set_ylim(.75,1.25);axs[1].set_xlim(-.3,3.3)
    axs[1].set_xticks(x,['45','46','47','48']);axs[1].set_xlabel('Episode in the continuing history');axs[1].set_ylabel('Response correction factor\n1 = no remembered calibration applied')
    for j,r in enumerate(flow):
        label=('Applied' if r['calibration_applied'] else 'Rejected')+'\n'+(r['calibration_candidate'] or 'incompatible old response')
        axs[1].annotate(label,(j,scale[j]),xytext=(0,-35 if scale[j]>1.1 else -42),textcoords='offset points',ha='center',fontsize=8)
    fig.suptitle('Recall a recent experience → test it against the present → act or abstain',fontsize=14)
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(FIG/f'cases-and-decisions.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
