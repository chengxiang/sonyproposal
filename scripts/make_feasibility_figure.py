"""Render the language comparison and retain documented feasibility data."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
DATA = {
    "elf": {
        "source": "elf_l_grant_summary.md, matched epoch-12 comparison (2026-09-12)",
        "display_name": "Continuous diffusion language model",
        "synchronous_accuracy_percent": 54.1698,
        "asynchronous_accuracy_percent": 61.9409,
        "denoiser_parameters": 795205824,
        "denoiser_parameters_millions": 795,
        "parameter_count_source": "Exact denoiser count provided by the investigator; excludes the frozen prompt encoder",
        "prompt_encoder": "Frozen Qwen3-4B-Instruct-2507; encodes the prompt once",
        "training": "Math-trained denoiser generates continuous answer states",
        "ode_steps": 32, "distinct_test_questions": 1319,
        "generation_trials": 2638, "inference_seeds": [42, 123],
        "same_checkpoint": True, "ema": 0.9999,
    },
    "language_context": {
        "source": "Nie et al., Large Language Diffusion Models, arXiv v3, Table 1",
        "url": "https://arxiv.org/html/2502.09992v3",
        "comparison": "Published benchmark context; training and evaluation settings differ",
        "baselines": [
            {"name": "Llama-3-8B Base", "generation": "Autoregressive", "gsm8k_accuracy_percent": 48.7, "shots": 4},
            {"name": "LLaDA-8B Base", "generation": "Discrete diffusion", "gsm8k_accuracy_percent": 70.3, "shots": 4},
        ],
    },
    "schedule": {
        "source": "Learning When to Denoise, Table 2; Semantic Flow Diffusion, Table 2; matched AutoGuidance + dopri5 sampling",
        "url": "https://arxiv.org/html/2606.19662v1",
        "sfd_url": "https://arxiv.org/html/2512.04926v1",
        "lwd_epochs": 200, "sfd_xl_epochs": 800,
        "lwd_autoguidance_fid": 1.05, "sfd_xl_autoguidance_fid": 1.06,
        "sampler": "dopri5", "denoiser_parameters_millions": 675,
        "final_public_result": {
            "source": "Learning When to Denoise, arXiv v1, Table 2",
            "lwd": {"epochs": 600, "parameters_millions": 675, "autoguidance_fid": 1.02},
            "sfd_xxl": {"epochs": 800, "parameters_millions": 1000, "autoguidance_fid": 1.04},
        },
    },
    "sharing": {
        "source": "experiment1.pdf, page 6, matched-budget architecture table",
        "baseline": {"name": "DiT F1024", "parameters": 10215472, "fid_50k": 17.574},
        "shared": {"name": "Shared DiT-MLP H14/d32/W256", "parameters": 10239536, "fid_50k": 14.304},
        "dataset": "CelebA64", "training_seeds": 1, "checkpoint_epoch": 400,
        "sampler_steps": 50,
    },
}


def build_feasibility():
    """The same 7-inch composition is used for the PNG, SVG, and print PDF."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#203747", "text.color": "#203747",
        "xtick.color": "#405666", "ytick.color": "#405666",
    })
    fig, ax = plt.subplots(figsize=(7, 2.2))
    fig.subplots_adjust(left=.095, right=.985, top=.84, bottom=.31)
    vals = [DATA["language_context"]["baselines"][0]["gsm8k_accuracy_percent"],
            DATA["elf"]["asynchronous_accuracy_percent"],
            DATA["language_context"]["baselines"][1]["gsm8k_accuracy_percent"]]
    ticks = ["Llama 3–8B Base\nAutoregressive",
             "Ours · 795M denoiser\nContinuous latent diffusion",
             "LLaDA–8B Base\nDiscrete diffusion"]
    fig.text(.095, .953, "Language reasoning through continuous latent diffusion",
             fontsize=11, weight="bold", ha="left", va="center")
    bars = ax.bar(range(3), vals, width=.48,
                  color=["#8793A3", "#167E83", "#8793A3"])
    ax.set_xticks(range(3), ticks, fontsize=10)
    ax.set_ylim(0, 82)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_ylabel("GSM8K accuracy (%)", fontsize=10, labelpad=6)
    ax.tick_params(axis="both", labelsize=10, length=0, pad=5)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#c9d3dc")
    ax.yaxis.grid(True, color="#e6ebef", lw=.7)
    ax.set_axisbelow(True)
    for bar, value, label in zip(bars, vals, ["48.7", "61.94", "70.3"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2,
                label, ha="center", va="bottom", fontsize=11, weight="bold")
    fig.text(.095, .035, "Published benchmark context; training and evaluation settings differ.",
             fontsize=10, color="#536575", ha="left", va="center")
    return fig


if __name__ == "__main__":
    (OUT / "feasibility_data.json").write_text(json.dumps(DATA, indent=2) + "\n")
    fig = build_feasibility()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"fig4_existing_feasibility.{ext}", dpi=220, facecolor="white")
    plt.close(fig)
