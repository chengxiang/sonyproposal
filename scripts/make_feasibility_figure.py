"""Render documented language, schedule, and parameter-sharing results."""
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
        "denoiser_parameters_millions": 795,
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
    fig, axes = plt.subplots(1, 3, figsize=(7, 3.0),
                             gridspec_kw={"width_ratios": [1.2, 1, 1]})
    fig.subplots_adjust(left=.078, right=.988, top=.70, bottom=.28, wspace=.65)
    vals = [[DATA["language_context"]["baselines"][0]["gsm8k_accuracy_percent"],
             DATA["elf"]["asynchronous_accuracy_percent"],
             DATA["language_context"]["baselines"][1]["gsm8k_accuracy_percent"]],
            [DATA["schedule"]["sfd_xl_epochs"], DATA["schedule"]["lwd_epochs"]],
            [DATA["sharing"]["baseline"]["fid_50k"], DATA["sharing"]["shared"]["fid_50k"]]]
    titles = ["A. Language reasoning", "B. Learned schedules", "C. Shared MLPs"]
    subs = ["Published context;\nsettings differ", "Comparable FID (~1.05)\n675M parameters", "CelebA64\n10.2M parameters"]
    ticks = [["Llama-3\n8B Base", "Ours", "LLaDA\n8B Base"],
             ["SFD-XL\nFID 1.06", "LWD\nFID 1.05"], ["DiT", "Shared\nDiT-MLP"]]
    ylabels = ["GSM8K accuracy (%)", "Training epochs", "FID-50k"]
    limits = [85, 1000, 22]
    colors = [["#7497BD", "#167E83", "#8793A3"], ["#7497BD", "#167E83"], ["#7497BD", "#167E83"]]
    for k, ax in enumerate(axes):
        pos = ax.get_position()
        fig.text(pos.x0 + pos.width / 2, .956, titles[k], fontsize=10.5,
                 weight="bold", ha="center")
        fig.text(pos.x0 + pos.width / 2, .827, subs[k], fontsize=10,
                 color="#536575", ha="center", va="center", linespacing=1.2)
        positions = range(len(vals[k]))
        bars = ax.bar(positions, vals[k], width=.58, color=colors[k])
        ax.set_xticks(list(positions), ticks[k], fontsize=10)
        ax.set_ylim(0, limits[k])
        ax.set_ylabel(ylabels[k], fontsize=10, labelpad=3)
        ax.tick_params(axis="both", labelsize=10, length=2, pad=3)
        ax.yaxis.grid(True, color="#e6ebef", lw=.6)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, vals[k]):
            label = f"{value:.1f}" if k == 0 else f"{value:.0f}" if k == 1 else f"{value:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, value + limits[k] * .025,
                    label, ha="center", va="bottom", fontsize=10.5, weight="bold")
    return fig


if __name__ == "__main__":
    (OUT / "feasibility_data.json").write_text(json.dumps(DATA, indent=2) + "\n")
    fig = build_feasibility()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"fig4_existing_feasibility.{ext}", dpi=220, facecolor="white")
    plt.close(fig)
