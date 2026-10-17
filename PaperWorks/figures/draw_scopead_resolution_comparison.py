"""Plot Q1 resolution results from the table in outlines/Experiments.md."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parent.parent
    text = (root / "outlines/Experiments.md").read_text()
    section = text.split("### Q1.", 1)[1].split("### Q2.", 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if line.startswith("|") and len(cells) == 3:
            if cells[0] in {"448", "672", "896", "1120"} or cells[0].startswith("4 해상도"):
                rows.append(cells)
    assert len(rows) == 5, "Expected four single-resolution rows and one fusion row"
    resolutions = [int(row[0]) for row in rows[:4]]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharex=True, sharey=True)
    for column, (ax, shot) in enumerate(zip(axes, [2, 16]), start=1):
        scores = [float(row[column].split("/")[0]) for row in rows[:4]]
        fusion = float(rows[4][column].split("/")[0])
        ax.plot(resolutions, scores, "o-", color="#2764A5", lw=1.8, ms=5, label="Single resolution")
        ax.axhline(fusion, color="#C05732", ls=(0, (5, 3)), lw=1.8, label="4-resolution fusion")
        best = max(range(4), key=scores.__getitem__)
        arrow_x = resolutions[best] + (62 if best < 3 else -62)
        ax.annotate("", xy=(arrow_x, fusion), xytext=(arrow_x, scores[best]),
                    arrowprops={"arrowstyle": "<->", "color": "#444444", "lw": 1.1})
        ax.text(arrow_x - 22, (fusion + scores[best]) / 2, f"+{fusion - scores[best]:.2f} pp",
                ha="right", va="center", fontsize=9, color="#333333")
        for x, y in zip(resolutions, scores):
            ax.annotate(f"{y:.2f}", (x, y), xytext=(0, -15), textcoords="offset points",
                        ha="center", fontsize=8.5, color="#24496F")
        ax.text(1145, fusion + 0.28, f"{fusion:.2f}", ha="right", color="#A34425", fontsize=9)
        ax.set_title(f"({'ab'[column - 1]}) {shot}-shot", fontsize=11)
        ax.set_xticks(resolutions)
        ax.set_xlim(395, 1173)
        ax.set_ylim(49.5, 61.3)
        ax.set_yticks([50, 52, 54, 56, 58, 60])
        ax.set_xlabel("Single input resolution (short side, px)", fontsize=9)
        ax.grid(axis="y", color="#E4E7EB", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("AUPRO@0.05 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.53, 1.01))
    fig.text(0.52, 0.025, "MVTec AD 2 public | cap 256 per resolution",
             ha="center", fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.79, bottom=0.22, wspace=0.14)
    for extension in ("png", "svg"):
        output = root / "figures" / f"fig_scopead_resolution_comparison.{extension}"
        fig.savefig(output, dpi=240, facecolor="white")
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
