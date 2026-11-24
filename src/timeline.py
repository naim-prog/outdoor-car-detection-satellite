import datetime as dt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"],
    "svg.fonttype": "path",
})

INK, MUTED, GRID = "#1F2937", "#6B7280", "#E5E7EB"
COLORS = {
    "Research": "#94A3B8",
    "Data": "#14B8A6",
    "Training": "#F97316",
    "Web map": "#F59E0B",
    "Inference": "#3B82F6",
    "Evaluation": "#8B5CF6",
}

D = lambda d, m: dt.date(2026, m, d)
TASKS = [
    ("Model benchmarking", "YOLO OBB 8x, 11x, 26x · DOTAv1.0 pre-trained", "Research", D(17, 9), D(19, 9)),
    ("Imagery download", "1,000+ CNIG images · province of Cádiz", "Data", D(21, 9), D(22, 9)),
    ("Labeling, round 1", "Auto-label + manual review · ~1,100 images", "Data", D(25, 9), D(29, 9)),
    ("Training experiments", "Layer freezing, epochs, hyperparameters", "Training", D(29, 9), D(30, 9)),
    ("MBTiles creation", "gdal_translate", "Web map", D(30, 9), D(15, 10)),
    ("Zoom levels", "gdaladdo overviews", "Web map", D(15, 10), D(21, 10)),
    ("PMTiles conversion", "Size reduction, PMTiles export, web testing", "Web map", D(21, 10), D(23, 10)),
    ("Labeling, round 2", "+600 images · ~1,800 total (80/20 train/val)", "Data", D(24, 10), D(2, 11)),
    ("Baseline inference", "YOLO26x-OBB base · whole dataset", "Inference", D(3, 11), D(7, 11)),
    ("Fine-tuning", "YOLO26x-OBB · own labeled dataset", "Training", D(7, 11), D(15, 11)),
    ("Fine-tuned inference", "Fine-tuned YOLO26x-OBB · whole dataset", "Inference", D(15, 11), D(22, 11)),
    ("Results export", "Test dataset", "Evaluation", D(24, 11), D(24, 11)),
]

def span_label(s, e):
    days = (e - s).days + 1
    if s == e:
        rng = f"{s:%b} {s.day}"
    elif s.month == e.month:
        rng = f"{s:%b} {s.day} – {e.day}"
    else:
        rng = f"{s:%b} {s.day} – {e:%b} {e.day}"
    return f"{rng}  ·  {days} d"

n = len(TASKS)
fig = plt.figure(figsize=(12, 7.0), facecolor="white")
ax = fig.add_axes([0.285, 0.085, 0.69, 0.72], facecolor="white")

start0, end0 = D(14, 9), D(3, 12)
ax.set_xlim(mdates.date2num(start0), mdates.date2num(end0))
ax.set_ylim(n - 0.45, -0.65)

# weekly grid (Mondays)
mondays = [start0 + dt.timedelta(days=i) for i in range((end0 - start0).days)
           if (start0 + dt.timedelta(days=i)).weekday() == 0]
for m in mondays:
    ax.axvline(mdates.date2num(m), color=GRID, lw=0.8, zorder=0)
ax.set_xticks([mdates.date2num(m) for m in mondays])
ax.set_xticklabels([f"{m:%b} {m.day}" for m in mondays], fontsize=8, color=MUTED)
ax.xaxis.tick_top()
ax.tick_params(axis="x", length=0, pad=6)
ax.set_yticks([])
for side in ("left", "right", "bottom", "top"):
    ax.spines[side].set_visible(False)

yt = ax.get_yaxis_transform()
for i, (title, detail, cat, s, e) in enumerate(TASKS):
    x0 = mdates.date2num(s)
    w = (e - s).days + 1
    ax.barh(i, w, left=x0, height=0.52, color=COLORS[cat], zorder=3)
    ax.text(x0 + w + 0.5, i, span_label(s, e), va="center", ha="left",
            fontsize=8.5, color=INK, fontweight=600, zorder=4)
    ax.text(-0.015, i - 0.12, title, transform=yt, ha="right", va="center",
            fontsize=10, color=INK, fontweight=700)
    ax.text(-0.015, i + 0.22, detail, transform=yt, ha="right", va="center",
            fontsize=7.8, color=MUTED)

total = (D(24, 11) - D(17, 9)).days + 1
fig.text(0.04, 0.945, "Outdoor Car Detection from Satellite Imagery", fontsize=18,
         fontweight=700, color=INK, ha="left", va="center")
fig.text(0.04, 0.905, f"Project timeline  ·  Sep 17 – Nov 24, 2026  ·  {total} days",
         fontsize=10.5, color=MUTED, ha="left", va="center")

handles = [Patch(facecolor=c, label=k) for k, c in COLORS.items()]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.04, 0.855), ncol=6,
           frameon=False, fontsize=9, handlelength=1.1, handleheight=1.1,
           columnspacing=1.6, labelcolor=INK)

fig.savefig(r"images\project-timeline.svg", facecolor="white")
print("ok", total)
