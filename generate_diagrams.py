"""
generate_diagrams.py
--------------------------------------------------------------------------------
Generates two presentation-ready PNG diagrams used in the README, research
paper, and viva presentation:

    1. diagrams/system_architecture.png  -> end-to-end system architecture
    2. diagrams/ml_workflow.png          -> ML training/inference workflow

Built with pure matplotlib (no external diagramming dependency) so it runs
anywhere the rest of the project runs.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PRIMARY = "#4F46E5"
SECONDARY = "#10B981"
ACCENT = "#F59E0B"
LIGHT = "#EEF2FF"
TEXT = "#1F2937"


def box(ax, x, y, w, h, text, color=LIGHT, edge=PRIMARY, fontsize=10, fontweight="bold"):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                           linewidth=1.8, edgecolor=edge, facecolor=color, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, fontweight=fontweight, color=TEXT, zorder=3, wrap=True)


def arrow(ax, xy_from, xy_to, color="#6B7280"):
    a = FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>", mutation_scale=16,
                         linewidth=1.6, color=color, zorder=1)
    ax.add_patch(a)


# ==============================================================================
# 1. SYSTEM ARCHITECTURE DIAGRAM
# ==============================================================================
def make_architecture_diagram():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("EduPro -- System Architecture", fontsize=16, fontweight="bold", color=PRIMARY, pad=20)

    # Data layer
    box(ax, 0.4, 6.2, 2.4, 1.0, "users.csv", color="#DBEAFE")
    box(ax, 3.1, 6.2, 2.4, 1.0, "courses.csv", color="#DBEAFE")
    box(ax, 5.8, 6.2, 2.4, 1.0, "transactions.csv", color="#DBEAFE")
    ax.text(0.4, 7.35, "RAW DATA LAYER", fontsize=10, fontweight="bold", color="#1E40AF")

    # ML pipeline layer
    box(ax, 2.0, 4.5, 5.0, 1.0, "train_model.py\n(Feature Engineering + K-Means)", color="#EDE9FE", edge="#7C3AED")
    ax.text(2.0, 5.65, "MACHINE LEARNING PIPELINE", fontsize=10, fontweight="bold", color="#5B21B6")

    box(ax, 8.6, 6.2, 3.8, 1.0, "models/scaler.pkl\nmodels/kmeans_model.pkl", color="#FEF3C7", edge=ACCENT)

    # Output data
    box(ax, 2.0, 2.9, 5.0, 1.0, "data/clustered_learners.csv", color="#D1FAE5", edge=SECONDARY)

    # Recommendation engine
    box(ax, 8.6, 2.9, 3.8, 1.0, "recommendation.py\n(Hybrid Recommender)", color="#EDE9FE", edge="#7C3AED")

    # Application layer
    box(ax, 2.0, 1.0, 5.0, 1.0, "sample_recommendation.csv", color="#D1FAE5", edge=SECONDARY)
    box(ax, 8.6, 1.0, 3.8, 1.0, "app.py\n(Streamlit Dashboard)", color="#FCE7F3", edge="#DB2777")

    ax.text(0.4, 0.05, "End user: Student / Admin / Government Stakeholder", fontsize=9,
            style="italic", color="#6B7280")

    # Arrows
    arrow(ax, (1.6, 6.2), (3.5, 5.5))
    arrow(ax, (4.3, 6.2), (4.5, 5.5))
    arrow(ax, (7.0, 6.2), (5.5, 5.5))
    arrow(ax, (7.0, 5.0), (8.6, 6.5))
    arrow(ax, (4.5, 4.5), (4.5, 3.9))
    arrow(ax, (7.0, 3.4), (8.6, 3.4))
    arrow(ax, (4.5, 2.9), (4.5, 2.0))
    arrow(ax, (10.5, 2.9), (10.5, 2.0))
    arrow(ax, (7.0, 1.5), (8.6, 1.5))

    plt.tight_layout()
    plt.savefig("diagrams/system_architecture.png", dpi=160, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 2. ML WORKFLOW DIAGRAM
# ==============================================================================
def make_ml_workflow_diagram():
    fig, ax = plt.subplots(figsize=(6, 12))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 22)
    ax.axis("off")
    ax.set_title("EduPro -- ML Training Workflow", fontsize=15, fontweight="bold", color=PRIMARY, pad=15)

    steps = [
        ("1. Load Raw Data", "users.csv, courses.csv, transactions.csv", "#DBEAFE", PRIMARY),
        ("2. Merge Datasets", "Join transactions + course metadata", "#DBEAFE", PRIMARY),
        ("3. Feature Engineering", "Engagement, Preference, Behavioural features", "#EDE9FE", "#7C3AED"),
        ("4. Encode Categoricals", "Gender, PreferredCategory, PreferredLevel", "#EDE9FE", "#7C3AED"),
        ("5. Scale Features", "StandardScaler -> mean 0, std 1", "#FEF3C7", ACCENT),
        ("6. Elbow Method", "Test K=2..9, plot inertia + silhouette", "#FEF3C7", ACCENT),
        ("7. Train K-Means", "Fit final model on optimal K", "#D1FAE5", SECONDARY),
        ("8. Name Clusters", "Z-score profiling -> business labels", "#D1FAE5", SECONDARY),
        ("9. Persist Artefacts", "clustered_learners.csv, scaler.pkl, kmeans_model.pkl", "#FCE7F3", "#DB2777"),
    ]

    y = 20.2
    step_h = 1.9
    gap = 0.3
    for title, desc, color, edge in steps:
        box(ax, 0.3, y - step_h, 5.4, step_h, f"{title}\n{desc}", color=color, edge=edge, fontsize=9.5)
        if y - step_h - gap > 0.3:
            arrow(ax, (3.0, y - step_h), (3.0, y - step_h - gap))
        y -= (step_h + gap)

    plt.tight_layout()
    plt.savefig("diagrams/ml_workflow.png", dpi=160, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    make_architecture_diagram()
    make_ml_workflow_diagram()
    print("Diagrams saved to diagrams/system_architecture.png and diagrams/ml_workflow.png")
