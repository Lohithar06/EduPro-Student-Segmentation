"""
train_model.py
================================================================================
EduPro - Student Segmentation & Personalized Course Recommendation System
--------------------------------------------------------------------------------
End-to-end Machine Learning training pipeline.

This script is the single source of truth for the clustering stage of the
project. It is intentionally self-contained and re-runnable: executing it
regenerates `clustered_learners.csv`, `models/scaler.pkl`, and
`models/kmeans_model.pkl` from the three raw CSVs (users, courses,
transactions), so the whole pipeline is fully reproducible and auditable
during evaluation / viva.

Pipeline stages
--------------------------------------------------------------------------------
    1. Load raw data                      (users, courses, transactions)
    2. Merge datasets                     (transaction-level fact table)
    3. Feature engineering                (learner-level behavioural features)
    4. Encode categorical features        (preferred category / level)
    5. Scale numerical features           (StandardScaler)
    6. Determine optimal K                (Elbow Method, saved as a chart)
    7. Train final K-Means model
    8. Assign human-readable cluster names (based on cluster centroid profile)
    9. Persist artefacts                  (clustered_learners.csv, *.pkl)

Output contract (must stay compatible with recommendation.py and app.py)
--------------------------------------------------------------------------------
    data/clustered_learners.csv columns:
        UserID, UserName, Age, Gender, Cluster, ClusterName,
        TotalCourses, TotalCategories, AvgSpending, TotalSpending,
        AvgRating, DiversityScore, LearningDepthIndex,
        PreferredCategory, PreferredLevel, EnrollmentFrequencyDays

    models/scaler.pkl    -> fitted sklearn StandardScaler
    models/kmeans_model.pkl -> fitted sklearn KMeans model
================================================================================
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering (no display needed on server / CI)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Configuration constants
# --------------------------------------------------------------------------
DATA_DIR = "data"
MODEL_DIR = "models"
DIAGRAM_DIR = "diagrams"
RANDOM_STATE = 42
MAX_K_TO_TEST = 9  # elbow method searches K = 2..MAX_K_TO_TEST


def ensure_directories():
    """Create output directories if they do not already exist."""
    for directory in (MODEL_DIR, DIAGRAM_DIR):
        os.makedirs(directory, exist_ok=True)


# --------------------------------------------------------------------------
# 1. LOAD RAW DATA
# --------------------------------------------------------------------------
def load_raw_data():
    """
    Load users, courses, and transactions CSVs from the data/ directory.

    Returns
    -------
    tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame)
    """
    try:
        users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
        courses = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
        transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"Required dataset missing: {err}. "
            f"Ensure users.csv, courses.csv, and transactions.csv exist in '{DATA_DIR}/'."
        ) from err

    # Basic sanity checks so failures surface early with a clear message
    required_cols = {
        "users": {"UserID", "UserName", "Age", "Gender", "Email"},
        "courses": {"CourseID", "CourseCategory", "CourseType", "CourseLevel", "CourseRating"},
        "transactions": {"UserID", "CourseID", "TransactionDate", "Amount"},
    }
    for name, df, cols in (
        ("users", users, required_cols["users"]),
        ("courses", courses, required_cols["courses"]),
        ("transactions", transactions, required_cols["transactions"]),
    ):
        missing = cols - set(df.columns)
        if missing:
            raise ValueError(f"'{name}.csv' is missing required columns: {missing}")

    transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"], errors="coerce")

    return users, courses, transactions


# --------------------------------------------------------------------------
# 2 & 3. MERGE + FEATURE ENGINEERING
# --------------------------------------------------------------------------
def build_learner_features(users, courses, transactions):
    """
    Merge the three raw tables and aggregate them into ONE row per learner
    (UserID) containing behavioural, engagement, and preference features.

    Feature groups (as required by the project spec):
        Engagement  : TotalCourses, EnrollmentFrequencyDays
        Preference  : PreferredCategory, PreferredLevel, AvgRating
        Behavioural : AvgSpending, TotalSpending, DiversityScore,
                      LearningDepthIndex

    Returns
    -------
    pd.DataFrame  (one row per UserID; users with zero transactions are kept
                   with NaN behavioural stats replaced by sensible defaults so
                   every learner in users.csv still gets a cluster assignment)
    """
    # Merge transactions with course metadata to know category/level/rating
    # of every purchase.
    merged = transactions.merge(courses, on="CourseID", how="left")

    # ---- Engagement features -------------------------------------------------
    agg = merged.groupby("UserID").agg(
        TotalCourses=("CourseID", "nunique"),
        TotalCategories=("CourseCategory", "nunique"),
        AvgSpending=("Amount", "mean"),
        TotalSpending=("Amount", "sum"),
        AvgRating=("CourseRating", "mean"),
        FirstTransaction=("TransactionDate", "min"),
        LastTransaction=("TransactionDate", "max"),
    ).reset_index()

    # Enrollment frequency: average number of days between enrollments.
    # A learner with many enrollments spread over a short window is highly
    # active; one course = default large gap (treated as low engagement).
    span_days = (agg["LastTransaction"] - agg["FirstTransaction"]).dt.days.clip(lower=1)
    agg["EnrollmentFrequencyDays"] = (span_days / agg["TotalCourses"]).round(1)
    agg.loc[agg["TotalCourses"] <= 1, "EnrollmentFrequencyDays"] = span_days

    # ---- Diversity score (how many distinct categories explored, normalised) --
    total_available_categories = courses["CourseCategory"].nunique()
    agg["DiversityScore"] = (agg["TotalCategories"] / total_available_categories).round(3)

    # ---- Learning depth index (advanced-vs-beginner ratio) --------------------
    level_counts = (
        merged.groupby(["UserID", "CourseLevel"]).size().unstack(fill_value=0)
    )
    for lvl in ["Beginner", "Intermediate", "Advanced"]:
        if lvl not in level_counts.columns:
            level_counts[lvl] = 0
    advanced_weighted = level_counts["Advanced"] * 1.0 + level_counts["Intermediate"] * 0.5
    total_lvl = level_counts[["Beginner", "Intermediate", "Advanced"]].sum(axis=1).replace(0, 1)
    learning_depth = (advanced_weighted / total_lvl).round(3).rename("LearningDepthIndex")
    agg = agg.merge(learning_depth, on="UserID", how="left")

    # ---- Preferred category / level (mode per learner) -------------------------
    preferred_category = (
        merged.groupby("UserID")["CourseCategory"]
        .agg(lambda x: x.value_counts().idxmax())
        .rename("PreferredCategory")
    )
    preferred_level = (
        merged.groupby("UserID")["CourseLevel"]
        .agg(lambda x: x.value_counts().idxmax())
        .rename("PreferredLevel")
    )
    agg = agg.merge(preferred_category, on="UserID", how="left")
    agg = agg.merge(preferred_level, on="UserID", how="left")

    # ---- Join back onto the FULL user list (so every registered learner is
    #      represented, even those with zero enrollments) ------------------------
    learner_df = users.merge(agg, on="UserID", how="left")

    fill_defaults = {
        "TotalCourses": 0, "TotalCategories": 0, "AvgSpending": 0.0,
        "TotalSpending": 0.0, "AvgRating": 0.0, "EnrollmentFrequencyDays": 0.0,
        "DiversityScore": 0.0, "LearningDepthIndex": 0.0,
        "PreferredCategory": "None", "PreferredLevel": "None",
    }
    learner_df = learner_df.fillna(fill_defaults)
    learner_df.drop(columns=["FirstTransaction", "LastTransaction"], inplace=True, errors="ignore")

    return learner_df


# --------------------------------------------------------------------------
# 4. ENCODE CATEGORICAL FEATURES
# --------------------------------------------------------------------------
def encode_categorical(learner_df):
    """
    Label-encode PreferredCategory, PreferredLevel, and Gender so they can be
    fed into K-Means (which requires purely numeric input). Encoders are kept
    only in-memory here since clustering does not need to invert them, but
    the mapping is printed for transparency during training.
    """
    encoders = {}
    for col in ["Gender", "PreferredCategory", "PreferredLevel"]:
        le = LabelEncoder()
        learner_df[col + "_enc"] = le.fit_transform(learner_df[col].astype(str))
        encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    return learner_df, encoders


# --------------------------------------------------------------------------
# 5. SCALE FEATURES
# --------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "Age", "TotalCourses", "TotalCategories", "AvgSpending", "TotalSpending",
    "AvgRating", "DiversityScore", "LearningDepthIndex",
    "EnrollmentFrequencyDays", "Gender_enc", "PreferredCategory_enc", "PreferredLevel_enc",
]


def scale_features(learner_df):
    """Fit a StandardScaler on FEATURE_COLUMNS and return the scaled matrix + scaler.

    WHY StandardScaler?
    K-Means uses Euclidean distance. Features like TotalSpending (range: 0-20,000+)
    would dominate low-range features like DiversityScore (range: 0-1) if left
    unscaled, biasing the clusters toward "spending" alone. StandardScaler
    transforms every feature to mean=0, std=1 so each contributes fairly.
    """
    X = learner_df[FEATURE_COLUMNS].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# --------------------------------------------------------------------------
# 6. ELBOW METHOD
# --------------------------------------------------------------------------
def find_optimal_k(X_scaled, max_k=MAX_K_TO_TEST):
    """
    Run K-Means for K = 2..max_k, record inertia (WCSS) and silhouette score
    for each, save an elbow-curve chart to diagrams/, and return the chosen K.

    WHY Elbow Method?
    Inertia (within-cluster sum of squares) always decreases as K increases,
    so we can't just pick the K with lowest inertia. The elbow method looks
    for the point where adding another cluster stops giving a meaningful
    improvement (the "elbow" in the curve) -- balancing model simplicity
    against explained variance. We cross-check with silhouette score, which
    measures how well-separated clusters are, to avoid an ambiguous elbow.
    """
    inertias = []
    silhouettes = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Plot elbow curve
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(list(k_range), inertias, marker="o", color="#4F46E5", label="Inertia (WCSS)")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia (WCSS)", color="#4F46E5")
    ax1.tick_params(axis="y", labelcolor="#4F46E5")

    ax2 = ax1.twinx()
    ax2.plot(list(k_range), silhouettes, marker="s", color="#10B981", label="Silhouette Score")
    ax2.set_ylabel("Silhouette Score", color="#10B981")
    ax2.tick_params(axis="y", labelcolor="#10B981")

    plt.title("Elbow Method & Silhouette Score for Optimal K")
    fig.tight_layout()
    plt.savefig(os.path.join(DIAGRAM_DIR, "elbow_method.png"), dpi=150)
    plt.close()

    # Choose K with the best silhouette score among a sensible business range
    # (3-6 segments is interpretable for a dashboard; avoids 2-cluster trivial
    # split and avoids >6 which is hard for stakeholders to act on).
    candidate_range = [k for k in k_range if 3 <= k <= 6]
    if not candidate_range:
        candidate_range = list(k_range)
    best_k = max(candidate_range, key=lambda k: silhouettes[list(k_range).index(k)])

    print(f"[Elbow Method] Inertia by K: {dict(zip(k_range, np.round(inertias, 1)))}")
    print(f"[Silhouette]   Score by K:  {dict(zip(k_range, np.round(silhouettes, 3)))}")
    print(f"[Selected K]   -> {best_k}")

    return best_k


# --------------------------------------------------------------------------
# 7. TRAIN FINAL MODEL
# --------------------------------------------------------------------------
def train_kmeans(X_scaled, k):
    """Train the final K-Means model with the chosen number of clusters."""
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"[Final Model]  K={k} | Silhouette Score={score:.3f}")
    return model, labels


# --------------------------------------------------------------------------
# 8. NAME CLUSTERS BASED ON CENTROID PROFILE
# --------------------------------------------------------------------------
def assign_cluster_names(learner_df):
    """
    Translate numeric cluster centroids into business-friendly names.

    Approach: standardise each behavioural metric ACROSS clusters (z-score),
    then greedily assign the most distinctive label to the cluster that
    stands out most on that metric, highest-priority metric first. Any
    cluster left unclaimed falls back to a generic "Balanced Learners" label.
    This keeps naming fully data-driven (works for any K), avoids duplicate
    names, and produces meaningful, board-room-ready segment labels.
    """
    profile = learner_df.groupby("Cluster").agg(
        AvgSpending=("AvgSpending", "mean"),
        DiversityScore=("DiversityScore", "mean"),
        TotalCourses=("TotalCourses", "mean"),
        LearningDepthIndex=("LearningDepthIndex", "mean"),
    )

    z = (profile - profile.mean()) / profile.std(ddof=0).replace(0, 1)

    # Priority-ordered candidate labels: (metric column, ascending?, label)
    candidates = [
        ("AvgSpending", False, "Premium / Career-Focused Learners"),
        ("DiversityScore", False, "Explorers (Multi-Category Learners)"),
        ("LearningDepthIndex", False, "Deep Specialists"),
        ("TotalCourses", True, "Casual / Low-Engagement Learners"),
    ]

    assigned = {}
    claimed_clusters = set()
    for metric, ascending, label in candidates:
        ordered = z[metric].sort_values(ascending=ascending)
        for cluster_id in ordered.index:
            if cluster_id not in claimed_clusters:
                assigned[cluster_id] = label
                claimed_clusters.add(cluster_id)
                break

    # Any leftover clusters (only possible when K > number of candidate labels)
    remaining = [c for c in profile.index if c not in claimed_clusters]
    for i, cluster_id in enumerate(remaining, start=1):
        assigned[cluster_id] = f"Balanced Learners (Segment {i})"

    return learner_df["Cluster"].map(assigned)


# --------------------------------------------------------------------------
# 9. VISUAL CLUSTER SUMMARY (bonus diagnostic chart)
# --------------------------------------------------------------------------
def plot_cluster_summary(learner_df):
    """Save a bar chart of average spending per cluster -- useful sanity check
    that also doubles as an artefact for the research paper / README."""
    fig, ax = plt.subplots(figsize=(8, 5))
    summary = learner_df.groupby("ClusterName")["AvgSpending"].mean().sort_values()
    summary.plot(kind="barh", ax=ax, color="#6366F1")
    ax.set_xlabel("Average Spending (INR)")
    ax.set_title("Average Spending per Learner Segment")
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGRAM_DIR, "cluster_spending_summary.png"), dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("EduPro Student Segmentation -- Training Pipeline")
    print("=" * 80)

    ensure_directories()

    print("\n[1/7] Loading raw data...")
    users, courses, transactions = load_raw_data()
    print(f"      users={len(users)} rows | courses={len(courses)} rows | "
          f"transactions={len(transactions)} rows")

    print("\n[2/7] Merging datasets and engineering learner-level features...")
    learner_df = build_learner_features(users, courses, transactions)
    print(f"      Learner feature table shape: {learner_df.shape}")

    print("\n[3/7] Encoding categorical features...")
    learner_df, encoders = encode_categorical(learner_df)
    print(f"      Encoded columns: Gender, PreferredCategory, PreferredLevel")

    print("\n[4/7] Scaling numerical features (StandardScaler)...")
    X_scaled, scaler = scale_features(learner_df)

    print("\n[5/7] Finding optimal K via Elbow Method + Silhouette Score...")
    best_k = find_optimal_k(X_scaled)

    print(f"\n[6/7] Training final K-Means model with K={best_k}...")
    model, labels = train_kmeans(X_scaled, best_k)
    learner_df["Cluster"] = labels
    learner_df["ClusterName"] = assign_cluster_names(learner_df)

    print("\n[7/7] Saving artefacts (clustered_learners.csv, scaler.pkl, kmeans_model.pkl)...")
    output_cols = [
        "UserID", "UserName", "Age", "Gender", "Cluster", "ClusterName",
        "TotalCourses", "TotalCategories", "AvgSpending", "TotalSpending",
        "AvgRating", "DiversityScore", "LearningDepthIndex",
        "PreferredCategory", "PreferredLevel", "EnrollmentFrequencyDays",
    ]
    learner_df[output_cols].to_csv(os.path.join(DATA_DIR, "clustered_learners.csv"), index=False)

    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "kmeans_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    plot_cluster_summary(learner_df)

    print("\nCluster distribution:")
    print(learner_df["ClusterName"].value_counts().to_string())

    print("\n" + "=" * 80)
    print("Training pipeline complete. Artefacts saved to data/ and models/.")
    print("=" * 80)


if __name__ == "__main__":
    main()
