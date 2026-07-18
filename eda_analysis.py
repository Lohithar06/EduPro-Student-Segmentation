"""
eda_analysis.py
--------------------------------------------------------------------------------
Exploratory Data Analysis (EDA) supporting the EduPro research paper.
Generates additional charts (beyond train_model.py's diagnostic plots) and
prints summary statistics used directly in the research paper's EDA section.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

DATA_DIR = "data"
DIAGRAM_DIR = "diagrams"
os.makedirs(DIAGRAM_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
PALETTE = ["#4F46E5", "#10B981", "#F59E0B", "#DB2777", "#0EA5E9", "#7C3AED"]


def main():
    users = pd.read_csv(f"{DATA_DIR}/users.csv")
    courses = pd.read_csv(f"{DATA_DIR}/courses.csv")
    transactions = pd.read_csv(f"{DATA_DIR}/transactions.csv")
    clustered = pd.read_csv(f"{DATA_DIR}/clustered_learners.csv")
    transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"])

    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print(f"Users: {users.shape} | Courses: {courses.shape} | Transactions: {transactions.shape}")
    print(f"\nMissing values (users):\n{users.isnull().sum()}")
    print(f"\nMissing values (transactions):\n{transactions.isnull().sum()}")
    print(f"\nAge distribution:\n{users['Age'].describe()}")
    print(f"\nGender distribution:\n{users['Gender'].value_counts()}")
    print(f"\nRevenue total: Rs. {transactions['Amount'].sum():,.2f}")
    print(f"Avg transaction value: Rs. {transactions['Amount'].mean():,.2f}")

    # ---- Chart 1: Age distribution -----------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(users["Age"], bins=20, color=PALETTE[0], edgecolor="white")
    ax.set_title("Learner Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Learners")
    plt.tight_layout()
    plt.savefig(f"{DIAGRAM_DIR}/eda_age_distribution.png")
    plt.close()

    # ---- Chart 2: Gender distribution --------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    gender_counts = users["Gender"].value_counts()
    ax.pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%",
           colors=PALETTE, startangle=90, wedgeprops={"edgecolor": "white"})
    ax.set_title("Gender Distribution")
    plt.tight_layout()
    plt.savefig(f"{DIAGRAM_DIR}/eda_gender_distribution.png")
    plt.close()

    # ---- Chart 3: Course category popularity (by enrollment count) --------
    merged = transactions.merge(courses, on="CourseID", how="left")
    cat_counts = merged["CourseCategory"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(cat_counts.index, cat_counts.values, color=PALETTE[1])
    ax.set_title("Enrollment Count by Course Category")
    ax.set_xlabel("Number of Enrollments")
    plt.tight_layout()
    plt.savefig(f"{DIAGRAM_DIR}/eda_category_enrollment.png")
    plt.close()

    # ---- Chart 4: Monthly revenue trend ------------------------------------
    tx = transactions.copy()
    tx["Month"] = tx["TransactionDate"].dt.to_period("M").astype(str)
    monthly = tx.groupby("Month")["Amount"].sum()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(monthly.index, monthly.values, marker="o", color=PALETTE[3], linewidth=1.8)
    ax.set_title("Monthly Revenue Trend")
    ax.set_ylabel("Revenue (INR)")
    plt.xticks(rotation=75, fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{DIAGRAM_DIR}/eda_revenue_trend.png")
    plt.close()

    # ---- Chart 5: Correlation heatmap of learner features -------------------
    feature_cols = ["Age", "TotalCourses", "TotalCategories", "AvgSpending",
                     "TotalSpending", "AvgRating", "DiversityScore",
                     "LearningDepthIndex", "EnrollmentFrequencyDays"]
    corr = clustered[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_cols)))
    ax.set_yticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=90, fontsize=8)
    ax.set_yticklabels(feature_cols, fontsize=8)
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{DIAGRAM_DIR}/eda_correlation_heatmap.png")
    plt.close()

    # ---- Chart 6: PCA 2D cluster visualization ------------------------------
    from sklearn.preprocessing import StandardScaler
    X = clustered[feature_cols].copy()
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cname in enumerate(clustered["ClusterName"].unique()):
        mask = clustered["ClusterName"] == cname
        ax.scatter(coords[mask, 0], coords[mask, 1], s=22, alpha=0.7,
                   label=cname, color=PALETTE[i % len(PALETTE)])
    ax.set_title(f"PCA Projection of Learner Segments "
                 f"(explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{DIAGRAM_DIR}/eda_pca_clusters.png")
    plt.close()

    print("\nAll EDA charts saved to diagrams/ directory.")

    # ---- Print cluster profile table for the research paper ----------------
    print("\n" + "=" * 70)
    print("CLUSTER PROFILE SUMMARY (means per segment)")
    print("=" * 70)
    profile = clustered.groupby("ClusterName")[feature_cols].mean().round(2)
    profile["LearnerCount"] = clustered["ClusterName"].value_counts()
    print(profile.to_string())
    profile.to_csv(f"{DATA_DIR}/cluster_profile_summary.csv")


if __name__ == "__main__":
    main()
