"""
recommendation.py
================================================================================
EduPro - Personalized Course Recommendation Engine
--------------------------------------------------------------------------------
Cluster-aware, content + collaborative hybrid recommender.

Logic
--------------------------------------------------------------------------------
    1. Load users, courses, transactions, and clustered_learners (from train_model.py)
    2. For a target learner, identify their assigned Cluster
    3. Find "peer" learners in the SAME cluster (collaborative signal)
    4. Aggregate peer enrollments into a popularity ranking (CourseID -> count)
    5. Filter out courses the target learner is already enrolled in
    6. Blend in content-based signal (CourseRating, category match with
       PreferredCategory) to compute a final RecommendationScore
    7. Return / persist the Top-N recommended courses

This module is imported directly by app.py (Streamlit) and can also be run
standalone from the command line to generate a sample CSV for a given user.

Usage (standalone):
    python recommendation.py --user_id U0001 --top_n 5
================================================================================
"""

import os
import argparse
import pandas as pd
import numpy as np

DATA_DIR = "data"

# Tunable blend weights for the final RecommendationScore.
# Popularity = how many similar (same-cluster) learners took this course.
# Rating     = intrinsic course quality (content-based signal).
# CategoryMatch = bonus if the course matches the learner's PreferredCategory.
WEIGHT_POPULARITY = 0.5
WEIGHT_RATING = 0.3
WEIGHT_CATEGORY_MATCH = 0.2


def load_data(data_dir=DATA_DIR):
    """Load all datasets required for recommendation, including the ML output
    from train_model.py (clustered_learners.csv)."""
    try:
        users = pd.read_csv(os.path.join(data_dir, "users.csv"))
        courses = pd.read_csv(os.path.join(data_dir, "courses.csv"))
        transactions = pd.read_csv(os.path.join(data_dir, "transactions.csv"))
        clustered = pd.read_csv(os.path.join(data_dir, "clustered_learners.csv"))
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"{err}. Run train_model.py first to generate clustered_learners.csv."
        ) from err
    return users, courses, transactions, clustered


def get_user_cluster(user_id, clustered_df):
    """Return the Cluster ID and ClusterName for a given UserID, or (None, None)
    if the user is not found."""
    row = clustered_df.loc[clustered_df["UserID"] == user_id]
    if row.empty:
        return None, None
    return int(row.iloc[0]["Cluster"]), row.iloc[0]["ClusterName"]


def get_enrolled_courses(user_id, transactions_df):
    """Return the set of CourseIDs the learner has already purchased/enrolled in."""
    return set(transactions_df.loc[transactions_df["UserID"] == user_id, "CourseID"])


def recommend_for_user(user_id, users, courses, transactions, clustered, top_n=5):
    """
    Generate the Top-N personalized course recommendations for `user_id`.

    Returns
    -------
    pd.DataFrame with columns:
        UserID, CourseID, CourseCategory, CourseType, CourseLevel,
        CourseRating, PeerPopularity, RecommendationScore
    Empty DataFrame (with correct columns) if the user is unknown or every
    course is already taken.
    """
    output_cols = ["UserID", "CourseID", "CourseCategory", "CourseType",
                   "CourseLevel", "CourseRating", "PeerPopularity", "RecommendationScore"]

    cluster_id, cluster_name = get_user_cluster(user_id, clustered)
    if cluster_id is None:
        print(f"[WARN] UserID '{user_id}' not found in clustered_learners.csv.")
        return pd.DataFrame(columns=output_cols)

    # Step 1: peers = everyone else in the same cluster
    peer_ids = clustered.loc[
        (clustered["Cluster"] == cluster_id) & (clustered["UserID"] != user_id), "UserID"
    ]

    # Step 2: peer course popularity (collaborative filtering signal)
    peer_transactions = transactions[transactions["UserID"].isin(peer_ids)]
    popularity = peer_transactions.groupby("CourseID").size().rename("PeerPopularity")

    # Step 3: exclude courses the learner already owns
    already_enrolled = get_enrolled_courses(user_id, transactions)
    candidates = courses[~courses["CourseID"].isin(already_enrolled)].copy()
    candidates = candidates.merge(popularity, on="CourseID", how="left")
    candidates["PeerPopularity"] = candidates["PeerPopularity"].fillna(0)

    if candidates.empty:
        return pd.DataFrame(columns=output_cols)

    # Step 4: content-based signal -- category match with learner preference
    learner_row = clustered.loc[clustered["UserID"] == user_id].iloc[0]
    preferred_category = learner_row.get("PreferredCategory", "None")
    candidates["CategoryMatch"] = (candidates["CourseCategory"] == preferred_category).astype(int)

    # Step 5: normalize each signal to 0-1 range before blending
    def normalize(series):
        rng = series.max() - series.min()
        return (series - series.min()) / rng if rng > 0 else series * 0

    norm_pop = normalize(candidates["PeerPopularity"])
    norm_rating = normalize(candidates["CourseRating"])
    norm_category = candidates["CategoryMatch"]  # already 0/1

    candidates["RecommendationScore"] = (
        WEIGHT_POPULARITY * norm_pop +
        WEIGHT_RATING * norm_rating +
        WEIGHT_CATEGORY_MATCH * norm_category
    ).round(4)

    candidates["UserID"] = user_id
    result = candidates.sort_values("RecommendationScore", ascending=False).head(top_n)

    return result[output_cols].reset_index(drop=True)


def recommend_for_all_users(users, courses, transactions, clustered, top_n=5):
    """Batch-generate recommendations for every learner (used to build
    sample_recommendation.csv for the repository / offline evaluation)."""
    all_recs = []
    for user_id in clustered["UserID"]:
        recs = recommend_for_user(user_id, users, courses, transactions, clustered, top_n)
        all_recs.append(recs)
    return pd.concat(all_recs, ignore_index=True) if all_recs else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="EduPro Recommendation Engine")
    parser.add_argument("--user_id", type=str, default=None,
                         help="Generate recommendations for a single UserID (e.g. U0001).")
    parser.add_argument("--top_n", type=int, default=5, help="Number of recommendations to return.")
    parser.add_argument("--all", action="store_true",
                         help="Generate recommendations for ALL users (sample_recommendation.csv).")
    args = parser.parse_args()

    users, courses, transactions, clustered = load_data()

    if args.all or args.user_id is None:
        print("Generating recommendations for ALL learners (this may take a moment)...")
        all_recs = recommend_for_all_users(users, courses, transactions, clustered, args.top_n)
        out_path = os.path.join(DATA_DIR, "sample_recommendation.csv")
        all_recs.to_csv(out_path, index=False)
        print(f"Saved {len(all_recs)} recommendation rows -> {out_path}")
    else:
        recs = recommend_for_user(args.user_id, users, courses, transactions, clustered, args.top_n)
        if recs.empty:
            print(f"No recommendations available for {args.user_id}.")
        else:
            print(f"\nTop {args.top_n} recommendations for {args.user_id}:\n")
            print(recs.to_string(index=False))


if __name__ == "__main__":
    main()
