"""
generate_data.py
-----------------
One-time synthetic data generator for the EduPro Student Segmentation project.

This script creates three realistic, internally-consistent CSV datasets:
    1. users.csv          -> learner demographics
    2. courses.csv        -> course catalogue
    3. transactions.csv   -> enrollment / purchase history

The data is generated with fixed random seeds so results are 100% reproducible
for evaluation, viva, and GitHub submission.

NOTE: This script only needs to be run ONCE to seed the `data/` folder.
      The actual ML pipeline (train_model.py) consumes these files.
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_USERS = 600
N_COURSES = 80

# ----------------------------------------------------------------------------
# 1. USERS.CSV
# ----------------------------------------------------------------------------
FIRST_NAMES = ["Aarav","Vihaan","Aditya","Vivaan","Arjun","Sai","Reyansh","Ayaan",
               "Krishna","Ishaan","Ananya","Diya","Saanvi","Aadhya","Myra","Aarohi",
               "Anika","Kavya","Riya","Pihu","Rohan","Karthik","Naveen","Suresh",
               "Divya","Meera","Priya","Sneha","Pooja","Neha","Arun","Vikram",
               "Deepak","Manoj","Sanjay","Kiran","Lakshmi","Swathi","Harini","Gokul"]
LAST_NAMES = ["Sharma","Verma","Iyer","Nair","Reddy","Rao","Menon","Pillai","Gupta",
              "Singh","Patel","Kumar","Krishnan","Subramanian","Raman","Das","Mishra",
              "Chopra","Bose","Kapoor"]

def make_name(used):
    while True:
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        if name not in used:
            used.add(name)
            return name

used_names = set()
user_ids = [f"U{str(i).zfill(4)}" for i in range(1, N_USERS + 1)]
names = [make_name(used_names) for _ in range(N_USERS)]
ages = np.random.randint(17, 45, size=N_USERS)
genders = np.random.choice(["Male", "Female", "Other"], size=N_USERS, p=[0.52, 0.45, 0.03])
emails = [f"{n.lower().replace(' ', '.')}{i}@edupromail.com" for i, n in enumerate(names)]

users_df = pd.DataFrame({
    "UserID": user_ids,
    "UserName": names,
    "Age": ages,
    "Gender": genders,
    "Email": emails
})

# ----------------------------------------------------------------------------
# 2. COURSES.CSV
# ----------------------------------------------------------------------------
CATEGORIES = ["Data Science", "Web Development", "Cloud Computing", "Cybersecurity",
              "AI & Machine Learning", "Business Analytics", "Mobile Development",
              "Design (UI/UX)", "Digital Marketing", "Finance & Accounting"]
COURSE_TYPES = ["Free", "Paid", "Certification"]
COURSE_LEVELS = ["Beginner", "Intermediate", "Advanced"]

course_ids = [f"C{str(i).zfill(3)}" for i in range(1, N_COURSES + 1)]
course_category = np.random.choice(CATEGORIES, size=N_COURSES)
course_type = np.random.choice(COURSE_TYPES, size=N_COURSES, p=[0.25, 0.55, 0.20])
course_level = np.random.choice(COURSE_LEVELS, size=N_COURSES, p=[0.4, 0.4, 0.2])
course_rating = np.round(np.random.normal(loc=4.2, scale=0.4, size=N_COURSES).clip(2.5, 5.0), 1)

courses_df = pd.DataFrame({
    "CourseID": course_ids,
    "CourseCategory": course_category,
    "CourseType": course_type,
    "CourseLevel": course_level,
    "CourseRating": course_rating
})

# ----------------------------------------------------------------------------
# 3. TRANSACTIONS.CSV
# ----------------------------------------------------------------------------
# Learner archetypes are baked in so that clustering later finds *real* structure
# rather than pure noise:
#   - Explorers   : many courses, many categories, low avg spending
#   - Specialists : few categories, deep enrollment in one domain
#   - Career-focused / Premium: fewer but expensive certification courses, high spend
archetype_pool = np.random.choice(
    ["explorer", "specialist", "career_focused"],
    size=N_USERS, p=[0.4, 0.35, 0.25]
)

start_date = datetime(2023, 1, 1)
end_date = datetime(2025, 12, 31)
date_range_days = (end_date - start_date).days

def random_date():
    return start_date + timedelta(days=random.randint(0, date_range_days))

def price_for(course_type_):
    if course_type_ == "Free":
        return 0.0
    if course_type_ == "Paid":
        return round(np.random.uniform(299, 2499), 2)
    return round(np.random.uniform(1999, 7999), 2)  # Certification

transactions = []
courses_by_category = courses_df.groupby("CourseCategory")["CourseID"].apply(list).to_dict()
all_course_ids = courses_df["CourseID"].tolist()

for uid, archetype in zip(user_ids, archetype_pool):
    if archetype == "explorer":
        n_tx = np.random.randint(8, 20)
        chosen_courses = np.random.choice(all_course_ids, size=n_tx, replace=False)
    elif archetype == "specialist":
        fav_category = np.random.choice(CATEGORIES)
        pool = courses_by_category.get(fav_category, all_course_ids)
        n_tx = np.random.randint(4, 10)
        n_tx = min(n_tx, len(pool)) if len(pool) > 0 else 0
        chosen_courses = np.random.choice(pool, size=max(n_tx, 1), replace=False) if pool else []
    else:  # career_focused
        cert_courses = courses_df[courses_df["CourseType"] == "Certification"]["CourseID"].tolist()
        n_tx = np.random.randint(2, 6)
        pool = cert_courses if cert_courses else all_course_ids
        n_tx = min(n_tx, len(pool))
        chosen_courses = np.random.choice(pool, size=max(n_tx, 1), replace=False)

    for cid in chosen_courses:
        ctype = courses_df.loc[courses_df["CourseID"] == cid, "CourseType"].values[0]
        transactions.append({
            "UserID": uid,
            "CourseID": cid,
            "TransactionDate": random_date().strftime("%Y-%m-%d"),
            "Amount": price_for(ctype)
        })

transactions_df = pd.DataFrame(transactions)
transactions_df = transactions_df.sort_values(["UserID", "TransactionDate"]).reset_index(drop=True)

# ----------------------------------------------------------------------------
# Save all datasets
# ----------------------------------------------------------------------------
users_df.to_csv("data/users.csv", index=False)
courses_df.to_csv("data/courses.csv", index=False)
transactions_df.to_csv("data/transactions.csv", index=False)

print(f"users.csv          -> {users_df.shape}")
print(f"courses.csv        -> {courses_df.shape}")
print(f"transactions.csv   -> {transactions_df.shape}")
print("Synthetic datasets generated successfully.")
