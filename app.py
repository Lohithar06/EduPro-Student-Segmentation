"""
app.py
================================================================================
EduPro - Student Segmentation & Personalized Course Recommendation Dashboard
--------------------------------------------------------------------------------
A professional multi-page Streamlit dashboard that presents:

    1. Dashboard        -> KPI overview (revenue, learners, courses, segments)
    2. Student Explorer  -> Search & inspect any learner's profile
    3. Recommendations   -> Personalized course suggestions per learner
    4. Analytics         -> Segment / category / rating / revenue visual analytics

Run with:
    streamlit run app.py

Data dependency:
    This app reads data/clustered_learners.csv, which is produced by
    train_model.py. If that file is missing, the app shows a clear one-line
    instruction instead of crashing.
================================================================================
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from recommendation import load_data, recommend_for_user

# ------------------------------------------------------------------------------
# Page configuration & global style
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="EduPro | Learner Intelligence Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --edu-primary: #4F46E5;
    --edu-secondary: #10B981;
    --edu-bg: rgba(255, 255, 255, 0.55);
}
.stApp {
    background: linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 40%, #ECFEFF 100%);
}
.glass-card {
    background: var(--edu-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 18px;
    border: 1px solid rgba(255, 255, 255, 0.35);
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.10);
    padding: 1.1rem 1.4rem;
    margin-bottom: 1rem;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #312E81;
}
.kpi-label {
    font-size: 0.85rem;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #312E81 0%, #4F46E5 100%);
}
section[data-testid="stSidebar"] * { color: #F5F3FF !important; }
h1, h2, h3 { color: #312E81; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DATA_DIR = "data"


# ------------------------------------------------------------------------------
# Cached data loading
# ------------------------------------------------------------------------------
@st.cache_data
def load_all():
    """Load every dataset once per session; cached for performance."""
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    courses = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
    transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    clustered_path = os.path.join(DATA_DIR, "clustered_learners.csv")
    if not os.path.exists(clustered_path):
        return users, courses, transactions, None
    clustered = pd.read_csv(clustered_path)
    return users, courses, transactions, clustered


users_df, courses_df, transactions_df, clustered_df = load_all()

if clustered_df is None:
    st.error(
        "`data/clustered_learners.csv` not found. Please run `python train_model.py` "
        "first to train the segmentation model and generate learner clusters."
    )
    st.stop()


# ------------------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------------------
st.sidebar.markdown("## 🎓 EduPro")
st.sidebar.caption("Learner Intelligence Platform")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "🔎 Student Explorer", "🎯 Recommendations", "📈 Analytics"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Total learners: **{len(users_df)}**")
st.sidebar.caption(f"Total courses: **{len(courses_df)}**")
st.sidebar.caption(f"Total transactions: **{len(transactions_df)}**")


def kpi_card(label, value, col):
    col.markdown(
        f"""<div class="glass-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )


# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================
if page == "📊 Dashboard":
    st.title("📊 EduPro Learner Intelligence Dashboard")
    st.caption("A real-time overview of platform health, learner segments, and revenue.")

    total_revenue = transactions_df["Amount"].sum()
    active_learners = transactions_df["UserID"].nunique()
    total_courses = len(courses_df)
    total_transactions = len(transactions_df)
    total_segments = clustered_df["ClusterName"].nunique()
    avg_spending = transactions_df["Amount"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi_card("Total Revenue", f"₹{total_revenue:,.0f}", c1)
    kpi_card("Active Learners", f"{active_learners:,}", c2)
    kpi_card("Course Count", f"{total_courses:,}", c3)
    kpi_card("Transactions", f"{total_transactions:,}", c4)
    kpi_card("Segments", f"{total_segments}", c5)
    kpi_card("Avg. Spending", f"₹{avg_spending:,.0f}", c6)

    st.markdown("### Learner Segment Overview")
    seg_counts = clustered_df["ClusterName"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Learners"]
    fig = px.pie(seg_counts, names="Segment", values="Learners", hole=0.45,
                 color_discrete_sequence=px.colors.qualitative.Prism)
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Recent Transactions")
    recent = transactions_df.sort_values("TransactionDate", ascending=False).head(10)
    st.dataframe(recent, use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 2: STUDENT EXPLORER
# ==============================================================================
elif page == "🔎 Student Explorer":
    st.title("🔎 Student Explorer")
    st.caption("Search for any learner by ID, name, or email to view their full profile.")

    search_type = st.radio("Search by:", ["UserID", "Name", "Email"], horizontal=True)
    query = st.text_input("Enter search value:")

    match = pd.DataFrame()
    if query:
        if search_type == "UserID":
            match = clustered_df[clustered_df["UserID"].str.contains(query, case=False, na=False)]
        elif search_type == "Name":
            match = clustered_df[clustered_df["UserName"].str.contains(query, case=False, na=False)]
        else:
            merged = clustered_df.merge(users_df[["UserID", "Email"]], on="UserID", how="left")
            match = merged[merged["Email"].str.contains(query, case=False, na=False)]

    if query and match.empty:
        st.warning("No learner found matching that search.")
    elif not match.empty:
        selected_id = st.selectbox("Select learner:", match["UserID"].tolist())
        profile = clustered_df[clustered_df["UserID"] == selected_id].iloc[0]
        user_email = users_df.loc[users_df["UserID"] == selected_id, "Email"].values
        user_email = user_email[0] if len(user_email) else "N/A"

        st.markdown(
            f"""<div class="glass-card">
                <h3>{profile['UserName']} <span style="font-size:0.9rem;color:#6B7280;">
                ({selected_id})</span></h3>
                <p>📧 {user_email} &nbsp;|&nbsp; 🎂 Age: {profile['Age']} &nbsp;|&nbsp;
                ⚧ {profile['Gender']}</p>
                <p><b>Segment:</b> {profile['ClusterName']}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        kpi_card("Total Courses", int(profile["TotalCourses"]), m1)
        kpi_card("Total Spending", f"₹{profile['TotalSpending']:,.0f}", m2)
        kpi_card("Avg. Rating Taken", f"{profile['AvgRating']:.2f}", m3)
        kpi_card("Diversity Score", f"{profile['DiversityScore']:.2f}", m4)

        st.markdown("#### Enrollment History")
        history = transactions_df[transactions_df["UserID"] == selected_id].merge(
            courses_df, on="CourseID", how="left"
        ).sort_values("TransactionDate", ascending=False)
        st.dataframe(
            history[["TransactionDate", "CourseID", "CourseCategory", "CourseLevel",
                     "CourseType", "Amount"]],
            use_container_width=True, hide_index=True,
        )


# ==============================================================================
# PAGE 3: RECOMMENDATIONS
# ==============================================================================
elif page == "🎯 Recommendations":
    st.title("🎯 Personalized Course Recommendations")
    st.caption("Cluster-aware hybrid recommendation engine (collaborative + content-based).")

    selected_id = st.selectbox("Select a learner:", clustered_df["UserID"].tolist())
    top_n = st.slider("Number of recommendations:", min_value=3, max_value=10, value=5)

    if st.button("Generate Recommendations", type="primary"):
        recs = recommend_for_user(selected_id, users_df, courses_df, transactions_df,
                                    clustered_df, top_n=top_n)
        if recs.empty:
            st.info("No new recommendations available -- this learner may already be "
                     "enrolled in every relevant course.")
        else:
            r1, r2, r3 = st.columns(3)
            kpi_card("Recommendations Generated", len(recs), r1)
            kpi_card("Avg. Recommended Rating", f"{recs['CourseRating'].mean():.2f}", r2)
            kpi_card("Top Score", f"{recs['RecommendationScore'].max():.2f}", r3)

            st.markdown("#### Recommended Courses")
            st.dataframe(recs, use_container_width=True, hide_index=True)

            csv_bytes = recs.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Recommendations (CSV)",
                data=csv_bytes,
                file_name=f"recommendations_{selected_id}.csv",
                mime="text/csv",
            )


# ==============================================================================
# PAGE 4: ANALYTICS
# ==============================================================================
elif page == "📈 Analytics":
    st.title("📈 Learning Analytics")
    st.caption("Deep-dive visual analytics across learners, courses, and revenue.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Segment Distribution", "Course Insights", "Revenue Trend", "Ratings"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            seg_counts = clustered_df["ClusterName"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Learners"]
            fig = px.bar(seg_counts, x="Segment", y="Learners", color="Segment",
                         color_discrete_sequence=px.colors.qualitative.Prism,
                         title="Learner Segment Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            spend_by_segment = clustered_df.groupby("ClusterName")["AvgSpending"].mean().reset_index()
            fig2 = px.bar(spend_by_segment, x="ClusterName", y="AvgSpending", color="ClusterName",
                          color_discrete_sequence=px.colors.qualitative.Prism,
                          title="Average Spending per Segment")
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            cat_counts = courses_df["CourseCategory"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Courses"]
            fig3 = px.pie(cat_counts, names="Category", values="Courses", hole=0.4,
                          title="Course Category Distribution")
            st.plotly_chart(fig3, use_container_width=True)
        with col2:
            level_counts = courses_df["CourseLevel"].value_counts().reset_index()
            level_counts.columns = ["Level", "Courses"]
            fig4 = px.bar(level_counts, x="Level", y="Courses", color="Level",
                          title="Course Level Distribution")
            st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        tx = transactions_df.copy()
        tx["TransactionDate"] = pd.to_datetime(tx["TransactionDate"])
        tx["Month"] = tx["TransactionDate"].dt.to_period("M").astype(str)
        monthly_revenue = tx.groupby("Month")["Amount"].sum().reset_index()
        fig5 = px.line(monthly_revenue, x="Month", y="Amount", markers=True,
                       title="Revenue Trend Over Time")
        st.plotly_chart(fig5, use_container_width=True)

    with tab4:
        fig6 = px.histogram(courses_df, x="CourseRating", nbins=15, color="CourseType",
                            title="Course Rating Distribution")
        st.plotly_chart(fig6, use_container_width=True)


# ------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------
st.markdown("---")
st.caption("EduPro Student Segmentation & Personalized Course Recommendation System · "
           "Built with Streamlit, scikit-learn & Plotly")
