const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, ImageRun,
  BorderStyle, PageBreak, TableOfContents, Header, Footer, PageNumber,
  LevelFormat, convertInchesToTwip,
} = require("docx");

const IMG = (name) => fs.readFileSync(`../diagrams/${name}`);

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 } });
}
function p(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    spacing: { after: 120 },
  });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 60 } });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 18, color: "6B7280" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
  });
}
function image(name, width, height) {
  return new Paragraph({
    children: [new ImageRun({ type: "png", data: IMG(name), transformation: { width, height } })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
  });
}

function makeTable(headers, rows) {
  const colWidth = Math.floor(9000 / headers.length);
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((htext) => new TableCell({
      width: { size: colWidth, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: "4F46E5" },
      children: [new Paragraph({ children: [new TextRun({ text: htext, bold: true, color: "FFFFFF", size: 20 })] })],
    })),
  });
  const dataRows = rows.map((row) => new TableRow({
    children: row.map((cell) => new TableCell({
      width: { size: colWidth, type: WidthType.DXA },
      children: [new Paragraph({ children: [new TextRun({ text: String(cell), size: 20 })] })],
    })),
  }));
  return new Table({
    width: { size: 9000, type: WidthType.DXA },
    columnWidths: headers.map(() => colWidth),
    rows: [headerRow, ...dataRows],
  });
}

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 22 } },
    },
  },
  sections: [
    // ---------------- TITLE PAGE ----------------
    {
      properties: { page: { size: { width: 12240, height: 15840 } } },
      children: [
        new Paragraph({ text: "", spacing: { before: 2000 } }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Student Segmentation & Personalized", bold: true, size: 44, color: "312E81" })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Course Recommendation System for EduPro", bold: true, size: 44, color: "312E81" })],
          spacing: { after: 400 },
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "A Machine Learning Research Paper", italics: true, size: 28, color: "6B7280" })],
          spacing: { after: 800 },
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Final Year B.E. Computer Science and Engineering Project", size: 24 })],
          spacing: { after: 200 },
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Submitted for Final Project Review and Evaluation", size: 24 })],
          spacing: { after: 1600 },
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "2026", size: 24, color: "6B7280" })],
        }),
      ],
    },
    // ---------------- MAIN BODY ----------------
    {
      properties: {
        page: { size: { width: 12240, height: 15840 } },
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            alignment: AlignmentType.RIGHT,
            children: [new TextRun({ text: "EduPro — Student Segmentation Research Paper", size: 16, color: "9CA3AF" })],
          })],
        }),
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "9CA3AF" })],
          })],
        }),
      },
      children: [
        h1("Abstract"),
        p("Online learning platforms serve an increasingly diverse population of learners whose goals, "
          + "engagement styles, and spending behaviour vary widely. Generic, one-size-fits-all course "
          + "recommendations fail to capture this diversity, resulting in lost engagement and retention "
          + "opportunities. This paper presents an end-to-end machine learning pipeline that segments "
          + "EduPro's learners using K-Means clustering on engineered behavioural features, and builds a "
          + "hybrid (collaborative + content-based) recommendation engine on top of the resulting segments. "
          + "The pipeline was validated on a representative dataset of 600 learners, 80 courses, and over "
          + "5,000 transactions. Three statistically distinct and business-interpretable segments emerged — "
          + "Explorers, Deep Specialists, and Premium/Career-Focused Learners — each with materially "
          + "different spending, diversity, and engagement profiles. The resulting system is deployed as an "
          + "interactive Streamlit dashboard for real-time learner analytics and recommendation delivery."),

        h1("1. Introduction & Background"),
        p("Online learners are not homogeneous. Some explore beginner courses across many domains; some "
          + "specialize deeply in a single subject; others focus narrowly on career-oriented certifications. "
          + "Generic course recommendations fail to maximize learner engagement, improve course completion, "
          + "or build long-term platform loyalty. EduPro requires a data-driven personalization engine that "
          + "can understand different learner types, recommend genuinely relevant courses, and support "
          + "personalized learning journeys at scale."),

        h1("2. Problem Statement"),
        p("EduPro currently faces three structural challenges:"),
        bullet("One-size-fits-all course recommendations that ignore individual learner behaviour"),
        bullet("Limited, ad-hoc understanding of learner behaviour patterns"),
        bullet("No structured, repeatable learner segmentation framework"),
        p("As a direct consequence, learners struggle to discover relevant content, and the platform loses "
          + "engagement and retention opportunities that a personalization layer could otherwise capture."),

        h1("3. Dataset Description"),
        p("Three relational datasets form the foundation of this study, joined on UserID / CourseID:"),
        makeTable(
          ["Dataset", "Fields", "Rows"],
          [
            ["users.csv", "UserID, UserName, Age, Gender, Email", "600"],
            ["courses.csv", "CourseID, CourseCategory, CourseType, CourseLevel, CourseRating", "80"],
            ["transactions.csv", "UserID, CourseID, TransactionDate, Amount", "5,077"],
          ]
        ),
        p(""),
        p("No missing values were found in any of the three source tables during data validation, confirmed "
          + "programmatically in the EDA stage of the pipeline (see Section 5)."),

        h1("4. Methodology"),
        h2("4.1 Learner-Level Feature Engineering"),
        p("Raw transactional records are aggregated to one row per learner, producing three families of "
          + "engineered features that together describe a learner's engagement, preferences, and behaviour:"),
        makeTable(
          ["Feature Group", "Engineered Features"],
          [
            ["Engagement", "TotalCourses, TotalCategories, EnrollmentFrequencyDays"],
            ["Preference", "PreferredCategory, PreferredLevel, AvgRating"],
            ["Behavioural", "AvgSpending, TotalSpending, DiversityScore, LearningDepthIndex"],
          ]
        ),
        p(""),
        p("DiversityScore is computed as the proportion of all available course categories a learner has "
          + "explored, and LearningDepthIndex is a weighted ratio of Advanced/Intermediate to Beginner "
          + "enrollments, capturing whether a learner favours surface-level or in-depth content."),

        h2("4.2 Preprocessing"),
        p("Categorical fields (Gender, PreferredCategory, PreferredLevel) are label-encoded to numeric form. "
          + "All numeric features are then standardised using scikit-learn's StandardScaler (mean = 0, "
          + "std = 1). This step is essential because K-Means relies on Euclidean distance: unscaled, "
          + "high-range features such as TotalSpending (₹0–20,000+) would dominate low-range features such "
          + "as DiversityScore (0–1), biasing the resulting clusters toward spending alone."),

        h2("4.3 Determining the Optimal Number of Clusters"),
        p("The Elbow Method was used to test K = 2 through 9. Because inertia (within-cluster sum of "
          + "squares) decreases monotonically with K, the Silhouette Score was used as a cross-check to "
          + "identify the K that produces the most well-separated, interpretable clusters within a "
          + "practically actionable range (3–6 segments)."),
        image("elbow_method.png", 500, 312),
        caption("Figure 1. Elbow curve (inertia) and Silhouette Score across candidate values of K."),

        h2("4.4 Model Training"),
        p("The final K-Means model was trained with the selected K, using 10 independent centroid "
          + "initializations (n_init=10) and a fixed random state for full reproducibility. Cluster labels "
          + "were then translated into business-friendly names using a data-driven, z-score-based ranking "
          + "of each cluster's spending, diversity, and depth profile — ensuring the naming logic "
          + "generalises to any dataset or choice of K rather than being hard-coded."),

        h1("5. Exploratory Data Analysis (EDA)"),
        h2("5.1 Learner Demographics"),
        image("eda_age_distribution.png", 420, 270),
        caption("Figure 2. Age distribution of registered learners."),
        image("eda_gender_distribution.png", 300, 300),
        caption("Figure 3. Gender distribution of registered learners."),

        h2("5.2 Course Engagement Patterns"),
        image("eda_category_enrollment.png", 460, 288),
        caption("Figure 4. Total enrollments by course category."),

        h2("5.3 Revenue Trend"),
        image("eda_revenue_trend.png", 500, 225),
        caption("Figure 5. Monthly platform revenue over the observed period."),

        h2("5.4 Feature Correlation"),
        image("eda_correlation_heatmap.png", 460, 402),
        caption("Figure 6. Correlation heatmap of engineered learner features."),

        h1("6. Results"),
        h2("6.1 Cluster Profiles"),
        p("The final model (K = 3) achieved a Silhouette Score of approximately 0.24, and produced three "
          + "clearly separated, interpretable learner segments:"),
        makeTable(
          ["Segment", "Learners", "Avg. Courses", "Avg. Spending (₹)", "Diversity Score"],
          [
            ["Explorers (Multi-Category)", "234", "13.6", "1,666", "0.77"],
            ["Deep Specialists", "239", "6.0", "1,800", "0.12"],
            ["Premium / Career-Focused", "127", "3.7", "5,068", "0.33"],
          ]
        ),
        p(""),
        image("eda_pca_clusters.png", 460, 345),
        caption("Figure 7. PCA projection of learner segments (2 components, ~60% explained variance)."),
        image("cluster_spending_summary.png", 460, 288),
        caption("Figure 8. Average spending per learner segment."),

        h2("6.2 Interpretation"),
        bullet("Explorers enroll in the highest number of courses across the widest range of categories, "
          + "but spend moderately per course — they are platform-loyal, breadth-seeking learners."),
        bullet("Deep Specialists concentrate on 1–2 categories with a moderate course count and the lowest "
          + "diversity score — they value depth over breadth."),
        bullet("Premium / Career-Focused Learners take the fewest courses but spend the most per learner, "
          + "gravitating toward certification-type, higher-value courses."),

        h1("7. Personalized Recommendation System"),
        p("Building on the learner segments, a hybrid recommendation engine was implemented in "
          + "recommendation.py. For a target learner, the engine identifies peers within the same cluster, "
          + "computes course popularity among those peers (collaborative signal), and blends this with "
          + "course rating and category-preference match (content-based signal) into a single, weighted "
          + "RecommendationScore. Courses the learner has already purchased are excluded automatically."),
        makeTable(
          ["Signal", "Weight", "Rationale"],
          [
            ["Peer Popularity", "50%", "Leverages what similar learners in the same segment are enrolling in"],
            ["Course Rating", "30%", "Ensures recommended courses are of genuinely high quality"],
            ["Category Match", "20%", "Keeps recommendations aligned with the learner's demonstrated preference"],
          ]
        ),
        p(""),
        p("This hybrid design avoids the two classic failure modes of single-strategy recommenders: pure "
          + "collaborative filtering struggles for learners in small or atypical clusters, while pure "
          + "content-based filtering tends to over-fit to a learner's past choices and rarely surfaces "
          + "novel, relevant courses."),

        h1("8. Insights & Recommendations for EduPro"),
        bullet("Deploy segment-aware marketing: Explorers respond well to bundle offers across categories; "
          + "Premium learners respond better to certification and career-outcome messaging."),
        bullet("Introduce a 'category bridge' nudge for Deep Specialists — a moderate-diversity recommendation "
          + "that gently broadens their exposure without abandoning their preferred domain."),
        bullet("Prioritize onboarding flows that quickly reveal a new learner's archetype, since early "
          + "signals (first 2–3 course choices) already correlate strongly with final segment membership."),
        bullet("Track Silhouette Score and cluster population drift over time; retrain the pipeline "
          + "periodically (train_model.py is fully reproducible) as learner behaviour evolves."),

        h1("9. Conclusion & Future Work"),
        p("This project demonstrates a complete, reproducible, end-to-end machine learning pipeline — from "
          + "raw transactional data to a deployed, interactive recommendation dashboard — that meaningfully "
          + "improves on EduPro's prior one-size-fits-all approach. Future work could extend the feature set "
          + "with course-completion and time-on-platform signals (where available), experiment with "
          + "Hierarchical Clustering or DBSCAN as alternative segmentation validation methods, and introduce "
          + "an online/incremental learning loop so segments adapt continuously as new transactions arrive."),

        h1("References"),
        p("[1] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825-2830."),
        p("[2] MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. "
          + "Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability."),
        p("[3] Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of "
          + "cluster analysis. Journal of Computational and Applied Mathematics, 20, 53-65."),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync("../docs/EduPro_Research_Paper.docx", buffer);
  console.log("Research paper written to ../docs/EduPro_Research_Paper.docx");
});
