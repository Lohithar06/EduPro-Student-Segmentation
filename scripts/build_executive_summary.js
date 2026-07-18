const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, ImageRun,
  Header, Footer, PageNumber,
} = require("docx");

const IMG = (name) => fs.readFileSync(`../diagrams/${name}`);

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 260, after: 140 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 180, after: 90 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 120 } });
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
    spacing: { before: 100, after: 60 },
  });
}
function makeTable(headers, rows) {
  const colWidth = Math.floor(9000 / headers.length);
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((htext) => new TableCell({
      width: { size: colWidth, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: "10B981" },
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
  styles: { default: { document: { run: { font: "Calibri", size: 22 } } } },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 } } },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "EduPro — Executive Summary", size: 16, color: "9CA3AF" })],
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
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Executive Summary", bold: true, size: 40, color: "312E81" })],
        spacing: { after: 100 },
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({
          text: "Student Segmentation & Personalized Course Recommendation System for EduPro",
          size: 26, color: "6B7280",
        })],
        spacing: { after: 100 },
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Prepared for Government & Institutional Stakeholders", italics: true, size: 22 })],
        spacing: { after: 400 },
      }),

      h1("Purpose of This Report"),
      p("This executive summary presents the outcomes of a data-driven learner segmentation and "
        + "personalization initiative undertaken on the EduPro online learning platform. It is intended to "
        + "give non-technical, decision-making stakeholders a clear, concise view of the problem addressed, "
        + "the approach taken, the results achieved, and the recommended next steps — without requiring "
        + "familiarity with the underlying machine learning methodology."),

      h1("The Challenge"),
      p("Online learning platforms like EduPro serve large, diverse learner populations. Historically, "
        + "EduPro recommended the same generic set of courses to every learner, regardless of their "
        + "individual interests, spending capacity, or learning style. This approach has three consequences "
        + "relevant to platform sustainability and public value:"),
      bullet("Learners struggle to discover courses relevant to their needs, reducing engagement."),
      bullet("Course completion rates suffer when content does not match learner readiness or interest."),
      bullet("The platform loses opportunities to build long-term learner loyalty and retention."),

      h1("The Approach"),
      p("A complete, reproducible machine learning pipeline was built and deployed to address this "
        + "challenge, structured in three stages:"),
      makeTable(
        ["Stage", "What It Does", "Stakeholder Benefit"],
        [
          ["Segmentation", "Groups learners into behaviourally distinct segments using unsupervised machine learning (K-Means clustering)", "Reveals who is actually using the platform and how"],
          ["Personalization", "Recommends courses tailored to each learner's segment and preferences", "Improves relevance, engagement, and outcomes"],
          ["Analytics Dashboard", "Interactive, real-time dashboard for monitoring learners, revenue, and segments", "Enables transparent, ongoing oversight"],
        ]
      ),

      h1("Key Findings"),
      p("Analysis of 600 learners, 80 courses, and over 5,000 transactions revealed three clear, "
        + "actionable learner segments:"),
      makeTable(
        ["Segment", "Share of Learners", "Defining Characteristic"],
        [
          ["Explorers (Multi-Category Learners)", "39%", "Engage broadly across many course categories"],
          ["Deep Specialists", "40%", "Focus deeply within one or two subject areas"],
          ["Premium / Career-Focused Learners", "21%", "Fewer courses, but highest-value certifications"],
        ]
      ),
      image("eda_pca_clusters.png", 420, 315),
      caption("Visual confirmation that the three learner segments are statistically well-separated."),

      h1("Impact & Value"),
      bullet("A repeatable, auditable analytics pipeline that can be re-run as new learner data arrives."),
      bullet("A personalization engine that recommends genuinely relevant courses instead of generic lists."),
      bullet("A live, interactive dashboard giving administrators and stakeholders continuous visibility "
        + "into learner segments, engagement, and revenue — supporting transparent, evidence-based decisions."),
      bullet("A foundation that generalizes beyond EduPro to any public or institutional e-learning "
        + "initiative seeking to personalize instruction at scale."),

      h1("Recommendations"),
      bullet("Adopt segment-aware learner communication and course promotion strategies."),
      bullet("Periodically retrain the segmentation model as learner behaviour evolves over time."),
      bullet("Extend data collection (e.g., course completion, time-on-platform) to sharpen future segments."),
      bullet("Use the dashboard as a standing tool for ongoing programme monitoring and reporting."),

      h1("Conclusion"),
      p("This initiative demonstrates that meaningful, personalized learning experiences can be delivered "
        + "at scale using transparent, well-understood machine learning methods. The resulting system is "
        + "fully documented, reproducible, and ready for institutional adoption or further extension."),
    ],
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync("../docs/EduPro_Executive_Summary.docx", buffer);
  console.log("Executive summary written to ../docs/EduPro_Executive_Summary.docx");
});
