import pandas as pd
import pickle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)
import logging
import os
from datetime import datetime


logging.basicConfig(
    filename='logs/pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_results():
    print("Step 1: Loading all results...")
    df = pd.read_csv('data/processed/cleaned_data.csv')
    summary = pd.read_csv('data/processed/summary_statistics.csv')
    correlations = pd.read_csv('data/processed/correlations.csv')
    print("   All results loaded successfully")
    logging.info("Results loaded for report generation")
    return df, summary, correlations


def generate_excel_report(df, summary, correlations):
    print("\nStep 2: Generating Excel report...")
    excel_path = 'reports/patient_analysis_report.xlsx'

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

        # Sheet 1 — Summary Statistics
        summary.to_excel(
            writer,
            sheet_name='Summary Statistics',
            index=False
        )

        # Sheet 2 — Correlation Results
        correlations.to_excel(
            writer,
            sheet_name='Correlations',
            index=False
        )

        # Sheet 3 — Readmission by Age
        age_readmission = df.groupby('age')['readmitted'].mean() * 100
        age_readmission = age_readmission.reset_index()
        age_readmission.columns = ['Age Group', 'Readmission Rate (%)']
        age_readmission.to_excel(
            writer,
            sheet_name='Readmission By Age',
            index=False
        )

        # Sheet 4 — Gender Distribution
        gender_dist = df['gender'].value_counts().reset_index()
        gender_dist.columns = ['Gender', 'Count']
        gender_dist.to_excel(
            writer,
            sheet_name='Gender Distribution',
            index=False
        )

        # Sheet 5 — Inpatient Visits Analysis
        inpatient_data = df.groupby('number_inpatient')['readmitted'].mean() * 100
        inpatient_data = inpatient_data.reset_index()
        inpatient_data.columns = ['Inpatient Visits', 'Readmission Rate (%)']
        inpatient_data.to_excel(
            writer,
            sheet_name='Inpatient Analysis',
            index=False
        )

        # Sheet 6 — Diagnoses vs Medications
        diag_med = df.groupby('number_diagnoses')['num_medications'].mean()
        diag_med = diag_med.reset_index()
        diag_med.columns = ['Number of Diagnoses', 'Avg Medications']
        diag_med.to_excel(
            writer,
            sheet_name='Diagnoses vs Medications',
            index=False
        )

        # Sheet 7 — Model Performance
        model_performance = pd.DataFrame({
            'Metric': [
                'Total Patients',
                'Readmission Rate',
                'Model Used',
                'Model Accuracy',
                'Recall (Readmitted)',
                'Training Patients',
                'Testing Patients'
            ],
            'Value': [
                f"{len(df):,}",
                f"{df['readmitted'].mean() * 100:.2f}%",
                'Logistic Regression',
                '66.89%',
                '52%',
                '81,412',
                '20,354'
            ]
        })
        model_performance.to_excel(
            writer,
            sheet_name='Model Performance',
            index=False
        )

    print(f"    Excel report saved to {excel_path}")
    logging.info("Excel report generated successfully")


def generate_pdf_report(df):
    print("\nStep 3: Generating PDF report...")

    pdf_path = 'reports/patient_analysis_report.pdf'
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    content = []

    
    content.append(Paragraph(
        "Patient Readmission Risk Analysis Report",
        styles['Title']
    ))
    content.append(Spacer(1, 10))
    content.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles['Normal']
    ))
    content.append(Spacer(1, 20))

    content.append(Paragraph("1. Project Overview", styles['Heading1']))
    content.append(Spacer(1, 10))
    content.append(Paragraph(
        "This report presents an end-to-end analysis of diabetic patient "
        "readmission risk using real hospital data from 130 US hospitals. "
        "The goal is to predict whether a patient will be readmitted within "
        "30 days of discharge using machine learning.",
        styles['Normal']
    ))
    content.append(Spacer(1, 20))

   
    content.append(Paragraph("2. Dataset Summary", styles['Heading1']))
    content.append(Spacer(1, 10))

    dataset_data = [
        ['Metric', 'Value'],
        ['Total Patients', f"{len(df):,}"],
        ['Total Features', str(df.shape[1])],
        ['Readmitted Patients', f"{df['readmitted'].sum():,}"],
        ['Not Readmitted', f"{(df['readmitted'] == 0).sum():,}"],
        ['Readmission Rate', f"{df['readmitted'].mean() * 100:.2f}%"],
        ['Average Hospital Stay', f"{df['time_in_hospital'].mean():.2f} days"],
        ['Average Medications', f"{df['num_medications'].mean():.2f}"],
        ['Average Lab Procedures', f"{df['num_lab_procedures'].mean():.2f}"],
        ['Average Diagnoses', f"{df['number_diagnoses'].mean():.2f}"],
    ]

    table = Table(dataset_data, colWidths=[250, 200])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.steelblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
            [colors.white, colors.lightblue]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    content.append(table)
    content.append(Spacer(1, 20))

  
  
    content.append(Paragraph("3. Key Findings", styles['Heading1']))
    content.append(Spacer(1, 10))

    findings = [
        "Patients with more inpatient visits are most likely to be readmitted",
        "Readmitted patients take significantly more medications (p-value < 0.05)",
        "Average hospital stay is 4.40 days across all patients",
        "Female patients slightly outnumber male patients in this dataset",
        "Only 11.16% of patients were readmitted within 30 days",
        "Number of diagnoses strongly correlates with number of medications",
        "Elderly patients (60-70) show highest readmission rates",
        "Patients with more emergency visits have higher readmission risk",
    ]
    for finding in findings:
        content.append(Paragraph(f"• {finding}", styles['Normal']))
        content.append(Spacer(1, 8))

    content.append(Spacer(1, 20))

    content.append(Paragraph("4. Statistical Analysis", styles['Heading1']))
    content.append(Spacer(1, 10))

    stats_data = [
        ['Metric', 'Value'],
        ['Hypothesis Test', 'Independent T-Test'],
        ['T-Statistic', '12.2690'],
        ['P-Value', '0.0000'],
        ['Result', 'Significant difference found'],
        ['Time in Hospital Skewness', '1.1340'],
        ['Medications Skewness', '1.3267'],
        ['Lab Procedures Skewness', '-0.2365'],
    ]

    stats_table = Table(stats_data, colWidths=[250, 200])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.steelblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
            [colors.white, colors.lightblue]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    content.append(stats_table)
    content.append(Spacer(1, 20))

    content.append(Paragraph("5. Model Performance", styles['Heading1']))
    content.append(Spacer(1, 10))

    model_data = [
        ['Metric', 'Value'],
        ['Model Used', 'Logistic Regression'],
        ['Training Patients', '81,412'],
        ['Testing Patients', '20,354'],
        ['Overall Accuracy', '66.89%'],
        ['Recall (Readmitted)', '52%'],
        ['ROC-AUC Score', '0.6464'],
    ]

    model_table = Table(model_data, colWidths=[250, 200])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.steelblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
            [colors.white, colors.lightblue]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    content.append(model_table)
    content.append(Spacer(1, 20))

    content.append(PageBreak())
    content.append(Paragraph("6. Visualizations", styles['Heading1']))
    content.append(Spacer(1, 10))

    charts = [
        ('reports/readmission_by_age.png',
            'Readmission Rate by Age Group'),
        ('reports/readmission_pie.png',
            'Readmission Distribution'),
        ('reports/gender_donut.png',
            'Gender Distribution'),
        ('reports/correlation_heatmap.png',
            'Feature Correlation Heatmap'),
        ('reports/confusion_matrix.png',
            'Model Confusion Matrix'),
        ('reports/medication_distribution.png',
            'Medication Distribution'),
        ('reports/scatter_medications_hospital.png',
            'Medications vs Hospital Stay'),
        ('reports/violin_medications.png',
            'Medication Violin Plot'),
        ('reports/line_diagnoses_medications.png',
            'Diagnoses vs Medications Trend'),
        ('reports/area_lab_procedures.png',
            'Lab Procedures Distribution'),
        ('reports/inpatient_readmission.png',
            'Inpatient Visits vs Readmission'),
        ('reports/pairplot_key_features.png',
            'Pair Plot of Key Features'),
        ('reports/time_in_hospital.png',
            'Time in Hospital vs Readmission'),
    ]

    
        

    for chart_path, chart_title in charts:
        if os.path.exists(chart_path):
            content.append(Paragraph(chart_title, styles['Heading2']))
            content.append(Spacer(1, 10))
            img = Image(chart_path, width=450, height=280)
            content.append(img)
            content.append(Spacer(1, 20))


    doc.build(content)
    print(f"    PDF report saved to {pdf_path}")
    logging.info("PDF report generated successfully")


if __name__ == "__main__":
    df, summary, correlations = load_results()
    generate_excel_report(df, summary, correlations)
    generate_pdf_report(df)
    print("\n Reports Generated Successfully!")
    print("   Check your reports/ folder!")

