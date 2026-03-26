import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import os


logging.basicConfig(
    filename='logs/pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_cleaned_data(filepath='data/processed/cleaned_data.csv'):
    logging.info("Loading cleaned data for visualization")
    df = pd.read_csv(filepath)
    print(f" Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def plot_readmission_by_age(df):
    logging.info("Plotting readmission rate by age group")

    age_readmission = df.groupby('age')['readmitted'].mean() * 100
    age_readmission = age_readmission.reset_index()
    age_readmission.columns = ['Age Group', 'Readmission Rate (%)']

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=age_readmission,
        x='Age Group',
        y='Readmission Rate (%)',
        palette='Blues_d'
    )
    plt.title('Readmission Rate by Age Group', fontsize=14)
    plt.xlabel('Age Group')
    plt.ylabel('Readmission Rate (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/readmission_by_age.png')
    plt.close()
    print(" Readmission by Age plot saved!")
    logging.info("Readmission by age plot saved")

def plot_correlation_heatmap(df):
    logging.info("Plotting correlation heatmap")

    numerical_df = df.select_dtypes(include='number')
    corr_matrix = numerical_df.corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap='coolwarm',
        linewidths=0.5
    )
    plt.title('Feature Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig('reports/correlation_heatmap.png')
    plt.close()
    print(" Correlation Heatmap saved!")
    logging.info("Correlation heatmap saved")


def plot_medication_distribution(df):
    logging.info("Plotting medication distribution")

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df,
        x='num_medications',
        hue='readmitted',
        bins=30,
        palette='Set1'
    )
    plt.title('Medication Distribution by Readmission Status', fontsize=14)
    plt.xlabel('Number of Medications')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('reports/medication_distribution.png')
    plt.close()
    print(" Medication Distribution plot saved!")
    logging.info("Medication distribution plot saved")


def plot_time_in_hospital(df):
    logging.info("Plotting time in hospital boxplot")

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df,
        x='readmitted',
        y='time_in_hospital',
        hue='readmitted',
    
        palette='Set2',
        legend=False
    )
    plt.title('Time in Hospital vs Readmission Status', fontsize=14)
    plt.xlabel('Readmitted (0=No, 1=Yes)')
    plt.ylabel('Time in Hospital (days)')
    plt.tight_layout()
    plt.savefig('reports/time_in_hospital.png')
    plt.close()
    print(" Time in Hospital plot saved!")
    logging.info("Time in hospital plot saved")


def plot_interactive_dashboard(df):
    logging.info("Creating interactive Plotly dashboard")

    
    age_readmission = df.groupby('age')['readmitted'].mean() * 100
    age_readmission = age_readmission.reset_index()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Readmission Rate by Age',
            'Time in Hospital Distribution',
            'Gender Distribution',
            'Inpatient Visits vs Readmission'
        )
    )

    fig.add_trace(
        go.Bar(
            x=age_readmission['age'],
            y=age_readmission['readmitted'],
            name='Readmission Rate',
            marker_color='steelblue'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Histogram(
            x=df['time_in_hospital'],
            name='Time in Hospital',
            marker_color='coral'
        ),
        row=1, col=2
    )


    gender_counts = df['gender'].value_counts()
    fig.add_trace(
        go.Bar(
            x=gender_counts.index,
            y=gender_counts.values,
            name='Gender',
            marker_color='mediumseagreen'
        ),
        row=2, col=1
    )

   
    fig.add_trace(
        go.Box(
            x=df['readmitted'].astype(str),
            y=df['number_inpatient'],
            name='Inpatient Visits',
            marker_color='mediumpurple'
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=800,
        title_text="🏥 Patient Readmission Interactive Dashboard",
        showlegend=False
    )

    fig.write_html('reports/dashboard.html')
    print(" Interactive Dashboard saved to reports/dashboard.html!")
    logging.info("Interactive dashboard saved")

def plot_readmission_pie(df):
    logging.info("Plotting readmission pie chart")

    readmission_counts = df['readmitted'].value_counts()
    labels = ['Not Readmitted', 'Readmitted']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0, 0.1)

    plt.figure(figsize=(8, 8))
    plt.pie(
        readmission_counts,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct='%1.1f%%',
        shadow=True,
        startangle=140
    )
    plt.title('Patient Readmission Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig('reports/readmission_pie.png')
    plt.close()
    print(" Readmission Pie Chart saved!")
    logging.info("Readmission pie chart saved")

def plot_gender_donut(df):
    logging.info("Plotting gender donut chart")

    gender_counts = df['gender'].value_counts()
    colors = ['#3498db', '#e91e8c', '#95a5a6']

    plt.figure(figsize=(8, 8))
    plt.pie(
        gender_counts,
        labels=gender_counts.index,
        colors=colors,
        autopct='%1.1f%%',
        pctdistance=0.85,
        startangle=140
    )

    
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.title('Gender Distribution of Patients', fontsize=14)
    plt.tight_layout()
    plt.savefig('reports/gender_donut.png')
    plt.close()
    print(" Gender Donut Chart saved!")
    logging.info("Gender donut chart saved")


def plot_scatter_age_hospital(df):
    logging.info("Plotting scatter plot")

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df['num_medications'],
        df['time_in_hospital'],
        c=df['readmitted'],
        cmap='RdYlGn_r',
        alpha=0.3,
        s=10
    )
    plt.colorbar(scatter, label='Readmitted (1=Yes, 0=No)')
    plt.title('Medications vs Time in Hospital\n(colored by readmission)', fontsize=14)
    plt.xlabel('Number of Medications')
    plt.ylabel('Time in Hospital (days)')
    plt.tight_layout()
    plt.savefig('reports/scatter_medications_hospital.png')
    plt.close()
    print(" Scatter Plot saved!")
    logging.info("Scatter plot saved")


def plot_violin_medications(df):
    logging.info("Plotting violin plot")

    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df,
        x='readmitted',
        y='num_medications',
        hue='readmitted',
        palette={0: '#2ecc71', 1: '#e74c3c'},
        legend=False
    )
    plt.title('Medication Distribution by Readmission Status', fontsize=14)
    plt.xlabel('Readmitted (0=No, 1=Yes)')
    plt.ylabel('Number of Medications')
    plt.tight_layout()
    plt.savefig('reports/violin_medications.png')
    plt.close()
    print(" Violin Plot saved!")
    logging.info("Violin plot saved")


def plot_line_diagnoses_medications(df):
    logging.info("Plotting line chart")

    line_data = df.groupby('number_diagnoses')['num_medications'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(
        line_data.index,
        line_data.values,
        color='steelblue',
        linewidth=2,
        marker='o',
        markersize=5
    )
    plt.fill_between(
        line_data.index,
        line_data.values,
        alpha=0.2,
        color='steelblue'
    )
    plt.title('Average Medications vs Number of Diagnoses', fontsize=14)
    plt.xlabel('Number of Diagnoses')
    plt.ylabel('Average Number of Medications')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/line_diagnoses_medications.png')
    plt.close()
    print(" Line Chart saved!")
    logging.info("Line chart saved")


def plot_area_lab_procedures(df):
    logging.info("Plotting area chart")

    lab_data = df['num_lab_procedures'].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    plt.fill_between(
        lab_data.index,
        lab_data.values,
        alpha=0.5,
        color='mediumpurple'
    )
    plt.plot(
        lab_data.index,
        lab_data.values,
        color='mediumpurple',
        linewidth=2
    )
    plt.title('Distribution of Lab Procedures', fontsize=14)
    plt.xlabel('Number of Lab Procedures')
    plt.ylabel('Number of Patients')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/area_lab_procedures.png')
    plt.close()
    print(" Area Chart saved!")
    logging.info("Area chart saved")


def plot_pairplot(df):
    logging.info("Plotting pair plot")

    pair_cols = [
        'time_in_hospital',
        'num_medications',
        'num_lab_procedures',
        'number_diagnoses',
        'readmitted'
    ]

    plt.figure(figsize=(12, 10))
    pair_plot = sns.pairplot(
        df[pair_cols],
        hue='readmitted',
        palette={0: '#2ecc71', 1: '#e74c3c'},
        plot_kws={'alpha': 0.3},
        diag_kind='kde'
    )
    pair_plot.fig.suptitle(
        'Pair Plot of Key Features',
        y=1.02,
        fontsize=14
    )
    plt.savefig('reports/pairplot_key_features.png')
    plt.close()
    print(" Pair Plot saved!")
    logging.info("Pair plot saved")


def plot_inpatient_readmission(df):
    logging.info("Plotting inpatient visits bar chart")

    inpatient_data = df.groupby('number_inpatient')['readmitted'].mean() * 100
    inpatient_data = inpatient_data.reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=inpatient_data,
        x='number_inpatient',
        y='readmitted',
        hue='number_inpatient',
        palette='Reds',
        legend=False
    )
    plt.title('Readmission Rate by Previous Inpatient Visits', fontsize=14)
    plt.xlabel('Number of Previous Inpatient Visits')
    plt.ylabel('Readmission Rate (%)')
    plt.tight_layout()
    plt.savefig('reports/inpatient_readmission.png')
    plt.close()
    print(" Inpatient Visits Chart saved!")
    logging.info("Inpatient visits chart saved")

if __name__ == "__main__":

    df = load_cleaned_data()

    # Original plots
    plot_readmission_by_age(df)
    plot_correlation_heatmap(df)
    plot_medication_distribution(df)
    plot_time_in_hospital(df)
    plot_interactive_dashboard(df)

    # New plots
    plot_readmission_pie(df)
    plot_gender_donut(df)
    plot_scatter_age_hospital(df)
    plot_violin_medications(df)
    plot_line_diagnoses_medications(df)
    plot_area_lab_procedures(df)
    plot_pairplot(df)
    plot_inpatient_readmission(df)

    print("\n🎉 All Visualizations Complete!")
    print("   Check your reports/ folder!")
