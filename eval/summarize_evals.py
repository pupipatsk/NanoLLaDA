import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define the directory containing the score CSV files
SCORE_DIR = "data/score"
REPORT_FILE = "data/report/score_summary_report.txt"
VISUAL_DIR = "data/visualizations"

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def compute_statistics(df, score_column='score'):
    """Compute statistical metrics for the score column."""
    scores = df[score_column]
    return {
        'Mean': scores.mean(),
        'Median': scores.median(),
        'Std': scores.std(),
        'Min': scores.min(),
        'Max': scores.max(),
        'Count': len(scores)
    }

def plot_histogram(scores, filename, title):
    """Generate and save a histogram for the scores."""
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, edgecolor='black')
    plt.title(f"Histogram of Scores: {title}")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()

def plot_boxplot(scores, filename, title, labels=None):
    """Generate and save a box plot for the scores."""
    plt.figure(figsize=(10, 6))
    if labels:
        plt.boxplot(scores, labels=labels)
    else:
        plt.boxplot(scores)
    plt.title(f"Box Plot of Scores: {title}")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()

def main():
    # Ensure visualization directory exists
    ensure_directory_exists(VISUAL_DIR)
    ensure_directory_exists(os.path.dirname(REPORT_FILE))

    # Initialize report content
    report_content = ["Summary of Scores in data/score Folder\n", "="*40 + "\n"]

    # Collect all CSV files in the score directory
    score_files = [f for f in os.listdir(SCORE_DIR) if f.endswith('.csv')]
    
    if not score_files:
        report_content.append("No CSV files found in data/score folder.\n")
        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            f.writelines(report_content)
        return

    # Store data for combined visualization
    all_scores = []
    all_labels = []

    # Process each CSV file
    for file in score_files:
        file_path = os.path.join(SCORE_DIR, file)
        report_content.append(f"\nFile: {file}\n")
        report_content.append("-"*40 + "\n")

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            if 'score' not in df.columns:
                report_content.append("Error: 'score' column not found.\n")
                continue

            # Compute statistics
            stats = compute_statistics(df)
            for key, value in stats.items():
                report_content.append(f"{key}: {value:.4f}\n" if isinstance(value, float) else f"{key}: {value}\n")

            # Generate histogram
            plot_histogram(
                df['score'],
                os.path.join(VISUAL_DIR, f"histogram_{file.replace('.csv', '.png')}"),
                file
            )

            # Generate individual box plot
            plot_boxplot(
                df['score'],
                os.path.join(VISUAL_DIR, f"boxplot_{file.replace('.csv', '.png')}"),
                file
            )

            # Store scores for combined box plot
            all_scores.append(df['score'])
            all_labels.append(file)

        except Exception as e:
            report_content.append(f"Error processing file: {e}\n")

    # Generate combined box plot for all files
    if len(all_scores) > 0:
        plot_boxplot(
            all_scores,
            os.path.join(VISUAL_DIR, "combined_boxplot.png"),
            "Comparison of Scores Across All Files",
            labels=[os.path.basename(label).replace('.csv', '') for label in all_labels]
        )

    # Write the report to a file
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.writelines(report_content)

    print(f"Summary report saved to {REPORT_FILE}")
    print(f"Visualizations saved to {VISUAL_DIR}")

if __name__ == "__main__":
    main()