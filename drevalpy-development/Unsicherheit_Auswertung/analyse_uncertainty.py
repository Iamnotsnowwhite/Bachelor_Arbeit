import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
import random  

# Define a default path for convenience, which can be overridden by the command line argument.
DEFAULT_FILE_PATH = "/home/mi/zhaow01/dokumente/BA/drevalpy-development/results/GDSC2/baseline_models_ale_epic/exclude_target_ale_epic/kombinierte_results.csv"

def analyze_uncertainty(file_path):
    """
    Reads uncertainty data, selects ONE random cell line, calculates metrics, 
    and generates plots for that specific cell line.
    
    :param file_path: Path to the input CSV file.
    """
    
    # --- Check and Load Data ---
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return

    try:
        df_full = pd.read_csv(file_path) # <<< GEÄNDERT: Lade in 'df_full'
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Required input columns: ['y_true', 'y_pred', 'sigma_epi', 'sigma_ale', 'sigma_tot']
    # <<< GEÄNDERT: Wir brauchen jetzt auch 'cell_line'
    required_cols_input = ['y_true', 'y_pred', 'sigma_epi', 'sigma_ale', 'sigma_tot', 'cell_line']
    if not all(col in df_full.columns for col in required_cols_input):
        print(f"Error: CSV file must contain columns: {required_cols_input}")
        print(f"Columns found: {df_full.columns.tolist()}")
        return

    # --- NEU: Wähle eine zufällige Zelllinie aus ---
    unique_cell_lines = df_full['cell_line'].unique()
    if len(unique_cell_lines) == 0:
        print("Fehler: Keine Zelllinien in der Spalte 'cell_line' gefunden.")
        return
    
    selected_cell_line = random.choice(unique_cell_lines)
    print(f"==================================================")
    print(f"Zufällige Zelllinie ausgewählt: {selected_cell_line}")
    print(f"==================================================")

    # Filter den DataFrame, um NUR Daten dieser Zelllinie zu enthalten
    df = df_full[df_full['cell_line'] == selected_cell_line].copy()
    
    # Prüfen, ob genügend Datenpunkte für die Analyse vorhanden sind
    if len(df) < 2:
        print(f"Fehler: Nicht genügend Daten (nur {len(df)} Zeile/n) für Zelllinie '{selected_cell_line}', um die Analyse durchzuführen.")
        return
    # --- Ende des neuen Blocks ---


    # --- Data Preprocessing: Calculate metrics for plotting/analysis ---
    # (Dieser Teil bleibt gleich, wird aber jetzt auf dem gefilterten 'df' ausgeführt)
    
    # 1. Calculate Absolute Error
    df['abs_error'] = np.abs(df['y_true'] - df['y_pred'])
    
    # 2. Calculate Variance (sigma^2) from Standard Deviation (sigma) for PIE CHART and PROPORTION
    df['var_epi'] = df['sigma_epi']**2
    df['var_ale'] = df['sigma_ale']**2
    df['var_total'] = df['sigma_tot']**2
    
    # 3. Define Standard Deviation ('std_') variables for BOX PLOT, SCATTER, and STATS
    df['std_epi'] = df['sigma_epi']
    df['std_ale'] = df['sigma_ale']
    total_std = df['sigma_tot']
    
    # Calculate Correlation Data
    correlation = total_std.corr(df['abs_error'])
    
    # --- Plotting Setup ---
    try:
        # Set font style (fallback included)
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    except:
        pass 
        
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # <<< NEU: Füge eine Hauptüberschrift mit dem Namen der Zelllinie hinzu
    fig.suptitle(f"Unsicherheitsanalyse für Zelllinie: {selected_cell_line}", fontsize=16, fontweight='bold')


    # --- 1. Pie Chart: Uncertainty Sources Proportion (USES VARIANCE) ---
    total_var_epi = df['var_epi'].sum()
    total_var_ale = df['var_ale'].sum()
    total_variance = total_var_epi + total_var_ale

    axes[0, 0].pie([total_var_epi, total_var_ale],
                   labels=['Epistemic\n(Model Uncertainty)', 'Aleatoric\n(Data Uncertainty)'],
                   autopct='%1.1f%%',
                   startangle=90,
                   colors=['#FF6B6B', '#4ECDC4'],
                   explode=(0.05, 0))
    axes[0, 0].set_title('Uncertainty Sources Distribution (by Variance)', fontweight='bold')

    # --- 2. Box Plot Comparison (USES STANDARD DEVIATION) ---
    uncertainty_data = [df['std_epi'], df['std_ale']]
    box = axes[0, 1].boxplot(uncertainty_data, patch_artist=True,
                            labels=['Epistemic', 'Aleatoric'])
    colors = ['#FF6B6B', '#4ECDC4']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    axes[0, 1].set_ylabel('Standard Deviation ($\sigma$)')
    axes[0, 1].set_title('Uncertainty Comparison (Standard Deviation)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # --- 3. Scatter Plot: Relationship Between Uncertainties (USES STANDARD DEVIATION) ---
    max_val = max(df['std_epi'].max(), df['std_ale'].max())
    axes[1, 0].scatter(df['std_epi'], df['std_ale'], 
                               alpha=0.7, s=60, c='#45B7D1')
    axes[1, 0].plot([0, max_val], [0, max_val], 
                   'r--', alpha=0.7, label='Equal Line')
    axes[1, 0].set_xlabel('Epistemic Uncertainty ($\sigma_{epi}$)')
    axes[1, 0].set_ylabel('Aleatoric Uncertainty ($\sigma_{ale}$)')
    axes[1, 0].set_title('Relationship Between Uncertainties', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # --- 4. Error vs. Total Uncertainty (USES TOTAL STANDARD DEVIATION) ---
    axes[1, 1].scatter(total_std, df['abs_error'], alpha=0.7, s=60, c='#F9A602')
    axes[1, 1].set_xlabel('Total Uncertainty ($\sigma_{tot}$)')
    axes[1, 1].set_ylabel('Absolute Prediction Error')
    axes[1, 1].set_title('Uncertainty vs Prediction Error', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    # Display Correlation
    axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=axes[1, 1].transAxes, fontsize=12,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # --- Save and Show Plot ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # <<< GEÄNDERT: Platz für suptitle
    
    # Generate output filename based on input path
    # <<< GEÄNDERT: Dateiname enthält jetzt die Zelllinie
    output_filename = f'uncertainty_analysis_{selected_cell_line}.png'
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot gespeichert in: {output_filename}")
    plt.show()

    # --- Output Key Statistics (USES STANDARD DEVIATION) ---
    print("=" * 50)
    # <<< GEÄNDERT: Titel angepasst
    print(f"KEY UNCERTAINTY STATISTICS (Zelllinie: {selected_cell_line})")
    print("=" * 50)
    print(f"Total samples: {len(df)}")
    # Use standard text for sigma and subscripts
    print(f"Epistemic uncertainty (Mean sigma_epi): {df['std_epi'].mean():.3f} ± {df['std_epi'].std():.3f}")
    print(f"Aleatoric uncertainty (Mean sigma_ale): {df['std_ale'].mean():.3f} ± {df['std_ale'].std():.3f}")

    if total_variance > 0:
        print(f"Epistemic proportion (Variance): {(total_var_epi/total_variance)*100:.1f}%")
    else:
        print("Epistemic proportion: 0.0% (Total Variance is Zero)")

    corr_epi_ale = df[['std_epi', 'std_ale']].corr().iloc[0,1]
    # Use standard text for sigma and subscripts
    print(f"Correlation (sigma_epi vs sigma_ale): {corr_epi_ale:.3f}")
    # Use standard text for sigma and subscripts
    print(f"Uncertainty-Error correlation (sigma_tot vs Abs Error): {correlation:.3f}")
    print("=" * 50)


def main():
    """Main function to parse arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize epistemic and aleatoric uncertainties from a CSV file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Define the file path argument (optional, uses default path if not provided)
    parser.add_argument(
        'file_path', 
        nargs='?', 
        type=str, 
        default=DEFAULT_FILE_PATH, 
        help=f"The path to the input CSV file. \n(Default: {DEFAULT_FILE_PATH})"
    )
    
    args = parser.parse_args()
    
    # Execute the core logic
    analyze_uncertainty(args.file_path)

if __name__ == '__main__':
    main()