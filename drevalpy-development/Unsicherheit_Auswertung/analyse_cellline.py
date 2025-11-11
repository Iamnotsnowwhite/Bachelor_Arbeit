import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

try:
    from adjustText import adjust_text
except ImportError:
    print("="*60)
    print("FEHLER: 'adjustText' Bibliothek nicht gefunden.")
    print("Diese Bibliothek ist ZWINGEND ERFORDERLICH für lesbare Labels.")
    print("Bitte installiere sie mit:   pip install adjustText")
    print("="*60)
    adjust_text = None


# Standard-Pfad
DEFAULT_FILE_PATH = "/home/mi/zhaow01/dokumente/BA/drevalpy-development/results/GDSC2/baseline_models_ale_epic/exclude_target_ale_epic/kombinierte_results.csv"


def generate_prediction_scatter_plot(df, cell_line_name, label_col='drug_id'):
    """
    Erstellt einen detaillierten Scatter-Plot (Prediction vs. Ground Truth),
    der Unsicherheiten als Farbe und Größe darstellt.
    (Diese Funktion ist unverändert)
    """
    
    # --- 1. Datenprüfung ---
    required_cols = ['y_true', 'y_pred', 'sigma_epi', 'sigma_ale', 'sigma_tot', label_col]
    if not all(col in df.columns for col in required_cols):
        print(f"Plotting-Fehler: Dem DataFrame fehlen Spalten.")
        print(f"Benötigt: {required_cols}")
        print(f"Gefunden: {df.columns.tolist()}")
        return
        
    if df.empty:
        print(f"Plotting-Fehler: Leerer DataFrame für {cell_line_name}.")
        return

    # --- 2. Metriken berechnen (für die Textbox) ---
    rmse = np.sqrt(mean_squared_error(df['y_true'], df['y_pred']))
    mae = mean_absolute_error(df['y_true'], df['y_pred'])
    r2 = r2_score(df['y_true'], df['y_pred'])
    pearson_corr, _ = pearsonr(df['y_true'], df['y_pred'])

    stats_text = (
        f"From these points:\n"
        f"RMSE={rmse:.3f}\n"
        f"MAE={mae:.3f}\n"
        f"$R^2$={r2:.3f}\n"
        f"Pearson={pearson_corr:.3f}"
    )

    # --- 3. Plot-Setup ---
    plt.figure(figsize=(10, 8))
    sns.set_style("ticks")
    
    ax = sns.scatterplot(
        data=df,
        x='y_true',
        y='y_pred',
        hue='sigma_epi',
        size='sigma_ale',
        sizes=(40, 400),
        palette='plasma',
        alpha=0.8,
        legend=False,
        edgecolors=None,
        linewidth=0,
        zorder=2
    )

    # --- 4. y=x Referenzlinie ---
    min_val = min(df['y_true'].min(), df['y_pred'].min()) - 1
    max_val = max(df['y_true'].max(), df['y_pred'].max()) + 1
    ax.plot([min_val, max_val], [min_val, max_val], 
            ls='--', color='grey', alpha=0.8, 
            zorder=1)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # --- 5. Titel und Achsenbeschriftungen ---
    ax.set_title(f'Predictions vs. Ground Truth — {cell_line_name}\n(Epistemic Color, Aleatoric Size)', 
                 fontsize=14, weight='bold')
    ax.set_xlabel('Ground Truth ($y_{true}$)', fontsize=12)
    ax.set_ylabel('Prediction ($y_{pred}$)', fontsize=12)

    # --- 6. Statistik-Box ---
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='grey')
    ax.text(0.03, 0.97, stats_text, 
            transform=ax.transAxes, 
            fontsize=10, 
            verticalalignment='top', 
            bbox=props)

    # --- 7. Manuelle Legenden ---
    norm = plt.Normalize(df['sigma_epi'].min(), df['sigma_epi'].max())
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.85, aspect=20, pad=0.03)
    cbar.set_label('Epistemic Uncertainty ($\sigma_{epi}$) - COLOR', 
                   fontsize=10, weight='bold')

    avg_ale_val = df['sigma_ale'].mean()
    max_ale_val = df['sigma_ale'].max()
    avg_size = np.interp(avg_ale_val, [df['sigma_ale'].min(), df['sigma_ale'].max()], [40, 400])
    max_size = np.interp(max_ale_val, [df['sigma_ale'].min(), df['sigma_ale'].max()], [40, 400])

    l1 = plt.scatter([], [], s=avg_size, color='grey', alpha=0.7, 
                     label=f'Avg $\sigma_{{ale}}$ ({avg_ale_val:.2f})')
    l2 = plt.scatter([], [], s=max_size, color='grey', alpha=0.7,
                     label=f'Max $\sigma_{{ale}}$ ({max_ale_val:.2f})')
    
    ax.legend(handles=[l1, l2],
              title='Aleatoric Uncertainty (SIZE)',
              loc='lower right',
              frameon=True,
              facecolor='white',
              framealpha=0.9,
              labelspacing=1.5,
              title_fontproperties={'weight':'bold'})

    # --- 8. Punkt-Beschriftungen ---
    top_uncertain_points = df.nlargest(3, 'sigma_tot')
    
    texts = []
    
    if not adjust_text:
        print("\nWARNUNG: 'adjustText' nicht installiert. Labels werden überlappen.")
        for _, row in top_uncertain_points.iterrows():
            ax.text(row['y_true'], row['y_pred'] + 0.1, str(row[label_col]), 
                    fontsize=9, ha='center', color='black', weight='bold', 
                    zorder=3)
    else:
        for _, row in top_uncertain_points.iterrows():
            texts.append(
                ax.text(row['y_true'], row['y_pred'], str(row[label_col]), 
                        fontsize=9, 
                        ha='center',
                        color='black',
                        weight='bold',
                        zorder=3)
            )
        
        adjust_text(texts, 
                    ax=ax,
                    arrowprops=dict(arrowstyle='->', color='grey', lw=0.5, alpha=0.7)
                   )

    # --- 9. Speichern und Anzeigen ---
    plt.tight_layout()
    output_filename = f'scatter_uncertainty_{cell_line_name}.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Plot 1 (4D-Scatter) gespeichert in: {output_filename}")
    plt.show()


def analyze_uncertainty(file_path, cell_line_name):
    """
    Liest Unsicherheitsdaten, filtert nach einer *bestimmten* Zelllinie, 
    und generiert Plots für diese.
    """
    
    # --- 1. Daten laden und prüfen ---
    if not os.path.exists(file_path):
        print(f"Fehler: Datei nicht gefunden unter: {file_path}")
        return

    try:
        df_full = pd.read_csv(file_path)
    except Exception as e:
        print(f"Fehler beim Lesen der CSV-Datei: {e}")
        return

    required_cols_input = ['y_true', 'y_pred', 'sigma_epi', 'sigma_ale', 'sigma_tot', 'cell_line', 'drug_id']
    if not all(col in df_full.columns for col in required_cols_input):
        print(f"Fehler: CSV-Datei muss Spalten enthalten: {required_cols_input}")
        print(f"Gefundene Spalten: {df_full.columns.tolist()}")
        return

    # --- 2. Zelllinie auswählen ---
    unique_cell_lines = df_full['cell_line'].unique()
    
    if cell_line_name not in unique_cell_lines:
        print(f"FEHLER: Zelllinie '{cell_line_name}' nicht in der Datei gefunden.")
        print("Verfügbare Zelllinien sind (Auszug):")
        print(unique_cell_lines[:20])
        return

    selected_cell_line = cell_line_name
    print(f"==================================================")
    print(f"Analyse wird für Zelllinie '{selected_cell_line}' gestartet...")
    print(f"==================================================")

    df = df_full[df_full['cell_line'] == selected_cell_line].copy()
    
    if len(df) < 2:
        print(f"Fehler: Nicht genügend Daten (nur {len(df)} Zeile/n) für Zelllinie '{selected_cell_line}', um die Analyse durchzuführen.")
        return

    # --- 3. AUFRUF PLOT 1: 4D Scatter Plot ---
    try:
        generate_prediction_scatter_plot(df, selected_cell_line, label_col='drug_id')
    except Exception as e:
        print(f"Fehler beim Erstellen des Scatter-Plots: {e}")

    # --- 4. AUFRUF PLOT 2: 2x2 Grid Plot (Original-Analyse) ---
    print("\n--- Erstelle 2x2 Grid-Plot (Originalskript) ---")

    # --- Datenvorverarbeitung (für 2x2 Grid) ---
    df['abs_error'] = np.abs(df['y_true'] - df['y_pred'])
    df['var_epi'] = df['sigma_epi']**2
    df['var_ale'] = df['sigma_ale']**2
    df['var_total'] = df['sigma_tot']**2
    df['std_epi'] = df['sigma_epi']
    df['std_ale'] = df['sigma_ale']
    total_std = df['sigma_tot']
    
    # Korrelation (Error vs Uncertainty)
    correlation = np.nan
    if len(df) > 1 and total_std.std() > 0 and df['abs_error'].std() > 0:
        correlation = total_std.corr(df['abs_error'])
        
    # Korrelation (Epi vs Ale)
    corr_epi_ale = np.nan
    if len(df) > 1 and df['std_epi'].std() > 0 and df['std_ale'].std() > 0:
        corr_epi_ale = df[['std_epi', 'std_ale']].corr().iloc[0,1]
    
    
    # --- Plotting Setup (2x2 Grid) ---
    try:
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    except:
        pass 
        
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Umfassende Unsicherheitsanalyse für: {selected_cell_line}", fontsize=16, fontweight='bold')

    # 1. Pie Chart (Unverändert)
    total_var_epi = df['var_epi'].sum()
    total_var_ale = df['var_ale'].sum()
    total_variance = total_var_epi + total_var_ale
    axes[0, 0].pie([total_var_epi, total_var_ale],
                   labels=['Epistemic\n(Model Uncertainty)', 'Aleatoric\n(Data Uncertainty)'],
                   autopct='%1.1f%%', startangle=90, colors=['#FF6B6B', '#4ECDC4'], explode=(0.05, 0))
    axes[0, 0].set_title('Uncertainty Sources Distribution (by Variance)', fontweight='bold')

    # 2. Box Plot (Unverändert)
    uncertainty_data = [df['std_epi'], df['std_ale']]
    box = axes[0, 1].boxplot(uncertainty_data, patch_artist=True, labels=['Epistemic', 'Aleatoric'])
    colors = ['#FF6B6B', '#4ECDC4']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    axes[0, 1].set_ylabel('Standard Deviation ($\sigma$)')
    axes[0, 1].set_title('Uncertainty Comparison (Standard Deviation)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Scatter Plot (Epi vs Ale) (Unverändert)
    sns.scatterplot(
        x=df['std_epi'],
        y=df['std_ale'],
        alpha=0.7, s=60, color='#45B7D1',
        ax=axes[1, 0],
        label='Datenpunkt (Drug)'
    )
    sns.regplot(
        x=df['std_epi'],
        y=df['std_ale'],
        ax=axes[1, 0],
        scatter=False, 
        line_kws={'color': 'blue', 'alpha': 0.7},
        label='Korrelationslinie'
    )
    max_val = max(df['std_epi'].max(), df['std_ale'].max())
    axes[1, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='Equal Line')
    axes[1, 0].set_xlabel('Epistemic Uncertainty ($\sigma_{epi}$)')
    axes[1, 0].set_ylabel('Aleatoric Uncertainty ($\sigma_{ale}$)')
    axes[1, 0].set_title(f'Relationship Between Uncertainties\n(Pearson: {corr_epi_ale:.3f})', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Error vs. Total Uncertainty
    sns.scatterplot(
        x=total_std,
        y=df['abs_error'],
        alpha=0.7, s=60, color='#F9A602',
        ax=axes[1, 1],
        label='Datenpunkt (Drug)'
    )
    sns.regplot(
        x=total_std,
        y=df['abs_error'],
        ax=axes[1, 1],
        scatter=False, 
        line_kws={'color': 'red', 'alpha': 0.7},
        label='Korrelationslinie'
    )
    axes[1, 1].set_xlabel('Total Uncertainty ($\sigma_{tot}$)')
    axes[1, 1].set_ylabel('Absolute Prediction Error')
    axes[1, 1].set_title('Uncertainty vs Prediction Error', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Textbox (bleibt oben links)
    axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=axes[1, 1].transAxes, fontsize=12,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                   verticalalignment='top') # Sicherstellen, dass es oben ausgerichtet ist

    # ******************************************************************
    # <<< HIER IST DIE ÄNDERUNG >>>
    # Die Legende wird explizit unten rechts platziert, um die 
    # Textbox oben links nicht zu verdecken.
    # ******************************************************************
    axes[1, 1].legend(loc='lower right') 


    # --- Speichern und Anzeigen (2x2 Grid) ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = f'grid_analysis_{selected_cell_line}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot 2 (2x2-Grid) gespeichert in: {output_filename}")
    plt.show()

    # --- Statistische Ausgabe ---
    print("=" * 50)
    print(f"KEY UNCERTAINTY STATISTICS (Zelllinie: {selected_cell_line})")
    print("=" * 50)
    print(f"Total samples: {len(df)}")
    print(f"Epistemic uncertainty (Mean sigma_epi): {df['std_epi'].mean():.3f} ± {df['std_epi'].std():.3f}")
    print(f"Aleatoric uncertainty (Mean sigma_ale): {df['std_ale'].mean():.3f} ± {df['std_ale'].std():.3f}")
    if total_variance > 0:
        print(f"Epistemic proportion (Variance): {(total_var_epi/total_variance)*100:.1f}%")
    else:
        print("Epistemic proportion: 0.0% (Total Variance is Zero)")
    
    print(f"Correlation (sigma_epi vs sigma_ale): {corr_epi_ale:.3f}")
    print(f"Uncertainty-Error correlation (sigma_tot vs Abs Error): {correlation:.3f}")
    print("=" * 50)


def main():
    """Hauptfunktion zum Parsen der Argumente und Starten der Analyse."""
    # (Unverändert)
    parser = argparse.ArgumentParser(
        description="Analysiert und visualisiert Unsicherheiten für eine BESTIMMTE Zelllinie aus einer CSV-Datei.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        'cell_line_name', 
        type=str, 
        help="Der Name der Zelllinie, die analysiert werden soll (z.B. 'A-204' oder 'JHH-1')."
    )
    
    parser.add_argument(
        '-f', '--file_path', 
        type=str, 
        default=DEFAULT_FILE_PATH, 
        help=f"Der Pfad zur Input-CSV-Datei. \n(Standard: {DEFAULT_FILE_PATH})"
    )
    
    args = parser.parse_args()
    
    analyze_uncertainty(args.file_path, args.cell_line_name)

if __name__ == '__main__':
    main()