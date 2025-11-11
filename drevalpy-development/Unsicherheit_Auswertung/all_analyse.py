'''
Erstellt die Abbildung 3.4 im Paper
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

# Imports für Metriken und Korrelationen
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# (adjustText Import wurde entfernt, da der Plot, der es brauchte, gelöscht wurde)

# Standard-Pfad
DEFAULT_FILE_PATH = "/home/mi/zhaow01/dokumente/BA/drevalpy-development/results/GDSC2/baseline_models_ale_epic/exclude_target_ale_epic/kombinierte_results.csv"


# ******************************************************************
# <<< PLOT 1 (ehemals Plot 2): "Kompakter" Binned Plot (Trend-Analyse) >>>
# ******************************************************************
def generate_binned_uncertainty_vs_groundtruth_plot(df, analysis_name, num_bins=20):
    """
    Erstellt ein 1x2-Grid, das den TREND der Unsicherheiten gegen 
    den Ground Truth 'y_true' plottet (mittels Binning).
    """
    
    if df.empty:
        print(f"Plotting-Fehler (Binned Plot): Leerer DataFrame.")
        return

    print(f"\n--- Erstelle 'kompakten' Binned-Plot (Trend) für {analysis_name} ---")

    # --- 2. Binning der Daten ---
    df_binned = df.copy()
    try:
        df_binned['y_true_bin'] = pd.cut(df_binned['y_true'], bins=num_bins)
    except Exception as e:
        print(f"Fehler beim Binning: {e}. Eventuell zu wenige Datenpunkte?")
        return

    binned_ale = df_binned.groupby('y_true_bin')['sigma_ale'].agg(['mean', 'sem']).dropna()
    binned_epi = df_binned.groupby('y_true_bin')['sigma_epi'].agg(['mean', 'sem']).dropna()

    bin_centers_ale = [interval.mid for interval in binned_ale.index]
    bin_centers_epi = [interval.mid for interval in binned_epi.index]

    # --- 3. Plot-Setup (1 Zeile, 2 Spalten) ---
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True)
    fig.suptitle(f'Uncertainty vs. Ground Truth (Trend) — {analysis_name}', fontsize=16, weight='bold')
    
    # --- Plot 1: Aleatoric Uncertainty Trend ---
    axes[0].plot(bin_centers_ale, binned_ale['mean'], color='#4ECDC4', marker='o', label='Mittlere Aleatorische Unsicherheit')
    axes[0].fill_between(
        bin_centers_ale,
        binned_ale['mean'] - binned_ale['sem'],
        binned_ale['mean'] + binned_ale['sem'],
        color='#4ECDC4', alpha=0.2, label='± 1 Standardfehler (SEM)'
    )
    axes[0].set_title('Aleatoric Uncertainty vs. Ground Truth', fontsize=14)
    axes[0].set_xlabel('Ground Truth ($y_{true}$)', fontsize=12)
    axes[0].set_ylabel('Mittlere Aleatorische Unsicherheit ($\sigma_{ale}$)', fontsize=12)
    axes[0].legend()

    # --- Plot 2: Epistemic Uncertainty Trend ---
    axes[1].plot(bin_centers_epi, binned_epi['mean'], color='#FF6B6B', marker='o', label='Mittlere Epistemische Unsicherheit')
    axes[1].fill_between(
        bin_centers_epi,
        binned_epi['mean'] - binned_epi['sem'],
        binned_epi['mean'] + binned_epi['sem'],
        color='#FF6B6B', alpha=0.2, label='± 1 Standardfehler (SEM)'
    )
    axes[1].set_title('Epistemic Uncertainty vs. Ground Truth', fontsize=14)
    axes[1].set_xlabel('Ground Truth ($y_{true}$)', fontsize=12)
    axes[1].set_ylabel('Mittlere Epistemische Unsicherheit ($\sigma_{epi}$)', fontsize=12)
    axes[1].legend()

    # --- 4. Speichern und Anzeigen ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = f'binned_uncertainty_vs_groundtruth_{analysis_name}.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Plot 1 (Binned Trend) gespeichert in: {output_filename}")
    plt.show()


# ******************************************************************
# <<< PLOT 2: 2x2 Grid Plot (basiert auf gemittelten Metriken) >>>
# <<< HIER WURDE DIE LEGENDEN-LOGIK KORRIGIERT >>>
# ******************************************************************
def generate_averaged_grid_plot(cell_line_metrics, analysis_name):
    """
    Erstellt ein 2x2-Grid, das die Verteilungen der
    Pro-Zelllinie-Metriken zeigt.
    JETZT MIT KORREKTER LEGENDEN-ZUWEISUNG.
    """
    
    if cell_line_metrics.empty:
        print("Plotting-Fehler (Averaged Grid): Keine Zelllinien-Metriken vorhanden.")
        return

    print("\n--- Erstelle 2x2 Grid-Plot (basierend auf Pro-Zelllinien-Metriken) ---")

    try:
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    except:
        pass 
        
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Zusammenfassende Analyse (gemittelt pro Zelllinie) — {analysis_name}", fontsize=16, fontweight='bold')

    # --- 1. Pie Chart (unverändert) ---
    if 'var_proportion_epi' in cell_line_metrics.columns:
        mean_epi_prop = cell_line_metrics['var_proportion_epi'].mean()
        mean_ale_prop = 1.0 - mean_epi_prop
        
        axes[0, 0].pie([mean_epi_prop, mean_ale_prop],
                       labels=['Epistemic\n(Model Uncertainty)', 'Aleatoric\n(Data Uncertainty)'],
                       autopct='%1.1f%%', startangle=90, colors=['#FF6B6B', '#4ECDC4'], explode=(0.05, 0))
        axes[0, 0].set_title('Mittlere Unsicherheits-Proportion\n(pro Zelllinie gemittelt)', fontweight='bold')
    else:
        axes[0, 0].text(0.5, 0.5, "Konnte Varianz-Proportion nicht berechnen", ha='center')
        axes[0, 0].set_title('Unsicherheits-Proportion', fontweight='bold')


    # --- 2. Box Plot (unverändert) ---
    if 'mean_epi' in cell_line_metrics.columns and 'mean_ale' in cell_line_metrics.columns:
        uncertainty_data = [cell_line_metrics['mean_epi'].dropna(), cell_line_metrics['mean_ale'].dropna()]
        box = axes[0, 1].boxplot(uncertainty_data, patch_artist=True, labels=['Epistemic', 'Aleatoric'])
        colors = ['#FF6B6B', '#4ECDC4']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        axes[0, 1].set_ylabel('Mittlere Standard Abweichung ($\sigma$)')
        axes[0, 1].set_title('Verteilung der mittleren Unsicherheiten\n(jeder Datenpunkt = 1 Zelllinie)', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, "Keine Daten für Boxplot", ha='center')
        axes[0, 1].set_title('Verteilung der mittleren Unsicherheiten', fontweight='bold')


    # --- 3. Scatter Plot (Epi vs Ale) mit KORRIGIERTER LEGENDE ---
    if 'mean_epi' in cell_line_metrics.columns and 'mean_ale' in cell_line_metrics.columns:
        mean_corr = cell_line_metrics['mean_epi'].corr(cell_line_metrics['mean_ale'])
        
        # *** KORREKTUR: Punkte separat plotten und labeln ***
        sns.scatterplot(
            x=cell_line_metrics['mean_epi'],
            y=cell_line_metrics['mean_ale'],
            ax=axes[1, 0],
            alpha=0.7, 
            s=30, 
            color='#45B7D1',
            label='Zelllinie' # <-- Label für die Punkte
        )
        
        # *** KORREKTUR: Regressionslinie separat plotten und labeln ***
        sns.regplot(
            x=cell_line_metrics['mean_epi'],
            y=cell_line_metrics['mean_ale'],
            ax=axes[1, 0],
            scatter=False, # <-- Punkte nicht nochmal zeichnen
            line_kws={'color': 'red', 'alpha': 0.7},
            label='Korrelationslinie' # <-- Label für die Linie
        )
        
        # Gleichheitslinie (y=x) als Referenz beibehalten
        max_val = max(cell_line_metrics['mean_epi'].max(), cell_line_metrics['mean_ale'].max())
        axes[1, 0].plot([0, max_val], [0, max_val], 'grey', ls=':', alpha=0.7, label='Gleichheitslinie')
        
        axes[1, 0].set_xlabel('Mittlere Epistemic Uncertainty ($\sigma_{epi}$)')
        axes[1, 0].set_ylabel('Mittlere Aleatoric Uncertainty ($\sigma_{ale}$)')
        axes[1, 0].set_title(f'Mittlere Unsicherheiten pro Zelllinie\n(Pearson: {mean_corr:.3f})', fontweight='bold')
        axes[1, 0].legend() # Legende wird jetzt korrekt erstellt
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, "Keine Daten für Scatter", ha='center')
        axes[1, 0].set_title('Mittlere Unsicherheiten pro Zelllinie', fontweight='bold')

    # --- 4. Error vs. Total Uncertainty mit KORRIGIERTER LEGENDE ---
    if 'mean_abs_error' in cell_line_metrics.columns and 'mean_sigma_tot' in cell_line_metrics.columns:
        mean_corr_err = cell_line_metrics['mean_abs_error'].corr(cell_line_metrics['mean_sigma_tot'])
        
        # *** KORREKTUR: Punkte separat plotten und labeln ***
        sns.scatterplot(
            x=cell_line_metrics['mean_sigma_tot'],
            y=cell_line_metrics['mean_abs_error'],
            ax=axes[1, 1],
            alpha=0.7,
            s=30,
            color='#F9A602',
            label='Zelllinie' # <-- Label für die Punkte
        )
        
        # *** KORREKTUR: Regressionslinie separat plotten und labeln ***
        sns.regplot(
            x=cell_line_metrics['mean_sigma_tot'],
            y=cell_line_metrics['mean_abs_error'],
            ax=axes[1, 1],
            scatter=False, # <-- Punkte nicht nochmal zeichnen
            line_kws={'color': 'blue', 'alpha': 0.7},
            label='Korrelationslinie' # <-- Label für die Linie
        )
        
        axes[1, 1].set_xlabel('Mittlere Totale Unsicherheit ($\sigma_{tot}$)')
        axes[1, 1].set_ylabel('Mittlerer Absoluter Fehler')
        axes[1, 1].set_title('Mittlerer Fehler vs. Mittlere Unsicherheit\n(pro Zelllinie)', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend() # Legende hier auch hinzufügen
        
        # Textbox mit Korrelation ist weiterhin nützlich
        axes[1, 1].text(0.05, 0.95, f'Pearson Corr: {mean_corr_err:.3f}', transform=axes[1, 1].transAxes, fontsize=12,
                       bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    else:
        axes[1, 1].text(0.5, 0.5, "Keine Daten für Scatter", ha='center')
        axes[1, 1].set_title('Mittlerer Fehler vs. Mittlere Unsicherheit', fontweight='bold')

    # --- Speichern und Anzeigen (2x2 Grid) ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = f'grid_analysis_averaged_with_reg_{analysis_name}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot 2 (Averaged 2x2-Grid) gespeichert in: {output_filename}")
    plt.show()


# ******************************************************************
# <<< HILFSFUNKTION: Metriken pro Zelllinie berechnen (unverändert) >>>
# ******************************************************************
def calculate_cell_line_metrics(group):
    """
    Berechnet einen Satz von Metriken für eine einzelne Zelllinien-Gruppe.
    Wird mit .apply() aufgerufen.
    """
    MIN_SAMPLES = 5 
    
    metrics = {
        'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'pearson': np.nan,
        'mean_epi': np.nan, 'mean_ale': np.nan, 'mean_sigma_tot': np.nan,
        'mean_abs_error': np.nan, 'var_proportion_epi': np.nan,
        'corr_err_unc': np.nan
    }
    
    if not group.empty:
        metrics['mean_epi'] = group['sigma_epi'].mean()
        metrics['mean_ale'] = group['sigma_ale'].mean()
        metrics['mean_sigma_tot'] = group['sigma_tot'].mean()
        metrics['mean_abs_error'] = np.abs(group['y_true'] - group['y_pred']).mean()
        
        var_epi = (group['sigma_epi']**2).sum()
        var_ale = (group['sigma_ale']**2).sum()
        total_var = var_epi + var_ale
        if total_var > 0:
            metrics['var_proportion_epi'] = var_epi / total_var
        else:
            metrics['var_proportion_epi'] = np.nan

    if len(group) >= MIN_SAMPLES and group['y_true'].std() > 0 and group['y_pred'].std() > 0:
        try:
            metrics['rmse'] = np.sqrt(mean_squared_error(group['y_true'], group['y_pred']))
            metrics['mae'] = mean_absolute_error(group['y_true'], group['y_pred'])
            metrics['r2'] = r2_score(group['y_true'], group['y_pred'])
            metrics['pearson'] = pearsonr(group['y_true'], group['y_pred'])[0]
        except ValueError:
            pass 

        try:
            abs_error = np.abs(group['y_true'] - group['y_pred'])
            if abs_error.std() > 0 and group['sigma_tot'].std() > 0:
                metrics['corr_err_unc'] = abs_error.corr(group['sigma_tot'])
        except ValueError:
            pass

    return pd.Series(metrics)


# ******************************************************************
# <<< ANALYSEFUNKTION (Aufruf von Plot 1 entfernt) >>>
# ******************************************************************
def perform_averaged_analysis(file_path):
    """
    Liest Unsicherheitsdaten, berechnet Metriken PRO Zelllinie,
    und plottet dann die gemittelten/zusammengefassten Ergebnisse.
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
        return

    analysis_name = "ALL_CELL_LINES"
    print(f"==================================================")
    print(f"Analyse (gemittelt pro Zelllinie) wird gestartet...")
    print(f"==================================================")

    # --- 2. Metriken pro Zelllinie berechnen ---
    print("Berechne Metriken für jede Zelllinie...")
    cell_line_metrics = df_full.groupby('cell_line').apply(calculate_cell_line_metrics)
    
    original_count = len(cell_line_metrics)
    cell_line_metrics = cell_line_metrics.dropna(subset=['rmse', 'r2', 'pearson', 'corr_err_unc'])
    valid_count = len(cell_line_metrics)
    print(f"{valid_count} von {original_count} Zelllinien hatten genügend Daten für die vollständige Metrik-Berechnung.")

    if valid_count == 0:
        print("FEHLER: Keine Zelllinie hatte genügend Daten. Stoppe die Analyse.")
        return

    # --- 3. Globale gemittelte Metriken berechnen ---
    averaged_metrics = {
        'rmse': cell_line_metrics['rmse'].mean(),
        'rmse_std': cell_line_metrics['rmse'].std(),
        'mae': cell_line_metrics['mae'].mean(),
        'mae_std': cell_line_metrics['mae'].std(),
        'r2': cell_line_metrics['r2'].mean(),
        'r2_std': cell_line_metrics['r2'].std(),
        'pearson': cell_line_metrics['pearson'].mean(),
        'pearson_std': cell_line_metrics['pearson'].std(),
        'corr_err_unc': cell_line_metrics['corr_err_unc'].mean(),
        'corr_err_unc_std': cell_line_metrics['corr_err_unc'].std(),
        'var_proportion_epi': cell_line_metrics['var_proportion_epi'].mean(),
        'var_proportion_epi_std': cell_line_metrics['var_proportion_epi'].std(),
    }

    # --- 4. Statistische Ausgabe (unverändert) ---
    print("\n" + "=" * 50)
    print(f"ZUSAMMENFASSENDE STATISTIK (gemittelt über {valid_count} Zelllinien)")
    print("=" * 50)
    print(f"Mean RMSE:           {averaged_metrics['rmse']:.3f} ± {averaged_metrics['rmse_std']:.3f}")
    print(f"Mean R²:             {averaged_metrics['r2']:.3f} ± {averaged_metrics['r2_std']:.3f}")
    print(f"Mean Pearson:        {averaged_metrics['pearson']:.3f} ± {averaged_metrics['pearson_std']:.3f}")
    print(f"Mean Error-Unc. Corr: {averaged_metrics['corr_err_unc']:.3f} ± {averaged_metrics['corr_err_unc_std']:.3f}")
    print(f"Mean Epistemic Prop: {averaged_metrics['var_proportion_epi']*100:.1f}% ± {averaged_metrics['var_proportion_epi_std']*100:.1f}%")
    print("=" * 50 + "\n")


    # --- 5. Plots aufrufen ---

    # PLOT 1 (ehemals 2): "Kompakter" Binned-Plot (Zeigt den Trend)
    try:
        generate_binned_uncertainty_vs_groundtruth_plot(df_full, analysis_name, num_bins=20)
    except Exception as e:
        print(f"Fehler beim Erstellen des Binned Uncertainty Plots: {e}")

    # PLOT 2 (ehemals 3): 2x2 Grid (Zeigt die Verteilung der Pro-Zelllinien-Metriken)
    try:
        generate_averaged_grid_plot(cell_line_metrics, analysis_name)
    except Exception as e:
        print(f"Fehler beim Erstellen des 2x2 Averaged Grid Plots: {e}")


# ******************************************************************
# <<< MAIN-FUNKTION (unverändert) >>>
# ******************************************************************
def main():
    """Hauptfunktion zum Parsen der Argumente und Starten der globalen Analyse."""
    parser = argparse.ArgumentParser(
        description="Analysiert und visualisiert Unsicherheiten (pro Zelllinie gemittelt) aus einer CSV-Datei.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '-f', '--file_path', 
        type=str, 
        default=DEFAULT_FILE_PATH, 
        help=f"Der Pfad zur Input-CSV-Datei. \n(Standard: {DEFAULT_FILE_PATH})"
    )
    
    args = parser.parse_args()
    
    perform_averaged_analysis(args.file_path)

if __name__ == '__main__':
    main()