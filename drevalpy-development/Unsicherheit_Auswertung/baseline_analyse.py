import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Standard-Pfad (kann über die Kommandozeile geändert werden)
DEFAULT_FILE_PATH = "/home/mi/zhaow01/dokumente/BA/drevalpy-development/results/GDSC2/baseline_models_ale_epic/exclude_target_ale_epic/global_baseline_preds.csv"

# --- Plot 1: Predicted vs. True ---
def plot_predicted_vs_true(df, metrics_text, analysis_name="GLOBAL"):
    """
    Erstellt einen Scatter-Plot von Predicted vs. True Werten.
    """
    print(f"\n--- Erstelle Plot 1: Predicted vs. True ({analysis_name}) ---")
    
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    # Scatter-Plot
    sns.scatterplot(
        data=df,
        x='y_true',
        y='y_pred',
        alpha=0.5,
        s=20, # Kleinere Punkte für eine globale Ansicht
        label='Datenpunkte'
    )
    
    # y=x Linie (Perfekte Vorhersage)
    min_val = min(df['y_true'].min(), df['y_pred'].min()) - 1
    max_val = max(df['y_true'].max(), df['y_pred'].max()) + 1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
             label='Perfekte Vorhersage ($y=x$)')
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Titel und Labels
    plt.title(f'Baseline Performance: Predicted vs. Ground Truth\n({analysis_name})', 
              fontsize=16, weight='bold')
    plt.xlabel('Ground Truth ($y_{true}$)', fontsize=12)
    plt.ylabel('Vorhersage ($y_{pred}$)', fontsize=12)
    
    # Metriken-Box
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='grey')
    plt.text(0.03, 0.97, metrics_text, 
             transform=plt.gca().transAxes, 
             fontsize=12, 
             verticalalignment='top', 
             bbox=props)
    
    plt.legend(loc='lower right')
    
    # Speichern und Anzeigen
    output_filename = f'baseline_predicted_vs_true_{analysis_name}.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Plot 1 gespeichert in: {output_filename}")
    plt.show()

# --- Plot 2: Residual Plot ---
def plot_residuals_vs_predicted(df, analysis_name="GLOBAL"):
    """
    Erstellt einen Scatter-Plot der Residuen (Fehler) vs. der Vorhersagen.
    """
    print(f"\n--- Erstelle Plot 2: Residual Plot ({analysis_name}) ---")
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Scatter-Plot
    sns.scatterplot(
        data=df,
        x='y_pred',
        y='residuals',
        alpha=0.5,
        s=20
    )
    
    # Null-Fehler-Linie
    plt.axhline(0, color='red', linestyle='--', label='Null-Fehler-Linie')
    
    # Titel und Labels
    plt.title(f'Baseline Diagnose: Residual Plot\n({analysis_name})', 
              fontsize=16, weight='bold')
    plt.xlabel('Vorhersage ($y_{pred}$)', fontsize=12)
    plt.ylabel('Residual (Fehler: $y_{true} - y_{pred}$)', fontsize=12)
    
    plt.legend(loc='upper right')
    
    # Speichern und Anzeigen
    output_filename = f'baseline_residuals_vs_predicted_{analysis_name}.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Plot 2 gespeichert in: {output_filename}")
    plt.show()

# --- Plot 3: Histogramm der Residuen ---
def plot_residuals_histogram(df, analysis_name="GLOBAL"):
    """
    Erstellt ein Histogramm der Residuen (Fehler), um die Verteilung zu sehen.
    """
    print(f"\n--- Erstelle Plot 3: Histogramm der Residuen ({analysis_name}) ---")
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Histogramm mit Dichtekurve (KDE)
    sns.histplot(
        data=df,
        x='residuals',
        kde=True, # Fügt die Dichtekurve hinzu
        bins=50   # Anzahl der Balken
    )
    
    # Null-Fehler-Linie
    plt.axvline(0, color='red', linestyle='--', label='Null-Fehler (Perfekt)')
    
    # Mittelwert und Median des Fehlers
    mean_err = df['residuals'].mean()
    median_err = df['residuals'].median()
    
    plt.axvline(mean_err, color='blue', linestyle=':', 
                label=f'Mittlerer Fehler: {mean_err:.3f}')
    
    # Titel und Labels
    plt.title(f'Baseline Diagnose: Fehlerverteilung (Histogramm)\n({analysis_name})', 
              fontsize=16, weight='bold')
    plt.xlabel('Residual (Fehler: $y_{true} - y_{pred}$)', fontsize=12)
    plt.ylabel('Anzahl / Dichte', fontsize=12)
    
    # Statistik-Text
    stats_text = f"Mean Error = {mean_err:.3f}\nMedian Error = {median_err:.3f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='grey')
    plt.text(0.03, 0.97, stats_text, 
             transform=plt.gca().transAxes, 
             fontsize=12, 
             verticalalignment='top', 
             bbox=props)
    
    plt.legend(loc='upper right')
    
    # Speichern und Anzeigen
    output_filename = f'baseline_residuals_histogram_{analysis_name}.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Plot 3 gespeichert in: {output_filename}")
    plt.show()

# --- Haupt-Analysefunktion ---
def generate_baseline_plots(file_path):
    """
    Liest die Daten und ruft alle Plot-Funktionen für eine globale Analyse auf.
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

    # Benötigte Spalten prüfen
    required_cols_input = ['y_true', 'y_pred']
    if not all(col in df_full.columns for col in required_cols_input):
        print(f"Fehler: CSV-Datei muss Spalten enthalten: {required_cols_input}")
        return

    analysis_name = "GLOBAL"
    print(f"==================================================")
    print(f"Baseline-Analyse ({analysis_name}) wird gestartet...")
    print(f"Datenquelle: {file_path}")
    print(f"Anzahl Datenpunkte: {len(df_full)}")
    print(f"==================================================")

    # --- 2. Daten vorbereiten ---
    
    # Residuen (Fehler) berechnen
    df_full['residuals'] = df_full['y_true'] - df_full['y_pred']
    
    # Globale Metriken berechnen
    try:
        rmse = np.sqrt(mean_squared_error(df_full['y_true'], df_full['y_pred']))
        r2 = r2_score(df_full['y_true'], df_full['y_pred'])
        mae = mean_absolute_error(df_full['y_true'], df_full['y_pred'])
        
        metrics_text = (
            f"Globale Metriken:\n"
            f"RMSE = {rmse:.3f}\n"
            f"$R^2$ = {r2:.3f}\n"
            f"MAE = {mae:.3f}"
        )
        print(metrics_text)
        
    except Exception as e:
        print(f"Fehler bei der Metrik-Berechnung: {e}")
        metrics_text = "Metriken konnten nicht berechnet werden."

    # --- 3. Plots aufrufen ---
    
    # Plot 1: Predicted vs. True
    plot_predicted_vs_true(df_full, metrics_text, analysis_name)
    
    # Plot 2: Residual Plot
    plot_residuals_vs_predicted(df_full, analysis_name)
    
    # Plot 3: Histogramm der Residuen
    plot_residuals_histogram(df_full, analysis_name)
    
    print("\nAlle Baseline-Plots wurden erfolgreich erstellt.")

# --- Main-Funktion (Argument Parser) ---
def main():
    """Hauptfunktion zum Parsen der Argumente und Starten der Analyse."""
    parser = argparse.ArgumentParser(
        description="Erstellt drei Baseline-Performance-Plots (Global) für ein Regressionsmodell.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '-f', '--file_path', 
        type=str, 
        default=DEFAULT_FILE_PATH, 
        help=f"Der Pfad zur Input-CSV-Datei. \n(Standard: {DEFAULT_FILE_PATH})"
    )
    
    args = parser.parse_args()
    
    generate_baseline_plots(args.file_path)

if __name__ == '__main__':
    main()