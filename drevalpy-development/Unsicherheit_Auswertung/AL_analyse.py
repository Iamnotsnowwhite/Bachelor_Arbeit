'''
Erstellt die Abbildung 3.5 im Paper
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import sys

# --- Konfiguration ---
FILE_PATH = '/home/mi/zhaow01/dokumente/BA/drevalpy-development/results/GDSC2/baseline_models_ale_epic/exclude_target_ale_epic/kombinierte_results.csv'
# ---------------------

def load_data(filepath):
    """Lädt die CSV-Daten und behandelt Ladefehler."""
    try:
        # index_col=0 geht davon aus, dass die erste Spalte (wie in Ihrem Beispiel) 
        # der Index ist und nicht als Daten geladen werden soll.
        data = pd.read_csv(filepath, index_col=0)
        print(f"Daten erfolgreich aus '{filepath}' geladen.")
        return data
    except FileNotFoundError:
        print(f"FEHLER: Die Datei '{filepath}' wurde nicht gefunden.", file=sys.stderr)
        print("Bitte passen Sie die Variable 'FILE_PATH' im Skript an.", file=sys.stderr)
        sys.exit(1) # Beendet das Skript mit einem Fehlercode
    except Exception as e:
        print(f"Ein unerwarteter Fehler beim Laden der Datei ist aufgetreten: {e}", file=sys.stderr)
        sys.exit(1)

def calculate_rmse(group):
    """Berechnet den RMSE für eine gegebene Gruppe."""
    # Stellt sicher, dass keine NaN-Werte die Berechnung stören
    valid_data = group.dropna(subset=['y_true', 'y_pred'])
    if valid_data.empty:
        return np.nan
        
    return np.sqrt(mean_squared_error(valid_data['y_true'], valid_data['y_pred']))

def analyze_performance(data):
    """Gruppiert Daten und berechnet die Leistung über Zyklen."""
    # Überprüfen, ob die notwendigen Spalten vorhanden sind
    required_cols = ['strategy', 'cycle', 'y_true', 'y_pred']
    if not all(col in data.columns for col in required_cols):
        print(f"FEHLER: Die CSV-Datei muss die Spalten {required_cols} enthalten.", file=sys.stderr)
        sys.exit(1)

    print("Berechne Modelleistung (RMSE) pro Strategie und Zyklus...")
    
    # Gruppieren nach Strategie und Zyklus, dann RMSE berechnen
    performance_over_time = data.groupby(['strategy', 'cycle']).apply(calculate_rmse)
    
    # Die resultierende Serie in einen DataFrame umwandeln und umbenennen
    performance_df = performance_over_time.rename('RMSE').reset_index()
    
    return performance_df

def plot_performance(performance_df, output_filename='model_performance_over_cycles.png'):
    """Erstellt ein Liniendiagramm der Leistung über Zyklen."""
    if performance_df.empty or performance_df['RMSE'].isnull().all():
        print("Keine gültigen Daten zum Plotten vorhanden.")
        return

    print(f"Erstelle Plot und speichere als '{output_filename}'...")
    
    plt.figure(figsize=(12, 7))
    
    # Seaborn eignet sich hervorragend für gruppierte Liniendiagramme
    sns.lineplot(
        data=performance_df,
        x='cycle',
        y='RMSE',
        hue='strategy',   # Erzeugt eine Linie pro Strategie
        style='strategy', # Verwendet unterschiedliche Linienstile
        markers=True,     # Fügt Marker an den Datenpunkten hinzu
        markersize=8,
        linewidth=2.5
    )
    
    plt.title('Modelleistung (RMSE) über Active Learning Zyklen', fontsize=16)
    plt.xlabel('Active Learning Zyklus', fontsize=12)
    plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=12)
    plt.legend(title='Strategie')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Setzt die X-Achse so, dass sie nur ganze Zahlen für Zyklen anzeigt
    # (besonders nützlich, wenn es viele Zyklen gibt)
    max_cycle = performance_df['cycle'].max()
    if max_cycle > 1:
        plt.xticks(ticks=range(1, int(max_cycle) + 1))
        
    plt.tight_layout()
    plt.savefig(output_filename)
    print("Plot erfolgreich gespeichert.")

# --- Hauptausführung ---
if __name__ == "__main__":
    data = load_data(FILE_PATH)
    performance_results = analyze_performance(data)
    
    print("\n--- Berechnete Leistung ---")
    print(performance_results)
    print("----------------------------\n")
    
    plot_performance(performance_results)