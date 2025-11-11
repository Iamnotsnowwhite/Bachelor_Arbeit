import pandas as pd

# (Dies ist der "sauberste" Datensatz, um die Korrelation im Start-Pool zu sehen)
BASELINE_PREDS_FILE = " /home/mi/zhaow01/dokumente/BA/drevalpy-development/results/GDSC2/baseline_models_ale_epic/exclude_target_ale_epic/global_baseline_results.csv"

# Pfad zur kombinierten Datei (alternativ, falls du Zyklen > 1 prüfen willst)
ALL_RESULTS_FILE = "/home/mi/zhaow01/dokumente/BA/drevalpy-development/results/GDSC2/baseline_models_ale_epic/exclude_target_ale_epic/kombinierte_results.csv"

def check_uncertainty_correlation(filepath: str):
    """
    Lädt eine Ergebnisdatei und berechnet die Pearson-Korrelation
    zwischen epistemischer und aleatorischer Unsicherheit.
    """
    print(f"--- Korrelationsprüfung für: {filepath} ---")
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"[FEHLER] Datei nicht gefunden: {filepath}")
        print("Bitte stelle sicher, dass das Skript im richtigen Verzeichnis ausgeführt wird.")
        return
    except Exception as e:
        print(f"[FEHLER] Konnte Datei nicht laden: {e}")
        return

    # Stelle sicher, dass die Spalten vorhanden sind
    if 'sigma_epi' not in df.columns or 'sigma_ale' not in df.columns:
        print("[FEHLER] Spalten 'sigma_epi' oder 'sigma_ale' nicht in der Datei gefunden.")
        return
        
    # Entferne NaN-Werte, um die Korrelation berechnen zu können
    df_clean = df.dropna(subset=['sigma_epi', 'sigma_ale'])
    
    if len(df_clean) == 0:
        print("[FEHLER] Keine gültigen (Nicht-NaN) Datenzeilen für die Korrelation gefunden.")
        return

    # Berechne die Pearson-Korrelation
    correlation = df_clean['sigma_epi'].corr(df_clean['sigma_ale'], method='pearson')
    
    print("\n" + "="*50)
    print(f" Pearson-Korrelation (sigma_epi vs. sigma_ale): {correlation:.4f}")
    print("="*50)

    # Interpretation
    if correlation > 0.7:
        print("\nInterpretation: SEHR HOHE KORRELATION.")
        print("Das bestätigt die Hypothese: Beide Strategien (epistemic, aleatoric)")
        print("identifizieren dieselben oder sehr ähnliche Datenpunkte als 'am unsichersten'.")
    elif correlation > 0.4:
        print("\nInterpretation: MITTLERE KORRELATION.")
        print("Die Strategien überschneiden sich, wählen aber auch unterschiedliche Punkte aus.")
    else:
        print("\nInterpretation: NIEDRIGE KORRELATION.")
        print("Die Strategien wählen unterschiedliche Punkte aus.")

if __name__ == "__main__":
    check_uncertainty_correlation("/home/mi/zhaow01/dokumente/BA/drevalpy-development/results/GDSC2/baseline_models_ale_epic/exclude_target_ale_epic/kombinierte_results.csv")