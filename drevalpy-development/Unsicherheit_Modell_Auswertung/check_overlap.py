import pandas as pd
import sys
import numpy as np

# Pfad zu deiner kombinierten Ergebnisdatei
ALL_RESULTS_FILE = "/home/mi/zhaow01/dokumente/BA/drevalpy-development/results/GDSC2/baseline_models_ale_epic/exclude_target_ale_epic/kombinierte_results.csv"

def check_selection_overlap_aggregated(filepath: str):
    """
    Analysiert die all_results.csv, um den MITTELWERT und die STANDARDABWEICHUNG
    der Überschneidung (NEU und GESAMT) pro Zyklus über ALLE ZELLLINIEN
    zwischen den Strategien zu berechnen.
    """
    print(f"--- Aggregierte Überprüfung der Strategie-Überschneidung für: {filepath} ---")
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"[FEHLER] Datei nicht gefunden: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"[FEHLER] Konnte Datei nicht laden: {e}")
        sys.exit(1)

    # (Daten-Lade- und Verarbeitungs-Teil: Unverändert)
    required_cols = ['cycle', 'strategy', 'Known_Drug', 'cell_line', 'drug_id']
    if not all(col in df.columns for col in required_cols):
        print(f"[FEHLER] Es fehlen Spalten. Benötigt: {required_cols}")
        return
        
    df['cell_line'] = df['cell_line'].astype(str)
    df['drug_id'] = df['drug_id'].astype(str)

    all_cell_lines = sorted(df['cell_line'].dropna().unique())
    if not all_cell_lines:
        print("[FEHLER] Keine Zelllinien in der Datei gefunden.")
        return
        
    print(f"Analysiere und aggregiere {len(all_cell_lines)} Zelllinien...")

    all_metrics_list = []

    for i, cell_line in enumerate(all_cell_lines):
        sys.stdout.write(f"\r  Verarbeite Zelllinie {i+1}/{len(all_cell_lines)}: {cell_line.ljust(20)}")
        sys.stdout.flush()
        
        df_cell = df[df['cell_line'] == cell_line]
        cycles = sorted(df_cell['cycle'].dropna().unique())
        if not cycles:
            continue

        prev_sets = {'epistemic': set(), 'aleatoric': set()}
        current_cumulative_sets = {}

        for cycle in cycles:
            df_cycle = df_cell[df_cell['cycle'] == cycle]
            new_sets = {}
            cycle_metrics = {'cell_line': cell_line, 'cycle': cycle}

            for strategy in ['epistemic', 'aleatoric']:
                df_strat_cycle = df_cycle[
                    (df_cycle['strategy'] == strategy) & 
                    (df_cycle['Known_Drug'] == True)
                ]
                current_cumulative_set = set(df_strat_cycle['drug_id'])
                current_cumulative_sets[strategy] = current_cumulative_set
                
                newly_acquired_set = current_cumulative_set - prev_sets[strategy]
                new_sets[strategy] = newly_acquired_set
                
                prev_sets[strategy] = current_cumulative_set

                cycle_metrics[f'neu_{strategy}'] = len(newly_acquired_set)
                cycle_metrics[f'gesamt_{strategy}'] = len(current_cumulative_set)

            if 'epistemic' not in new_sets or 'aleatoric' not in new_sets:
                continue

            new_epi, new_ale = new_sets['epistemic'], new_sets['aleatoric']
            new_overlap = new_epi.intersection(new_ale)
            new_union = new_epi.union(new_ale)
            cycle_metrics['overlap_neu'] = len(new_overlap)
            cycle_metrics['jaccard_neu'] = (len(new_overlap) / len(new_union)) * 100 if new_union else 0.0
            
            total_epi, total_ale = current_cumulative_sets['epistemic'], current_cumulative_sets['aleatoric']
            total_overlap = total_epi.intersection(total_ale)
            total_union = total_epi.union(total_ale)
            cycle_metrics['overlap_gesamt'] = len(total_overlap)
            cycle_metrics['jaccard_gesamt'] = (len(total_overlap) / len(total_union)) * 100 if total_union else 0.0
            
            all_metrics_list.append(cycle_metrics)

    sys.stdout.write("\n")
    print("Verarbeitung abgeschlossen. Berechne finale aggregierte Tabelle...\n")

    # ================== ÄNDERUNG START: Kombinierte, kompakte Tabelle ==================
    if not all_metrics_list:
        print("[FEHLER] Keine Metriken zum Aggregieren gefunden.")
        return

    df_results = pd.DataFrame(all_metrics_list)
    
    # Gruppiere nach Zyklus und berechne Mittelwert und Standardabweichung
    grouped = df_results.groupby('cycle')
    mean_stats = grouped.mean(numeric_only=True)
    std_stats = grouped.std(numeric_only=True).fillna(0) # Fülle NaNs bei Std=0 auf

    # --- 1. SELBSTERKLÄRENDE EINLEITUNG ---
    print("="*120)
    print(" Aggregierte Analyse der Strategie-Überschneidung".center(120))
    print("="*120)
    print("Diese Tabelle fasst zusammen, wie ähnlich sich die Strategien 'epistemic' (Epi) und 'aleatoric' (Ale) bei der Auswahl neuer Medikamente verhalten.")
    print(f"Alle Werte sind als `Mittelwert (Standardabweichung)` über {len(all_cell_lines)} Zelllinien angegeben.\n")
    print("SPALTENERKLÄRUNG:")
    print("  - Cyc:         Active Learning Zyklus.")
    print("  - Neu (Epi/Ale): Avg. Anzahl *neu* ausgewählter Medikamente.")
    print("  - Ges (Epi/Ale): Avg. Anzahl *insgesamt* ausgewählter Medikamente.")
    print("  - Olap (Neu):  Avg. Anzahl *identischer* Medikamente, die beide Strategien *neu* ausgewählt haben.")
    print("  - Jacc (Neu)%:  Ähnlichkeit der *neuen* Auswahl (0%=verschieden, 100%=identisch).")
    print("  - Jacc (Ges)%:  Ähnlichkeit der *gesamten* gesammelten Sets.")
    
    # --- Vorbereiten der formatierten Strings ---
    # Hilfsfunktion für `Avg (Std)` Format
    def fmt(m, s, p=1):
        return f"{m:.{p}f} ({s:.{p}f})"

    # Wende Formatierung für jede Spalte an
    mean_stats['neu_epi_str'] = [fmt(m, s) for m, s in zip(mean_stats['neu_epistemic'], std_stats['neu_epistemic'])]
    mean_stats['neu_ale_str'] = [fmt(m, s) for m, s in zip(mean_stats['neu_aleatoric'], std_stats['neu_aleatoric'])]
    mean_stats['ges_epi_str'] = [fmt(m, s) for m, s in zip(mean_stats['gesamt_epistemic'], std_stats['gesamt_epistemic'])]
    mean_stats['ges_ale_str'] = [fmt(m, s) for m, s in zip(mean_stats['gesamt_aleatoric'], std_stats['gesamt_aleatoric'])]
    mean_stats['olap_neu_str'] = [fmt(m, s) for m, s in zip(mean_stats['overlap_neu'], std_stats['overlap_neu'])]
    mean_stats['jacc_neu_str'] = [f"{m:.1f} ({s:.1f}) %" for m, s in zip(mean_stats['jaccard_neu'], std_stats['jaccard_neu'])]
    mean_stats['jacc_ges_str'] = [f"{m:.1f} ({s:.1f}) %" for m, s in zip(mean_stats['jaccard_gesamt'], std_stats['jaccard_gesamt'])]

    # --- 2. KOMBINIERTE TABELLE DRUCKEN ---
    
    # Definiere einen Divider für die Tabelle
    div = f"|{'-'*5}|{'-'*12}|{'-'*12}|{'-'*14}|{'-'*14}|{'-'*12}|{'-'*15}|{'-'*15}|"
    print("\n" + div)
    print(f"| {'Cyc':<3} | {'Neu(Epi)':<10} | {'Neu(Ale)':<10} | {'Ges(Epi)':<12} | {'Ges(Ale)':<12} | {'Olap(Neu)':<10} | {'Jacc(Neu) %':<13} | {'Jacc(Ges) %':<13} |")
    print(div)
    
    for cycle in mean_stats.index:
        row = mean_stats.loc[cycle]
        print(
            f"| {int(cycle):<3} | "
            f"{row['neu_epi_str']:<10} | "
            f"{row['neu_ale_str']:<10} | "
            f"{row['ges_epi_str']:<12} | "
            f"{row['ges_ale_str']:<12} | "
            f"{row['olap_neu_str']:<10} | "
            f"{row['jacc_neu_str']:<13} | "
            f"{row['jacc_ges_str']:<13} |"
        )
    print(div)
    
    # --- 3. ZUSAMMENFASSUNG / INTERPRETATION ---
    print("\n Erklärung:")
    print("Der Jaccard-Index misst die Ähnlichkeit von zwei Sets:")
    print("  - **0%** = Völlig verschieden (keine gemeinsamen Elemente).")
    print("  - **100%** = Perfekt identisch (beide Strategien wählen exakt dasselbe aus).")
    print(f"\nEin **NIEDRIGER Jaccard-Wert** (< 20%) bedeutet, die Strategien sind 'divers'.")
    print(f"Ein **HOHER Jaccard-Wert** (> 50%) bedeutet, die Strategien sind 'ähnlich'.")
    print("="*120)
    # ================== ÄNDERUNG ENDE ==================


if __name__ == "__main__":
    check_selection_overlap_aggregated(ALL_RESULTS_FILE)