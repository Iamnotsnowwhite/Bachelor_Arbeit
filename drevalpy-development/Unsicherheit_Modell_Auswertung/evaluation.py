import pandas as pd
import numpy as np
import argparse
from typing import Dict, List, Optional, Tuple # 'Tuple' ist für die geänderte Funktion
import sys
from scipy import stats # Import für t-Tests

# --- Metrik-Funktionen (Unverändert) ---
def metrics_all(y_true, y_pred) -> Dict[str, float]:
    """Calculates standard regression metrics (MSE, RMSE, MAE, R2, Pearson correlation)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {"MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "Pearson": np.nan}

    mse  = float(np.mean((y_true - y_pred)**2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    
    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = float(1 - (ss_res / (ss_tot + 1e-12)))
    
    if y_true.std() < 1e-12 or y_pred.std() < 1e-12:
        pr = 0.0
    else:
        pr = float(np.corrcoef(y_true, y_pred)[0,1])
        
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "Pearson": pr}

def calculate_baseline_metrics(baseline_df: pd.DataFrame) -> Dict[str, float]:
    """Calculates metrics for the baseline dataframe."""
    print("Calculating metrics for Baseline...")
    if baseline_df.empty:
        return {"MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "Pearson": np.nan}
    return metrics_all(baseline_df['y_true'], baseline_df['y_pred'])

def calculate_metrics_for_cycle(strategy_df: pd.DataFrame, strategy_name: str, cycle: float) -> Dict[str, float]:
    """Filters for a SPECIFIC cycle of an AL strategy and calculates metrics."""
    if 'cycle' not in strategy_df.columns:
         print(f"[Warning] No 'cycle' column for {strategy_name}. Returning NaNs.")
         return {"MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "Pearson": np.nan}
    cycle_df = strategy_df[strategy_df['cycle'] == cycle]
    if cycle_df.empty:
        return {"MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "Pearson": np.nan}
    return metrics_all(cycle_df['y_true'], cycle_df['y_pred'])

def calculate_final_cycle_metrics(strategy_df: pd.DataFrame, strategy_name: str) -> Dict[str, float]:
    """Filters for the FINAL cycle of an AL strategy and calculates metrics."""
    print(f"Calculating metrics for {strategy_name} (Final Cycle)...")
    if strategy_df.empty:
        return {"MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "Pearson": np.nan}
    if 'cycle' not in strategy_df.columns or strategy_df['cycle'].isnull().all():
        print(f"[Warning] No 'cycle' column found for {strategy_name}. Using all data (assuming baseline).")
        return metrics_all(strategy_df['y_true'], strategy_df['y_pred'])
    max_cycle = strategy_df['cycle'].max()
    if pd.isna(max_cycle):
        print(f"[Error] Could not determine final cycle for {strategy_name}. Skipping.")
        return {"MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "Pearson": np.nan}
    print(f"Found final cycle: {int(max_cycle)}")
    final_cycle_df = strategy_df[strategy_df['cycle'] == max_cycle].copy()
    final_cycle_df['y_true'] = pd.to_numeric(final_cycle_df['y_true'], errors='coerce')
    final_cycle_df['y_pred'] = pd.to_numeric(final_cycle_df['y_pred'], errors='coerce')
    return metrics_all(final_cycle_df['y_true'], final_cycle_df['y_pred'])

# --- MODIFIZIERTE P-WERT-FUNKTION (Gibt p-Wert UND t-Statistik zurück) ---
def calculate_p_value(df_base: pd.DataFrame, df_comp: pd.DataFrame) -> Tuple[float, float]:
    """
    Führt einen gepaarten t-Test auf den quadratischen Fehlern (SE) von zwei Modellen
    auf einem gemeinsamen Satz von Datenpunkten durch.
    
    Vergleicht df_comp (z.B. Epistemic) mit df_base (z.B. Random).
    
    NEU: Gibt (p_value, t_statistic) zurück.
    """
    
    # Notwendige Spalten
    cols = ['cell_line', 'drug_id', 'y_true', 'y_pred']
    if not all(c in df_base.columns for c in cols) or not all(c in df_comp.columns for c in cols):
        print("[Warning] p-Wert-Berechnung übersprungen: Spalten 'cell_line', 'drug_id', 'y_true', 'y_pred' nicht in beiden DFs.")
        return np.nan, np.nan 

    try:
        # Setze Indizes für perfektes Paaren
        df_base_indexed = df_base.set_index(['cell_line', 'drug_id'])
        df_comp_indexed = df_comp.set_index(['cell_line', 'drug_id'])
        
        # Führe einen inneren Join durch, um *nur* die gemeinsamen Punkte zu erhalten
        joined = df_base_indexed.join(df_comp_indexed, how='inner', lsuffix='_base', rsuffix='_comp')
        
        if len(joined) < 10: # Zu wenige Paare für einen sinnvollen t-Test
            return np.nan, np.nan 
            
        # Berechne die quadratischen Fehler (SE) für jeden Punkt
        se_base = (joined['y_true_base'] - joined['y_pred_base'])**2
        se_comp = (joined['y_true_comp'] - joined['y_pred_comp'])**2
        
        # Entferne NaNs, falls welche durch Berechnungen entstanden sind
        mask = ~np.isnan(se_base) & ~np.isnan(se_comp)
        se_base = se_base[mask]
        se_comp = se_comp[mask]

        if len(se_base) < 10:
            return np.nan, np.nan 
            
        # Wenn die Fehler identisch sind, ist p = 1.0 (und t = 0.0)
        if np.allclose(se_base, se_comp):
            return 1.0, 0.0 
            
        # Führe den gepaarten (related) t-Test durch
        t_statistic, p_value = stats.ttest_rel(se_base, se_comp)
        
        # Wir testen, ob der MSE_base != MSE_comp ist (zweiseitiger Test)
        return p_value, t_statistic 
        
    except Exception as e:
        print(f"[Error] T-Test fehlgeschlagen: {e}")
        return np.nan, np.nan

# --- MODIFIZIERTE TABELLENFUNKTION (Mit t-Statistik Spalte) ---
def print_summary_table(all_metrics: List[Dict[str, float]], all_names: List[str], title: str, base_name: str):
    """
    Druckt eine Zusammenfassungstabelle.
    NEU: Zeigt jetzt auch p-Wert und t-Statistik (vs. Base) an.
    """
    
    metrics_to_show = ["MSE", "RMSE", "MAE"]
    base_comparison_metric = "MSE"
    
    # Spaltenbreiten
    name_col_width = 20
    metric_col_width = 15
    perc_col_width = 15 # "% vs [base_name]"
    pval_col_width = 15 # "p-Wert (vs Base)"
    tval_col_width = 15 # "t-Statistik" <<< NEU
    
    # Gesamtbreite der Tabelle
    width = name_col_width + 4
    width += (metric_col_width + 3) * len(metrics_to_show)
    width += (perc_col_width + 3)
    width += (pval_col_width + 3) 
    width += (tval_col_width + 3) # <<< NEU
    
    print("\n" + "="*width)
    print(f"| {title:^{width-4}} |")
    print("="*width)

    base_metrics = None
    try:
        base_index = all_names.index(base_name)
        base_metrics = all_metrics[base_index]
    except (ValueError, IndexError):
        pass

    if base_metrics is None or pd.isna(base_metrics.get(base_comparison_metric)):
        print(f"[Warning] Base strategy '{base_name}' metrics not found or are NaN. Cannot calculate relative improvements.")
        base_metrics = {}
            
    # Erstelle Header
    header = f"| {'Method':<{name_col_width}} |"
    divider = f"|{'-'*(name_col_width+2)}|"
    
    for metric in metrics_to_show:
        header += f" {metric:<{metric_col_width}} |"
        divider += f"{'-'*(metric_col_width+2)}|"
        
        if metric == base_comparison_metric:
            perc_header = f"% vs {base_name}"
            header += f" {perc_header:<{perc_col_width}} |"
            divider += f"{'-'*(perc_col_width+2)}|"
            
            # p-Wert Spaltenkopf
            pval_header = f"p-Wert (vs {base_name})"
            header += f" {pval_header:<{pval_col_width}} |"
            divider += f"{'-'*(pval_col_width+2)}|"
            
            # <<< NEU: t-Statistik Spaltenkopf >>>
            tval_header = "t-Statistik"
            header += f" {tval_header:<{tval_col_width}} |"
            divider += f"{'-'*(tval_col_width+2)}|"
    
    print(header)
    print(divider)

    # Schleife durch jede Methode (Zeile)
    for name, metrics in zip(all_names, all_metrics):
        line_str = f"| {name:<{name_col_width}} |"
        
        for metric in metrics_to_show:
            val = metrics.get(metric, np.nan)
            line_str += f" {val:<{metric_col_width}.6f} |"
            
            if metric == base_comparison_metric:
                # %-Wert
                perc_str = "N/A"
                base_val = base_metrics.get(base_comparison_metric, np.nan)
                if name == base_name:
                    perc_str = "(Base)"
                elif not np.isnan(base_val) and not np.isnan(val) and base_val != 0:
                    try:
                        improvement = (base_val - val) / base_val * 100
                        perc_str = f"{improvement:+.2f}%"
                    except (ValueError, ZeroDivisionError):
                        pass
                line_str += f" {perc_str:<{perc_col_width}} |"

                # p-Wert Zelle
                pval_str = "N/A"
                if name == base_name:
                    pval_str = "-"
                else:
                    pval = metrics.get("p_value", np.nan) 
                    if not pd.isna(pval):
                        if pval < 0.001:
                            pval_str = "< 0.001" 
                        else:
                            pval_str = f"{pval:.3f}"
                line_str += f" {pval_str:<{pval_col_width}} |"

                # <<< NEU: t-Statistik Zelle >>>
                tval_str = "N/A"
                if name == base_name:
                    tval_str = "-"
                else:
                    tval = metrics.get("t_stat", np.nan) # t-Statistik aus dem dict holen
                    if not pd.isna(tval):
                        tval_str = f"{tval:.3f}" # Auf 3 Dezimalstellen runden
                line_str += f" {tval_str:<{tval_col_width}} |"
        
        print(line_str)
        
    print("="*width)

# --- MODIFIZIERTE ANALYSEFUNKTION (Speichert p-Wert UND t-Statistik) ---
def run_full_analysis(df_baseline: pd.DataFrame, 
                        df_random: pd.DataFrame, 
                        df_epistemic: pd.DataFrame, 
                        df_aleatoric: pd.DataFrame, 
                        analysis_title_suffix: str,
                        strategies_to_show: Optional[List[str]] = None):
    """
    Führt die Analyse durch UND berechnet p-Werte & t-Statistiken für Vergleiche.
    """

    # 1. Berechne Baseline-Metriken (nur einmal)
    metrics_baseline = calculate_baseline_metrics(df_baseline)
    
    # 2. PER-ZYKLUS-ANALYSE
    print("\n\n" + "="*80)
    print(f"| {f'PER-CYCLE ANALYSIS {analysis_title_suffix} (vs. Random)':^78} |")
    print("="*80)
    
    al_dfs = [df_random, df_epistemic, df_aleatoric]
    all_cycles = pd.concat([df['cycle'] for df in al_dfs if 'cycle' in df.columns]).dropna().unique()
    all_cycles.sort()
    
    if len(all_cycles) == 0:
        print("\n[Warning] No cycles found in AL data. Skipping per-cycle analysis.")
    else:
        print(f"\nFound cycles to analyze: {all_cycles}")
        
        for cycle in all_cycles:
            cycle_int = int(cycle)
            
            # Hole die Daten für DIESEN Zyklus
            df_random_cycle = df_random[df_random['cycle'] == cycle]
            df_epistemic_cycle = df_epistemic[df_epistemic['cycle'] == cycle]
            df_aleatoric_cycle = df_aleatoric[df_aleatoric['cycle'] == cycle]

            # Berechne Metriken
            all_metrics_data_c = {
                "Baseline": metrics_baseline.copy(), # .copy() ist sicherer
                "Random": calculate_metrics_for_cycle(df_random, "Random", cycle),
                "Epistemic": calculate_metrics_for_cycle(df_epistemic, "Epistemic", cycle),
                "Aleatoric": calculate_metrics_for_cycle(df_aleatoric, "Aleatoric", cycle)
            }
            
            # <<< GEÄNDERT: p-Werte & t-Statistiken berechnen >>>
            p_val, t_stat = calculate_p_value(df_random_cycle, df_baseline)
            all_metrics_data_c["Baseline"]["p_value"] = p_val
            all_metrics_data_c["Baseline"]["t_stat"] = t_stat

            p_val, t_stat = calculate_p_value(df_random_cycle, df_epistemic_cycle)
            all_metrics_data_c["Epistemic"]["p_value"] = p_val
            all_metrics_data_c["Epistemic"]["t_stat"] = t_stat
            
            p_val, t_stat = calculate_p_value(df_random_cycle, df_aleatoric_cycle)
            all_metrics_data_c["Aleatoric"]["p_value"] = p_val
            all_metrics_data_c["Aleatoric"]["t_stat"] = t_stat
            
            all_metrics_data_c["Random"]["p_value"] = 1.0 # (Basis vs. Basis)
            all_metrics_data_c["Random"]["t_stat"] = 0.0 # (Basis vs. Basis)
            
            # Filterlogik (unverändert)
            if strategies_to_show is None:
                strategies_to_print_c = ["Baseline", "Random", "Epistemic", "Aleatoric"]
            else:
                strategies_to_print_c = strategies_to_show
            all_metrics_c = [all_metrics_data_c[name] for name in strategies_to_print_c if name in all_metrics_data_c]
            all_names_c = [name for name in strategies_to_print_c if name in all_metrics_data_c]

            print_summary_table(
                all_metrics_c, 
                all_names_c, 
                title=f"Summary for Cycle {cycle_int} {analysis_title_suffix}", 
                base_name="Random"
            )
    
    # 3. FINALE ZUSAMMENFASSUNG (vergleicht nur den LETZTEN Zyklus)
    print("\n\n" + "="*80)
    print(f"| {f'FINAL SUMMARY ANALYSIS {analysis_title_suffix} (Comparing Final Cycles)':^78} |")
    print("="*80)
    
    # Hole die Daten für den FINALEN Zyklus
    df_random_final = df_random[df_random['cycle'] == df_random['cycle'].max()]
    df_epistemic_final = df_epistemic[df_epistemic['cycle'] == df_epistemic['cycle'].max()]
    df_aleatoric_final = df_aleatoric[df_aleatoric['cycle'] == df_aleatoric['cycle'].max()]

    # Berechne Metriken
    all_metrics_data_final = {
        "Baseline": metrics_baseline.copy(), # .copy() ist sicherer
        "Random (Final)": calculate_final_cycle_metrics(df_random, "Random"),
        "Epistemic (Final)": calculate_final_cycle_metrics(df_epistemic, "Epistemic"),
        "Aleatoric (Final)": calculate_final_cycle_metrics(df_aleatoric, "Aleatoric")
    }
    
    # <<< GEÄNDERT: p-Werte & t-Statistiken berechnen >>>
    p_val, t_stat = calculate_p_value(df_random_final, df_baseline)
    all_metrics_data_final["Baseline"]["p_value"] = p_val
    all_metrics_data_final["Baseline"]["t_stat"] = t_stat

    p_val, t_stat = calculate_p_value(df_random_final, df_epistemic_final)
    all_metrics_data_final["Epistemic (Final)"]["p_value"] = p_val
    all_metrics_data_final["Epistemic (Final)"]["t_stat"] = t_stat
            
    p_val, t_stat = calculate_p_value(df_random_final, df_aleatoric_final)
    all_metrics_data_final["Aleatoric (Final)"]["p_value"] = p_val
    all_metrics_data_final["Aleatoric (Final)"]["t_stat"] = t_stat

    all_metrics_data_final["Random (Final)"]["p_value"] = 1.0 # (Basis vs. Basis)
    all_metrics_data_final["Random (Final)"]["t_stat"] = 0.0 # (Basis vs. Basis)

    # Filterlogik (unverändert)
    if strategies_to_show is None:
        default_names = ["Baseline", "Random (Final)", "Epistemic (Final)", "Aleatoric (Final)"]
        strategies_to_print_final = default_names
    else:
        strategies_to_print_final = []
        for s in strategies_to_show:
            if s == "Baseline":
                strategies_to_print_final.append("Baseline")
            else:
                strategies_to_print_final.append(f"{s} (Final)")
    all_metrics_final = [all_metrics_data_final[name] for name in strategies_to_print_final if name in all_metrics_data_final]
    all_names_final = [name for name in strategies_to_print_final if name in all_metrics_data_final]
    
    print_summary_table(
        all_metrics_final, 
        all_names_final, 
        title=f"FINAL SUMMARY {analysis_title_suffix} (All comparisons vs. Random)", 
        base_name="Random (Final)"
    )

# --- Main-Funktion (Unverändert) ---
def main():
    """Main function to parse arguments and run the analysis."""
    
    # === [START] DATENLADE-TEIL ===
    # HINWEIS: Pfade müssen an Ihre Dateistruktur angepasst werden!
    baseline_file_path = "/home/mi/zhaow01/dokumente/BA/drevalpy-development/results/GDSC2/baseline_models_ale_epic/exclude_target_ale_epic/global_baseline_preds.csv" 
    al_results_file_path = "/home/mi/zhaow01/dokumente/BA/drevalpy-development/results/GDSC2/baseline_models_ale_epic/exclude_target_ale_epic/kombinierte_results.csv"

    print(f"Loading AL results from: {al_results_file_path}")
    print(f"Loading Baseline results from: {baseline_file_path}")

    try:
        df_al_all = pd.read_csv(al_results_file_path)
        print("\nSuccessfully loaded the Active Learning (AL) result file.")
        df_baseline = pd.read_csv(baseline_file_path)
        print("Successfully loaded the separate Baseline result file.")
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e.filename}")
        print("Please check your file paths defined in 'baseline_file_path' and 'al_results_file_path'")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file(s): {e}")
        sys.exit(1)
        
    df_random = df_al_all[df_al_all['strategy'] == 'random'].copy()
    df_epistemic = df_al_all[df_al_all['strategy'] == 'epistemic'].copy()
    df_aleatoric = df_al_all[df_al_all['strategy'] == 'aleatoric'].copy()

    print("\nDataFrames split by strategy:")
    print(f"  Baseline rows:  {len(df_baseline)}")
    print(f"  Random rows:    {len(df_random)}")
    print(f"  Epistemic rows: {len(df_epistemic)}")
    print(f"  Aleatoric rows: {len(df_aleatoric)}")
    
    # === [ENDE] DATENLADE-TEIL ===


    # === [START] ANALYSE 1: ALLE DROGEN (Mit Data Leakage) ===
    print("\n\n" + "#"*80)
    print(f"| {'Running Analysis for ALL DRUGS (INCLUDES LEAKAGE)':^78} |")
    print("#"*80)
    all_pairs_analysis_1 = df_baseline[['cell_line', 'drug_id']].drop_duplicates()
    run_full_analysis(df_baseline, df_random, df_epistemic, df_aleatoric, 
                      analysis_title_suffix=f"(All Drugs, N={len(all_pairs_analysis_1)})")
    # === [ENDE] ANALYSE 1 ===


    # === [START] ANALYSE 3: GEMEINSAMER HOLD-OUT (FAIRER VERGLEICH) ===
    print("\n\n" + "#"*80)
    print(f"| {'Running Analysis for COMMON HOLDOUT SET (FAIREST COMPARISON)':^78} |")
    print("#"*80)

    if 'Known_Drug' not in df_al_all.columns:
        print("\n[ERROR] 'Known_Drug' column not found. Cannot run common holdout analysis.")
    else:
        df_al_only = df_al_all[df_al_all['strategy'].notnull()].copy()
        acquired_pairs = df_al_only[df_al_only['Known_Drug'] == True][['cell_line', 'drug_id']].drop_duplicates()
        if 'cell_line' not in df_baseline.columns or 'drug_id' not in df_baseline.columns:
             print("\n[ERROR] 'cell_line' or 'drug_id' columns not found in Baseline. Cannot create common holdout.")
        elif len(acquired_pairs) == 0:
            print("\n[Warning] No 'Known_Drug == True' pairs found. Common Holdout is identical to 'All Drugs'.")
        else:
            print(f"Found {len(acquired_pairs)} unique (cell, drug) pairs acquired by ANY strategy.")
            all_pairs = df_baseline[['cell_line', 'drug_id']].drop_duplicates()
            common_unknown_pairs = all_pairs.merge(
                acquired_pairs, on=['cell_line', 'drug_id'], how='left', indicator=True
            )
            common_unknown_pairs = common_unknown_pairs[common_unknown_pairs['_merge'] == 'left_only'][['cell_line', 'drug_id']]
            print(f"Created Common Holdout set with {len(common_unknown_pairs)} pairs.")

            if len(common_unknown_pairs) > 0:
                df_baseline_common = df_baseline.merge(
                    common_unknown_pairs, on=['cell_line', 'drug_id'], how='inner'
                ).copy()
                df_random_common = df_random.merge(
                    common_unknown_pairs, on=['cell_line', 'drug_id'], how='inner'
                ).copy()
                df_epistemic_common = df_epistemic.merge(
                    common_unknown_pairs, on=['cell_line', 'drug_id'], how='inner'
                ).copy()
                df_aleatoric_common = df_aleatoric.merge(
                    common_unknown_pairs, on=['cell_line', 'drug_id'], how='inner'
                ).copy()
                
                print("\nFiltered DataFrames for 'Common Holdout':")
                print(f"  Baseline (Common) rows:  {len(df_baseline_common)}")
                print(f"  Random (Common) rows:    {len(df_random_common)}")
                print(f"  Epistemic (Common) rows: {len(df_epistemic_common)}")
                print(f"  Aleatoric (Common) rows: {len(df_aleatoric_common)}")

                run_full_analysis(
                    df_baseline_common, 
                    df_random_common, 
                    df_epistemic_common, 
                    df_aleatoric_common, 
                    analysis_title_suffix=f"(Common Holdout - FAIREST, N={len(common_unknown_pairs)})"
                )
            else:
                print("\n[Warning] Common Holdout set is empty. Skipping analysis.")
    # === [ENDE] ANALYSE 3 ===


    # === [START] ANALYSE 4: PAIRWISE-HOLD-OUT (Random vs. Epistemic) ===
    print("\n\n" + "#"*80)
    print(f"| {'Running Analysis for PAIRWISE HOLDOUT (Random vs. Epistemic)':^78} |")
    print("#"*80)

    if 'Known_Drug' not in df_al_all.columns:
        print("\n[ERROR] 'Known_Drug' column not found. Cannot run pairwise holdout analysis.")
    else:
        df_al_only = df_al_all[df_al_all['strategy'].notnull()].copy()
        acquired_pairs_R_E = df_al_only[
            (df_al_only['strategy'].isin(['random', 'epistemic'])) & (df_al_only['Known_Drug'] == True)
        ][['cell_line', 'drug_id']].drop_duplicates()
        print(f"Found {len(acquired_pairs_R_E)} unique pairs acquired by Random OR Epistemic.")
        all_pairs = df_baseline[['cell_line', 'drug_id']].drop_duplicates()
        common_holdout_R_E = all_pairs.merge(
            acquired_pairs_R_E, on=['cell_line', 'drug_id'], how='left', indicator=True
        )
        common_holdout_R_E = common_holdout_R_E[common_holdout_R_E['_merge'] == 'left_only'][['cell_line', 'drug_id']]
        print(f"Created (R vs E) Holdout set with {len(common_holdout_R_E)} pairs.")
        
        if len(common_holdout_R_E) > 0:
            df_baseline_R_E = df_baseline.merge(common_holdout_R_E, on=['cell_line', 'drug_id'], how='inner').copy()
            df_random_R_E = df_random.merge(common_holdout_R_E, on=['cell_line', 'drug_id'], how='inner').copy()
            df_epistemic_R_E = df_epistemic.merge(common_holdout_R_E, on=['cell_line', 'drug_id'], how='inner').copy()
            df_aleatoric_R_E = df_aleatoric.merge(common_holdout_R_E, on=['cell_line', 'drug_id'], how='inner').copy() # Dummy

            print("\n[INFO] Running (R vs E) Analysis. Focus on 'Baseline', 'Random', 'Epistemic' rows.")
            run_full_analysis(
                df_baseline_R_E, 
                df_random_R_E, 
                df_epistemic_R_E, 
                df_aleatoric_R_E, # Wird intern ignoriert
                analysis_title_suffix=f"(Holdout R vs E, N={len(common_holdout_R_E)})",
                strategies_to_show=["Baseline", "Random", "Epistemic"]
            )
        else:
            print("\n[Warning] (R vs E) Holdout set is empty. Skipping analysis.")
    # === [ENDE] ANALYSE 4 ===


    # === [START] ANALYSE 5: PAIRWISE-HOLD-OUT (Random vs. Aleatoric) ===
    print("\n\n" + "#"*80)
    print(f"| {'Running Analysis for PAIRWISE HOLDOUT (Random vs. Aleatoric)':^78} |")
    print("#"*80)

    if 'Known_Drug' not in df_al_all.columns:
        print("\n[ERROR] 'Known_Drug' column not found. Cannot run pairwise holdout analysis.")
    else:
        df_al_only = df_al_all[df_al_all['strategy'].notnull()].copy()
        acquired_pairs_R_A = df_al_only[
            (df_al_only['strategy'].isin(['random', 'aleatoric'])) & (df_al_only['Known_Drug'] == True)
        ][['cell_line', 'drug_id']].drop_duplicates()
        print(f"Found {len(acquired_pairs_R_A)} unique pairs acquired by Random OR Aleatoric.")
        all_pairs = df_baseline[['cell_line', 'drug_id']].drop_duplicates()
        common_holdout_R_A = all_pairs.merge(
            acquired_pairs_R_A, on=['cell_line', 'drug_id'], how='left', indicator=True
        )
        common_holdout_R_A = common_holdout_R_A[common_holdout_R_A['_merge'] == 'left_only'][['cell_line', 'drug_id']]
        print(f"Created (R vs A) Holdout set with {len(common_holdout_R_A)} pairs.")
        
        if len(common_holdout_R_A) > 0:
            df_baseline_R_A = df_baseline.merge(common_holdout_R_A, on=['cell_line', 'drug_id'], how='inner').copy()
            df_random_R_A = df_random.merge(common_holdout_R_A, on=['cell_line', 'drug_id'], how='inner').copy()
            df_epistemic_R_A = df_epistemic.merge(common_holdout_R_A, on=['cell_line', 'drug_id'], how='inner').copy() # Dummy
            df_aleatoric_R_A = df_aleatoric.merge(common_holdout_R_A, on=['cell_line', 'drug_id'], how='inner').copy()

            print("\n[INFO] Running (R vs A) Analysis. Focus on 'Baseline', 'Random', 'Aleatoric' rows.")
            run_full_analysis(
                df_baseline_R_A, 
                df_random_R_A, 
                df_epistemic_R_A, # Wird intern ignoriert
                df_aleatoric_R_A, 
                analysis_title_suffix=f"(Holdout R vs A, N={len(common_holdout_R_A)})",
                strategies_to_show=["Baseline", "Random", "Aleatoric"]
            )
        else:
            print("\n[Warning] (R vs A) Holdout set is empty. Skipping analysis.")
    # === [ENDE] ANALYSE 5 ===


if __name__ == "__main__":
    main()