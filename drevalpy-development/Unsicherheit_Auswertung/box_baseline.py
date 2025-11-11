import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# --- EINSTELLUNGEN ---
# IHR PFAD:
file_path = '/home/mi/zhaow01/dokumente/BA/drevalpy-development/results/GDSC2/baseline_models_ale_epic/exclude_target_ale_epic/global_baseline_preds.csv'

# Mindestanzahl an Datenpunkten pro drug_id, um sie zu berücksichtigen.
MIN_SAMPLES = 10 

# Legt den "Zoom"-Bereich für die Y-Achse fest, um Ausreißer auszublenden.
Y_AXIS_LIMITS = (-1.0, 1.1) 

# Filtert numerische Artefakte. R^2-Werte unter diesem Wert werden ignoriert.
R2_ARTIFACT_THRESHOLD = -100.0
# --- ENDE EINSTELLUNGEN ---


# --- 1. Daten laden ---
try:
    df = pd.read_csv(file_path)
    print(f"Daten erfolgreich geladen von: {file_path}")
    print(f"Insgesamt {len(df)} Zeilen geladen.")
except FileNotFoundError:
    print(f"FEHLER: Datei nicht gefunden unter: {file_path}")
    exit()
except Exception as e:
    print(f"Ein Fehler ist beim Laden der Datei aufgetreten: {e}")
    exit()


# --- 2. R²-Berechnung PRO MEDIKAMENT ---
# (Die Bootstrap-Analyse wurde entfernt)

per_drug_results = []
unique_drugs = df['drug_id'].unique()

print(f"\nStarte R²-Berechnung für {len(unique_drugs)} unique drug_ids...")

for i, drug in enumerate(unique_drugs):
    drug_df = df[df['drug_id'] == drug]
    
    # PRÜFUNG: Überspringe Drogen mit zu wenigen Datenpunkten
    if len(drug_df) < MIN_SAMPLES:
        # print(f"Warnung: drug_id {drug} hat nur {len(drug_df)} Samples (Min={MIN_SAMPLES}) und wird übersprungen.")
        continue
        
    y_true_drug = drug_df['y_true'].values
    y_pred_drug = drug_df['y_pred'].values
    
    # PRÜFUNG: Verhindert Division durch Null bei R², wenn y_true keine Varianz hat
    if np.var(y_true_drug) < 1e-6:
        # print(f"Warnung: drug_id {drug} hat keine Varianz in y_true und wird übersprungen.")
        continue

    try:
        # Berechne EINEN R²-Wert für dieses Medikament
        r2 = r2_score(y_true_drug, y_pred_drug)
        
        # PRÜFUNG: Filtert numerische Artefakte/Ausreißer
        if r2 > R2_ARTIFACT_THRESHOLD:
            per_drug_results.append({'drug_id': drug, 'R2_Score': r2})
            
    except ValueError:
        pass # Ignoriere Fehler bei der R²-Berechnung

# --- 3. Daten für Plot vorbereiten ---
if not per_drug_results:
    print("\nFEHLER: Keine Ergebnisse zum Plotten vorhanden. (Alle Drogen hatten zu wenige Samples?)")
    exit()

# Erstellt einen DataFrame, bei dem JEDE ZEILE ein Medikament und dessen R²-Wert ist
results_df = pd.DataFrame(per_drug_results)
results_df['drug_id'] = results_df['drug_id'].astype(str)

print(f"\nBerechnung abgeschlossen. R²-Werte für {len(results_df)} Medikamente werden geplottet.")


# --- 4. Plotting ---
# (Plot-Logik wurde vereinfacht, um EINEN Boxplot für alle Drogen zu erstellen)

print("\nErstelle Plot...")
plt.figure(figsize=(7, 9)) # Schmaler, höherer Plot

# 1. Der Boxplot: Zeigt die Hauptverteilung (Quartile, Median)
sns.boxplot(
    y='R2_Score', 
    data=results_df, 
    color='skyblue', 
    width=0.3,
    showfliers=False # Wir blenden die Standard-Ausreißer aus...
)

# 2. Der Stripplot: Zeigt JEDEN EINZELNEN PUNKT (jeder Punkt = 1 Medikament)
# Dies ist, was Sie mit "jeder Punkt ein R²" meinten.
sns.stripplot(
    y='R2_Score', 
    data=results_df, 
    color='black', 
    alpha=0.4,       # Transparenz für die Punkte
    jitter=0.2,      # Streut die Punkte horizontal, damit sie sich nicht überlappen
    s=3              # Kleinere Punktgröße
)

# Referenzlinie
plt.axhline(0, color='red', linestyle='--', linewidth=1.0, label='R² = 0')

# Achsen-Limits ("Zoom")
plt.ylim(Y_AXIS_LIMITS) 

plt.title(f'Verteilung der R²-Performance pro Medikament\n(n = {len(results_df)} Medikamente)', fontsize=14)
plt.xlabel('Alle berücksichtigten Medikamente', fontsize=12)
plt.ylabel(f'R² Score (pro Medikament)\n(Y-Achse begrenzt auf {Y_AXIS_LIMITS})', fontsize=12)
plt.legend()

# X-Achsen-Beschriftung entfernen, da sie nur eine Kategorie darstellt
plt.xticks([]) 
plt.tight_layout()

# Speichern der Abbildung
plot_filename = 'r2_performance_PER_DRUG_boxplot.png'
plt.savefig(plot_filename)

print(f"\nBoxplot gespeichert als: {plot_filename}")

# Zeige Median und Mittelwert in der Konsole an
print("\n--- Statistik der R²-Werte (pro Medikament) ---")
print(f"Median R²:  {results_df['R2_Score'].median():.4f}")
print(f"Mittelw. R²: {results_df['R2_Score'].mean():.4f}")
print(f"Std.Abw. R²: {results_df['R2_Score'].std():.4f}")
print(f"Min R²:     {results_df['R2_Score'].min():.4f}")
print(f"Max R²:     {results_df['R2_Score'].max():.4f}")