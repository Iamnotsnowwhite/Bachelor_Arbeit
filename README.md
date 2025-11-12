# Quantifying the Impact of Aleatoric and Epistemic Uncertainty on Drug Response Prediction in Active Learning Settings

## Summary

Due to the pronounced biological heterogeneity of cancer, drug response varies significantly from patient to patient. Models that provide general predictions are often insufficient for individualized therapies. In personalized medicine, precisely predicting individual drug efficacy is crucial. As generating new data is both time-consuming and costly, we utilize Active Learning (AL) to design the learning process more efficiently by selectively querying only the most informative data points. This paper investigates how two types of uncertainty—aleatoric uncertainty and epistemic uncertainty—impact the performance of an AL system for drug response prediction. We model aleatoric uncertainty using a Mean-Variance network, while epistemic uncertainty is estimated using Monte Carlo Dropout. Both sampling strategies are tested on a neural network trained on the GDSC2 dataset. The model predicts ln(IC50) values, receiving drug fingerprints and gene expression data as inputs. We compare our uncertainty-based sampling strategies against a random sampling baseline. The objective is to quantify which form of uncertainty more effectively improves model performance within an AL framework. Our results show that the epistemic strategy achieves a 28.06\% improvement over the random strategy, while the aleatoric strategy achieves 27.18\%. These similar outcomes are attributed to the strong correlation between the two uncertainty types, which consequently led to a very similar selection of samples. This work demonstrates that uncertainty-based AL approaches can effectively enhance model performance, thereby contributing to cancer research.


## Modellsturktur
<img width="842" height="406" alt="截屏2025-11-05 16 37 42" src="https://github.com/user-attachments/assets/157cb40d-1d3f-42c0-9744-4f37686b330f" />

## Relevant folders/files
- drevalpy-development/Unsicherheit_Auswertung
- drevalpy-development/Unsicherheit_Hauptscript/
- drevalpy-development/Unsicherheit_Modell_Auswertung
- drevalpy-development/Unsicherheit_Optuna_Baseline
- drevalpy-development/results/GDSC2
