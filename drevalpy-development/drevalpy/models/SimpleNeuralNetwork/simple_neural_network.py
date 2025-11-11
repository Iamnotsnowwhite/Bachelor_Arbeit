"""Contains the SimpleNeuralNetwork model."""
import json
import os
import platform
import warnings

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from ..drp_model import DRPModel
from ..utils import load_and_select_gene_features, load_drug_fingerprint_features, scale_gene_expression
from .utils import FeedForwardNetwork

class SimpleNeuralNetwork(DRPModel):
    """Simple Feedforward Neural Network model with dropout using only gene expression data."""

    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]
    early_stopping = True

    def __init__(self):
        """Initializes the SimpleNeuralNetwork.

        The model is built in train(). The gene_expression_scalar is set to the StandardScaler() and later fitted
        using the training data only.
        """
        super().__init__()
        self.model = None
        self.hyperparameters = None
        self.optimizer = None 
        self.gene_expression_scaler = StandardScaler()

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: SimpleNeuralNetwork
        """
        return "SimpleNeuralNetwork"

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.

        :param hyperparameters: includes units_per_layer and dropout_prob.
        """
        self.hyperparameters = hyperparameters
        self.hyperparameters.setdefault("input_dim_gex", None)
        self.hyperparameters.setdefault("input_dim_fp", None)
        # ----------------我加的-------------------
        # Aleatoric should be on - for hyperparameter settings
        self.hyperparameters.setdefault("aleatoric", False) 
        # -----------------------------------------

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
        warm_start_path: str | None = None,      # <<< NEU
        # selected_drugs: list[str] | None = None,   # <<< NEU weights 
        # weight_factor: float = 1.0,            # <<< NEU weights 
    ) -> None:
        """
        First scales the gene expression data and trains the model.

        The gene expression data is first arcsinh transformed. Afterward, the StandardScaler() is fitted on the
        training gene expression data only. Then, it transforms all gene expression data.
        :param output: training data associated with the response output
        :param cell_line_input: cell line omics features
        :param drug_input: drug omics features
        :param output_earlystopping: optional early stopping dataset
        :param model_checkpoint_dir: directory to save the model checkpoints
        :raises ValueError: if drug_input (fingerprints) is missing

        """
        if drug_input is None:
            raise ValueError("drug_input (fingerprints) are required for SimpleNeuralNetwork.")

        # Apply arcsinh transformation and scaling to gene expression features
        if "gene_expression" in self.cell_line_views:
            cell_line_input = scale_gene_expression(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(output.cell_line_ids),
                training=True,
                gene_expression_scaler=self.gene_expression_scaler,
            )

        dim_gex = next(iter(cell_line_input.features.values()))["gene_expression"].shape[0]
        dim_fingerprint = next(iter(drug_input.features.values()))["fingerprints"].shape[0]
        self.hyperparameters["input_dim_gex"] = dim_gex
        self.hyperparameters["input_dim_fp"] = dim_fingerprint

        self.model = FeedForwardNetwork(
            hyperparameters=self.hyperparameters,
            input_dim=dim_gex + dim_fingerprint,
        )

        # ----- optional warm start，我加的 -----
        if warm_start_path is not None:
            import os, torch # 导入放在这里可以避免全局污染
            model_pt = os.path.join(warm_start_path, "model.pt")
            if os.path.exists(model_pt):
                print(f"[WARM-START] Loading weights from: {model_pt}")
                try:
                    state_dict = torch.load(model_pt, map_location="cpu", weights_only=True)  # torch>=2.4
                except TypeError:
                    state_dict = torch.load(model_pt, map_location="cpu")  # 兼容老版本
                load_res = self.model.load_state_dict(state_dict, strict=False)
                missing = getattr(load_res, "missing_keys", [])
                unexpected = getattr(load_res, "unexpected_keys", [])
                if missing:   print("[WARM-START] missing_keys:", missing)
                if unexpected:print("[WARM-START] unexpected_keys:", unexpected)
            else:
                print(f"[WARM-START] No model.pt under {warm_start_path}, skipping.")
        # -------------------------------------

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*does not have many workers which may be a bottleneck.*",
            )
            warnings.filterwarnings(
                "ignore",
                message="Starting from v1\\.9\\.0, `tensorboardX` has been removed.*",
            )
            if (output_earlystopping is not None) and len(output_earlystopping) == 0:
                output_earlystopping = output
                print("SimpleNeuralNetwork: Early stopping dataset empty. Using training data for early stopping")

                print("Probably, your training dataset is small.")

        # ------------------ 修正的 self.model.fit 调用 ------------------
        print(len(output))
        print(drug_input.meta_info)
        print(cell_line_input.meta_info)
        self.model.fit(
        output_train=output,
        drug_input=drug_input, 
        cell_line_input = cell_line_input,

        # cell_line_input=scale_gene_expression(  
        #     cell_line_input=cell_line_input,
        #     cell_line_ids=np.unique(output.cell_line_ids),
        #     training=False,
        #     #gene_expression_scaler=self.gene_expression_scaler,
        # ), 

        cell_line_views=self.cell_line_views,
        drug_views=self.drug_views,
        output_earlystopping=output_earlystopping,
        trainer_params={
            "max_epochs": self.hyperparameters.get("max_epochs", 100),
            "progress_bar_refresh_rate": self.hyperparameters.get("progress_bar_refresh_rate", 20),
            "precision" : 16, 
        },
        batch_size=self.hyperparameters.get("batch_size", 32), # wenn nicht dann 32
        patience=self.hyperparameters.get("patience", 5), # wenn nicht dann 5      
        num_workers=self.hyperparameters.get("num_workers", 1 if platform.system() == "Windows" else 8), 
        model_checkpoint_dir=model_checkpoint_dir,
        #selected_drugs=selected_drugs,                  # <<< NEU weights 
        #weight_factor=weight_factor,                    # <<< NEU weights 
        )

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts the response for the given input.

        :param cell_line_ids: IDs of the cell lines to be predicted
        :param drug_ids: IDs of the drugs to be predicted
        :param cell_line_input: gene expression of the test data
        :param drug_input: fingerprints of the test data
        :returns: the predicted drug responses
        """
        # Apply arcsinh transformation and scaling to gene expression features
        if "gene_expression" in self.cell_line_views:
            cell_line_input = scale_gene_expression(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(cell_line_ids),
                training=False,
                gene_expression_scaler=self.gene_expression_scaler,
            )

        x = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view="fingerprints",
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )

        return self.model.predict(x)

    # --------------------------------------------------------------------------
    ## High-level Uncertainty Prediction Wrapper (for Active Learning)
    # --------------------------------------------------------------------------
    def predict_uncertainty_by_ids(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        T: int = 50,
        keep_bn_eval: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        [Active Learning Wrapper] Performs MC-Dropout inference using IDs and FeatureDataset
        and returns the final statistics.
        
        This method handles feature scaling and concatenation before calling the underlying
        predict_uncertainty method.
        
        :returns: (y_pred_mean, sigma_epi, sigma_ale) all in the standardized space (mu_bar, sigma_epi, sigma_ale)
        """
        if drug_input is None:
            raise ValueError("drug_input (fingerprints) are required for uncertainty prediction.")

        # 1. 1. Feature Scaling (using the fitted self.gene_expression_scaler from training)
        if "gene_expression" in self.cell_line_views:
            cell_line_input = scale_gene_expression(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(cell_line_ids),
                training=False, # must be False: only transform
                gene_expression_scaler=self.gene_expression_scaler,
            )

        # 2. Feature Concatenation
        x_np = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view="fingerprints",
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        
        # 3. caculate uncertainty
        mean, sigma_epi, sigma_ale = self.model.predict_uncertainty(
            x_np=x_np, 
            T=T,
            keep_bn_eval=keep_bn_eval
        )
        return mean, sigma_epi, sigma_ale 


    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features.

        :param data_path: Path to the gene expression and landmark genes
        :param dataset_name: name of the dataset
        :return: FeatureDataset containing the cell line gene expression features, filtered through the landmark genes
        """
        return load_and_select_gene_features(
            feature_type="gene_expression",
            gene_list="landmark_genes",
            data_path=data_path,
            dataset_name=dataset_name,
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the fingerprint data.

        :param data_path: Path to the fingerprints, e.g., data/
        :param dataset_name: name of the dataset, e.g., GDSC1
        :returns: FeatureDataset containing the fingerprints
        """
        return load_drug_fingerprint_features(data_path, dataset_name, fill_na=True)

    def save(self, directory: str) -> None:
        """
        Save the trained model, hyperparameters, and gene expression scaler to the given directory.

        This enables full reconstruction of the model using `load`.

        Files saved:
        - model.pt: PyTorch state_dict of the trained model
        - hyperparameters.json: Dictionary containing all relevant model hyperparameters
        - scaler.pkl: Fitted StandardScaler for gene expression features

        :param directory: Target directory to store all model artifacts
        """
        os.makedirs(directory, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(directory, "model.pt")) # noqa: S614

        with open(os.path.join(directory, "hyperparameters.json"), "w") as f:
            json.dump(self.hyperparameters, f)

        joblib.dump(self.gene_expression_scaler, os.path.join(directory, "scaler.pkl"))

    @classmethod
    def load(cls, directory: str) -> "SimpleNeuralNetwork":
        """
        Load a trained SimpleNeuralNetwork instance from disk.

        This includes:
        - model.pt: PyTorch state_dict of the trained model
        - hyperparameters.json: Dictionary with model hyperparameters
        - scaler.pkl: Fitted StandardScaler for gene expression features

        :param directory: Directory containing the saved model files
        :return: An instance of SimpleNeuralNetwork with restored state
        :raises FileNotFoundError: if any required file is missing
        """
        hyperparam_file = os.path.join(directory, "hyperparameters.json")
        scaler_file = os.path.join(directory, "scaler.pkl")
        model_file = os.path.join(directory, "model.pt")

        if not all(os.path.exists(f) for f in [hyperparam_file, scaler_file, model_file]):
            raise FileNotFoundError("Missing model files. Required: model.pt, hyperparameters.json, scaler.pkl")

        instance = cls()

        with open(hyperparam_file) as f:
            instance.hyperparameters = json.load(f)

        instance.gene_expression_scaler = joblib.load(scaler_file)

        dim_gex = instance.hyperparameters["input_dim_gex"]
        dim_fp = instance.hyperparameters["input_dim_fp"]

        # ----------------我加的----------------
        try:
            # torch>=2.4 安全加载
            state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
        except (TypeError, RuntimeError): 
            # 兼容老版本或 weights_only 无法使用的情况
            state_dict = torch.load(model_file, map_location="cpu")

        has_logvar = any(k.startswith("logvar_head.") for k in state_dict.keys())

        # 若超参没写 aleatoric，则按checkpoint决定；若写了True但权重没有，也自动关掉
        if "aleatoric" not in instance.hyperparameters:
            instance.hyperparameters["aleatoric"] = bool(has_logvar)
        elif instance.hyperparameters["aleatoric"] and not has_logvar:
            print("[WARN] Checkpoint has no logvar_head.*; disabling aleatoric for compatibility.")
            instance.hyperparameters["aleatoric"] = False

        # 构建模型（会根据 aleatoric 开关决定是否创建 logvar_head）
        instance.model = FeedForwardNetwork(instance.hyperparameters, input_dim=dim_gex + dim_fp)

        # 宽松加载，忽略缺失/多余键
        res = instance.model.load_state_dict(state_dict, strict=False)
        if getattr(res, "missing_keys", None):    print("[INFO] Ignored missing keys:", res.missing_keys)
        if getattr(res, "unexpected_keys", None): print("[INFO] Ignored unexpected keys:", res.unexpected_keys)

        instance.model.eval()
        # ------------------------------------
        # instance.model = FeedForwardNetwork(instance.hyperparameters, input_dim=dim_gex + dim_fp)
        # instance.model.load_state_dict(torch.load(model_file))
        # instance.model.eval()
        return instance
    
    # --------------------------------------------------------------------------
    ## Self-Check for MC Dropout Randomness 
    # --------------------------------------------------------------------------
    def check_mc_randomness(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> bool:
        """
        Performs a self-check to ensure MC Dropout randomness is active.
        It runs MC inference twice (T=2) on a single sample and checks if the predictions differ.
        
        :returns: True if randomness is detected, False otherwise.
        """
        if self.model is None:
            print("[MC_CHECK] ERROR: Model has not been trained (self.model is None).")
            return False
            
        # 1. Select the first available sample for a quick check (N=1)
        # We assume the input IDs arrays are non-empty
        check_cl_id = cell_line_ids[0:1]
        check_drug_id = drug_ids[0:1]

        # 2. Perform Feature Concatenation for the single sample
        try:
            x_np_single = self.get_concatenated_features(
                cell_line_view="gene_expression",
                drug_view="fingerprints",
                cell_line_ids_output=check_cl_id,
                drug_ids_output=check_drug_id,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
            )
        except Exception as e:
            print(f"[MC_CHECK] ERROR during feature preparation: {e}")
            return False

        # 3. Perform two MC runs (T=2) to test for variance
        try:
            # Note: We must ensure input features are scaled correctly before this check, 
            # but predict_uncertainty_by_ids handles scaling, so we'll use a modified check here.
            # We call the underlying method directly with a small T=2 for speed.
            
            # Scale features manually just for the check, as the model expects scaled input
            # If GEX is used, ensure it is scaled by self.gene_expression_scaler
            if "gene_expression" in self.cell_line_views:
                check_input = scale_gene_expression(
                    cell_line_input=cell_line_input.copy(), # Use copy for safety
                    cell_line_ids=check_cl_id,
                    training=False,
                    gene_expression_scaler=self.gene_expression_scaler,
                )
                x_np_single = self.get_concatenated_features(
                    cell_line_view="gene_expression", drug_view="fingerprints",
                    cell_line_ids_output=check_cl_id, drug_ids_output=check_drug_id,
                    cell_line_input=check_input, drug_input=drug_input,
                )

            # Call core logic using T=2
            mean, sigma_epi, _ = self.model.predict_uncertainty(
                x_np=x_np_single, 
                T=2,
                keep_bn_eval=True
            )
            
            # sigma_epi is the standard deviation across T=2 runs. 
            # If sigma_epi > 0, randomness is present.
            is_random = bool(sigma_epi[0] > 1e-6) 
            
            if is_random:
                print(f"[MC_CHECK] SUCCESS: Randomness detected (sigma_epi > 1e-6). MC Dropout is active.")
            else:
                print("[MC_CHECK] FAILURE: No randomness detected (sigma_epi ≈ 0). Check model dropout settings.")
            
            return is_random

        except Exception as e:
            print(f"[MC_CHECK] CRITICAL ERROR during prediction: {e}")
            return False

    def set_learning_rate(self, new_lr: float):
        """
        Passt die Lernrate des internen Optimierers manuell an.

        Dies ist entscheidend für das Fine-Tuning, um die Lernschritte
        zu verkleinern und "katastrophales Vergessen" zu verhindern.

        :param new_lr: Die neue Lernrate, z.B. 1e-5.
        """
        if self.optimizer is None:
            # Dieser Fall sollte nicht eintreten, wenn das Modell geladen wurde,
            # aber es ist eine gute Sicherheitsüberprüfung.
            warnings.warn("[WARN] Optimizer has not been created yet. Cannot set learning rate.")
            return

        print(f"[INFO] Lernrate des Optimierers wird auf {new_lr} aktualisiert.")
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr