"""Utility functions for the simple neural network models."""

import os
import secrets

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from torch import nn
from torch.utils.data import DataLoader, Dataset

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset


class RegressionDataset(Dataset):
    """Dataset for regression tasks for the data loader."""

    def __init__(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset,
        cell_line_views: list[str],
        drug_views: list[str],
        selected_drugs: dict[str, set[str]] | None = None,
        # weight_factor: float = 1.0,  #>>>> NEU wegen den Weights
    ):
        """
        Initializes the regression dataset.

        :param output: response values
        :param cell_line_input: input omics data
        :param drug_input: input fingerprint data
        :param cell_line_views: either gene expression for the SimpleNeuralNetwork or all omics data for the
            MultiOMICSNeuralNetwork
        :param drug_views: fingerprints
        :raises AssertionError: if the views are not found in the input data
        """
        self.cell_line_views = cell_line_views
        self.drug_views = drug_views
        self.output = output
        self.cell_line_input = cell_line_input
        self.drug_input = drug_input
        for cl_view in self.cell_line_views:
            if cl_view not in cell_line_input.view_names:
                raise AssertionError(f"Cell line view {cl_view} not found in cell line input")
        for d_view in self.drug_views:
            if d_view not in drug_input.view_names:
                raise AssertionError(f"Drug view {d_view} not found in drug input")

        #######################################################
        # --- Gewichte initialisieren ---
        #      es ändert die weights 
        #######################################################        
        # print(f"Gewichtsfaktor im Datensatz ist: {weight_factor}")
        # self.weights = np.ones(len(output.response), dtype=np.float32)
        # if selected_drugs is not None:
        #     for i, (cl, drug) in enumerate(zip(output.cell_line_ids, output.drug_ids)):
        #         if cl in selected_drugs and drug in selected_drugs[cl]:
        #             self.weights[i] = weight_factor

    def __getitem__(self, idx):
        """
        Overwrites the getitem method from the Dataset class.

        Retrieves the cell line and drug features and the response for the given index.
        :param idx: index of the sample of interest
        :returns: the cell line feature(s) and the response
        :raises TypeError: if the features are not numpy arrays
        """
        cell_line_id = self.output.cell_line_ids[idx]
        drug_id = self.output.drug_ids[idx]
        response = self.output.response[idx]
        cell_line_features = None
        drug_features = None
        for cl_view in self.cell_line_views:
            feature_mat = self.cell_line_input.features[cell_line_id][cl_view]

            if cell_line_features is None:
                cell_line_features = feature_mat
            else:
                cell_line_features = np.concatenate((cell_line_features, feature_mat))
        for d_view in self.drug_views:
            if drug_features is None:
                drug_features = self.drug_input.features[drug_id][d_view]
            else:
                drug_features = np.concatenate((drug_features, self.drug_input.features[drug_id][d_view]))
        if not isinstance(cell_line_features, np.ndarray):
            raise TypeError(f"Cell line features for {cell_line_id} are not numpy array")
        if not isinstance(drug_features, np.ndarray):
            raise TypeError(f"Drug features for {drug_id} are not numpy array")
        data = np.concatenate((cell_line_features, drug_features))
        # cast to float32
        data = data.astype(np.float32)
        response = np.float32(response)
        #weight = self.weights[idx]   #Änderung
        return data, response #weight

    def __len__(self):
        """
        Overwrites the len method from the Dataset class.

        :returns: the length of the output
        """
        return len(self.output.response)


class FeedForwardNetwork(pl.LightningModule):
    """Feed forward neural network for regression tasks with basic architecture."""

    def __init__(self, hyperparameters: dict[str, int | float | list[int]], input_dim: int) -> None:
        """
        Initializes the feed forward network.

        The model uses a simple architecture with fully connected layers, batch normalization, and dropout. An MSE
        loss is used.

        :param hyperparameters: hyperparameters
        :param input_dim: input dimension, for SimpleNeuralNetwork it is the sum of the gene expression and
            fingerprint, for MultiOMICSNeuralNetwork it is the sum of all omics data and fingerprints
        :raises TypeError: if the hyperparameters are not of the correct type
        """
        super().__init__()
        self.save_hyperparameters()

        if not isinstance(hyperparameters["units_per_layer"], list):
            raise TypeError("units_per_layer must be a list of integers")
        if not isinstance(hyperparameters["dropout_prob"], float):
            raise TypeError("dropout_prob must be a float")

        n_units_per_layer: list[int] = hyperparameters["units_per_layer"]
        dropout_prob: float = hyperparameters["dropout_prob"]
        self.n_units_per_layer = n_units_per_layer
        self.dropout_prob = dropout_prob
        self.loss = nn.MSELoss()
        # self.checkpoint_callback is initialized in the fit method
        self.checkpoint_callback: pl.callbacks.ModelCheckpoint | None = None
        self.fully_connected_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.dropout_layer = None

        self.fully_connected_layers.append(nn.Linear(input_dim, self.n_units_per_layer[0]))
        self.batch_norm_layers.append(nn.BatchNorm1d(self.n_units_per_layer[0]))

        for i in range(1, len(self.n_units_per_layer)):
            self.fully_connected_layers.append(nn.Linear(self.n_units_per_layer[i - 1], self.n_units_per_layer[i]))
            self.batch_norm_layers.append(nn.BatchNorm1d(self.n_units_per_layer[i]))

        self.fully_connected_layers.append(nn.Linear(self.n_units_per_layer[-1], 1))
        if self.dropout_prob is not None:
            self.dropout_layer = nn.Dropout(p=self.dropout_prob)
        #---------------我加的-------------------
        # === PATCH: aleatoric 开关与 logvar 头（极小增量）
        self.aleatoric: bool = bool(hyperparameters.get("aleatoric", False))
        self.logvar_head: nn.Linear | None = None
        self.LOGVAR_MIN = float(hyperparameters.get("logvar_min", -5.0))
        self.LOGVAR_MAX = float(hyperparameters.get("logvar_max",  2.0))

        if self.aleatoric:
            # create log-variance head：Input dimension = number of hidden units in the last layer
            self.logvar_head = nn.Linear(self.n_units_per_layer[-1], 1)
            # Initialization: 
            # Give the bias a conservative negative value to avoid 
            # excessively large or small variance in the early stages.
            init_logvar = float(hyperparameters.get("logvar_init", -2.0))
            nn.init.constant_(self.logvar_head.bias, init_logvar)
            # Optional: Initialize weights with a small range to reduce 
            # fluctuations in the early stages of training
            # 可选：把权重小范围初始化，减小训练初期波动
            nn.init.zeros_(self.logvar_head.weight)


    def fit(
        self,
        output_train: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None,
        cell_line_views: list[str],
        drug_views: list[str],
        output_earlystopping: DrugResponseDataset | None = None,
        trainer_params: dict | None = None,
        batch_size=32,
        patience=5,
        num_workers: int = 2,
        model_checkpoint_dir: str = "checkpoints",
        #selected_drugs: dict[str, set[str]] | None = None,   # <<< NEU weights 
        #weight_factor: float = 1.0,            # <<< NEU weights
    ) -> None:
        """
        Fits the model.

        First, the data is loaded using a DataLoader. Then, the model is trained using the Lightning Trainer.
        :param output_train: Response values for training
        :param cell_line_input: Cell line features
        :param drug_input: Drug features
        :param cell_line_views: Cell line info needed for this model
        :param drug_views: Drug info needed for this model
        :param output_earlystopping: Response values for early stopping
        :param trainer_params: custom parameters for the trainer
        :param batch_size: batch size for the DataLoader, default is 32
        :param patience: patience for early stopping, default is 5
        :param num_workers: number of workers for the DataLoader, default is 2
        :param model_checkpoint_dir: directory to save the model checkpoints
        :raises ValueError: if drug_input is missing
        """
        if trainer_params is None:
            trainer_params = {
                "max_epochs": 100,
                "progress_bar_refresh_rate": 50,
            }
        if drug_input is None:
            raise ValueError(
                "Drug input (fingerprints) are required for SimpleNeuralNetwork and " "MultiOMICsNeuralNetwork."
            )

        train_dataset = RegressionDataset(
            output=output_train,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            cell_line_views=cell_line_views,
            drug_views=drug_views,
            #selected_drugs=selected_drugs,        # <<< NEU weights
            #weight_factor=weight_factor,  # <<< NEU weights
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=False, #True, ACHTUNG ÄNderung vom Original
            drop_last=True,  # to avoid batch norm errors, if last batch is smaller than batch_size, it is not processed
        )

        val_loader = None
        if output_earlystopping is not None:
            val_dataset = RegressionDataset(
                output=output_earlystopping,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
                cell_line_views=cell_line_views,
                drug_views=drug_views,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                persistent_workers=True,
            )

        # Train the model
        monitor = "train_loss" if (val_loader is None) else "val_loss"

        early_stop_callback = EarlyStopping(monitor=monitor, mode="min", patience=patience)

        unique_subfolder = os.path.join(model_checkpoint_dir, "run_" + secrets.token_hex(8))
        os.makedirs(unique_subfolder, exist_ok=True)

        # prevent conflicts
        name = "version-" + "".join([secrets.choice("0123456789abcdef") for _ in range(10)])
        self.checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=unique_subfolder,
            monitor=monitor,
            mode="min",
            save_top_k=1,
            filename=name,
        )

        progress_bar = TQDMProgressBar(refresh_rate=trainer_params["progress_bar_refresh_rate"])
        trainer_params_copy = trainer_params.copy()
        del trainer_params_copy["progress_bar_refresh_rate"]

        # Initialize the Lightning trainer
        trainer = pl.Trainer(
            callbacks=[
                early_stop_callback,
                self.checkpoint_callback,
                progress_bar,
            ],
            default_root_dir=model_checkpoint_dir,
            devices=1,
            **trainer_params_copy,
        )
        if val_loader is None:
            trainer.fit(self, train_loader)
        else:
            trainer.fit(self, train_loader, val_loader)

        # # load best model
        if self.checkpoint_callback.best_model_path is not None:
            checkpoint = torch.load(self.checkpoint_callback.best_model_path, weights_only=True)  # noqa: S614
            self.load_state_dict(checkpoint["state_dict"])
        else:
            print("checkpoint_callback: No best model found, using the last model.")

        # # 我加的：von mir 
        # ckpt_path = self.checkpoint_callback.best_model_path
        # if ckpt_path and os.path.isfile(ckpt_path):
        #     ckpt = torch.load(ckpt_path, map_location=self.device)
        #     self.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        # else:
        #     print("checkpoint_callback: No best model found, using the last model.")


    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the model.

        :param x: input data
        :returns: predicted response
        """
        for i in range(len(self.fully_connected_layers) - 2):
            x = self.fully_connected_layers[i](x)
            x = self.batch_norm_layers[i](x)
            if self.dropout_layer is not None:
                x = self.dropout_layer(x)
            x = torch.relu(x)

        x = torch.relu(self.fully_connected_layers[-2](x))
        x = self.fully_connected_layers[-1](x)

        return x.squeeze()

#-----------------我加的-------------------

    # def forward(self, x) -> torch.Tensor:
    #     """
    #     Forward pass of the model.
    #     :returns: predicted response (mu)
    #     """
    #     h = self._forward_hidden(x)
    #     x = self.fully_connected_layers[-1](h)
    #     return x.squeeze()


    # === PATCH: Gaussian NLL（for aleatoric noice)
    def _forward_loss_and_log(self, x, y, log_as: str, w: torch.Tensor | None = None):
        mu, logv = self._forward_mu_logvar(x)
        if self.aleatoric and (logv is not None):
            inv_var = torch.exp(-logv)
            loss = 0.5 * (logv + (y - mu)**2 * inv_var)
            # Normalisierung（Range: 1e-5~1e-4）
            loss = loss + 1e-5 * (logv**2) # muss man nicht
            loss = loss.mean()
        else:
            loss = self.loss(mu, y)
        self.log(log_as, loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # def _forward_loss_and_log(self, x, y, log_as: str, w: torch.Tensor | None = None):
        # Änderung wegen weights
        # mu, logv = self._forward_mu_logvar(x)
        
        # if self.aleatoric and (logv is not None):
        #     # --- Aleatoric (NLL) Pfad ---
        #     inv_var = torch.exp(-logv)
            
        #     # 1. Per-Sample-Loss berechnen (kein .mean() am Ende!)
        #     loss_per_sample = 0.5 * (logv + (y - mu)**2 * inv_var)
        #     loss_per_sample = loss_per_sample + 1e-5 * (logv**2) # Deine Normalisierung

        #     # 2. Gewichtet oder ungewichtet mitteln
        #     if w is not None:
        #         # Sicherstellen, dass w die gleiche Form wie der Loss hat
        #         w = w.view_as(loss_per_sample)
        #         # Gewichteten Mittelwert berechnen: (Summe(loss * w)) / (Summe(w))
        #         loss = (loss_per_sample * w).sum() / (w.sum() + 1e-8) # +1e-8 für numerische Stabilität
        #     else:
        #         # Standard-Mittelwert, falls keine Gewichte gegeben sind
        #         loss = loss_per_sample.mean()

        # else:
        #     # --- Standard-Loss Pfad (z.B. MSE) ---
            
        #     # WICHTIGE ANNAHME:
        #     # Damit dies funktioniert, muss self.loss im Konstruktor (__init__)
        #     # mit reduction='none' initialisiert worden sein!
        #     # z.B.: self.loss = torch.nn.MSELoss(reduction='none')
            
        #     # 1. Per-Sample-Loss berechnen (gibt einen Tensor zurück, keinen Skalar)
        #     loss_per_sample = self.loss(mu, y)
            
        #     # 2. Gewichtet oder ungewichtet mitteln
        #     if w is not None:
        #         w = w.view_as(loss_per_sample)
        #         loss = (loss_per_sample * w).sum() / (w.sum() + 1e-8)
        #     else:
        #         loss = loss_per_sample.mean()
                
        # self.log(log_as, loss, on_step=True, on_epoch=True, prog_bar=True)
        # return loss
        
#------------------------
    # original _forword_loss_and_log
    # def _forward_loss_and_log(self, x, y, log_as: str):
    #     """
    #     Forward pass, calculates the loss, and logs the loss.

    #     :param x: input data
    #     :param y: response
    #     :param log_as: either train_loss or val_loss
    #     :returns: loss
    #     """
    #     y_pred = self.forward(x)
    #     result = self.loss(y_pred, y)
    #     self.log(log_as, result, on_step=True, on_epoch=True, prog_bar=True)
    #     return result

    def training_step(self, batch):
        """
        Overwrites the training step from the LightningModule.

        Does a forward pass, calculates the loss and logs the loss.
        :param batch: batch of data
        :returns: loss
        """
        x, y = batch
        return self._forward_loss_and_log(x, y, "train_loss")

        # # Änderung wegen weights - weights muss noch übergeben werden 
        # if len(batch) == 2:
        #     x , y = batch
        #     w = None
        # else:
        #     x, y ,w = batch
        # return self._forward_loss_and_log(x, y, "train_loss", w)

    def validation_step(self, batch):
        """
        Overwrites the validation step from the LightningModule.

        Does a forward pass, calculates the loss and logs the loss.
        :param batch: batch of data
        :returns: loss
        """
        x, y = batch
        return self._forward_loss_and_log(x, y, "val_loss")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the response for the given input.

        :param x: input data
        :returns: predicted response
        """
        is_training = self.training
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(torch.from_numpy(x).float().to(self.device))
        self.train(is_training)
        return y_pred.cpu().detach().numpy()

    # Habe ich ergänzt
    @torch.no_grad()
    def predict_uncertainty(self, x_np: np.ndarray, T: int , keep_bn_eval: bool = True
                        ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """
        Returns (mean, sigma_epi, sigma_ale), all in the 'training-normalized space'.
        does not add σ to μ; it is up to the caller to decide whether to save/use it for sorting/plotting intervals.
        """
        # Preparation (activate MC Dropout)
        X = torch.from_numpy(x_np).float().to(self.device) #X_np -> features as numpy array 
        # close dropout
        self.eval()
        # falls module dropout wäre, wird es zu train zurück -> ermöglicht die loop
        for m in self.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
            elif keep_bn_eval and isinstance(m, torch.nn.BatchNorm1d): # falls module batchnorm layer wäre, wäre eval() an.
                m.eval()
        mus = [] #all prediction points
        logvs = [] if self.aleatoric else None
        for _ in range(T):
            mu, logv = self._forward_mu_logvar(X)
            mus.append(mu.detach().cpu().numpy())
            if self.aleatoric:
                logvs.append(logv.detach().cpu().numpy())
        mu_T = np.stack(mus, axis=0)            # [T, N]
        mean = mu_T.mean(axis=0)              # [N]
        epi_std = mu_T.std(axis=0, ddof=1)    # [N]
        sigma_ale = None
        if self.aleatoric:
            logv_bar = np.stack(logvs, axis=0).mean(axis=0)  # [N]
            sigma_ale = np.sqrt(np.exp(logv_bar)) 
            #print(f"Aleatoric wird aufgerufen, mit: logvbar:{logv_bar}, {sigma_ale}")
        return mean, epi_std, sigma_ale

#-----------------------------------------------------------------------
    # original configure_configure_optimizers
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Overwrites the configure_optimizers from the LightningModule.

        :returns: Adam optimizer
        """
        return torch.optim.Adam(self.parameters())

    # 改进：优化器要吃到 lr / weight_decay
    # Improvement: The optimizer needs to incorporate learning rate (lr) and weight decay.
    # def configure_optimizers(self):
    #     hp = self.hparams["hyperparameters"]
    #     lr = float(hp.get("lr", 1e-3))
    #     wd = float(hp.get("weight_decay", 0.0))
    #     return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

#------------------我加的----------------------
        # === 抽取隐向量:
        # Extract hidden vectors (features before the last layer)
    def _forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.fully_connected_layers) - 2):
            x = self.fully_connected_layers[i](x)
            x = self.batch_norm_layers[i](x)
            x = torch.relu(x)
            if self.dropout_layer is not None:
                x = self.dropout_layer(x)
        x = torch.relu(self.fully_connected_layers[-2](x))
        return x

    # === 提供一个统一取 (mu, logvar) 的函数，供预测/损失共用
    # Provide a unified function to get (mu, logvar), for shared use by prediction and loss.
    def _forward_mu_logvar(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self._forward_hidden(x)
        mu = self.fully_connected_layers[-1](h).squeeze()
        if self.aleatoric and (self.logvar_head is not None):
            logv = self.logvar_head(h).squeeze()
            logv = torch.clamp(logv, min=self.LOGVAR_MIN, max=self.LOGVAR_MAX)
            return mu, logv
        else:
            return mu, None