import torch
import numpy as np
from src.base.engine import BaseEngine
from tqdm import tqdm
from src.utils.metrics import masked_mape, masked_rmse, compute_all_metrics

class IGSTGNN_Engine(BaseEngine):
    """
    Incident-aware IGSTGNN engine
    """
    def __init__(self, cl_step, warm_step, horizon, incident=False, time=None, module_name=None, **args):
        super(IGSTGNN_Engine, self).__init__(**args)
        self._cl_step = cl_step
        self._warm_step = warm_step
        self._horizon = horizon
        self._cl_len = 0
        self._incident = incident

    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader['train_loader'].shuffle()

        # Get iterator and total number of batches
        iterator = self._dataloader['train_loader'].get_iterator()
        total_batches = self._dataloader['train_loader'].num_batch
        
        # Create progress bar
        progress_bar = tqdm(iterator, total=total_batches, desc="Training",
                            unit="batch", leave=False, position=0, 
                            dynamic_ncols=True, colour="green")
        try:
            for batch_idx, batch in enumerate(progress_bar):
                self._optimizer.zero_grad()
                
                if self._incident and isinstance(batch, dict):
                    X = batch['x_data']
                    label = batch['y_data']
                    incident_data = {
                        'incident': batch['incident_features'],
                        'position': batch['incident_position'],
                        'distances': batch['incident_distances'],
                        'durations': batch['durations']
                    }
                    
                    X, label = self._to_device(self._to_tensor([X, label]))
                    for key in incident_data:
                        incident_data[key] = self._to_device(self._to_tensor(incident_data[key]))
                    
                    sensor_data = None
                    if 'sensor_data' in batch:
                        sensor_data = {}
                        for key, value in batch['sensor_data'].items():
                            sensor_data[key] = self._to_device(value)
                    
                    pred = self.model(X, label, incident_data=incident_data, sensor_data=sensor_data)
                else:
                    if isinstance(batch, tuple) and len(batch) == 2:
                        X, label = batch
                    else:
                        X = batch[0]
                        label = batch[1]
                    X, label = self._to_device(self._to_tensor([X, label]))
                    pred = self.model(X, label)

                pred, label = self._inverse_transform([pred, label])

                mask_value = torch.tensor(0)
                if label.min() < 1:
                    mask_value = label.min()
                if self._iter_cnt == 0:
                    print('check mask value', mask_value)

                self._iter_cnt += 1
                if self._iter_cnt < self._warm_step:
                    self._cl_len = self._horizon
                elif self._iter_cnt == self._warm_step:
                    self._cl_len = 1
                else:
                    if (self._iter_cnt - self._warm_step) % self._cl_step == 0 and self._cl_len < self._horizon:
                        self._cl_len += 1

                pred = pred[:, :self._cl_len, :, :]
                label = label[:, :self._cl_len, :, :]

                loss = self._loss_fn(pred, label, mask_value)
                mape = masked_mape(pred, label, mask_value).item()
                rmse = masked_rmse(pred, label, mask_value).item()

                loss.backward()
                if self._clip_grad_value != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
                self._optimizer.step()

                train_loss.append(loss.item())
                train_mape.append(mape)
                train_rmse.append(rmse)

                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "mape": f"{mape:.4f}",
                    "rmse": f"{rmse:.4f}"
                })

        finally:
            progress_bar.close()

        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)

    def _model_predict(self, batch):
        if self._incident and isinstance(batch, dict):
            X = batch['x_data']
            label = batch['y_data']
            incident_data = {
                'incident': batch['incident_features'],
                'position': batch['incident_position'],
                'distances': batch['incident_distances'],
                'durations': batch['durations'],
            }
            X, label = self._to_device(self._to_tensor([X, label]))
            for key in incident_data:
                incident_data[key] = self._to_device(self._to_tensor(incident_data[key]))
            sensor_data = None
            if 'sensor_data' in batch:
                sensor_data = {k: self._to_device(v) for k, v in batch['sensor_data'].items()}
            pred = self.model(X, label, incident_data=incident_data, sensor_data=sensor_data)
        else:
            if isinstance(batch, tuple) and len(batch) == 2:
                X, label = batch
            else:
                X = batch[0]
                label = batch[1]
            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self.model(X, label)
        return pred, label

    def evaluate(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in self._dataloader[mode + '_loader'].get_iterator():
                pred, label = self._model_predict(batch)
                pred, label = self._inverse_transform([pred, label])
                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse
        elif mode == 'test':
            test_mae, test_mape, test_rmse = [], [], []
            for i in range(self._horizon):
                res = compute_all_metrics(preds[:, i, :], labels[:, i, :], mask_value)
                self._logger.info(f'Horizon {i+1:d}, Test MAE: {res[0]:.4f}, Test RMSE: {res[2]:.4f}, Test MAPE: {res[1]:.4f}')
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])
            self._logger.info(f'Average Test MAE: {np.mean(test_mae):.4f}, Test RMSE: {np.mean(test_rmse):.4f}, Test MAPE: {np.mean(test_mape):.4f}')
            np.savez(self._save_path + 'test_predictions.npz',
                     preds=preds.numpy(), labels=labels.numpy())
