import torch


def _observation_mask(preds, labels, observed_mask):
    if preds.shape != labels.shape:
        raise ValueError(
            f"prediction shape {preds.shape} does not match label shape {labels.shape}"
        )
    if observed_mask.shape != labels.shape:
        raise ValueError(
            f"observation mask shape {observed_mask.shape} does not match "
            f"label shape {labels.shape}"
        )
    mask = observed_mask.to(device=labels.device, dtype=torch.bool)
    if not torch.any(mask):
        raise RuntimeError("metric has no observed target values")
    if not torch.isfinite(labels[mask]).all():
        raise RuntimeError("observed labels contain non-finite values")
    if not torch.isfinite(preds[mask]).all():
        raise RuntimeError("predictions contain non-finite values at observed targets")
    return mask


def masked_mse(preds, labels, observed_mask):
    mask = _observation_mask(preds, labels, observed_mask)
    return ((preds - labels) ** 2)[mask].mean()


def masked_rmse(preds, labels, observed_mask):
    return torch.sqrt(
        masked_mse(
            preds=preds,
            labels=labels,
            observed_mask=observed_mask,
        )
    )


def masked_mae(preds, labels, observed_mask):
    mask = _observation_mask(preds, labels, observed_mask)
    return torch.abs(preds - labels)[mask].mean()


def masked_mape(preds, labels, observed_mask):
    mask = _observation_mask(preds, labels, observed_mask)
    percentage_error = torch.abs(preds - labels) / labels.abs().clamp_min(1e-6)
    return percentage_error[mask].mean()


def compute_all_metrics(preds, labels, observed_mask):
    mae = masked_mae(preds, labels, observed_mask).item()
    mape = masked_mape(preds, labels, observed_mask).item()
    rmse = masked_rmse(preds, labels, observed_mask).item()
    return mae, mape, rmse
