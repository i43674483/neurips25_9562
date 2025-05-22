import torch.nn as nn
from .utils_metric import img2mse


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()

    def forward(self, model_output, target_data, log_scalars):
        """
        Compute the training loss
        """
        predicted_rgb = model_output.get("rgb")  # torch.Size([256, 1])
        predicted_mask = model_output.get("mask", None)
        if predicted_mask is not None:
            predicted_mask = predicted_mask.float()

        ground_truth_rgb = target_data["rgb"]  # torch.Size([256, 1])

        computed_loss = img2mse(predicted_rgb, ground_truth_rgb, predicted_mask)

        return computed_loss, log_scalars
