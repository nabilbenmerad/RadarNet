import torch
import numpy as np

class WeightedSpatialMSELoss(torch.nn.Module):
    def __init__(self):
        super(WeightedSpatialMSELoss, self).__init__()
        self.loss = torch.nn.MSELoss(reduce=False, size_average=False)

    def forward(self, input, target, weights=1):
        return self.loss(input, target).mean(3).mean(2).mean(1) * weights

# Compute root mean squared error between targets
def rmse(predictions, targets):
    return np.sqrt(((predictions[0] - targets[0]) ** 2).mean())

