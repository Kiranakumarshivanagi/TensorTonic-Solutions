import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """

    p = np.array(p, dtype=float)
    y = np.array(y, dtype=float)

    # Intersection
    intersection = np.sum(p * y)

    # Dice coefficient
    dice = (2 * intersection + eps) / (np.sum(p) + np.sum(y) + eps)

    # Dice loss
    loss = 1 - dice

    return float(loss)