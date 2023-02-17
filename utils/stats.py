import numpy as np
from scipy import linalg

"""NOTE: Code is adapted from mseitzer's implementation of fretchet inception distance,
 https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py"""


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, (
        "Training and test mean vectors have different lengths "
        + str(mu1.shape)
        + " "
        + str(mu2.shape)
    )
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; adding %s to diagonal of cov estimates"
            % eps
        )
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calc_stats(traj):
    """Calculates the statistics used by FID
    Args:
        traj: Batch trajectories torch.tensor, shape: (N, T, F), dtype: torch.float32
    Returns:
        mu:     mean over the batch of trajectories at each time step
        sigma:  covariance over the batch of trajectories at each time step.
    """
    flat_traj = np.reshape(traj, (traj.shape[0], -1))
    mu = np.mean(flat_traj, axis=0)
    sigma = np.cov(flat_traj, rowvar=False)
    return mu, sigma


def calculate_fid(gt_traj, pred_traj):
    """Calculate FID between gt and pred trajectories
    Args:
        gt_traj: np.array, shape: (N, T, F), dtype: np.float32
        pred_traj: np.array, shape: (N, T, F), dtype: np.float32
    Returns:
        FID (scalar)
    """
    mu1, sigma1 = calc_stats(gt_traj)
    if len(pred_traj.shape) == 4 and pred_traj.shape[0] == 1:
        pred_traj = pred_traj.squeeze(0)
    mu2, sigma2 = calc_stats(pred_traj)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid