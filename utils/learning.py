""" Learning tools for the VRNN """
import numpy as np
import torch
import torch.distributions.normal as Normal

def frange_cycle_linear(n_iter, beta_min, beta_max, cycle, R):
    L = np.ones(n_iter) * beta_max
    period = n_iter / cycle
    step = (beta_max - beta_min) / (period * R)  # linear schedule

    for c in range(cycle):
        v, i = beta_min, 0
        while v <= beta_max and (int(i + c * period) < n_iter):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L

def gmm_loss(batch, mus, sigmas, logpi, reduce=True):
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return -torch.mean(log_prob)
    return -log_prob