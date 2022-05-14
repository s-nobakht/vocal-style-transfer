def zero_mean_unit_var_norm(x, dim=None, keepdim=False):
    if dim is None:
        return (x - x.mean()) / x.std()
    return (x - x.mean(dim=dim, keepdim=keepdim)) / x.std(dim=dim, keepdim=keepdim)
