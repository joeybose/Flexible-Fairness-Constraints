def compute_rank(enrgs, target, mask_observed=None):
    enrg = enrgs[target]
    if mask_observed is not None:
        mask_observed[target] = 0
        enrgs = enrgs + 100*mask_observed

    return (enrgs < enrg).sum() + 1


def create_or_append(d, k, v, v2np=None):
    if v2np is None:
        if k in d:
            d[k].append(v)
        else:
            d[k] = [v]
    else:
        if k in d:
            d[k].append(v2np(v))
        else:
            d[k] = [v2np(v)]
