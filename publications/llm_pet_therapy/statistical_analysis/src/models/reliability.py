import math,random

def icc_absolute(pairs):
    """Two-way absolute-agreement ICC(A,1) and ICC(A,2), k=2.

    A1=(MSR-MSE)/(MSR+(k-1)*MSE+k*(MSC-MSE)/n)
    A2=(MSR-MSE)/(MSR+(MSC-MSE)/n). Negative estimates are retained.
    Both constant calls are explicitly nonestimable, including unequal constants.
    """
    n, k = len(pairs), 2
    empty = {'ICC_A1': None, 'ICC_A2': None, 'n_clips': n}
    if n < 2:
        return {**empty, 'status': 'not_estimable_fewer_than_two_clips'}
    if all(len({row[j] for row in pairs}) == 1 for j in range(k)):
        return {**empty, 'status': 'not_estimable_both_calls_constant'}
    row_means = [(a + b) / k for a, b in pairs]
    column_means = [sum(row[j] for row in pairs) / n for j in range(k)]
    grand = sum(row_means) / n
    msr = k * sum((value - grand) ** 2 for value in row_means) / (n - 1)
    msc = n * sum((value - grand) ** 2 for value in column_means) / (k - 1)
    mse = sum((row[j] - row_means[i] - column_means[j] + grand) ** 2
              for i, row in enumerate(pairs) for j in range(k)) / ((n - 1) * (k - 1))
    denominators = (msr + (k - 1) * mse + k * (msc - mse) / n,
                    msr + (msc - mse) / n)
    estimates = [(msr - mse) / d if abs(d) > 1e-12 else None for d in denominators]
    return {'ICC_A1': estimates[0], 'ICC_A2': estimates[1], 'n_clips': n,
            'MSR': msr, 'MSC': msc, 'MSE': mse,
            'status': 'estimated' if all(v is not None for v in estimates)
                      else 'not_estimable_zero_denominator'}

def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

def participant_bootstrap(cluster_pairs, replicates, seed):
    """Resample participants, keeping every available clip from each draw together."""
    if replicates != 20000:
        raise ValueError('the prescribed ICC bootstrap uses exactly 20000 draws')
    ids = sorted(cluster_pairs)
    samples = {'ICC_A1': [], 'ICC_A2': []}
    generator = random.Random(seed)
    for _ in range(replicates):
        pairs = [pair for _ in ids for pair in cluster_pairs[generator.choice(ids)]]
        result = icc_absolute(pairs)
        for key in samples:
            if result[key] is not None and math.isfinite(result[key]):
                samples[key].append(result[key])
    return {'method': 'participant_cluster_percentile_linear_interpolation',
            'seed': seed, 'planned_replicates': replicates, 'participant_n': len(ids),
            **{key: {'valid_replicates': len(values),
                     'ci95': [percentile(values, .025), percentile(values, .975)] if values else None}
               for key, values in samples.items()}}
