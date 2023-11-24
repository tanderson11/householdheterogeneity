if __name__ == '__main__':
    import src.utilities as utilities
    import src.traits as traits
    import numpy as np
    import pandas as pd

    s80_axis = np.linspace(0.01, 0.78, 78)
    trait_rows = []
    for s80 in s80_axis:
        sus_variance = utilities.lognormal_s80_solve(s80)
        assert(sus_variance.success is True)
        sus_variance = sus_variance.x[0]
        sus_dist = traits.LognormalTrait.from_natural_mean_variance(1., sus_variance)
        trait_rows.append([
            (float(f"{s80:.3f}")),
            sus_dist.mu,
            sus_dist.sigma,
        ])
        print(trait_rows[-1])

    trait_frame = pd.DataFrame(trait_rows, columns=('s80','mu','sigma')).set_index(['s80'])
    trait_frame.to_csv('./new_s80_lookup.csv')