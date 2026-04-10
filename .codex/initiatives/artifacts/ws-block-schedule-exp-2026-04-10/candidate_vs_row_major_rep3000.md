# Candidate Schedule vs Row-Major

- candidate: `band_n_20_row_major`
- rounds: `16`
- rep per timing call: `3000`
- manual warmup calls before each timing: `5`
- order: alternated every round

## Summary

- mean row_major: `0.34270173 ms`
- mean band_n_20_row_major: `0.34060572 ms`
- mean delta (`row_major - candidate`): `0.00209601 ms`
- mean relative edge for candidate: `0.6116%`
- median delta: `0.00216452 ms`
- delta stddev: `0.00043346 ms`
- bootstrap 95% CI for mean delta: `[0.00188117, 0.00228858] ms`
- sign test (two-sided) p-value: `0.000031`
- candidate wins / losses / ties: `16 / 0 / 0`
- CI-excludes-zero: `yes`

## Interpretation

The repeated long-run comparison supports a statistically clear difference between `row_major` and `band_n_20_row_major` under this setup.

Bootstrap settings: `20000` resamples, seed `0`.

See the sibling CSV for raw per-round timings.
