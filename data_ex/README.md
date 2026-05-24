Example repo-local data layout.

- `svs/`: small example WSI inputs for documentation-driven runs
- `tiles/`: example tiling outputs produced by `wsi_read.py`
- `proteome/`: example proteome TSV and derived vector artifacts

Recommended usage:

1. Keep lightweight example assets under `data_ex/`.
2. Keep real datasets outside the repo and optionally symlink them as `./data`.
3. For real runs, pass explicit CLI paths instead of relying on defaults.
