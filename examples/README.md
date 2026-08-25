# Official HELENA examples

The v0.1.0 public example set contains only:

- the [CPU quickstart](../quickstart/helena_cpu_quickstart.ipynb);
- the [GPU quickstart](../quickstart/helena_gpu_quickstart.ipynb); and
- the [TXS 0506+056 2017 canonical example](txs0506_2017/txs0506_2017_canonical.ipynb).

CPU and GPU use the same public API and scientific semantics. The TXS notebook
is the only public real-source workflow. Legacy, deep-private, diagnostic, and
profiling notebooks remain in the preserved developer archive.

BH-enabled sections require the official validated Bethe--Heitler cache. The
one-time workflow is `helena cache fetch` followed by the normal example. Use
`helena cache status` to verify it, or select a complete cache copied to
offline/HPC storage with `cache_dir=` or `HELENA_CACHE_DIR`. Normal examples
never download or rebuild the cache. The release-candidate CLI is implemented,
but its official asset URL/archive checksum binding is not yet published.
