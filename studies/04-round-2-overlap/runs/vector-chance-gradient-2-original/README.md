# Direct computational update with initialized CUDA statistics

The sole controller `weight-patterns-vector-gradient-2.service` launched at
04:02:56 AM CT November16, declaration SHA256
`8b485ad8bb5c81f2c239ccf9567f03fea588526c01d005fc0ed3873554124fa8`.

The update mathematics is unchanged: one full computational-loss backward,
one measured directional scale, complete native BF16 installation, then a
fresh computational path. The first attempt reproduced the exact original
baseline but failed before differentiation at an uninitialized CUDA peak reset.
Explicit CUDA initialization fixes that order; an actual cold-device smoke
and the independent reviewer's separate source-bound addendum pass. All earlier records remain.

The first update was completely installed at 04:12:42 AM CT and a fresh native
path was measured at 04:14:05. Its model SHA256 starts `e675a74686b4930c`;
`first-update.json` binds the complete records and all 94 unchanged source files.

| Computational measure | Original | After first native update |
| --- | ---: | ---: |
| Loss | 0.00029234223350039747 | 0.0002908526975024328 |
| Outside-tolerance cells | 4,386 | 5,753 |
| Complete pair/merge cells | 61,320 | 61,320 |

Loss fell 0.51%, while more cells crossed outside the fixed tolerance. This is
mixed first-update evidence, not closure or capability/transfer proof. The
controller continues from the reached model and fresh path with the same
original reference/tolerance, all 105 merges, and no text/token evaluation.

The calculation took 11.33 s backward, 4.42 s directional response and 8.14 s
rounding; peak allocated memory was 10,824,096,256 bytes on one GPU. Native
copying, verification and state preservation dominated the first transition's
elapsed time. The linear prediction remains separate from actual native loss.

The existing MLflow server at port 5000 publishes this run in
`computational-training`, parent `55a627154e3f4cdf888fa2617112fd5a`.

