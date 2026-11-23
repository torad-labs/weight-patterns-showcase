# Gemma 4 E2B mechanical replay

Purpose: let Marcos follow the actual recorded work of QGRE's Gemma 4 E2B decode through tangible mechanisms, then inspect how the same work uses the megakernel's parallel worker blocks. Keep the approved slate, brass, amber and teal cel-shaded world.

## Evidence and boundaries

The input is `runs/legs/internal-decode-replay/lifecycle-ready/bundle.json`. It contains independent captures at token positions 1067 and 1015 from `runs/legs/gemma4-e2b/long-context-capture-7263027c5-gpu1-2/`. They are not consecutive decoded tokens. Read the original replay normalizer and topology admission; reuse them when preparing the mechanical view. Check each tape and launch-table digest against the bundle before exporting.

Captured topology: 35 layers, repeating four sliding and one full attention; eight query heads and one K/V head; 256/512 head width; window capacities 512/131072. Layers 16–35 reuse K/V from layer 14 (sliding) or 15 (full). Launch tables carry 63 blocks, output split-K 2 and down split-K 6. FFN K-block counts are 192 for layers 1–15 and 384 for layers 16–35, or 6144/12288 channels at 32 channels per block.

Source anatomy is a separately labeled snapshot, revision `6880f93f75f0627180f8022ba28dcb04ed0f4b60`. Its config describes hidden width 1536, per-layer input width 256, TQ4-native FP4, and controller/loader/consumer warp counts 1/1/6. The bundle explicitly does not certify that this snapshot produced the captures. Do not substitute the shared checkout's changing policy for it.

Each capture has 1435 recorded step records, including 210 retired sentinels. Keep every record inspectable; retired markers are not active machines. Preserve parent/child overlap and individual participating block observations. Whole replay time spans the earliest participating entry through the latest exit; durations of nested intervals are never added. Source-only mechanisms and parts that share a measured span do not receive invented independent timings.

Not recorded: prompt text, token ID/text, selected output, tensor values, K/V contents, weight values, ring occupancy, physical SM placement, separately timed intake and final readout. Use neutral carriers and illustrative weight balls without fabricated numerical values. Instrumented intervals are not production latency benchmarks. Attention pass stamps describe the final tile, not the whole attention history. Waits are observed intervals, not proven producer-caused stalls.

## World and interactions

- A seven-by-five layer installation: four short-window readers followed by one full reader in each row. Connecting pipes preserve layer order. Number buttons and a layer selector open one layer without changing the clock.
- Eight attention rollers share a paired K/V drum. Shared layers use a visible return pipe to their earlier owner. The full reader exposes two measured channel passes.
- Two feed-forward presses fan out into gate/up products; a shaped gate combines them and a down press compresses the result. Late layers have a wider expansion bed.
- A double residual bypass carries hi/lo, and a separate PLE cartridge joins the layer tail. Norm jaws, paired RoPE rotors, partial-sum cups and readiness latches explain each operation. Clicking a mechanism shows its purpose, its recorded children and its evidence boundary.
- Fixed amber weight balls remain in their drawer. A staging carousel and multiplication press explain read/compute topology. No arbitrary ball diameter claims a measured weight, and no animation claims measured memory occupancy.
- A second view shows 63 reusable worker benches, not 63 physical SMs. Work intervals move pistons; wait intervals hold latches. Selection of a block reveals its observed intervals at the current time. Layer progress never triggers a separate fabricated kernel launch.
- Total viewing seconds scales one captured clock uniformly. Scrubbing reconstructs every mechanism and worker. Layer and operation navigation provide explicit seek actions; camera and selection remain independent. Reduced-motion mode starts paused.
- The old France story stays available at `teaching.html`, explicitly scripted. The recorded machine is the default entry. Unknown input/output remain unrecorded endpoints rather than a Paris simulation.

## Completion

Load both captures locally without a GPU run; verify topology and clock mapping with focused functional tests; run the production build; inspect the default world, layer and worker views at 3840×2050, then laptop and phone. Preserve the existing teaching demonstration. Complete the Impeccable finish review and documentation. No inference engine files are changed.
