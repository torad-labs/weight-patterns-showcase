# E2B decode observatory: 30,000 generated tokens

Production capture completed November 5, 2026, 21:41:35 CT. The 21-pass Nsight Compute capture and later-context diagnostic capture are complete. The later internal timeline failed admission because its recorded clock bound is inadequate; the workbench preserves that failure rather than displaying rejected timing.

## Workload and what actually ran

QGRE generated exactly **30,000 new tokens in one retained conversation over 187 turns**, on physical GPU 0, an RTX 5080. Earlier replies remained in subsequent prompts. Turns 1–186 stopped naturally; turn 187 stopped at the remaining 78-token budget. This is neither a single uninterrupted answer nor 187 independent prompts.

The initial prompt contained 86 tokens. The final request contained 39,494 input tokens, 39,442 reported as cached, and produced 78 new tokens. Its resulting context was 39,572 tokens. The workload requested a long catalogue of engineering observations and repeatedly asked for continuation. It is a sustained decode workload, not a semantic-quality benchmark.

The active conversation ran from **21:00:44 to 21:41:35 CT**, a span of **2,450.948 seconds**. This excludes server startup before the first request. Every completed request and response, emitted token IDs, cache counts, stop reason, host timing, server log and 500 ms GPU telemetry sample is retained under [conversation-normal](conversation-normal/).

| Measured scope | Result | Denominator / qualification |
|---|---:|---|
| New tokens | 30,000 | Includes first output tokens produced by prefill and emitted end-of-turn tokens |
| Requests | 187 | One sequential, retained conversation |
| Sum of request wall times | 2,442.492 s | Excludes gaps between requests and initial server startup |
| Persistent decode GPU time | 1,673.372 s | 29,813 decode-loop tokens; one output token per request belongs to prefill |
| Prefill GPU time | 724.745 s | 9,758 rows processed across initial and continuation prompts |
| Other time within requests | 44.375 s | Wall time minus reported prefill and decode GPU time; not a per-cause attribution |
| Mean decode-loop time | 56.129 ms/token | Weighted by the measured decode-loop token population |
| First reply mean decode time | 10.516 ms/token | 2,098 decode-loop tokens |
| Last reply mean decode time | 114.385 ms/token | 77 decode-loop tokens |

These are observations from this workload, not a controlled optimization A/B or a qualified peak-throughput claim. No inference policy, precision or kernel implementation was changed.

## Main finding: prioritize full-history attention distribution

The production reply averages fit context length closely: **2.725 ms additional decode time per 1,000 context tokens**, with R² **0.999525** over all 187 replies. The x-coordinate is the mean context position inside each persistent launch, not simply the turn number.

The seven early internal recordings independently show:

- Full-history attention tail growth: **390.007 ns per context token per full-attention layer**. Across seven such layers, that is **2.730 ms per 1,000 context tokens**.
- Sliding-window attention tails remain around **94 µs per layer after position 512**, with negligible growth over the captured positions.
- At position 2,183, full-attention layer 5 has **eight observed workers taking 860.6–863.3 µs**, while forty other observed workers each take less than 1 µs in that operation. There are 48 observations out of the 63 logical workers; absence of an observation is not an idle measurement.
- The actual captured [launch tables](first-turn-tapes/launch_tables.json) contain `attention_position_slices: "1"`.

Here, **tail means last recorded entry to last recorded exit**, averaged across layers of one kind. It is not the entire entry-to-exit envelope, a complete critical path, or a claim of recoverable time. Workers can enter at different times, and some entries perform almost no work.

The agreement between independent passes is strong evidence for choosing the next experiment. It does not establish a speedup or prove that one policy change alone resolves the cost. The next controlled experiment should vary the attention position-slice policy, retain the same precision and megakernel path, and measure the context-growth slope alongside short-context cost and the existing coherence gate. Sliding-window work and multi-row prefill need their own checks; an improvement in long-context batch-one decode does not qualify either automatically.

## What the other observations rule in or out

**GPU clocks did not fall with this slowdown.** The first request's sampled SM clock averaged 2,916 MHz; the final request's samples were all 2,932 MHz. Device memory used remained 4,792 MiB in these request samples. The final request was cooler than the first. This supports increasing work as the explanation for the growing latency, rather than a falling clock or exhausted device memory in those samples.

**Launch fragmentation is not the obvious target.** The early Nsight Systems capture contains two CUDA launches: prefill, followed by one persistent decode launch for the first reply. The many logical operations in the mechanical view are not separate CUDA launches. Its decode samples show approximately 75.0% SM activity, 6.5% issue activity, 1.1% tensor activity and 12.2% DRAM-read activity. These counters have distinct hardware definitions; none directly measures useful work. Low aggregate activity does not by itself identify the stalled instruction or dependency.

**Cached inputs do not make continuation prefill free.** Prefill accounts for about 29.7% of summed request time. In the final request, 52 fresh rows cost 5.750 s of GPU prefill, versus 8.808 s for 77 decode-loop tokens. The cache count proves prefix reuse; it does not eliminate processing of new prompt rows against the retained history.

**There is a host-capacity question worth measuring.** The early trace records a 2 GiB pinned-host allocation retained across decode and freed afterward. QGRE's log calls an allocation of the same size its conversation lot; the normal server reports it too. Allocation lifetime is recorded, but occupied bytes, first/last use, and reclaimable capacity are not. This is a lead for measuring the lot's utilization, not a finding that it is wasted memory.

**Token production is not semantic progress.** The later replies repeat substantial wording. Their complete token sequences differ, but the mean fraction of distinct eight-token sequences not seen in previous replies falls from 58.7% over the first ten replies to **7.66% over the final ten**. This explicit text-reuse metric is retained in the export. It does not diagnose numerical error, correctness or reasoning ability; a task-specific quality check is still needed.

## What the early Nsight Compute capture adds

The counter fixture uses the same initial 86-token prompt, two generated tokens and one selected persistent decode launch. It completed 21 application replay passes. QGRE's existing replay importer admitted **487 metric rows**, including device and launch attributes. The [native report](ncu-early.ncu-rep), [raw CSV](ncu-early.csv), [readable details](ncu-early-details.txt), and [SASS with source counters](ncu-early-sass.txt) are retained. These are early-context observations, not counter measurements of the entire 30k run.

| Counter observation | Recorded result | Implication to test |
|---|---:|---|
| Cycles with no eligible warp | 89.35% | Separate dependency readiness, load latency and intentional sleeping before changing scheduling |
| Active / eligible warps per scheduler | 2.00 / 0.12 | Resident work often cannot issue its next instruction |
| Sleeping stall contribution | 8.55 of 18.78 cycles per issued instruction, about 45.5% | Find which producer each sleeping consumer awaits; removing sleeps could merely replace waiting with polling |
| Long-scoreboard stall contribution | 3.12 cycles per issued instruction | Memory dependencies also matter; this is not the same as saturated bandwidth |
| Register count / rounded allocation | 242 / 248 per thread | Register allocation limits residency to one block per SM |
| Dynamic / static / driver shared memory | 51,200 / 44,032 / 1,024 bytes per block | Total allocation is 96,256 bytes; shared memory independently limits residency to one block |
| Theoretical / achieved occupancy | 16.67% / 16.71% | The kernel reaches its low resource-limited resident-warp budget |
| DRAM read / write bytes | 1,304,050,688 / 949,760 | Whole-launch traffic, not a byte map to individual weights or buffers |
| DRAM throughput | 23.70% of peak sustained elapsed | The early fixture is not saturating aggregate DRAM bandwidth |
| L2 sector hit rate | 14.04% | Aggregate cache behavior; it does not identify which object missed |
| Tensor / TMA instruction activity | 1.87% / 0.38% of their reported active-cycle peaks | Distinct metric denominators; neither measures useful-operation coverage |

The profiler's automatic estimated-speedup hints are not predicted or measured speedups. In particular, widening 63 blocks to 84 without assigning distinct work can add idle workers, and lowering register demand alone cannot create a second resident block while shared memory remains over budget. The next experiment must address the relevant work distribution and resource constraints together, through the policy plane.

The register finding also corrects an incomplete resource picture: **51.2 KB is the dynamic allocation, not total shared memory**. The run export now retains static and dynamic components separately; the counter report provides the additional driver allocation.

The SASS export contains 74,048 instruction locations, 15,648 with at least one sample. The [ranked instruction evidence](instruction-sites.json) retains the top 64 by not-issued warp-stall samples, with exact PCs and no invented logical-operation binding. The three leading locations are `WARPSYNC.COLLECTIVE` instructions: `0x10018b0a7f0` (91,729 not-issued samples), `0x10018b1e490` (77,177), and `0x10018b06220` (44,015). These are sampled instruction locations, not durations, root-cause proof or removable synchronization. Their producer dependencies and operation identity are the next missing join.

## Action order from the inspection ontology

These are proposed experiments, not implemented optimizations. They use the workbench's [architecture-area ontology](../../../../.dev/decode-machine/EVIDENCE-ONTOLOGY.md).

| Priority and architecture area | Evidence and hypothesis | Smallest next experiment | What would refute it |
|---|---|---|---|
| 0. Capture integrity | The later tape uses 2.73 GHz as a cycle upper bound, although the saved device report permits 3.135 GHz | Make the capture writer record a proven upper bound with its device provenance, or use an unambiguous clock representation; preserve the ordering checks and recapture the affected slice | A nominal or sampled frequency is still treated as a guaranteed maximum, or the old receipt is inflated merely to pass admission |
| 1. Coordination / work distribution | Production growth and full-history attention tail growth agree; one position slice leaves the long operation concentrated in eight observed workers | Build a known-revision baseline, then vary position slicing as a policy bundle with distinct owned work. Measure the context slope, early cost, and dependent completion; run the existing coherence gate | The slope does not fall, dependent completion does not advance, or overhead merely moves to another group |
| 2. Warp execution / local resources | Low eligibility, sleep and scoreboard samples, and two independent one-block residency limits | Bind the captured hot PCs to logical operations and readiness events. Test useful independent work and resource lifetime changes together where the data supports them | Occupancy or stall counters improve but the same production workload does not finish earlier, or spills / other regions regress |
| 3. Continuation prefill | 724.745 s of GPU prefill across the conversation, including 5.750 s for the final 52 fresh rows | Profile the real cached-prefix M-row continuation path separately, retaining its large history and row count | A short standalone prefill improves while actual continuation request time is unchanged |
| 4. Memory capacity and reuse | 2 GiB pinned lot survives the first decode; allocated size alone says nothing about occupied capacity | Add occupied-capacity and first/last-use events for this existing buffer, then assess allocation timing or reuse | Most capacity is needed, or smaller/later allocation adds transfer or request latency |
| 5. Output utility | Late replies reuse most eight-token phrases from earlier replies | Add a task-specific progress criterion alongside throughput and retain the actual text / token IDs | A faster configuration reduces useful task progress or required correctness, even if raw token rate rises |

Before any optimization leg, declare its quantitative expected band from the actual candidate and measure both arms with the same engine and instrument. This observational run does not provide a defensible speedup percentage to promise in advance.

## Captures and identity

### Later-context result and the capture defect it exposed

The saved turn-64 prompt contains **22,279 tokens**. A separate cold prefill processed that complete prompt, then generated two tokens. The recorded IDs **818 and 1626** match the first two IDs in normal turn 64. This is only a sampled correspondence; it does not establish identical cache state, every intermediate value, or a full matching trajectory.

The [later Nsight Systems trace](later-context/trace.nsys-rep) records a **69.645 ms CUDA decode kernel**; QGRE's host bracket around the instrumented step is **73.882 ms**. Cold prefill took about **811.4 s**, and is not compared with the normal server's cached continuation prefill. The diagnostic kernel uses **255 registers/thread**, versus 242 in the early production trace, another reason not to treat diagnostic timing as production performance.

During the later decode, 70 GPU samples report mean SM activity **74.1%**, issue activity **5.1%**, tensor activity **1.0%** and DRAM-read activity **2.4%**. They suggest that aggregate DRAM bandwidth alone does not explain this diagnostic's cost. They do not establish the exact waiting dependency, and the pass differs from the early production trace in context and instrumentation.

The existing range reader admitted the later stored-value dump: **527 stage/layer rows covering 28,992 active blocks**, retained in [range-22279.json](range-22279.json) and the relational export. Its timing binding is explicitly absent.

The phase reader rejected the internal clock tape: tap 0, layer 34, block 16 lies **202,001,458 cycles** from its block origin, beyond the receipt-derived **201,698,155-cycle** bound. The receipt records `sm_clock_khz=2730000`; the saved [device report](device.xml) lists a maximum SM clock of **3,135 MHz**. Current source explains the mismatch: `Device::sm_clock_khz_text` reads `CU_DEVICE_ATTRIBUTE_CLOCK_RATE`, and the phase reader multiplies it by the host bracket as if it were a guaranteed ceiling. Normal-run telemetry also observes clocks above 2,730 MHz.

The original tape and metadata remain unchanged. [capture-failures.json](capture-failures.json) records the failure; the viewer displays the unavailable position instead of inventing its animation. No later full-attention tail or worker-distribution claim is made from this rejected tape. The safe next repair is in capture provenance, followed by a new admitted capture, not removal of the ordering check.

NVIDIA's current [CUDA driver attribute documentation](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TYPES.html) describes `CU_DEVICE_ATTRIBUTE_CLOCK_RATE` as the **typical** clock frequency. That supports the source-level diagnosis: it is unsuitable as a guaranteed upper bound for boosted execution. The [NVML maximum-clock API](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html) is a different query and also documents small possible differences from current P0 clocks; a repaired capture must define and justify its bound rather than substitute an unexplained multiplier.

- [run-evidence.json](run-evidence.json): relational projection of sources, turns, launches, counters, internal captures, CUDA memory events, text reuse and evidence-to-experiment observations.
- [nsys-attempt-1.nsys-rep](nsys-attempt-1.nsys-rep), with [SQLite export](nsys-attempt-1.sqlite): separate first-reply production trace, not the normal conversation's timestamps.
- [first-turn-tapes](first-turn-tapes/): instrumented positions 86, 511, 512, 1,023, 1,024, 1,085 and 2,183. All seven sampled input/output token pairs match the corresponding positions in the normal first reply. This is a sampled correspondence, not a full-trajectory comparison.
- [checkpoint.sha256](checkpoint.sha256), [device.xml](device.xml), saved binaries under [bin](bin/), and the [capture plan](plan.md).

The checkpoint is `models/gemma4/gemma4-e2b/tq4-e2m1-nvme`. The emitted policy SHA is `09663b7f7fe5695c5bad0390cc3e82d64334d2792263ad2ec5eadc76fab65d2b`. Observed geometry is 63 blocks, 256 threads per block, controller/loader/consumer roles 1/1/6 warps, 51,200 bytes of dynamic shared memory and ring depth 4. Static and driver shared allocations are listed above. The early trace reports 242 registers per thread. Device identity is `GPU-b6d3869b-4794-4ff4-0398-47df517a8c82`.

Saved executable hashes:

- `qgre-infer`: `7bf8f6a1151b41887c6a9da6ec45a6db16b7a0ea91d9e27c8105c4be12dd1524`
- `qgre-serve`: `19a6f23ebeb143ac902e9f21c5336d82911431acd92338ba8a06b9c0d86287f3`

The exact source revision of these executables is not established. The replay's source-anatomy snapshot is labeled separately and must not be treated as that proof. A subsequent optimization A/B should build both arms from a known source revision. These diagnostic observations have not been promoted to a certified performance-ledger entry: the ledger requires a source-head and coherence binding that this saved binary does not establish. Their artifact identity is the run ID `gemma4-e2b-30k-2026-11-05`, plus executable, checkpoint, policy and tape hashes.

Nsight Systems is 2026.1.3; Nsight Compute is 2026.2.1. The Systems trace warns that the installed driver is newer than its supported tracing libraries. Its exported GPC-clock samples were invalid negative values and are excluded; nvidia-smi clocks remain a separate source. Diagnostic, profiler-replay and normal-production timings retain separate pass identities.

## Remaining evidence gaps

The capture does not provide generation-specific producer-ready events; per-application-buffer occupied capacity and first/last use; matched useful/issued element and requested/transferred byte populations per operation; individual production-token timestamps; or an exact PC-to-logical-operation map. Stored-code histograms do not establish clipping, so saturation remains unknown rather than zero. Aggregate activity and allocation events cannot fill those gaps by inference.

The workbench is useful for selecting a specific experiment: it connects the growing production cost to full-history readers, the few workers carrying that work, and the captured position-slice policy. It also prevents attractive but unsupported explanations such as launch fragmentation, falling clocks, or treating all cache reuse as free. The longer capture exposed a defect in the observability system itself. No speedup has yet been demonstrated, and the later internal animation remains unavailable pending capture repair.

## Workbench validation and presentation status

The workbench changes are committed in `5e6723207`, `842e30891` and `39d2e883b`. `bun run test` completed with 65 passing tests and exit 0; `bun run build` completed with exit 0, retaining the existing large-chunk warning. The final imported data reconciles 30,000 tokens, 187 turns, 7 sources and **1,417 relationships with no unresolved endpoint**. These validate the workbench, not the inference engine or a performance improvement.

Browser checks covered the real 30,000-token dataset, reply selection, capture selection with its matching stored-value evidence, and the new scheduler/resource comparison. Captures use 3840×2050, 1440×900 and 390×844 page areas. A fresh Sonnet visual judge scored the current screenshots 6.4, 7.2 and 7.7 respectively; its [report](design-review-final.md) is retained. It flags control proportions, typography hierarchy, explanatory text and missing-data presentation. This is functional diagnostic software with unfinished visual polish, not a completed design sign-off.
