# Frozen endpoint in the anchored graph and native engine

**The signed reversal occurs between the finite anchored graph and the native engine.** The anchored graph's finite forward moved the target in the same direction as its rounded-weight JVP, with a larger magnitude. The native engine moved in the opposite direction on the same bytes, input history and computational correspondence. [Result](result.json) · [Native predecessor](../directional-intervention-projected-native/README.md)

| Held fitted readout | Signed target movement | Full capped loss |
|---|---:|---:|
| Original native and anchored origin | 0 | 0.000292459960 |
| Exact rounded-displacement JVP | $-4.91365422\times10^{-6}$ | 0.000291745221 |
| Anchored finite endpoint forward | $-7.43178444\times10^{-5}$ | 0.000291666883 |
| Recorded native finite endpoint | $+1.10394653\times10^{-4}$ | 0.000291695782 |

The JVP loss column squares the linearized residual with coefficient one for the complete native displacement. The recomputed signed JVP agreed with the preceding gradient contraction to $2.85\times10^{-10}$. The finite anchored forward kept the original observed and reconstructed anchors. At its origin, all incoming and outgoing residuals reproduced the original capture exactly. The replacement copied only the 31,552 changed native values into private tensors and recovered the recorded displacement norm squared, $4.24109242\times10^{-9}$. All reads used the same GPU, 578 historical input events, 584 registered pairs, and 105 merges. Implementation · Original residual map · Anchored endpoint residual map

## Where the discrepancies first appear

- **Block 0 attention, merge 0.** The finite anchored and native incoming states were bitwise equal across the target's entire causal prefix. Their attention-merge outputs already differed by 0.00478 in raw vector norm at the matched event and by up to 0.59165 across the prefix. The difference along the target's normalized turn coordinate at the matched event was only $+4.70\times10^{-8}$.
- **Block 9 feed-forward, merge 28.** This was the earliest upstream transform where the anchored–native gap on that same coordinate reached the magnitude of the gap at the target. The measured gap was $-2.33456\times10^{-4}$ at block 9 versus $+1.84712\times10^{-4}$ at the target. Incoming states already differed at block 9. The finding localizes when the gap reached that scale and leaves its causal route unresolved.
- **Block 20 feed-forward, historical target.** The native-minus-anchored difference in the output unit coordinate was $+1.19285\times10^{-4}$. The corresponding input difference was $-6.54278\times10^{-5}$. The local turn subtracts that input term, so both discrepancies contributed to the positive signed gap.

The comparison found where the anchored and native responses diverge at residual boundaries. It leaves unresolved which internal attention or feed-forward operations, rounding behavior, or cache reconstruction account for those differences. The run read the existing native capture and performed the fixed finite method without additional installs or generations. [Per-merge data](result.json) · [MLflow run](http://127.0.0.1:5000/#/experiments/6/runs/00da96a67d2749e59678bc9eab7f53c5)
