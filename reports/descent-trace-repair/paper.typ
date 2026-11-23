#set document(title: "Training on Local Divergence Between Computational Traces", author: "Marcos Damasceno", description: "Working paper on trace-guided weight updates, arithmetic repair and transfer in Gemma 4 E2B.", date: datetime(year: 2026, month: 11, day: 18))
#let web = sys.inputs.at("edition", default: "print") == "web"
#let ink = rgb("#192f37")
#let accent = rgb("#08665f")
#let grey = rgb("#496065")
#let rule = rgb("#c7d5d2")
#let sans = "DejaVu Sans"
#set page(paper: "us-letter", margin: (x: 0.88in, top: 0.8in, bottom: 0.77in), footer: context [
  #line(length: 100%, stroke: 0.5pt + rule)
  #v(2pt)
  #text(font: sans, size: 8pt, fill: grey)[Marcos Damasceno · Computational trace repair #h(1fr) #counter(page).display("1")]
])
#set text(font: "Libertinus Serif", size: 11.5pt, fill: ink, lang: "en")
#set par(leading: 0.64em, spacing: 0.8em, justify: false)
#show heading.where(level: 1): it => if web { html.elem("h2", it.body) } else { block(above: 0pt, below: 11pt, text(font: sans, size: 19pt, weight: "bold", fill: ink, it.body)) }
#show heading.where(level: 2): it => if web { html.elem("h3", it.body) } else { block(above: 11pt, below: 3pt, text(font: sans, size: 11pt, weight: "bold", fill: accent, it.body)) }
#let chart(file, alt, cap) = block(above: 9pt, below: 10pt)[
  #image("figures/" + file, width: 100%, alt: alt)
  #v(3pt)
  #text(size: 9.5pt, fill: grey)[#cap]
]
#let tab(cols, ..cells) = table(columns: cols, stroke: (x, y) => (bottom: 0.45pt + rule), inset: (x: 5pt, y: 4pt), align: (x,y) => if x==0 { left } else { center }, ..cells)
#let caption(body) = block(above: 4pt, below: 5pt, text(size: 9.5pt, fill: grey, body))
#let citekey(s) = text(size: 9pt, fill: grey, s)
#text(font: sans, fill: accent, size: 10pt, weight: "bold")[TORAD LABS · WORKING PAPER · 18 NOVEMBER 2026]
#v(13pt)
#let title-body = [Training on Local Divergence Between Computational Traces]
#if web { html.elem("h1", title-body) } else { text(font: sans, size: 27pt, weight: "bold", title-body) }
#v(8pt)
#text(size: 13pt)[Marcos Damasceno] #h(8pt) #text(size: 11pt, fill: grey)[Torad Labs]

== Abstract
Earlier weight edits repaired reasoning on problems outside their fit and sometimes damaged computations that had worked. We study how to make the repair procedure reproducible. We measure the change in direction through each residual merge during generation, align a failing execution to a fixed successful execution, and differentiate each local discrepancy beyond the variation seen in correct traces. The same rule supplies the gradient to shared weights throughout the model. In Gemma 4 E2B, it changed a downhill energy calculation while preserving the subsequent subtraction of friction. Across eight new formulations of four physical configurations, a frozen endpoint produced three correct numerical chains, versus one at baseline. All three came from the same prompt template. Two earlier checkpoints each produced two correct chains even though the second had 56% lower fitted loss. On a separately recorded weight update, total trace loss fell by 0.29% while the pre-decision computation moved farther into the observed error direction. The unrounded gradient already predicted this local increase. In a second prospective panel, eleven updates incorporating a direction shared by two successful edits preserved the fitted calculation and produced no new correct descent on eight fresh prompts.

== The physics behind the sign
Let $D=m g d$ be the positive energy released on a drop of depth $d$, $Q$ the energy dissipated by friction, and $H$ the height of the subsequent climb. In this setup the block has enough energy to climb if
$ K_"available" = K_0 + D - Q space >= space m g H. $
On the 96 J problem, both 55 J and the correct 115 J exceed the requirement. A grader that sees only “yes” rewards either calculation. We therefore audit the sign with its operand convention, the intermediate quantities and the verdict. We also measure the trace loss independently. The arithmetic checks whether a response obeys conservation of energy. Only the internal trace discrepancy supplies the training gradient.

Our working hypothesis is that correct uses of a capability share measurable internal operations across different solutions, and that editing the weights responsible for a recurring departure can repair the capability. A fitted repair tests the intervention on one trajectory. Transfer asks whether a second trajectory reaches a corresponding operation and benefits from the same installed weights.

#pagebreak()
= 1. What reaches the weights
Earlier stages of this program isolated recurring trace signals for two errors in these energy problems (adding an earlier drop to the climb height and subtracting gravitational work on descent) and showed unseen transfer: `FK` and `A2` repaired 15 of 20 climb fields and 9 of 31 full chains on 40 excluded prompts, and 60-trace DTW overlap `a2-path` reversed 29 of 30 fitted downhill signs and 7 of 21 excluded failures (§13). Because those one-shot solves traded collateral regressions, we test iterative computational backpropagation here. #citekey("[P]")

We used one fixed reference and trained on one freely generated failing response from the same failing prompt at every update. The reference was the originally successful trace with the smallest mean squared computational difference to the other successes after alignment. It computes $16+40-12=44$ J and a 20 J climb correctly, although it calls the energy released by the descent “potential energy gained.” The fitted problem requires $64+40-13=91$ J. The original model reports 11 J. Fourteen other originally successful traces calibrate acceptable variation. Their answers never enter as further targets for the weight derivative. #citekey("[M]")

== Measuring the effect of one branch
At each consumed position $j$, the model adds three branches to the residual stream in each of its 35 transformer blocks. These additions carry the attention output, the feed-forward output and a per-layer input branch. For each of these 105 additions, indexed by $m$, we measure the change in direction between its incoming and outgoing 1,536-dimensional vectors:
$ T_(j m) = h_"out" / norm(h_"out") - h_"in" / norm(h_"in"). $
This “turn” retains the size of the change in direction. It discards the absolute sizes of the incoming and outgoing states. Two distinct internal calculations can produce the same turn. We flatten all 105 turns at each position and align the two sequences by dynamic time warping with cost one minus cosine similarity. The path $P$ spans both complete histories, including the prompt. Its entries pair positions by computation. The matcher receives no words or token-equality labels.

== Assigning the discrepancy to the weights
Write $R_(i m)$ for the frozen reference turn, $delta_(i j m)=T_(j m)-R_(i m)$, and $d_(i j m)=norm(delta_(i j m))$. The allowance $tau_(i m)$ is the square root of twice the mean squared distance of the fourteen calibration traces, first averaged within each trace’s matches. A larger deviation contributes its own squared excess. With $N=560$ reference positions and $n_i$ matches for position $i$,
$ w_i = 1/(105 N n_i), quad L = sum_((i,j) in P) sum_m w_i [d_(i j m)-tau_(i m)]_+^2. $
Here $[x]_+=max(x,0)$. Outside the allowance the local trace derivative is $2w_i(1-tau_(i m)/d_(i j m))delta_(i j m)$. Inside the allowance it is zero. A small excess therefore contributes a smaller trace derivative than a large excess at the same weight. On the current history and alignment, the chain rule sends all these contributions into shared weights through $g=J_T^top nabla_T L$. The Jacobian includes both sides of each measured residual addition. The expanded learner makes 538 parameter tensors eligible, including the accessed embedding rows. Earlier phases restricted the eligible weights, as described on the next page.

#pagebreak()
= 2. What changed during training
The update uses the measured response along $s=-g$. Stack the weighted vector excesses into $r$, so that $L=norm(r)^2$, and let $q=J_r s$. The stationary step along this direction in the local linear approximation has length $alpha=-(r^top q)/(q^top q)$. The proposed displacement is $eta alpha s$, with the empirically calibrated multiplier $eta=0.26989$. The installer rounds to each tensor’s native precision and carries the difference into the next update. The model then generates freely from the edited weights and the full computation is paired afresh. Differentiation follows the current history through a surrogate for finite-precision rounding. A new greedy choice can take the model onto a different history. #citekey("[M, H]")

#chart("fitted-history.svg", "Fitted loss repeatedly rises from about 0.000064 to 0.00074 as 91 J changes to 11 J, then recovers.", [*Figure 1.* Each point is a fresh generation on the fitted question, updates 317–390. Circles and diamonds give the audited available energy and verdict. Loss is on a logarithmic scale. All observations in this recorded interval are shown.])

At step 355 the model gave 91 J and loss 0.00006393. One update later the model took 631 positions instead of 612, gave 11 J and loss 0.00073685. On the next step it again gave 91 J. The error recurred at 371 and recovered at 372. The error returned at 387 and cleared at 388. The last step in that segment, 390, gave 91 J and loss 0.00006406, 78.09% below the original 0.00029234. The best loss in the 42 updates after 348 occurred at 384 (0.00006366). The training run measured only the fitted question at 390. Weight changes also improved a fixed computation. Across six earlier consecutive updates from 230 to 236, the same history and pairing scored 2.21% lower. Across 34 unchanged-route updates in a later audit, the rounded-weight forecast had mean absolute per-merge error $5.94 times 10^(-10)$, versus $2.88 times 10^(-9)$ for predicting no change. Those forecasts describe the current route. #citekey("[H]")

Reading the actual parameters at 348 gives a misleading account of where the earlier gains occurred. The first attention-output matrix held 99.988% of squared weight displacement at 348, after a phase that deliberately concentrated edits there. By 230, when it held 3.03%, the learner had already achieved 94.60% of the net original-to-348 loss decrease. The absolute reduction at 348 was 78.04%. When we supplied the later of two correct histories at frozen weights (175 and 189), the history swap accounted for 99.66–99.80% of the 56.26% loss decrease between them, depending on intervention order. Both histories had already reached 91 J. #citekey("[H, W]")

#pagebreak()
= 3. Which intervention repairs the sign?
We took an early checkpoint that subtracted on the 64 J problem and a descendant that added correctly. Into the same early weights we installed either all later attention-output matrices or all later feed-forward down-projection matrices. Each family spanned 35 blocks. With free generation, both single-family transplants produced $64+40=104$ J, then $104-13=91$ J and YES. The early control gave 24 J, 11 J and NO. Attention alone reproduced the later response word for word. Feed-forward alone gave the same arithmetic along a different continuation. Either substitution was sufficient on this background. We reacquired all four conditions on the same device. Its early control reproduced the historical error using different wording. The endpoints span many updates and leave the first responsible update unidentified. #citekey("[W]")

Across checkpoints 175, 189 and 348 the fitted question and the correct reference gave the right arithmetic. At 348 the reference’s trace loss rose from 0.000000130 to 0.000093520, although both solutions gave 44 J and the same correct verdict. The second development question still gave $96-30-11=55$ J at all three. Its climb requirement and verdict improved. At frozen checkpoint 175 we tested whether the same weights could perform the missing arithmetic. On each question, we supplied either run’s heading and setup with the receiving question’s own quantities, stopping before the sign. The model chose the sign and completed the calculation. Both natural-context controls reproduced the original sign-score vector and subsequent response. #citekey("[C]")

#chart("context-margins.svg", "On both questions the sign score favors addition with the fitted framing and subtraction with the second framing.", [*Figure 2.* The native difference between the addition and subtraction token scores. The first and fourth rows reproduce each question’s own context. The middle rows exchange the setup. The two second-question verdicts were both YES, at 115 J and 55 J respectively.])

Changing only the first heading choice disrupted the fitted calculation. The reciprocal one-word change on the second problem elicited a signed potential-energy change of $-30$ J, then the model computed $96+(-30)=66$ J at the bottom. Under that convention the correct calculation is $96-(-30)=126$ J. An isolated plus sign can encode the wrong physics. A decomposition of the sign decision assigned about half of the *pre-cap readout difference* between contexts to the block-33 feed-forward addition on each question. This decomposition uses each execution’s own final normalization. Earlier blocks supply its inputs, and we did not intervene on block 33 alone. The supplied contexts served only this diagnostic. #citekey("[C]")

#pagebreak()
= 4. A recurring difference before the sign
At the row where the model chose the first sign, the fitted error and the correct addition both lay inside tolerance at all 105 merges. All eight audited histories had zero local loss at that row. A later numerical repetition of the subtraction did incur loss, which can propagate back into weights used earlier. We looked upstream on the model’s own generations. A November 13 analysis had concentrated pre-sign departures in blocks 14–23. Removing a single correct example reduced the number of cells meeting its two criteria there from 43 to one. That sensitivity motivated looking for recurrence in individually observed turns. We examined nine motivated reference positions across 105 merges, or 945 cells. One cell was the feed-forward output of zero-indexed block 20 at reference row 312, one row before the fitted sign decision. Its direction $u$ was the normalized difference between the fitted wrong turn (averaged over its matches) and the correct reference. We projected each aligned turn difference onto $u$. Of the original 30 development traces, 28 were eligible for this analysis. Two known wrong members were kept outside site selection. One of the 13 included wrong responses reached its length limit after choosing the sign. #citekey("[G]")

#chart("precursor-geometry.svg", "Radial distance mixes correct and wrong groups. All thirteen wrong projections exceed the largest of the fifteen correct projections in this development sample.", [*Figure 3.* Same reference, same computation, two measurements. A dot or diamond denotes a development trace at the selected site. Dashed lines mark the original radius (top) and maximum observed net-correct projection (bottom). The site was selected on these data.])

The radius was 0.50603. All 13 wrong traces were inside it, and two net-correct traces were outside. One positive was 0.71061 from the reference, farther than the fitted wrong trace at 0.39333. Any radius that admitted that positive would admit the fit. Along $u$, the largest positive projection was 0.38018. Every wrong projection lay between 0.38941 and 0.42060. This establishes a geometric property of the aligned development traces. Searching 945 cells and calibrating on the observed positive extreme make these counts unsuitable as an estimate of out-of-sample accuracy. #citekey("[G]")

At the same block and pre-sign position we also compared separate parts of the feed-forward computation in eight matched pairs (13 distinct questions). Both the gated product and the normalized turn ordered the wrong member farther along their respective fitted directions in 8/8 pairs. The gate before its nonlinearity and the up projection each did so in 5/8. Three pairs reused the same correct question. We registered these pairs by the sign-relative positions from the earlier study. The 945-cell survey used computational alignment. We reconstructed the internal activations in CPU float32. The largest relative discrepancy between a reconstructed post-normalization branch and its saved addition was 0.00000630 over seven examined blocks. No activation was substituted. #citekey("[G]")

#pagebreak()
= 5. Freezing the signal and testing an update
Before evaluating edited weights we fixed one change to the objective. At row 312, merge 61, the permitted turn differences became the intersection of the original ball and a half-space:
$ cal(A) = {delta : norm(delta) <= tau, quad u^top delta <= 0.38018046}. $
The local loss used the squared Euclidean distance to that intersection. Every other site kept its original loss and mass. We compared the original model, one ordinary update and one update with the extra directional constraint. Both updates began at identical weights with zero rounding remainders. Their installed models differed at 2,056 native values across 84 tensors. A prospectively reserved question came from a prior public screen of original errors. Its trace and edited outcomes played no role in choosing the site, direction or step. #citekey("[U]")

#tab((1.5fr,1fr,1fr,1fr),
  table.header([*Fresh trace loss*],[*Original*],[*Ordinary*],[*Directional*]),
  [Fitted failure],[0.000292457],[0.001622357],[0.000291610],
  [Correct reference],[0.000000130],[0.002785691],[0.000000158],
  [Reserved failure],[0.011736433],[0.011737470],[0.011740480])
#caption([*Table 1.* All nine free generations scored with the ordinary objective on the same device. This original fitted acquisition has its own measured baseline. Values are rounded here.])
On the fitted question the ordinary update raised loss 5.55-fold and yielded $-9$ J. The directional update reduced loss 0.290% and yielded the original 11 J. At the targeted cell its excess *increased*, from 0.013111 to 0.013493. That cell’s extra term initially contributed about 0.0010% of total fitted loss. We reconstructed the weight derivative on the original consumed history. The full step predicted an increase of $3.56 times 10^(-5)$ in the signed pre-decision projection. Using the exact stored weight changes gave $3.61 times 10^(-5)$. Native execution on the same history increased it by $3.82 times 10^(-4)$. The fitted target’s own loss term favored correction, while the other terms together moved the target farther toward error. The directional model reproduced all original greedy choices on the fit, reference and reserved questions (406, 388 and 511 choices). Both rounded-step forecasts predicted lower fitted loss on the original history. The ordinary update changed that history and raised fresh loss. It retained the reference’s final quantities while introducing intermediate errors. #citekey("[U]")

We removed the part of this step that increased the signed error to first order, and installed the adjusted weights. On the original history, a full forward pass in the graph used for backpropagation predicted a decrease of $7.43 times 10^(-5)$ in the error projection. The native engine measured an increase of $1.10 times 10^(-4)$ at precisely those weights. Both forwards measured a reduction in total loss. In free generation the model still subtracted downhill work and reported $-9$ J. The correct control retained its 44 J answer. We localized one discrepancy in the reconstruction of attention. At the audited event, supplying the first block’s recorded keys to the same attention calculation reduced the relative error in its predicted mixed-output change from 102% to 2.65%. We then measured only the keys at the original weights, added the change computed by the training graph, and rounded the result to the native cache precision. This forecast also gave 2.65% error at the audited event. Across the full prefix through that event, including batched prompt processing, the error remained 45%. We then applied this rule at all 15 layers that produce keys. The finite forecast for the downstream pre-decision projection changed from $-7.43 times 10^(-5)$ to $+1.28 times 10^(-4)$, against the native measurement of $+1.10 times 10^(-4)$. Absolute error fell by 90.7%. The derivative evaluated along the rounded weight displacement still gave the wrong sign. On a second archived displacement with the same input history, both finite calculations predicted deterioration. Using the original native keys reduced absolute error by 9.94% and still underestimated the native magnitude. The revised forecast ranked the adjusted update as the more harmful of the two at this site. Native execution ranked it as the less harmful. Both displacements share one fitted history. #citekey("[U]")

== Trace direction before the generated histories separated

On the fitted 91 J problem, we crossed each successful transplant with the histories generated by the early and repaired weights. At the historical descent readout, early weights already entered the correct range on repaired history. Both transplanted models remained outside on the original history. The first 73 decoded predictions provide a separate comparison. All five weight states still received identical inputs there. At each prediction we measured each edit’s change in the turn between normalized incoming and outgoing residuals at all 105 merges. Across the shared prefix the two successful displacements had cosine 0.736. Their cosines with the unsuccessful displacement were 0.299 and 0.201. The successful pair ranked first separately at every prediction, and the ordering survived centering all endpoints on either starting model. At the later descent stage the two successes had opposite signs of mean direct change in squared reference distance. These controls isolate an earlier candidate signal. The 73 predictions belong to one fitted question, and the successful edits share a training lineage. #citekey("[S]")

== Where the held comparison actually landed
The reserved block began with 36 J, gained 20 J and lost 9 J, leaving 47 J for a 60 J climb. All three models subtracted 20, reported 7 J after friction, added the 60 J climb requirement back in, and answered YES with 67 J. The model had already selected the wrong sign at position 259. Whole-trace alignment mapped reference row 312 to position 296, where the model was writing an equals sign. At the actual precursor (258), the projection was 0.27370. At the paired row it was 0.14120. Across *all 93* decoded positions before the choice, the original maximum was 0.29950, below the frozen 0.38018 limit. The error therefore exposed two distinct problems. Correspondence reached the calculation late, and moving the same direction earlier still supplied no request for correction. #citekey("[U]")

#chart("reserved-projection.svg", "The maximum of 93 pre-sign projections, the immediately preceding row and the later paired row all fall below the frozen threshold.", [*Figure 4.* Projection on the fixed direction in the original reserved response. The model chose subtraction at row 259. The first mark is the maximum over all preceding decoded rows.])

#pagebreak()
= 6. Locating an operation during generation

The first frozen direction missed the reordered problem. Could a second correct response supply a better comparison? We examined the trajectories as the model freely generated them. The companion visualization shows a two-dimensional slice through one weight matrix for each held response and the aligned transformations across 105 merges. The largest joint discrepancy occurred at aligned pair 270. The annotated sign was at pair 143. Choosing the largest peak would have selected a computation well after the decision. This measured slice leaves the geometry of the full parameter space undetermined. #citekey("[R]")

At the feed-forward branch of block 20, we compared the change in direction and the log of the ratio of outgoing to incoming norm. On the reordered problem a direction-only comparison flagged two of the 94 prediction events before the erroneous choice, the two immediately preceding it. The version that included log gain flagged 28 and missed both of those two. Counting the first prediction from the prompt adds one event to the 93 decoded positions in Fig. 4. A feature that distinguishes two recorded trajectories can dilute a signal when combined with another feature. #citekey("[R]")

== Aligning the reference before reading the difference

Two successful reference trajectories had been aligned 32 and 33 events late. We compared consecutive block-20 turns across eligible reference trajectories and selected the reference event with the smallest median discrepancy. This recovered the annotated reference events in all 37 recorded comparisons (607 assignments with reuse). The query supplied its own past and current execution. We excluded donors with the same incoming kinetic energy and downhill work as the query. This retrospective revision followed inspection of the two failures. #citekey("[R]")

From the outgoing block-20 state we estimated the expected subsequent computation separately for correct and failing examples. Within each group we found the convex mixture of donor contexts closest to the query and carried the same coefficients through the remaining 43 merges at that event. A positive difference between squared distances to the two predictions favored the failing group. At the verified pre-descent event, the rule flagged 13 of 13 wrong and zero of 15 valid descents in the development bank. The incoming state alone gave the same 28 decisions. On the reordered question the incoming score was −0.00343 and the downstream score was +0.07148. Across the entire generation every one of the 15 valid responses triggered an alert somewhere. An operator for training would have acted on those alerts too. #citekey("[R]")

#pagebreak()
= 7. Which part of the calculation changed?

At checkpoint 387 the model arrived at the bottom of the hill with 24 J. At 388 it arrived with 104 J. It lost precisely 13 J to friction in both generations. The entire 80 J difference in final energy arose in the descent. We can read each operation against the energy the model actually brought to it.

#chart("descent-friction.svg", "At checkpoint 387 gravity takes the model from 64 to 24 joules and friction takes it to 11. At 388 gravity takes it to 104 joules and friction takes it to 91.", [*Figure 5.* Two freely generated answers on the same fitted question. Downhill work changed sign between the adjacent checkpoints. Both rough-section calculations correctly removed the stated loss from their incoming energy. The right panel reports the local matcher at the annotated events in the original cohort.])

Let $B$ and $A$ denote the bottom and post-friction energies the model reports. For a positive released energy $D$ and dissipated energy $Q$, the two local errors in joules satisfy
$ e_D = B - (K_0 + D), quad e_Q = A - (B - Q), $
$ A - (K_0 + D - Q) = e_D + e_Q. $
The residuals $(e_D,e_Q)$ were $(−80,0)$ at 387 and $(0,0)$ at 388. This decomposition also handles a model that writes a signed change in potential energy and subtracts it correctly, provided the operands are interpreted with their physical signs. The arithmetic labels evaluate behavior. They never enter the fitted trace gradient. #citekey("[O]")

All 30 original responses subtracted friction correctly given the energy that reached that step, including all 15 with a wrong descent. The same held in the reordered response and four fitted checkpoint captures. These 35 recordings include repeated prompts. Two different weight substitutions had already produced the fitted repair independently. The changes in their turn fields had cosine similarity 0.736 over 73 events with identical consumed histories. The similarity reached 0.9991 at the later block-20 site, by which time the histories had diverged. Those later observations characterize the repaired trajectory. They leave the causal origin of the repair open. #citekey("[W, R, O]")

#pagebreak()
= 8. How often would the learner act in the right place?

Three annotated recordings provided examples of downhill work and friction. Two used subtraction for both. At each event in a query response we compared two consecutive block-20 turns against the reference recordings. We also evaluated a variant that aligned the entire prefix. Each rule had to identify the operation as downhill and find a positive discrepancy in the subsequent 43 merges before requesting correction. The query’s words and annotated event were available to the evaluator only. #citekey("[O]")

#tab((1.65fr,1fr,1fr,1fr), table.header([*Reference matching*],[*Wrong descent*],[*Valid descent*],[*Valid friction*]), [Two successive turns], [15 / 15], [3 / 15], [0 / 30], [Full prefix], [13 / 15], [0 / 15], [2 / 30])
#caption([*Table 2.* Correction requests at the prediction immediately before each physically verified decision in the original responses. A request in either of the last two columns is a false alert. Each response supplies a downhill and a friction observation.])

The two-turn matcher found the reordered error but missed the wrong descent at 387. Its alert at 318 preceded addition of a negligible-friction term. The wrong subtraction occurred at 337. Widening the donor labels to include both symbolic and numerical applications let the prefix matcher catch 387 at 336. The same rule issued 14 false alerts during the valid downhill computation in each of 175, 189 and 388, and ten in the reordered response’s valid friction interval.

A further test compared the query with the donor stage selected by its aligned computation. False alerts on the three valid fitted descents fell to 11, 11 and 10. At 336 in the failing response every one of the 16 eligible witnesses was already three to seven events past its own operator choice. The score changed from +0.02726 to −0.75182, suppressing the real warning. Matching the broad operation still allowed a comparison of opposite sides of a decision. #citekey("[O]")

== A competing explanation

In every recording the rough section followed the descent. A detector could learn the order or equation form. Among 18 matched question pairs differing only in friction loss, nine reversed the earlier downhill choice, with five repairs and four regressions as friction increased. These pairs reuse 22 prompts across nine fixed configurations. In the nine stable pairs, just four showed a larger change in the full turn field before friction than before descent. Only two pairs isolated drop height, and neither contained two valid descents. This sample cannot identify two independently editable capabilities. #citekey("[O]")

#pagebreak()
= 9. Can overlap predict the scope of a repair?

A correct friction calculation can appear in a globally wrong response. Once this is visible, a prediction becomes possible. An edit aimed at the recurrent descent departure might repair the descent and preserve friction. On another trajectory, an erroneous operation shared by both groups might persist because their difference provides no training signal there. These are predictions about named operations. The 387→388 comparison suggested this distinction after the outcomes were available.

For a differentiable shared trace feature $S$ on a held history, an installed displacement $Delta theta$ gives a local approximation
$ Delta S approx J_S Delta theta. $
Even if the local loss at $S$ is zero, a weight shared with a divergent operation can move $S$. Regeneration can also change which history reaches the operation. Similar correct and wrong traces identify a candidate for preservation or persistence. Their overlap alone sets no bound on the size of the change. The earlier many-trace experiment made precisely such a preservation forecast and measured substantial sideways movement outside the targeted region (§13).

== A prediction that can fail

Before installing an edit, freeze the reference bank, stage matcher, candidate update and unseen problem families. For each operation in the original responses, record its starting validity, the measured trace evidence and a prediction of repair, persistence, preservation or regression. Record unresolved cases too. A correct friction step inherited from an erroneous descent starts in the preservation group. A second incorrect operation that both reference and query perform incorrectly starts in the persistence or repair group, depending on the evidence. An unexplained failure of the targeted descent remains a failure on that task.

Regenerate each held prompt freely from the installed weights and audit each operation from its own inputs. Report how often the forecasts were correct separately for each outcome, how often the method abstained, which operations disappeared or became unreadable, and how often a rule that predicts no change would have been right on the same observations. Include initially failing friction cases and examples with different orderings. In a sample where every friction calculation already works, universal preservation is an easy prediction. Hold whole physical configurations together and reserve model and domain changes for independent tests. The companion protocol specifies this comparison. No prospective operation-by-operation forecast accuracy is reported here.

#pagebreak()
= 10. What transfers under one update rule?

We froze the original model and three checkpoints. Each freely answered three known controls and eight new formulations. The four new physical configurations crossed two initial speeds (5 and 7 m/s) with two downhill drops (1 and 2 m). Each appeared in both existing prompt templates. The model was never fit on these numerical combinations. An assessor read the generated calculations with the checkpoint identity hidden. A generation that explicitly corrected its own intermediate mistake received credit for its adopted calculation. A contradiction it still relied on counted against the full chain. Four generations reached the token limit. The table retains them in the denominator and credits only observable operations. #citekey("[V]")

#tab((1fr,1.2fr,1fr,1fr,1fr,1fr,1fr),table.header([*State*],[*Fitted loss*],[*Downhill effect*],[*Friction*],[*Climb*],[*Verdict*],[*Full chain*]),[Original],[—],[4],[8],[5],[5],[1],[175],[0.00009736],[2],[8],[6],[6],[2],[189],[0.00004259],[3],[8],[6],[6],[2],[390],[0.00006404],[3],[8],[6],[6],[3])
#caption([*Table 3.* Correct readings out of eight unseen formulations. Fitted losses use all 105 merges on the common evaluation device. Each edited checkpoint’s generation on the fitted prompt reproduced the history of its full-loss measurement. The downhill column asks whether the model added gravity’s contribution to the energy it carried. The full chain also requires consistent intermediate energies and an accurate verdict.])

On the 5 m/s, 1 m channel problem all four states gave the correct verdict, NO. The original and 175 subtracted 20 J of downhill work from 25 J. Checkpoint 189 added the 20 J in its accounting but still called 25 J the bottom kinetic energy. Checkpoint 390 explicitly computed $25+20=45$ J, then $45-13=32$ J against a 60 J climb. The same checkpoint got the descent wrong on the other wording of the same physical problem.

Compared with the original, 390 changed two unequivocally wrong chains into correct ones, resolved one previously capped and ambiguous chain, and lost the one chain the original had completed correctly. Four initially wrong chains remained wrong. The 5 m/s, 2 m reordered question illustrates the loss. The original corrected its own initial subtraction, carried 65 J to the rough section, and finished the calculation. Each edited endpoint returned to the subtraction and reached the generation limit. All 32 new-case generations subtracted the friction term correctly from the input they actually used. These operation readings audit the same weight intervention. No physical-operation labels entered its training loss.

#pagebreak()
= 11. Can the repairs accumulate?

On the 115 J question the original model made two separable mistakes. It subtracted the downhill work and put the destination 60 J above the foot of the final climb. At 175, 189 and 348 it correctly required 30 J for the climb. It still computed $96-30=66$ J at the bottom and correctly subtracted 11 J of friction from that incoming value. The downhill mistake explains the entire remaining 60 J error in its available energy. These same weights solved the fitted 91 J problem. Under the earlier weight-family transplants (§3), the evidence for a descent repair also came from the fitted problem. Each observation names both the operation and the problem on which it changed. #citekey("[C, W]")

Could the previously sufficient downhill edit be installed in 175 to finish the 115 J calculation? The successful donor was checkpoint 174. Its recipient in the original transplant had been a much earlier checkpoint. We read the BF16 weights at those two endpoints and at the proposed recipients, reconstructing each from its verified common seed and sparse overrides. This determines exactly how much of each proposed edit is already present.

#tab((1.0fr,1.2fr,1.2fr,1fr,1.2fr), table.header([*Recipient*],[*Weight family*],[*Donor values already present*],[*Changed by replacement*],[*Changed by adding the edit*]), [175],[Attention output],[784,934 / 785,007],[73],[783,298], [175],[Feed-forward down],[1,368,141 / 1,368,221],[80],[1,365,074], [189],[Attention output],[730,693 / 785,007],[54,889],[783,294], [189],[Feed-forward down],[1,269,193 / 1,368,221],[100,125],[1,365,066])
#caption([*Table 4.* The third column counts equality to the late donor among coefficients that changed between the donor endpoints. The fourth counts changes anywhere in the family on replacing it with the late donor. The fifth applies the original displacement to the recipient and rounds to native precision. These are parameter readings. No combined model was generated in this comparison.])

Checkpoint 175 already had the late donor value at 99.9929% of the coordinates the donor had changed across both families. Its ordinary generation still failed on the 115 J calculation. Adding the entire displacement there tests a second dose on a descendant. An independent recipient exists in the historical record. The earlier three-matrix FK edit repaired the climb requirement on two documented questions that still lost energy on the descent. A four-condition comparison can put the original model, downhill edit alone, FK alone, and their combination on the same questions. Holding the three FK matrices fixed tests whether their learned change can survive a new downhill edit elsewhere in the network. Behavior under that combination remains a prospective question. #citekey("[T, P]")

#pagebreak()
= 12. Training on the shared direction

The two fitted repairs moved the trace in similar directions during 73 predictions before their histories diverged (§5). At each event and merge where their displacements agreed in direction, the smaller projection set a local floor. We added the squared shortfall below each floor to the ordinary trace loss. The two interventions supplied 7,299 such locations. A previously failed intervention also lowered this directional loss by 5.66%. #citekey("[S, N]")

== Eleven updates on the fitted computation

Checkpoint 390 already gave $64+40-13=91$ J for a 20 J climb. We selected the eleven-update endpoint before generating edited test answers. Every installed weight state reproduced the fitted calculation. With unchanged source coverage, ordinary loss fell 0.693% and auxiliary loss rose 2.461%. At baseline only 33 of 7,402 registered contributions fell below the floor. Six rounded steps were predicted to increase the auxiliary loss, and five did. Four continuous directions were already uphill at infinitesimal length. #citekey("[N]")

== Independent physics on new configurations

Four unseen configurations each appeared in two orders. Each fell one joule above or below the energy needed to climb. Two readers independently scored all sixteen free generations with checkpoint assignments concealed. Neither found a newly correct descent or a complete local chain. Six of eight generations at each checkpoint reached the decoding cap.

One answer derived the correct 45 J bottom kinetic energy while stating that gravity did negative work. Another chose the initial height as its energy datum. Its 41 J of total energy against 40 J of final potential gives the same margin as 61 J available against 60 J required. At 390 it reached the correct verdict. At 401 the same prompt ended at the cap before the verdict. All sixteen answers numerically subtracted the correct friction amount from their own operand. The two blind audits document differences in signed formulas, sequence and reuse. #citekey("[N]")

== The held-out traces

All 105 merges were captured. On source locations visited by both checkpoints, prefix loss rose in six of eight pairs. The 59 J problem gave both extremes across its two orderings. Neither acquired the missing downhill operation. The three unchanged answer sequences also had higher matched loss. #citekey("[N]")

#chart("prefix-transfer.svg", "Eight paired percentage changes in held-out directional loss on common source locations.", [*Figure 6.* Change from 390 to 401, restricted to the same source locations within each paired prompt. Each row is one physical configuration. Negative values indicate a lower prefix loss.])

#pagebreak()
= 13. Scope and evidence
== How this differs from the earlier edits
Earlier stages established stored-weight repair and unseen transfer on both energy failures using one-shot matrix solves. On the climb error, the single-pair rank-one edit `D1` (one attention-output matrix) preserved two repairs, a protected case and seven prior correct chains while cutting root-mean-square action on four unrelated background requests by 85.5% versus the initial edit, with no gain on four previously examined physics prompts. Using contextual trace targets, three-matrix `FK` repaired 15 of 20 wrong or missing climb fields across 40 excluded prompts (2 of 31 full chains; 8 of 9 prior successes damaged), and 46-matrix `A2` repaired 25 of 36 fitted failures and 9 of 31 unseen chains (15 of 20 climb fields) on that previously examined 40-question set while damaging 6 of 9 prior successes. Separately, a 105-merge conditional climb overlap from 13 wrong-descent donors separated 643 of 673 cells on the withheld 115 J climb error and dropped 86.9% across 670 shared cells after checkpoint 175 repaired that climb from 60 to 30 J. On the descent sign, 18-matrix `a2-path` (solved from a 60-trace computational DTW overlap without token labels) repaired 29 of 30 fitted signs (17 complete) and 7 of 21 excluded Round-1 failures (11 of 21 adding descent), while 12 of 30 overlap successes and 4 of 7 Round-1 successes regressed. Outside its targeted interval, regenerated traces drifted sideways before aligned position 142 by 2.6–2.7 local gaps; inside it, wrong traces moved 77% along the intended turn direction while full distance from the original correct pattern grew 16%. Those one-shot solves proved transferable trace signals while exposing collateral drift across coupled layers, motivating iterative computational backpropagation and operation-level localization. Their objectives, data and weights differ from this learner and cannot be pooled into one success rate. #citekey("[P]")

== When to stop following a single reference

The final 42 updates lowered the fitted loss by 0.23% from 348 to 390. The fitted arithmetic was already correct at 348 and failed temporarily again at 387. Even on a held 612-position history, one query position was paired to two reference positions with disjoint tolerance regions at 37 merges. Under that fixed correspondence, the resulting loss floor was 0.00000288720, or at most a 99.0124% decrease from the original fitted loss. A different alignment can change the floor. Neither this bound nor an arbitrary percentage determines when the model has acquired the operation. A stopping decision should use fresh loss and check the operation, stability under further updates and preservation on other prompts as separate outcomes. An operational stopping rule can measure the same whole-trace discrepancy on a fixed development panel, compare it with repeat-measurement variation, and save a checkpoint when gains stall or previously accepted computations move persistently outside tolerance. A domain evaluator can inspect the selected checkpoint independently. The longest observed fitted dry spell that subsequently recovered was ten updates in the 348–390 segment. That segment supplies an eleven-update check interval and a 42-update budget for this implementation. How long a first repair persists on new physical configurations remains unmeasured. #citekey("[H, V]")

== Relation to earlier methods
Romero et al. trained a student network using intermediate representations from a teacher. Here the reference is one successful trajectory from the same original model, and the inputs and generated histories can differ. Meng et al. used causal interventions to locate and edit factual associations in GPT. Our weight-family interventions ask a related sufficiency question about a multi-step calculation. Neither comparison supplies a benchmark for computational cost. We have measured one model, a narrow physics domain and deterministic generations. An advantage over reinforcement learning, or a general change in the geometry of the model’s loss landscape, remains unmeasured.

#pagebreak()
= Evidence and references

The original model was Gemma 4 E2B in BF16 GGUF. The model digest, reference captures, native reader and per-run source versions appear in the repository’s _Final Experiment_ and in its run declarations. The file-level index in #raw("reports/descent-trace-repair/README.md") links each key below to the observations used here. The prospective scoring protocol and the data for Figure 1 are supplied alongside it.

#text(size: 10pt)[
The source map beside this paper gives the measurements, declarations, figure generator and checkpoint identities. The earlier frozen comparisons use revision #raw("3285da16"). The continuation supplement identifies every added observation by file digest.

*M* · _Final Experiment_, §§2–8. Model, immutable pair, calibration, computational correspondence, eligible parameters, response scaling and native installation.\
*H* · _One-pair correction_, arithmetic audit and stopping calibration. Fresh losses, frozen-route forecasts, fitted arithmetic, fixed-alignment bound.\
*W* · _Direct weight read_, _Crossed histories_, and _Descent-sign weight transplants_. Direct parameter changes and native counterfactual generation.\
*C* · _Reference preservation_, _Descent context_ and _Descent transfer_. Frozen checkpoints, reciprocal contexts, first-heading choice and sign readout.\
*G* · _Native tolerance recurrence_ and _Natural descent decisions_. Full development cell survey, component comparisons, positive and wrong measurements.\
*U* · _Isolated pre-error device continuation_ and _Natural descent decisions_. Two native installations, nine fresh answers and pre-decision held measurements.\
*P* · Damasceno, M. (2026). _Fixing a Reasoning Mistake by Editing the Weights_, _How the Trace-Derived Edit Works_ and _Editing the Weights from the Overlap of Many Answers_. See also the project repair retrospective and _Joint climb conditional overlap_. _Iterative Learning from Computational Traces_ states the broader proposal.\
*R* · _One-pair landscape and traces_, _Descent log gain and repair controls_ and _Descent landmark route comparison_. Registered events, geometry and whole-response timing.\
*O* · _Descent operand identity_. Local arithmetic at 35 recordings, full-response role assignment, episode and stage follow-ups, and one-factor prompt contrasts.\
*V* · _Frozen energy generalization_, the same-device fit measurement and the development-operation stopping guide. Native responses, blind readings and paired changes across four physical configurations.\
*S* · _Same-input operation comparison_ and _Crossed successful histories_. Native residuals at shared predictions, history interventions and independent reconstruction.\
*N* · _Prefix-direction training 390–401_ and _Prefix-direction generalization_. Native losses, installed forecasts, blinded operation audits and held-out trace comparisons.\
*T* · _Descent donor on descendants_ and _Accumulating repairs_. Source-relative and recipient-relative BF16 changes across 70 matrices and the prospective four-condition design.

Romero, A. et al. (2015). _FitNets: Hints for Thin Deep Nets._ #link("https://arxiv.org/abs/1412.6550")[arXiv:1412.6550].\
Meng, K., Bau, D., Andonian, A. & Belinkov, Y. (2022). _Locating and Editing Factual Associations in GPT._ NeurIPS. #link("https://arxiv.org/abs/2202.05262")[arXiv:2202.05262].
]
