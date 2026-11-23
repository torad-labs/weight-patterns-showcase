# Local response to the archived directional update

The update moved the fitted pre-error computation farther outside tolerance in both forecasts and native execution. The total capped loss fell by 0.290%. The continuous derivative already had the adverse local sign. Applying the archived rounding preserved that sign, and native execution made the observed projection increase about 10.6 times the rounded forecast. The same fresh fitted answer still subtracted the descent. [Full result](result.json)

| Fitted measurement | Projection onto the historical error direction | Capped cell loss | Sum of all capped cells |
|---|---:|---:|---:|
| Original native | 0.3932914975 | 2.92345846 × 10⁻⁹ | 0.000292459960 |
| Continuous forecast | 0.3933270478 | 2.93933373 × 10⁻⁹ | 0.000291581798 |
| Forecast of exact rounded displacement | 0.3933275653 | 2.93956514 × 10⁻⁹ | 0.000291745383 |
| Native treatment, fixed history | 0.3936732616 | 3.09618621 × 10⁻⁹ | 0.000291613082 |
| Native treatment, fresh correspondence | 0.3936732616 | 3.09618621 × 10⁻⁹ | 0.000291613082 |

The allowance was 0.3801804580. The fitted excess rose by 0.271% in the continuous forecast, 0.275% in the rounded forecast, and 2.912% in native computation. The continuous first-order change of the historical cell loss was **+1.58538 × 10⁻¹¹** while the same update predicted **−1.01515 × 10⁻⁶** across the whole capped objective. In the differentiated graph, the update that descends the full objective therefore ascends this term. The selected row and merge come from the archived control. The program measures every paired event at every merge and supplies no runtime training selector. Fitted measures · Per-cell arrays

## What was held fixed

The original and treatment made independent native free acquisitions in the earlier experiment. The fitted and correct pairs each have identical consumed histories, full native callback indexes, engine bindings and GPUs. For each pair the original computational correspondence was frozen and applied to both recorded vectors. The fresh treatment correspondence was then read independently from its archived measurement. Both paths retained exactly the original generation, and each fresh correspondence matched the fixed one. The standard arm is present only as a recorded fresh comparison because it changed both histories.

The comparison restored the four exact native streams from the project archive, recomputed the original fitted full-graph derivative, and applied its sealed step length through the same native-anchored reconstruction as the earlier proposal. The rounded forecast used the original and treatment bytes from the archived XOR patch, converting the physical values into the graph's full parameter layout. The correct-path forecasts propagated those same updates through the correct response's original graph. No new weight installation, data selection or generation was used. [Declaration](declaration.json) · Measurement source

## Movement on the successful comparison

The correct free response retained its 44 J calculation. Its historical capped cell moved in the raw turn coordinates and stayed inside tolerance. Across all 58,800 correct-path cells, 57,456 started inside tolerance and 207 of those crossed it. The largest raw turn movement among initially tolerated cells was 0.07113, versus maxima of 0.02898 and 0.02697 in the continuous and rounded forecasts. The raw rounded movement had cosine 0.397 with the full measured movement and 0.378 among initially tolerated cells. Total native correct-path loss went from 1.30027 × 10⁻⁷ to 1.58379 × 10⁻⁷. The finite forecasts were 1.71594 × 10⁻⁷ and 1.72044 × 10⁻⁷. Both forecast increases include the second-order term from squaring the linearized residual. Correct-path measures · Per-cell arrays

On the fitted path, 262 of 55,558 initially tolerated cells also crossed their recorded tolerance despite lower total loss. The fitted continuous and rounded total forecasts reproduce the archived proposal. The continuous forecast establishes that the joint derivative already points away from the local target in this archived update. The loss and reference for the proposed successor and its performance on an independent route remain open. Fitted measures · Local operations log · [MLflow run](http://127.0.0.1:5000/#/experiments/6/runs/fb22294a36f5400c87a8eddc239ff2bc)
