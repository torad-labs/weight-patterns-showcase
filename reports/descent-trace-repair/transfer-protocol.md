# Predicting which operations an edit will change

This is a prospective test proposed in section 9 of the paper. The 387→388 pair motivated it and belongs to development. No independent cohort has yet been scored with this protocol.

## Before installing the update

Group related formulations and energy configurations before choosing a held evaluation set. Specify the source weights, proposed edit, source code, reference captures, stage matcher, tolerances, decoding configuration and planned checkpoint in a dated run declaration. Include original failures, original successes and contexts where the same written operation has a different physical role. To test whether operand identity transfers beyond this data, include changes of operation order and cases where the original friction calculation itself fails. Reserve model and domain changes for independently reported partitions.

From the original free generations, register each intended operation, its local inputs and output, the arithmetic convention and starting validity. For each query, derive correspondence and a whole-response map of local departures from its observed computation. Record ambiguous matching, multiple disjoint departures, and alerts elsewhere. Text can identify the physical operation for evaluation. Information about the annotated query stage or the desired words cannot enter the candidate detector.

Before applying the proposed update, assign each operation a forecast from the trace evidence. A recurrent divergence aligned to the same stage may support repair. A computation shared by correct and wrong solutions may support preservation when locally correct, or persistence when locally wrong. Unknown alignment, incompatible stage or unresolved sensitivity can warrant abstention. Record the actual signal, supporting record and competing explanation, including shared parameter effects. Define the rule for making each call in the declaration. The protocol prescribes no numeric threshold. Freeze both the predictions and their evidence independently of the edited outcomes.

## After installing the update

Regenerate the same prompts freely with the agreed decoding procedure. Score operations blind to the forecast using the input the generated solution supplies to each operation. A later operation can be locally correct on a wrong upstream input. Separately record global answer validity, collateral alerts across the full decode and fresh trace loss. Keep the original evaluation inventory even if an edited answer omits a step. Mark an absent operation or an unreadable computation explicitly. Reserve unmentioned behavior for exploratory analysis.

## Files and scoring

Both JSON files identify the same study, original model and specific planned edit. Identify the edit by immutable weights, or by an immutable recipe and exact source weights when the displacement is computed during the experiment. Store the digest of the completed forecast file in the prospective run declaration before training or installing the edit. The scorer verifies byte identity. Chronology comes from the separately recorded declaration. The following illustrates file structure and contains no experimental result.

```json
{"study":"example","source_model":"original-model-id","edit":"declared-edit-id","operations":[
  {"case":"held-A","operation":"descent-1","family":"family-A","before":"wrong","prediction":"repair","evidence":"reference to pre-edit trace comparison"},
  {"case":"held-A","operation":"friction-1","family":"family-A","before":"right","prediction":"preserve","evidence":"reference to shared computation and sensitivity assessment"}
]}
```

The outcome file uses the same three identifiers and one row for every declared operation. Each row supplies `case`, `operation`, and `after`. Use `right`, `wrong`, `absent`, or `unscorable` for the observed local result. Use `right`, `wrong`, or `unscorable` for the starting validity. A forecast is one of `repair`, `persist`, `preserve`, `regress`, or `abstain`. Repeated operations within a response need separate stable identifiers. Record the agreed matching convention in the declaration.

```sh
python3 score_forecasts.py forecast.json outcomes.json --forecast-sha256 "$registered_forecast_digest"
```

The output reports counts rather than suppressing small denominators. Forecast coverage is `forecast_issued / declared`. Accuracy and always-predict-unchanged accuracy use the same `scored` denominator, restricted to issued forecasts with observable before and after validity. The forecast-by-outcome table includes abstentions and unmeasurable transitions. For each prediction type, report how often it was correct and how often true outcomes of that type were missed. Report missing events, unreadable operations, coverage, and family counts alongside accuracy. Treat multiple operations in one response and multiple prompts in one family as related observations.
