# Change a repeated-key lookup preference through a weight edit

The goal is to deliberately change a reproducible model behavior and observe the intervention's propagation. It is not to fix QGRE. The user selected llama.cpp for subsequent experiments. The original Q4_K_M GGUF stays unchanged; any edit is published as a separate model under models/.

First measure default choice with a repeated key, alongside explicit first/last instructions and unique-key controls. Eight discovery families each have two layouts and three instruction modes (48 prompts). The prior twenty unique-key questions are preservation controls. Eight separate confirmation families provide 48 prompts withheld from inspection and candidate selection.

Choose the preference direction after seeing the default behavior. Inspect actual native weights and captured internal activity before specifying an edit. A candidate must name its intended mechanism, tensors and represented quantization change. No global output-token bias or claim that a successful behavioral edit proves a geometric hypothesis.

Use the installed local fork f089c9cb4bc285a11e510e1c4336b95aca449b08, model google_gemma-4-E2B-it-Q4_K_M.gguf (SHA256 923c4c86177d2ee173a7f5b4fa3d0ac65f5962ab15e6d6a5bc250aec4fd7bf7e), greedy generation, the prior explicit user/model framing, no penalties, and no prompt-cache reuse. Keep raw prompts, token IDs, outputs, captures, and edited-byte provenance. Report failures and preservation regressions separately. Withheld questions are evaluated after selecting a candidate, not used to choose its direction or strength.
