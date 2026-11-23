# Instruction-meaning contrast, original weights only

The recorded failing Lom=blue case already ranks the blue value first in five of layer 29's eight Q/K affinity lists, while outputting 2. Affinity is not a probability or proof of correct retrieval. A narrow competing explanation is that the wording "value" helps elicit a numerical or explanatory response after relevant information is available.

Change only the two occurrences of "value" to "color" in six existing development questions: three failures (01,03,10), and one successful red, green, and blue control (00,02,08). Use the unchanged original checkpoint and saved inference executable, GPU1, greedy, cap8, final prompt position and all35 named KV views. No weight changes, no new held-out questions.

Prediction: at least two of the three failures become blue, and all three controls retain their color. A smaller effect refutes this proposed wording explanation; unchanged errors argue against it. The captured prompt token count must equal the expected count before using its presumed answer-boundary tap; otherwise recapture at the native observed count. Compare changes in the existing layer readings after observing outcomes. Do not call a prompt repair a successful weight edit.
