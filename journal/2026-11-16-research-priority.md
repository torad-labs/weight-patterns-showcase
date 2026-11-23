# Research has first priority — November 16, 2026

Marcos directly instructed this session that nothing should stop work on the research and that it takes precedence over everything. This supersedes the earlier host-IO instruction to keep the experiment frozen. Competing archive and maintenance work yields resources to the research; it does not freeze the experiment.

The same hypothesis, papers, computational comparison and declared correction method continue. The first recovery service had already been killed while frozen (its exit record reports KILL); its lost inverse memory cannot be thawed. A new attempt uses the reviewed recovery entrypoint and the retained original native observation, preserving both earlier runs and restarting only the lost inverse calculation.

The implementation peer owns the sole experiment controller. Weight-research owns subordinate storage work and peer review; the independent reviewer checks the new attempt bindings with the goal and all four papers. No duplicate controller or Fable project work is introduced.

## Relay withdrawn and termination source established

Simon subsequently withdrew the stop-all relay: he said he had no authority to order stops and no record that Marcos superseded the direct research-priority instruction. The research process was left running. The previously relayed 20-percent IO-pressure stop rule was also withdrawn, without a replacement threshold.

The host journal identifies the old solver's killer: at November 15 23:55:12 CT, earlyoom PID 3521850 sent SIGTERM to research Python PID 3381054 (17,295 MiB RSS, oom_score 1419), which exited at 23:55:13. It then repeatedly targeted the launcher UV PID 3381001; the service recorded KILL at 23:56:08. This is a host memory-pressure interruption, not a result of the computational experiment.

At 23:58:36 the kernel reported global OOM and killed xdg-desktop-portal PID 4488 with 90,587,552 KiB anonymous RSS (about 86.4 GiB), with oom_score_adj -950. That later reading establishes a large host memory consumer; it does not measure the portal's earlier usage or fully explain the earlier disk-pressure episode. The frozen service's unexecuted exit child also blocked user systemd; thawing that exact cgroup restored the manager and wrote the retained exit record at 00:08:21.

Evidence commands: `journalctl --since '2026-11-15 23:50:00' --until '2026-11-16 00:00:00' --grep='3381001|3381054' --no-pager`; `journalctl -k --since '2026-11-15 23:50:00' --until '2026-11-16 00:00:00' --grep='global_oom|Out of memory: Killed process 4488' --no-pager`. These were bounded read-only checks while the new research inverse continued. No global host policy was changed.
