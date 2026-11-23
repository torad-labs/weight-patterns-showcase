# November 19, 2026: host memory bounded by the piece being worked on, and a studies folder that keeps only what is needed here

What Marcos directed during the night of November 18 to 19, as the seat that did the work understood it.
The records are `records/development/2026-11-19-streamed-working-state/README.md` and
`records/maintenance-runs/2026-11-19-memory-headroom/README.md`.

## Host memory

**The goal.** Research runs mostly on the card. The host holds the piece being worked on and no more:
what needs the CPU is streamed to it. Using the whole card is fine. This came after the joint
continuation took the machine into a memory emergency and died on a display error.

**What changes because of it.** The trainer's working arrays live in files and are paged in one piece at
a time; gradients arrive one leaf at a time; a trace is read one merge at a time; nothing opens the
display. The method, objective, matcher, step and endpoint are untouched. A resume is done when it
reproduces the recorded updates and its unit's memory peak is a few GB.

**How the cause is found.** From the history, the logs and the code, including the other seat's
transcripts. A run is not started to learn where the memory goes, and the work is not paused to watch
one. He said this more than once that night: sessions had answered him by starting runs and by disk work
when he had asked for the streaming build.

**The disk a check asks for is part of the problem.** A preflight that wants 130 GB free is a finding
about the build, not a fact to accept. The per-member trace files were removed for that reason.

## Disk

**The goal.** The studies folder stops holding what the work no longer needs on this machine. Checkpoints,
raw traces and logs of finished runs go to the private bucket with the repository's own tool; the most
recent state stays. This widens what `STATE.md` listed as kept locally on November 18: the 348 and 390
payloads and the update-27 origin are bucket-only now.

**What does not change.** A local copy leaves only after its bucket copy is read back. The sealed
directory and the no-fit answers are never read or archived. Nobody asks him a disk question.

## Where builds, checks and tests run

Not on this machine. They go to a secondary test host via the remote runner, or to a rented box. The trainer itself stays
here because it needs these cards.

## The model that did this work

Fable 5.2 is excluded from this project's code by `CLAUDE.md`. Marcos switched this session to it himself
on November 19 and asked it to take the streaming build over from the model that had started it, so that
rule was set aside for this work on his direction. The exclusion stands for everything else, and no
project task was delegated to that model.
