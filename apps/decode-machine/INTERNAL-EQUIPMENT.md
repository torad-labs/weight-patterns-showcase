# Inside the layer equipment

## Goal

Let Marcos open a layer operation and see physical machinery for its transformations: what arrives, what the equipment uses, what changes, and what leaves. Preserve the approved simple cel-shaded world and measured replay clock.

## Direction contract

THESIS: Opening a machine reveals equipment whose motion explains its operation.

OWN-WORLD: Broad slate housings, brass working parts, retained amber weight balls, teal information carriers, pale ground and dark outlines.

STORY: Select a layer, open a housing, follow its parts and descend into a press or mixer. Inspect what each receives, uses and produces.

FIRST VIEWPORT: Machinery fills the main scene beside the existing inspector. Parallel branches share rows. On a phone, focus one selected device above the flowing controls.

FORM: Code-led extension of the approved illustrated machine. Signature interaction: open equipment, then hand-crank its internal motions. Cranking pauses the recording without changing recorded time; recorded playback resumes the common clock. No new direction selection or comp.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

Inherited direction provenance: `.impeccable/surface.md` records Marcos's explicit correction from reflective watchmaking to simple cel-shaded machinery, and the subsequent accepted extensions. That source explicitly says no new direction roll was needed; no concept-roll seed exists to carry forward. Its Quality bar paragraph and the shipped `DESIGN.md` are the retained standard. This extension does not invent a seed or a new reference card.

## Coverage

All twelve housings open: input normalization, Q/K/V projection, position rotation, attention, K/V cache, output projection, residual join, gate/up, down projection, per-layer input, dependency latch and weight storage. Depending on layer kind, these contain 62–72 immediate pieces and 199–248 pieces including nested equipment. Repeated functions reuse the same equipment vocabulary.

This is a curated physical decomposition of the operations. It does not claim a unique 3D asset for every node in the separate 1,885-node source anatomy, nor instruction-level telemetry.

| Operation | Physical equipment and behavior |
| --- | --- |
| Gather and layout | Forks and rollers gather a carrier into the next machine's lanes. |
| Weight read | Amber balls remain in their bins; read copies feed the press. |
| Matrix multiplication | Crossing operand beds and a descending platen combine input and weights; the receiving fork gathers output. |
| Normalization | Balance pans and a gain mechanism represent measuring magnitude and rescaling. |
| Position rotation | Paired turntables rotate pairs together using position settings. |
| Attention score | A comb compares the query with retained key material. |
| Probability and value mix | A maximum selector, exponential cam, parallel value press and sum beam feed a final gathering mechanism. |
| GEGLU | Cubic, coefficient, tanh and half-gain mechanisms shape the gate. The up branch bypasses GELU and joins at multiplication. |
| Residual addition | Two arriving carriers meet at a combining beam. |
| Cache | Retained racks store owner-layer context; shared layers use owner couplings and read forks. |
| Dependency | A counter mechanism holds a latch until its recorded dependency completes. |

Q/K/V and gate/up branches retain their parallel structure. Shared-cache layers omit their own K/V projection presses. Full attention has two channel passes; sliding attention has one. The per-layer-input path describes its prepared side input and gated contribution.

## Interactions and evidence

- Select a device in the scene or equipment selector. Receives, uses and produces describe that device.
- Open parts to descend; breadcrumbs return to the enclosing machine. Previous/next and camera focus work at each depth.
- Seek recorded work explicitly. Selection alone leaves recorded time unchanged.
- Hand crank pauses the replay and animates an illustrative traversal. It never rewrites the recorded time or invents a measured duration.
- Play resumes the shared recorded clock and clears manual cranking.
- Source lookup searches the separately labeled source anatomy.

Recorded operations use their measured interval. Children of a fused operation share that interval and say so; individually timed internal instructions are unavailable. Untimed storage remains static in recorded mode. Ball values, follower heights and transformations are symbolic, not captured numerical tensors. The source snapshot is not certified as the capture's build.

## Validation

`bun run test`: 12 pass, 0 fail. `bun run build`: exit 0, with the existing large-bundle advisory. Functional coverage includes every housing's input/use/output descriptions, shared-cache ownership, attention variants, parallel gate/up edges, nested timing and the independent hand crank.

Browser checks exercised all twelve housings, nested navigation, both captures, shared-cache owners and hand-crank motion with recorded time unchanged. Captures use exact 3840 × 2050, 1440 × 900 and 390 × 844 viewports at scale 1. Evidence lives in `evidence/workshop/`.

The independent Impeccable reviewer returned `ship` for its scored fixes: inherited direction provenance, the authored teaching-link arrow, phone navigation targets and the arrow's wrapping regression. That verdict covers those fixes, not a fresh whole-surface certification.

A separate screenshot-only Sonnet Design Judge scored the educational experience 7.0 and 7.1 on the desktop views, 6.5 on the laptop and 7.2/7.3 on the two phone views. These advisory scores are below its operating-screen benchmark. Findings include unused stage space, small secondary labels and text-heavy inspector details; the laptop inspector requires scrolling to reach its lower controls. The report is preserved in `evidence/workshop/design-judge.md`.
