---
name: Decode machine
description: A cel-shaded connected question-to-answer machine with optional measured Gemma replay.
colors:
  ink: "#263e3d"
  muted: "#576864"
  line: "#b7c0b8"
  paper: "#f4f3eb"
  ground: "#e8e8e1"
  brass: "#9a6a29"
  token-teal: "#378b82"
  weight-amber: "#dc9c4a"
  structure-slate: "#91a4a7"
  structure-ivory: "#e4e3d5"
  structure-base: "#c6ceca"
  outline: "#33494c"
  residual: "#6c9e95"
  replay-muted: "#53645f"
  replay-teal: "#28796f"
  replay-brass: "#896029"
  machine-ink: "#344c4b"
  machine-base: "#b9c5bd"
  machine-paper: "#e9e9dd"
  machine-brass: "#c59854"
  machine-residual: "#729e92"
  machine-wait: "#8a78a6"
  machine-dark: "#567875"
  diagnostic-work: "#387e75"
  profile-teal-light: "#6d9e91"
  profile-amber-light: "#e1c181"
  profile-case-light: "#e7ece3"
  profile-teal: "#45877b"
  profile-teal-edge: "#2a675d"
  profile-amber: "#d2aa5d"
  profile-amber-edge: "#a2762c"
  profile-wait: "#9585ae"
  profile-wait-edge: "#786591"
  profile-paper: "#e7e8df"
  profile-case: "#d6ded5"
  profile-route: "#a8bcb3"
  profile-last-input: "#b4802c"
typography:
  profile-diagram:
    fontFamily: "Manrope, sans-serif"
    fontSize: "17px"
  profile-diagram-small:
    fontFamily: "Manrope, sans-serif"
    fontSize: "14px"
  diagnostic-title:
    fontFamily: "Manrope, system-ui, sans-serif"
    fontSize: "21px"
    fontWeight: 650
    letterSpacing: "-.4px"
  replay-display:
    fontFamily: "Manrope, system-ui, sans-serif"
    fontSize: "32px"
    fontWeight: 450
    lineHeight: 1.15
    letterSpacing: "-1px"
  replay-title:
    fontFamily: "Manrope, system-ui, sans-serif"
    fontSize: "24px"
    fontWeight: 500
    lineHeight: 1.25
    letterSpacing: "-.6px"
  replay-body:
    fontFamily: "Manrope, system-ui, sans-serif"
    fontSize: "14px"
    lineHeight: 1.7
  display:
    fontFamily: "Manrope, system-ui, sans-serif"
    fontSize: "30px"
    fontWeight: 450
    lineHeight: 1.2
    letterSpacing: "-1px"
  title:
    fontFamily: "Manrope, system-ui, sans-serif"
    fontSize: "23px"
    fontWeight: 500
  body:
    fontFamily: "Manrope, system-ui, sans-serif"
    fontSize: "13px"
    lineHeight: 1.7
  label:
    fontFamily: "Manrope, system-ui, sans-serif"
    fontSize: "12px"
rounded:
  profile-compact: "3px"
  profile-plot: "8px"
  track: "2px"
  control: "4px"
  panel: "6px"
spacing:
  compact: "8px"
  small: "12px"
  standard: "20px"
  panel: "24px"
components:
  trace-case-selected:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    padding: "20px 16px"
  trace-playback:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.control}"
    padding: "12px 16px"
  profile-action:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "10px 16px"
  profile-piece-selected:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.control}"
    padding: "7px 12px"
  diagnostic-action:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "9px 12px"
  replay-field:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "9px 10px"
  action-filled:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.control}"
    padding: "12px 20px"
  navigation-selected:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.control}"
    padding: "10px 15px"
  equipment-open:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.control}"
    padding: "10px 12px"
    width: "100%"
  transport:
    backgroundColor: "{colors.paper}"
    rounded: "{rounded.panel}"
    padding: "18px 20px"
---

# Design System: Decode machine

## Overview

**Creative North Star: "The Illustrated Layer Machine"**

Simple roller mechanisms, broad slate surfaces and outlined carriers make computation tangible. Three bands of cel shading describe depth; spherical amber weights and angular teal carriers communicate different roles through color and shape. Stored material remains in place while read copies move.

Quiet Manrope text and pale controls surround a full-width machine. The default illustrated France journey retains its input, current location and output through layer, equipment, worker, Cart & supplies, SM/warp and tile-machinery views. Fine machinery explains the relationships inside a representative SM while remaining explicitly illustrative. Measured recordings preserve their evidence boundaries. A synchronized worker timeline and evidence stories extend the same workbench for recorded diagnosis, using one page scroll with full-width evidence below the machine and timeline. Connected operations, chronological spans and observation dots expose relationships while preserving the distinction between captured evidence and proposed explanations. Eight profiling lenses extend the evidence inspector with one large relationship drawing and its connected facts, preserving the selected operation and distinguishing imported observations from an authored example. Connected weight-edit comparisons extend the same visual vocabulary with a measured boundary route, paired decisions and visible repaired, preserved and damaged outcomes. The separate `teaching.html` retains its numerical illustration and established material vocabulary. This system belongs to the local Decode machine viewer, not QGRE's overall identity.

**Key Characteristics:**
- Three-band shading, broad functional forms and restrained outlines.
- Amber stored weights and teal information carriers.
- One transport and reversible position across inspection depths.
- Spatial approach and return connect inspection depths without changing playback.
- Stable worker-floor overview with explicit active-worker following.
- Recorded work, source anatomy and authored teaching remain distinguishable.
- The same selected record and worker remain identifiable in the machine, timeline and evidence inspector.

## Colors

Muted green-grey structure sits on a pale ground; amber and teal distinguish the moving material.

### Primary
- **Token teal:** changing information carriers, branch hubs, legend square and timeline progress.
- **Weight amber:** reusable sample balls and the circular legend key.

Recorded replay uses darker replay-teal for progress, evidence text and field accents, replay-brass for hover/focus, and replay-muted for supporting text. Token teal remains the scene information material and legend key. The original muted, brass and residual tokens remain specific to teaching mode.

Diagnostic work spans use diagnostic-work, wait spans reuse machine-wait and mixed spans reuse structure-slate. The timeline legend repeats these interval colors with text. Ink outlines identify the selected operation independently of span color; the ink cursor identifies time.

The profiling drawings use their own muted teal, amber and violet material fills with darker matching edges. Teal denotes useful work, in-use material and one comparison population; amber denotes delivery, the last recorded prerequisite and another population; violet denotes a lens-specific wait, retained interval, bookkeeping or limit. Local captions establish the meaning. Pale profile-paper and profile-case surfaces house these relationships; profile-route joins them, while profile-last-input emphasizes the last recorded prerequisite. These SVG materials extend the scene palette rather than replacing it. Selected marks use an ink stroke, independent of their material fill.

The weight-edit route reuses machine-dark for the original baseline and round markers, replay-brass for the edited line and profile-amber-light for its square markers. Amber balls identify the edited weight projection; pale structure-base blocks identify the other boundaries. The labels and distinct marker shapes carry the comparison alongside color. Selection remains an ink surface with paper text, independent of whether the response was repaired, preserved or damaged.

### Secondary
- **Residual green:** preserved-information carrier and route, with matching route-label borders.
- **Brass:** teaching focus rings and the follow-state indicator, distinct from the brighter weight material.
- **Machine brass:** mechanical rollers and press chambers. **Machine wait:** violet recorded latches, legend key and waiting status markers.
- **Machine residual / dark:** replay bypass routes and side-input parts / cache drums and chassis.

### Neutral
- **Ink:** text, active navigation, active layer labels and filled controls.
- **Muted / line:** explanatory copy and quiet labels / field and annotation borders.
- **Paper / ground:** interface backing / scene backdrop.
- **Structure slate / ivory / base:** repeated module surfaces.
- **Outline:** teaching scene contours. Machine ink supplies replay crossbars, posts and drawer beds; machine base and paper supply replay foundations and reader channels. Material colors are shader inputs, not promises of identical rendered pixels.

**The Carrier Distinction Rule.** Keep weights spherical and amber, token information angular and teal, and the preserved residual route a quieter green.

## Typography

Locally bundled variable Manrope supplies headings, controls and body copy, with system-ui and sans-serif fallbacks. The compact ramp uses regular explanation, medium titles and stronger operation names; labels remain sentence case. Tabular numerals keep positions and time stable. Source paths, diagnostic tap names and provenance hashes use monospace.

The connected viewer uses a 32px heading, 24px inspector title, 14px body and controls, and 12px supporting text. At 2400px and above these become 42px, 30px, 18px and 16px. At 650px and below the heading is 23px; body and title remain 14px and 24px. Projected equipment labels wrap and center, at 12px normally, 18px on large desks and 13px on phones. Source paths use 12px/1.6, rising to 14px on large desks.

The frontmatter display/title/body roles retain teaching's 30px heading, 23px title and 13px copy; replay roles describe the connected viewer. Teaching headings step to 40px on large desks and 23px on phones; its inspector titles step to 30px and 22px respectively (20px in phone overview). Teaching retains its separate stylesheet.

Diagnostic panel and timeline headings share a compact 650-weight title, rising to 28px on large desks. Inspector subsection titles are 17px, rising to 23px at the same breakpoint; body and supporting copy inherit the replay ramp. Evidence paragraphs use 1.55 line height and a 70ch maximum. These remain functional headings, not display typography.

The profiling inspector retains Manrope and the inherited body/supporting ramp. Its section title is fluid (`clamp(26px, 1.6vw, 48px)`, tracking `-.035em`); lens headings use `clamp(22px, 1.1vw, 32px)`. These are local hierarchy roles, not new global display sizes. Repeated diagram labels use profile-diagram and profile-diagram-small; below 700px they become 15px and 13px respectively, while axis labels remain 14px. SVG text sizes are viewBox units and scale with the drawing. Large quantitative callouts use 38px at weight 650, reducing to 27px on phones. At 2400px and above, base labels use 11px, small and axis labels 9px, and quantitative callouts 26px in SVG viewBox units. The full-width SVG scales these values up in the rendered page; the smaller source sizes keep labels subordinate to the lens heading rather than establishing 9px body copy.

Weight-edit comparisons inherit the replay body ramp and Manrope family. Their heading uses 32px at weight 500, 42px on large desks and 25px on phones, with -.8px tracking. Outcome labels use weight 650; compact energy labels and phone controls use 12px. Paired decisions use 1.4em at weight 550. Passage and continuation text preserve line breaks but retain the body font; raw provenance alone uses the inherited monospace treatment.

## Layout

### Shared workbench

The connected viewer fills the width and 100svh, with a 740px minimum height. A flexible scene and 350px inspector share a row, separated by 28px; page padding is 26px 34px 18px. The inspector scrolls behind a left divider. Heading, input/location/output strip, view tabs, transport and footer span the workbench. No centered maximum-width column is introduced.

At 2400px and above, the inspector is 460px, gap 36px and padding 36px 48px 24px. From 651px to 1100px, these become 280px, 14px and 20px. The persistent story strip uses 1.2fr / 1fr / .8fr columns, with teal output and readiness chips. Its main text is 18px, rising to 25px on large desks.

At 650px and below the page flows in one column with 16px padding: story, tabs, canvas, transport, inspector and footer. The question spans two columns above current location and output; main story text is 16px. The decorative token strip and subtitle are hidden. The scene is 300px tall, or 380px for equipment, with its caption in a separate row. Timeline and duration each occupy a full transport row. The inspector uses a top rule and page scrolling.

### Recorded diagnostic workbench

Diagnostic mode uses one page scroll. From 651px upward, the machine and worker timeline occupy adjacent flexible columns. The timeline expands to expose all 63 logical workers in the document rather than trapping rows in an inner scroller. Shared transport follows this row, then a full-width evidence inspector behind a top rule. The inspector also expands in document flow.

Evidence stories use two columns from 651px and three from 2400px, beneath a full-width location trail. On large desks, worker chronology and recorded connections occupy the first column, arrivals the second and the investigation path the third; comparison, ranking and provenance follow within the same grid. Native layer, operation and worker selectors share a row above the stories.

At 650px and below the page flows through the machine, transport, complete worker timeline, evidence inspector and footer. The timeline has no fixed height or internal scrolling. Three axis labels at 12px show the beginning, midpoint and end. Worker-row heights are 20px on large desks, 24px at intermediate widths and 32px on phones; these describe the current plot density, not universal touch-target standards. The phone operation selector spans its own row, and evidence stories stack.

Underlined **Inspect selected evidence** and **Back to the machine** controls move through the page without changing the shared clock. These provide direct access across the longer document. This layout replaces the earlier three-pane diagnostic view and nested scrolling; final visual review scored the desk view 7.0 and compact view 6.5, both below the target of 8, so visual approval is not established.

### Profiling lenses

The profiling section spans the evidence inspector before the existing evidence stories. Eight illustrated lens buttons sit beside one large drawing and a connected-evidence rail. The regular grid is `minmax(200px, .68fr) minmax(0, 2.5fr) minmax(250px, .9fr)` with 36px gaps. At 2400px it becomes `minmax(260px, .62fr) minmax(0, 2.3fr) minmax(400px, 1fr)` with 50px gaps. At 1200px and below, navigation becomes a four-column row above the drawing and a 290px evidence rail. At 700px and below, two-column navigation precedes the drawing, controls and evidence rail in a single document column with 20px gaps. This local breakpoint is separate from the main workbench's 650px transition.

The paper plot has an origin strip, a responsive drawing and wrapping evidence-piece controls. The drawing switches from a wide to a compact composition when its host is narrower than 700px; it does not merely crop the desktop view. The SVG fills the stage width at automatic height, preserving its aspect ratio without the former 640px height cap. Captions, timeline and piece controls remain distinct rows. Plot minimum height is 390px, rising to 680px on large desks. The evidence rail uses ruled two-column facts with wrapping values, then a reading disclosure and the next investigation. Import and example-download actions wrap below the section.

### Connected weight-edit comparison

The comparison is a separate full-width section in document flow below the existing workbench, reached by an underlined masthead link. A top rule separates it from the machine. Four equal outcome buttons precede the expected result, problem disclosure and ruled playback row. At normal desktop widths the measured route and decision detail share a 1.65fr / minimum-330px column grid with 42px gaps. At 2400px and above the proportions become 1.8fr / minimum-700px with 64px gaps and 48px outer gutters. At 1600px and below the route precedes the decision detail in a single column.

Original and edited energy summaries sit side by side in each case above 1800px, stack below that width, and use proportional available/required bars within each case. At 650px and below all four case labels remain visible in a two-by-two grid; their repeated summaries move to one selected-case pair beneath the grid. Phone gutters are 18px. The download becomes an underlined text link, the introductory subtitle is hidden, the scrubber takes the first full control row, and timing remains in a disclosure. Complete continuations stack on phones.

The connected route preserves its 1100-unit drawing width at phone sizes inside a horizontal scroller, with an explicit scroll instruction. Surrounding copy and decision details wrap in normal document flow. This is a local choice for a sequential measured route, not a global requirement to scroll diagrams.

### Spatial inspection and labels

The layer installation, isolated layer, worker floor and internal equipment share the same surface and transport. Stage layouts fold into serpentine rows; parallel branches occupy separate lanes connected to their consumers. Counts and camera coordinates describe an instance rather than system tokens.

Projected paper labels avoid controls, each other and viewport edges; displaced labels receive muted dashed leaders. Overview density adapts to available space. Phone detail and short equipment canvases prioritize the selected label. The inspector and native selectors retain access to parts hidden by label reduction. Worker-floor framing stays stable until Follow active worker or an explicit closer view is chosen. Manual picking and camera dragging release automatic following; Fit restores the assembly. Phone Input / Weights / Output landmarks change the camera without seeking.

### Cart & supplies cutaway

The connected cutaway retains the workbench and transport. Six consumer branches occupy distinct parallel lanes between the incoming cartridge and collecting gate; context shelves, retained supplies, staging slots and final selection remain connected machinery. The inspector places its evidence description immediately below the topic heading and keeps “What happens here” collapsed initially.

On desktop the story strip aligns its output column with the inspector (350px, rising to 460px on large desks). The output has a compact teal border; pending output uses a dashed border and muted text. At phone width, the cutaway keeps a 300px minimum scene, wrapping tabs and actions, and playback beneath the machine. Its main heading, legend, provenance caption and story-strip text actions are hidden; topic evidence remains in the inspector. Option labels retain a 44px minimum height.

### Shared supply on the worker floor

The floor's source-defined installation includes a retained VRAM cabinet and one taller shared L2 bank beside the logical worker benches. Both reuse paper housings, ink compartments and amber stored balls. A brass link joins VRAM to L2; a distribution spine branches along worker rows and into individual supply inlets. The overview fit includes the cabinets and routes with the worker floor. This branching arrangement explains shared supply, not a physical SM map. Cabinet proportions, shelf counts, branch positions and packet sizes are instance geometry, not new design tokens. Desktop, laptop and phone captures for this extension are under evidence/cache-tma/.

### SM and warp cutaway

The representative SM occupies the existing full-width workbench, with a fixed outer foundation and rails around the selected block workspace. Shared staging and instruction issue sit to one side; shared execution sits opposite. The supply extension places a view of the same shared L2 cupboard outside the housing and an automatic TMA crane beside the local staging rack, joined by a brass delivery rail. The crane uses ink posts, a slate gantry and a brass moving fork. Source-role groups occupy parallel rows within the boundary, with role labels beside the lanes. Opening a warp hides the housing and other groups, expanding its 32 logical lanes from two rows of sixteen into four spaced rows of eight. An attached instruction rail connects the expanded rows; desktop projected numbers identify all 32 lanes. These are explanatory arrangements, not physical placement or reusable dimension tokens.

The inspector retains compact block and warp selectors, wrapping return navigation and the existing action family. Its lane selector has eight columns with 5px gaps and tabular numbered buttons, 36px minimum height on desktop and 44px on phones. On desktops at least 651px wide and no more than 1050px high, the hardware view uses one document scroll with a 580px scene sticky at 16px from the top; the inspector has no separate scroll. The desktop story strip aligns its output column with the inspector (350px, rising to 460px on large desks). At 650px and below the hardware viewport is 310px high, the view tabs form a two-by-two grid with thin outlined buttons and an ink-filled selected state, projected lane numbers are hidden, playback remains beneath the scene, and the inspector follows in page flow. The main heading, legend, provenance caption and story-strip text actions are hidden there; input, current position, output and the inspector's evidence boundary remain. Larger screens inherit the shared workbench and type ramp.

### Connected tile machinery

The tile cutaway expands across one broad workbench in the hardware view. Retained spherical weights occupy the reservoir and L2 shelves outside an outlined representative SM boundary. The staged route crosses a loading crane and reusable trays, turns through a layout table and private operand holders, then returns through the execution equipment, output jaws and completion counter. Teal information arrives on a distinct inlet. A ready-crew clutch and finite register/shared-memory drawers occupy the same connected installation. The depicted route is one staged operand example, not a floor plan or a claim about every GPU load.

Eleven inspectable stations have projected paper labels and matching native-selector options. Selecting a station changes its bed from paper to slate and its label to ink with paper text. Below 1600px, only the selected station label remains alongside the boundary and information-inlet landmarks. Desktop selection preserves the current framing; **Look closer** explicitly focuses the selected station and retains that focus across resize. On phones selection also frames that station. The overview uses an oblique view that exposes the working parts' height. The tile cutaway inherits the hardware view's desktop scroll and phone canvas/transport/inspector order. Station positions, camera angles, tray and producer counts, and the resource example are instance details rather than reusable design tokens.

### Retained teaching mode

Desktop fills the available width and 100svh with a 720px minimum height. The canvas occupies the workbench; masthead and caption sit upper left, inspector upper right, and navigation, transport and footer span the bottom. Lateral insets are 36–38px, increasing to 58–60px on large desks. Inspector widths are 258px normally, 235px on compact desktops and 350px on large desks. There is no centered maximum-width column.

At 650px and below, the final cascade uses a flex column with 568px top reservation. The canvas starts at 270px and is 290px high, leaving room above it for the question, token strip and answer. Playback sits directly beneath it, followed by inspector, navigation and footer. Side margins are 16px. Timeline and total-duration input each occupy a full transport row. The overview inspector retains its current-stage title and description. The active-layer readout spans the navigation width.

Desktop detail views suppress the scene caption and give masthead, navigation and footer paper backing. Layer inspection isolates the selected module and its local routes; the stack view restores context. Token cutter, Candidate selector, Return track and Weights annotations orient the overview. Overview and isolated-layer annotations try nearby positions within the canvas, avoiding the caption, inspector, masthead, transport, visible layer numbers and already placed annotations; labels without a clear position are hidden. Desktop inspectors scroll within the remaining height above the lower controls. Phones retain only the active layer's projected layer label. Repeated spacing clusters around compact, small, standard and panel steps rather than a strict baseline grid.

## Elevation & Depth

Three brightness bands (1.0, 0.81, 0.62), dark structural members and physical overlap give machinery depth. Teaching also uses explicit outlines and transparent reservoir walls. No reflection environment is used. The connected interface uses pale surfaces and fine borders; its transport has no ambient panel shadow.

Profiling brings the same banded material language into SVG machine housings and dependency trays. Discrete vertical gradients hold a light top band for 15% of the shape, the base material through 85%, and a dark lower band for the rest, with hard boundaries. Teal and amber reuse their profile base and edge colors with their light variants; housings use profile-case-light, profile-case and profile-route. Statistical marks retain flat fills so shading does not suggest another measured quantity. These bands create material depth without adding shadows.

Teaching retains a transport shadow (`0 6px 20px #263e3d0c`) and a floating guide shadow (`0 10px 35px #1b31241f`).

**The Banded Depth Rule.** Use broad lighting bands, outlines and physical overlap to describe machinery; keep interface elevation soft and sparse.

## Shapes

Modules combine simple roller cylinders, broad covers, feet and tubular routes. Cutter blades, sizing jaws, pin assemblies, paired K/V shelves, feed-forward presses and selector arms extend this same functional mechanical vocabulary. Tile machinery adds a loading crane, reusable trays, a sorting table, operand holders, a scheduling clutch and completion pegs. A tensor press, adding lever, maximum selector and shaped cam distinguish alternative execution tools; their arrangement does not imply four serial stages. Weight spheres contrast with box-shaped information carriers. Small rounded rectangles define interface controls and panels; circles identify playback, follow state and weight material. Fine outlines clarify geometry without ornamental screws or gear trains.

Profiling keeps small rectangular controls: profile-compact corners serve mode buttons and comparison rails, existing control corners serve evidence-piece and tool buttons, and profile-plot corners soften the large paper drawing. Mode containers reuse the panel radius. SVG boxes generally use the existing panel radius, with tighter corners for cards and interval strips. Circular evidence markers and drawn navigation icons carry meaning; these are not text glyph icons.

## Components

### Controls and inspector

One shared family serves every connected view: paper native fields with line borders and 4px corners; ink-filled primary actions; full-width outlined secondary actions; underlined text actions; and ink-selected view tabs. Primary and outline buttons have 44px minimum height and 10px 12px padding. Replay focus uses a 2px brass outline with 3px offset; disabled controls fade to .45 opacity. Icons are inline stroked SVG. Circular ink playback remains distinct from rectangular inspection actions.

The inspector uses restrained headings, two-column facts and native disclosures. Equipment navigation keeps a wrapping location path, Back, Previous/Next, and housing/part selectors. Receives / Uses / Produces rows have a 70px label column and 12px gap, rising to 90px and 16px on large desks; output is teal semibold. Source anatomy has a separate disclosure with search, wrapped paths and child navigation. Unknown measured values stay textual boundaries.

Active work labels use ink fill, while selected layer/mechanism labels use a green border; selected internal equipment also uses ink fill. Follow controls describe their target—tray, equipment or worker—and use an ink pressed state. Opening, selecting and focusing preserve pause. Watch this layer and transport actions explicitly control playback.

### Shared transport and motion

One rail presents play, step, restart, current phase, reversible scrub and duration. The illustrated journey separately retains whole-animation duration and inspection duration; the label names the currently edited scope. Opening a layer or housing magnifies the same story position, and returning to the whole journey releases the scope without silently resuming paused playback. Scrubbing and stepping pause. Camera changes do not seek. Duration changes viewing speed rather than the recorded intervals or story identity.

`ReplayPlayback` owns transport commands and time. `updateMachinery` in `machine-motion.ts` owns the layer-machine poses, with a shared single compression-and-release press stroke; carrier routes remain separate consumers of the same time. The worker-floor supply, SM and tile poses consume the same worker activity, story position and teaching choices through `tileSnapshotFor`. This ownership prevents competing playback clocks at different inspection depths. Reduced motion starts paused; explicit playback and exploration remain available.

Spatial inspection uses a short handoff independent of playback. An outgoing canvas frame scales toward the selected assembly while the destination camera approaches its fitted position; returning reverses the spatial direction. The camera arrival lasts 520 ms with cubic ease-out, while the outgoing frame dissolves over 360 ms using `cubic-bezier(.2,.7,.2,1)`. Projected labels fade in after the first 35% of arrival and remain noninteractive during the handoff. Resizing retargets the arrival, rapid navigation replaces it, and manual pointer or wheel input cancels it. Reduced motion retains a 120 ms dissolve and label reveal with no spatial scaling. No transition starts, stops or seeks the shared playback clock.

**The Continuous Journey Rule.** Retain the question, current location and output while magnifying the same story position; a change of camera must not become a change of time.

### Material movement and evidence

Amber balls stay stored while separate read copies feed weighted parts. Angular teal trays carry information through the physical intake, layer paths and printer. Concurrent branches remain parallel until joining. Recorded work uses teal activity and moving presses; waits use amber-brown lamps and held latches. Untimed equipment stays static. The worker overview summarizes working, waiting and idle logical blocks without implying physical SM placement.

Recorded packet and mechanism phases accumulate active time and park during gaps. Reverse seeking reconstructs position; staggered copies disappear at their outlet rather than visibly wrapping. Stack carriers remain in their machine through recorded completion before traversing the connector. Incoming pipes meet receiving-machine ports at matching positions. These routes, counts and speeds remain symbolic, not measured transfers.

**The Stored-and-Read Rule.** Preserve the visible stored balls while moving separate read copies; preserve token identity while changing its numerical pins.

**The Recorded Read Boundary Rule.** Keep stored balls visible while symbolic copies move during captured work spans; their routes, counts and speed do not establish measured transfer data. Keep unrecorded intake and readout physically present and labeled.

Measured equipment offers a separate, explicitly illustrative hand crank. It pauses captured time; Play clears cranking. Fused children traverse the parent's interval in dependency order, with same-stage lanes parallel. This illustrates staging without inventing child timings. The illustrated journey hides the crank and uses its shared scoped clock.

**The Captured-Time Boundary Rule.** Hand cranking pauses captured time; fused children share their recorded parent interval, never invented individual timings.

**The Evidence Separation Rule.** Keep recorded intervals, source anatomy and scripted teaching visibly distinct; a symbolic mechanism does not supply missing values or timings.

### Cart & supplies inspection

The Explore selector exposes nine topics: Inside the cart, Retained context, Split & collect, Weights & staging, Worker roles, Readiness & waits, Codes & scales, Choose the next token and Shared machinery. Topic and request selection preserve the playhead. Look closer changes the camera; the explicit Watch action starts the relevant illustrated process on the existing clock. Recorded mode hides Watch and keeps its captured timing. The explanation uses a native disclosure, followed by the current identity, options and topic-specific readings.

The cartridge table compares twelve incoming/current channel values using tabular, right-aligned numbers and paired signed bars. Both columns share a displayed symmetric range and center-zero line; current readings use teal and stronger weight, incoming readings use muted ink. Thin row rules keep the comparison compact. Pin height and the readout change together while cart identity remains.

Numbered context tiles form a three-column grid. Pale tiles are not stored yet; green tiles retain context outside the current query’s reach; teal-bordered tiles are stored and eligible. The readout names the K/V owner and counts retained and eligible positions. Reading sends copies while shelves remain. The optional three-position comparison appears only for illustrated sliding attention.

Partial contributions and reusable staging states use ruled label/output rows, with teal numerical results. The six contributions remain separate until collection; controller, loader and consumer descriptions use the incumbent material-flow rows. The rounding gauge pairs its physical notches with incoming value, sign/level/scale, stored value and rounding difference. Candidate rows pair native meters with explicit values and a selection/feedback state.

Optional requests A and B share the machinery and retained weights, identified by one and two notches. Separate ruled request sections retain each question, phase, layer, context count and answer; a companion status keeps B visible in the journey while either request is inspected. The overlap is an authored schedule. Recorded mode instead uses dashed unknown-reading boxes and explicit unmeasured states for missing channels, context contents, transfers, staging occupancy, numerical effects and final selection. Equal pins and unfilled shelves represent unknown readings, not measured zeros or an empty cache; a recorded wait alone does not identify its cause.

### SM, warp and lane inspection

**Open SM & warps** extends the selected worker into a representative fixed housing. The current separate source snapshot supplies eight groups: one controller, one loader and six consumers, each with 32 logical lanes. The model derives the group count from that snapshot; the eight-group arrangement is neither universal geometry nor a measured binding to the capture. Controller cams, loader tines and consumer levers distinguish roles while all groups connect to shared instruction issue and execution. Paper register holders belong to individual lanes; the staging rack belongs to the block. Warm retained operands and separate read copies reuse the established material vocabulary.

The selected warp bed becomes dark green. In the expanded view, a narrow ink strip locates the selected lane and an ink-filled, paper-text numbered button carries the matching pressed state. Recorded lane fingers use neutral paper; illustrated fingers retain teal. Recorded grid buttons have dashed borders for unknown lane activity, while the selected button retains a solid border. The marker denotes selection, not active-lane telemetry. Native block/warp selectors and the lane grid keep every target accessible alongside direct scene picking. Worker floor and SM bay text navigation restore the surrounding depth; Open 32 lanes magnifies the selected group. Selection, opening and returning retain the selected block, playhead and pause state. Watch this block explicitly seeks and plays on the existing clock; illustrated idle blocks show a disabled action instead.

A small square accompanies the textual Working, Waiting or Idle block status: teal, violet or neutral respectively. A persistent unknown-activity note precedes the controls; two initially collapsed native disclosures explain the machinery and evidence. Physical SM identity, other resident blocks, per-warp timing, active-lane masks, register values and role-to-thread binding stay unknown. The 63 captured logical blocks do not establish a physical SM count. The shared worker-activity resolver drives symbolic press, lever and finger strokes; group-wide movement does not assert measured lane execution. Pausing, reversing and changing viewing duration use that same playback position, with no hardware-specific clock.

### Tile handoff inspection

**Follow a tile through the machinery** opens the connected cutaway from the SM; related entry actions also reach it from the journey, equipment and cart views. The inspector retains the selected block and the input/location/output story. **SM bay** restores the containing view and the other return action names its originating view. **Inspect equipment** and direct scene selection change the explanation without seeking. **Watch one tile handoff** explicitly seeks and plays the relevant illustrated phase or captured worker interval; an illustrated idle block shows a disabled action. **Look closer** changes only the camera.

One muted evidence line precedes the equipment selector and its concise action description. Watch and Look closer use the existing filled/outlined button pair, wrapping when needed. Tile-specific hover retains ink at .85 opacity for Watch and gives Look closer paper backing with an ink border. A teal, brass or neutral square accompanies the textual working, held or idle cause. Three equal tray columns have 8px gaps and small muted state text beneath bold Tray A/B/C labels and drawn tray states. Open ink outlines hold brass balls: a raised incoming ball indicates filling, an upright peg means ready, a top bar means in use, and absent balls mean empty or released. The text distinguishes those last two states. Scene tray beds use brass while filling, residual green when ready, teal while in use and slate otherwise; raised readiness pegs reinforce the state.

The existing ruled material-flow rows show **Working crew**, **Inside the press** and **Receiving latch**, using muted labels and ink readings. Compact drawings carry the state alongside concise text: three outlined crew circles lift and fill the selected crew, four contribution cells rise and fill as results accumulate, and three completion pegs rise before a horizontal latch opens. These drawings are hidden from assistive technology because the adjacent text carries the same state. The contribution cells appear only for matrix work; alternative tools retain their own explanation. In the scene, matrix markers remain inside the press until output encoding, and producer pegs distinguish deposited results from completed dependent work. Stored weight balls remain visible while separate copies travel, and the sorting table rearranges marked values without changing their identity.

**Try the machinery** is an initially collapsed native disclosure containing the cache-source, instruction-kind and supply/workspace selectors. The four execution tools are alternatives. Supply and execution delays visibly hold the illustrative mechanism; register and shared-memory choices change the teaching reservation drawers and resident/queued blocks. These choices alter the cutaway only. A separate, initially collapsed **What is known here** disclosure names slot count, cache outcome, issue choice, producer count and resource budgets as examples, including in recorded mode. The repeated hardware-limit line is hidden in this depth; the visible illustrative label and scene caption remain. Fine machinery derives its pose from the current story phase or selected worker interval; it has no separate playback clock and supplies no missing transfer, issue, occupancy or wait-cause telemetry.

### L2 and TMA supply inspection

The floor inspector's **Supplying the workers** section uses thin top and bottom rules, 16px vertical padding and the existing body/label ramp. **Illustrate a cache** is a native Miss / Hit selector; the two wrapping outline actions are **Open L2 cache** and **Open TMA crane**. A muted footnote states that cache outcomes and deliveries are illustrative and physical SM placement was not captured. The SM inspector offers **Open shared L2** and **Open TMA crane** in the same outline-action family. Projected cabinet/crane labels and direct object picking provide equivalent spatial entry points. They open and focus the matching cache or loader station in the tile cutaway, retaining the selected block, playback time, pause state and cache choice. The existing zoom/dissolve and reduced-motion treatment apply.

One deterministic delivery state connects these depths. In the miss example, a small ink request travels from L2 toward VRAM, then an amber refill copy returns to L2 before onward delivery. In the hit example, delivery starts from L2. The floor renders one small amber packet on each active worker branch; the selected worker drives the shared request/refill marker. The SM view resolves delivery into its crane and local shared rack; tile inspection resolves the same phase into the crane and reusable trays. Stored balls remain present throughout. Loading one tray can overlap computation on another, and a tray receives its visible contents near the end of delivery. These movements explain relationships within the shared story or captured work envelope; route length, phase fractions and viewing speed do not establish cache outcomes, TMA latency or bandwidth. Browser checks on November 5, 2026 verified station entry and return, retained worker/cache selection, paused and running playback, reversible delivery poses, duration changes and reduced-motion return. Captures cover the 3840×2050 desktop, 1440×900 laptop and 390×844 phone viewports.

### Retained teaching actions

Filled dismissal and retry actions use ink and paper. Cover controls are bordered, full-width and paired with inline SVG. Follow the story uses an outlined circle which fills when pressed; the camera moves between intake, active layers, selector and overview with the story. Playback is circular with an ink fill and green hover. Shared keyboard focus uses a 2px brass outline offset by 4px. Disabled buttons fade to half opacity. Icons are stroked inline SVG.

### Teaching navigation and annotations

Named text buttons select inspection targets; selection receives ink fill and paper text. Numbered layer buttons use the same contrast for the active layer, with quieter green for completed layers. Active playback layer and selected inspection target are distinct states. Navigation wraps. Split, Join, Residual, Normalize, Feed-forward and individual Head labels explain isolated modules on desktop. Phones suppress the Head and route labels; story annotations and the stored-drawer label remain subject to collision placement. Intake, weights, layer and selector inspection hide unrelated mechanisms.

### Teaching fields and playback

Native select and number fields have pale backgrounds, thin borders and small corners. The duration field names total animation seconds for the entire question-to-answer story. A teal progress track and dark circular thumb scrub it; time reads elapsed / total seconds. Duration scales the shared animation while preserving progress; scrubbing pauses it and reconstructs cache and answer state. Question, Prefill, Paris, Period and End chapters seek and pause the same clock; the current chapter receives a teal underline. Replay restores the beginning and overview. Completion stops playback and retains the answer.

### Teaching inspector and concurrency lanes

A paper-backed bordered panel explains the selected mechanism. Normalization leads three equal overlapping head bars; join follows their interval, then rescaling and feed-forward work. A full-length residual bar accompanies the sequence. A separate count reports positions kept in the selected layer’s paired K/V shelves; moving read copies do not remove stored markers. The cursor shares the animation clock. Cover and camera inspection remain available while paused.

### Teaching question, tokens and answer

The fixed France question is a sentence-case heading, followed by a wrapping token strip. Tokens gain paper backing and a fine border as the cutter advances. They are buttons: selection adds a teal border, pale green backing and darker teal text. Selection opens the carrier readout without seeking playback; the token selector also includes Paris and the period. The Answer output grows from an ellipsis to Paris and its period; completion turns it green. A nearby note identifies the scripted example. This is presented as output, not an editable prompt. A second paper-backed answer label sits at the printer.

### Teaching stored-weight drawers and carrier readings

Native disclosures separate detail from the compact inspector. A thin rule, sentence-case summary and teal disclosure marker introduce each section. The weight section appears for layer inspection: opening it slides that layer’s drawer out and reveals twelve signed values in a three-column grid. Closing returns the drawer. Amber balls remain stored while local copies travel to the heads and press; their values are reused on each pass. Drawer and cover use the same interpolation and snap under reduced motion.

The carrier section preserves the selected token label and position identity. Four rows pair Layer input with Now, using tabular signed values and two adjacent bars: muted green-grey for the input and teal for the current value. The explicit illustration note remains beside the reading. The separate Show this carrier in a layer action pauses playback and can seek a waiting or completed carrier to its pass. Selecting alone does not move time. Desktop opening scrolls the reading into view within the inspector; the phone panel flows beneath playback. Phone disclosure summaries and token selects have a 44px minimum height, with 14px pin values. Concurrency lanes follow the disclosures.

### Teaching physical story mechanisms

The cutter reveals paper tokens, and the pin assembly gives teal carriers numerical pins. Paired shelves retain context while read copies travel toward the three attention heads. Sizing jaws and a press visibly change carrier height. Candidate bars use teal for the selected candidate and slate for alternatives; their heights are illustrative scores. The selected token prints and returns along the teal feedback route. End closes the return gate without printing an End word. These elements share the existing material and outline system; their instance dimensions are not design tokens.

### Teaching exploration guide

The raised guide uses the same body and title family and a filled dismissal action. Height is bounded with scrolling. Closing restores trigger focus. Reduced motion starts playback paused, removes loading-ring rotation and snaps camera/cover transitions; explicit playback remains available.

### Recorded evidence and worker timeline

Diagnosis keeps the cel-shaded machine and adds flat, ruled evidence surfaces. The same selection is visible as an ink-filled worker button, an outlined raw observation and an ink-filled machine label prefixed “Inspecting”. Participating machine labels receive a teal lower border. These distinguish the inspected record from concurrently active machinery.

Native layer, operation and worker selectors precede a connected location trail: recording position, layer, operation and worker. Operation spans its own selector row on phones and joins the other selectors in one row at wider widths. Paper-backed outline actions reuse small corners and the shared focus treatment. Ruled story sections ask ordinary questions about time, recorded connections, worker arrivals, comparison and investigation.

The selected worker's chronological strip shows resolved work, wait, mixed and unassigned time in its actual order. It shares the timeline category colors, ink observation outline and live cursor. Four tabular duration readouts accompany it, arranged in four columns normally and two on phones. Selecting a span selects its operation and seeks to its clipped entry; the explicit entry action is disabled when there is no valid observation. SVG spans and exit dots expose named button roles and keyboard activation, with ink stroke feedback on hover and focus.

Before, Inspecting and After operation columns sit on a connecting rail. The current operation has a heavier ink outline and ground backing. Each button carries its live stamp state with an outlined square, teal for an observed exit and violet while between entry and exit. An enclosing-operation bracket provides the parent link; a dashed branch explicitly names a missing producer connection. These links describe recorded order or missing evidence, never an inferred causal graph.

Worker exits are individual teal dots positioned by observed exit time and stacked to avoid collisions. The axis starts at the first exit, preserving the visibility of sub-microsecond differences. The selected worker is an enlarged ink dot. A dashed median line and an outlined median-to-last bracket show spread; first and last labels and the selected worker's entry, exit and elapsed values retain numeric context. Selecting a dot changes the worker without seeking. Comparison rows show the same operation from both captures as zero-based median-to-last spreads on one common duration scale, using teal ranges and endpoint dots. Opening the other recording explicitly pauses at the selected worker's entry; that action is disabled when its valid entry is missing. The nearby text states that different positions are not a controlled A/B. A disclosure ranks wider root-work exit spreads with proportional teal bars.

The plot uses alternating pale worker rows, darker backing for the selected worker, colored work/wait/mixed spans, ink observation outlines and an ink playhead. A textual legend identifies each interval category, including unassigned space. Row buttons select without seeking; clicking a plot span pauses at that time. Window selection magnifies the selected layer, operation or recording without moving the clock. In operation view the raw selected observation remains the displayed interval, including when it encloses nested work. A cursor outside the window is hidden and the time caption says so.

**The Shared Inspection Rule.** Keep record selection, worker selection and playback time distinct: selection identifies evidence across the machine and timeline; seeking changes the shared clock; changing the time window changes only its view.

**The Observed Evidence Rule.** Label exits as observations and links as recorded order. Keep the thread-0 stamp limitation and unknown wait producer visible; a longer exit spread is a place to investigate, not a claim of recoverable time or a complete critical path.

### Evidence states and investigation paths

Architecture areas form a branching selector with open circular markers; an ink ring identifies the selected area without filling its missing-evidence marker. Each architecture area has a conceptual checkpoint route, drawn with open dashed circles joined by dashed lines and explicitly labeled as missing event timings. The investigation path connects an observed statement, alternative explanations, required evidence and a conditional experiment. Known evidence uses a teal-filled circle; a hypothesis uses an outlined diamond; missing evidence has a dashed connection; a conditional experiment has a dashed circular marker. Alternative explanations branch from the same stem. Prediction, companion metrics and refutation remain in a native disclosure under the experiment.

The provenance disclosure repeats the same distinction: filled squares for the capture and tape/table evidence, dashed squares for absent exact-build and causal-producer bindings. Raw record identity, taps and hashes remain available in wrapping monospace text. Charts keep the existing Manrope family, with SVG text enlarged in phone viewBox coordinates rather than introducing a new display family. Stories use three columns on large desks and two at intermediate widths; phones retain the connected operation rail, wrap the location trail and use two-column duration and architecture groups. One-off SVG geometry is not a typography or spacing scale.

**The Evidence Shape Rule.** Use filled markers for present evidence, open or dashed markers for missing evidence, and a diamond for a proposed explanation. Keep the textual state beside the shape; selection does not turn a missing measurement into an observed fact.

### Eight-lens profiling inspector

**Follow the reason** retains layer, operation and worker context while switching among dependencies, instruction supply, useful work, buffer lifetimes, supply/activity, value distributions, tokens and instruments. Ruled navigation buttons pair a small inline SVG with text; the selected lens has paper backing, stronger ink text and a teal dot. An inactive lens gains ink text and an underline on hover, without the selected background. Recorded/example mode uses adjacent buttons with an ink-filled selected state. The origin strip states whether the drawing contains imported evidence, an authored example, or missing deeper measurements.

Relationship drawings use connected mechanisms, punched cards, proportional work cells, phased lifetime strips, interval-width samples, shared-bin histograms, token trajectories and instrument links. Text explains the units and evidence boundary. Dependency names remain identical in the drawing and piece controls at every viewport. Object names and supporting readiness captions share the small-label ladder with separate baselines; spaced prerequisite rows keep the captions clear of the next tray. Supply/activity drawings include an elapsed-time axis and a named example-time or recorded-time cursor while retaining each collection-pass label. A selected piece has a 3px ink outline on its own shape, including the first rectangle of a group, while its material color remains visible. Matching piece buttons use ink fill and paper text. A visible, wrapping chart key sits below the drawing and above piece controls. In dependencies it labels the amber last-required-input dot, ink consumer-start dot, violet time-between interval and thin example-time or recorded-time cursor. Other populated lenses label the ink outline as selected evidence. Key text inherits the supporting UI size; the meanings remain visible without opening a disclosure. Enter and Space activate the same entities as pointer selection; repeated shapes for one entity introduce one keyboard stop. Rerendering restores the corresponding focused mark, button or picker without scrolling. More than sixteen textual pieces use a native selector; drawing limits do not discard exported evidence. Selecting a distribution brings it and a compatible shared-bin population into view, including a third population outside the initial pair.

Recorded inspection actions explicitly select the linked operation or seek its interval. Opening the selected machine retains the established machine/selection model. The authored example has its own reversible scrub and duration controls; it never advances recorded time. Reduced motion disables automatic example playback and decorative transitions while retaining manual scrubbing. Profiling controls use a 2px ink focus outline with a 4px offset, while diagram focus shares the selected-mark stroke. Hover brightens no surface: interactive marks darken to .9 brightness over 160ms ease-out, removed under reduced motion.

Missing data uses open dashed shapes, explicit unmeasured text and an offered example. Present and missing provenance nodes remain filled and open/dashed respectively. Sample gaps are unobserved time, retained buffer endpoints are observation bounds, and different collection passes remain labeled. Share links carry lens, mode and selected piece; an imported bundle must be supplied again after reload. Export retains the complete evidence bundle and relationships, independent of the displayed subset.

**The Profiling Selection Rule.** Keep a selected evidence piece identifiable in its drawing, textual control and connected facts; selection does not imply that the evidence was recorded or change recorded time.

### Shared relational evidence

The JSON export preserves the visual reading order through location, worker time, recorded order, arrivals and investigation chapters. Each chapter points to identified entities and typed relationships rather than duplicating observations. The same focus records the run, layer, operation, logical worker, selected architecture area, clock position and window. Source record IDs remain intentional evidence identifiers, namespaced by capture in exported entity IDs.

The export separates raw observations, resolved segments and derived measurements from unproven hypotheses, missing evidence requirements and conditional experiments. Relations state their basis: recorded order, recorded enclosure, derivation, structural correspondence or analyst choice. Scope reports included and available operations; invalid observations retain their validity state instead of disappearing. Separate capture clocks and duration-only spread comparisons prevent the drawings from implying a shared absolute clock or controlled speedup. See `EVIDENCE-ONTOLOGY.md` for the schema and investigation semantics. These are evidence boundaries for the interface, not new visual tokens.

### Connected trace comparison

Outcome navigation pairs a concise case label with original/edited energy summaries on desktops. The ink-filled pressed state denotes the selected case. Inactive hover uses profile-case backing; teal focus outlines are 2px with a 4px offset throughout the section. Available and required bars have explicit quantities and answer text, so their colors do not carry the result alone. On phones only the selected pair appears beneath the four persistent case labels.

The route combines a horizontal original baseline with round markers and an edited polyline with square markers. Each boundary retains its label and percentage, with a visible key beneath the plot. A small amber weight holder identifies the projection among muted boundary housings. Dashed horizontal reference lines establish one fixed relative-change scale across cases. The route is a schematic of connected computation, not recovered hidden geometry.

A paper-backed incoming passage precedes two adjacent chosen-token readings and a ruled score table. The edited token uses brass emphasis. Native disclosures hold complete continuations, source evidence, the problem interpretation and timing options; evidence and continuations wrap rather than leaking beyond their columns. The source status remains visible near the heading, with a loading message and retry action when necessary.

**The Paired Trace Rule.** Keep the original and edited readings on one labeled scale and one shared incoming history; retain the first differing decision and identify later continuations as separate branches.

The playback control steps through recorded shared-history predictions over a user-selected viewing duration. Scrubbing, selecting a case or changing duration pauses playback; selecting a case opens its first changed token, or a late shared prediction when the case stays identical. The timing disclosure offers the first change or shared-history endpoint. Reduced motion changes Play into a single-prediction step and removes smooth scrolling on entry. No decorative interpolation is added between recorded boundary measurements.

## Do's and Don'ts

### Do:
- **Do** keep original and edited trace markers distinct in shape and label, with one fixed comparison scale.
- **Do** retain all four outcome selectors on phones and move repeated energy summaries into the selected case.
- **Do** preserve the weight/token distinction in both scene and legend.
- **Do** keep operation and context labels attached to isolated layer inspection.
- **Do** leave overview and detail annotations clear of interface panels and other visible labels.
- **Do** place playback directly beneath the visible canvas on phones.
- **Do** preserve pause during inspection, and label whole-animation versus inspection duration.
- **Do** keep the worker floor stable until active-worker following is explicitly enabled.
- **Do** run carriers and machinery from the shared clock and keep measured work, source anatomy and teaching distinguishable.
- **Do** keep fine machinery visibly illustrative, with retained storage, alternative execution tools and explicit teaching choices.
- **Do** preserve a short dissolve under reduced motion while removing spatial scaling.
- **Do** show the selected worker and record with both text and selection styling across diagnostic surfaces.
- **Do** retain observed-exit and recorded-order wording alongside their evidence limitations.
- **Do** pair evidence-state shapes with explicit labels, and keep selected architecture areas distinct from confirmed hardware causes.

- **Do** preserve profiling origin labels, local units and collection-pass identity alongside each relationship.

### Don't:
- **Don't** restore reflective miniature watchmaking or ornamental machinery; the chosen world uses functional, simplified mechanisms.
- **Don't** present illustrative counts or viewing durations as measured inference.
- **Don't** turn a camera pose or a one-off dimension into a reusable system token.

Not canonized: dense diagnostic worker-row targets remain an implementation detail, not a new touch-target standard. The earlier below-target machine framing, unused vertical space, alignment and phone density remain review context, not patterns for future surfaces to inherit. The latest correction removes nested timeline/inspector scrolling; final QA still identifies small-screen machine labels and framing, navigation distance to evidence, and a text-heavy action plan as visual gaps, not standards to inherit. Incidental heading/spacing values, SVG coordinates and the diagnostic swatch-border color are not promoted into new token primitives. The former inspector’s stacked summary and short exit underlines are superseded by chronological and relational evidence stories. Raw record-list arrow glyphs remain icon debt; instance dimensions, illustrative pin values, tiny overview labels and obsolete CSS values are not reusable tokens or legibility standards. SM housing proportions, source-snapshot group counts, symbolic lane strokes, tile-station coordinates, tray counts and teaching resource budgets are explanatory instances, not universal hardware geometry, telemetry or new system tokens.

Evidence: current `index.html`, `src/replay.css`, `src/replay-playback.ts`, `src/machine-motion.ts` and the retained teaching system; direction from `PRODUCT.md` and `.impeccable/surface.md`. The six inspected captures are `evidence/convergence/{user-3840,layer-3840,engine-3840,equipment-3840,laptop-1440,phone-390}.png`. The parent supplied an independent finish-review ship disposition with no material fixes, 21 browser checks, 23 tests and a successful build. Static screenshots do not prove temporal smoothness. This documentation pass did not rerun checks.

SM/warp extension evidence: `src/hardware-model.ts`, `src/hardware-world.ts`, `src/hardware-panel.ts` and the hardware rules in `src/replay.css`, reconciled with `PRODUCT.md`, `README.md` and `.impeccable/surface.md`. The parent reports a finish review finding visual and behavioral match. A subsequent screenshot-judge correction batch supplied the neutral recorded fingers, matching ink selection, attached instruction rail, projected lane numbers and responsive scroll/layout refinements recorded here; final captures are under `evidence/hardware/`, including `laptop-controls.png`. This documentation pass read source and did not rerun builds or tests. The historical concept-roll seed and corroborating record are unavailable: the accepted surface brief records a task-scoped provenance exception for this extension, not a reconstructed seed or new visual direction.

Tile-machinery extension evidence: `src/tile-flow.ts`, `src/tile-world.ts`, `src/tile-panel.ts`, `src/inspection-motion.ts`, the hardware integration and `src/replay.css`, reconciled with `MACHINERY-FLOW.md` and the accepted surface contract. This pass opened the final replacements of all five supplied captures under `evidence/machinery/`: `user-3840.png`, `desktop-tools.png`, `laptop.png`, `phone.png` and `phone-controls.png`. The final `desktop-tools.png` shows the execution-tool close-up; `laptop.png` retains the overview with only the selected station label. The independent `evidence/machinery/finish-review.md` records a ship disposition with no material fixes for the earlier batch. The builder reports 36 passing tests and build exit 0 after the final visual refinements. Static captures establish appearance, not temporal smoothness. This documentation pass did not run the browser, builds or tests.

L2/TMA extension evidence: the current `src/supply-world.ts`, `src/hardware-world.ts`, `src/hardware-panel.ts`, `src/hardware-model.ts`, `src/tile-flow.ts`, `src/tile-world.ts`, `src/replay-world.ts`, `index.html` and `src/replay.css`, read against the final L2/TMA paragraphs in `MACHINERY-FLOW.md`. The initial documentation pass was limited to source because the browser tool reported access disabled despite the enabled setting. Access recovered on November 5, 2026. Current captures are under `evidence/cache-tma/`: `floor-3840.png`, `sm-3840.png`, `tma-3840.png`, `tma-1440.png`, `tma-390.png` and `l2-controls-390.png`. Live checks verified floor/SM entry to both stations, return navigation, preserved paused and running playback, worker and cache selection, miss request/refill/delivery, hit delivery, overlapping tray use, reverse seeking, inactive transfers during waits/idle, duration scaling and reduced motion. Browser console warnings/errors were empty. A replay-source switch exposed a dropdown reset while the machinery kept its worker; `rebuildCapture` now restores the same valid worker selection, verified across both recordings and the France story. `bun run test` passed 38 tests (exit 0), `bun run build` passed (exit 0), and `git diff --check` passed. Captures establish appearance; live state and camera samples establish the checked motion behavior.

Browser-review refinement: the story strip keeps the same desktop column alignment across inspection depths, and unknown recorded output stays muted. The transport uses the same current operation as the location strip. Tile equipment fields stack label above a full-width select; paired actions stretch to equal height. Tile readout labels receive at least 110px and 40% of the row. Tray markers are 17px amber circles with a brass outline, matching the scene weights; waiting markers are violet. Tile disclosures share the body-size, semibold treatment. Block identifiers use two digits throughout hardware inspection. The revised captures are under evidence/cache-tma/r2/.

Recorded diagnostic extension evidence: `src/diagnostic.css`, `src/diagnostic-panel.ts`, `src/diagnostic-timeline.ts` and `src/diagnostic-model.ts`, read against `PRODUCT.md`, `DIAGNOSTICS.md` and the incumbent `.impeccable/surface.md` contract. The parent reports 51 passing tests, build exit 0 and a fresh functional review confirming three fixes resolved. This documentation pass inspected source only and did not rerun checks. The completed second visual review is below target; earlier extension approvals do not establish this extension’s visual approval. The parent reports no material functional finding after the final batch.

Diagnostic layout correction evidence (preceding evidence-story extension): the then-current `src/diagnostic.css` and `src/diagnostic-panel.ts` implement the three-column wide view, side-by-side machine/timeline with evidence below at intermediate widths, expanded phone timeline, reduced axis-label count and proportionate worker-time summary. The playhead uses the existing ink token. The completed review in `evidence/diagnostics/visual-review-r2.md` scores the 4K view 6.4, laptop entrance 6.5, laptop evidence 7.2, phone entrance 6.4, phone timeline 6.7 and phone evidence 7.6, all below the target of 8. Remaining valid issues concern machine framing and unused vertical space, nested timeline/inspector scrolling, alignment and phone density. The agreed two-round limit was reached; no further polish is included. The review’s row-count estimate is not authoritative: the capture has 63 logical workers. Source record IDs are intentional shared-evidence identifiers, not incorrect labels. Detector advisories alone do not establish visual approval.

Evidence-story extension evidence: `src/diagnostic-stories.ts`, `src/diagnostic-stories.css`, `src/inspection-report.ts` and their current panel integration, reconciled with `EVIDENCE-ONTOLOGY.md`. The parent reports the full suite passing with 55 tests, build exit 0, and browser checks passing for selection, export, shared-link restoration and worker-row mapping. Final visual review scored the desk view 7.0 and compact view 6.5, below the target of 8. No visual approval is claimed for this extension. This documentation pass read source and did not run builds or tests.

Evidence-story layout correction: current `src/diagnostic.css` and `src/diagnostic-stories.css` replace the three-pane diagnostic arrangement with a full-width evidence inspector below the machine/timeline and transport. All 63 worker rows are in page flow; the inspector and timeline have no internal scrollers. Story columns adapt from three to two to one. Page-jump controls retain clock position. Final QA is complete: desk 7.0 and compact 6.5 remain below the target of 8. Remaining visual gaps are small-screen machine labels and framing, navigation distance to evidence, and the text-heavy action plan. No builds or tests were run by this documentation pass.

Profiling extension evidence: `src/profiling-panel.ts`, `src/profiling-drawings.ts` and `src/profiling.css`, read against `PRODUCT.md` and `PROFILING-LENSES.md`. Final captures are `evidence/profiling/desktop-0-r2.png` through `desktop-7-r2.png`, `laptop-r2.png`, and `phone-0-r2.png` through `phone-7-r2.png`. This documentation pass sampled `desktop-5-r2.png` and `phone-0-r2.png`; the parent reports that the independent finish reviewer inspected all eight lenses and marked its four findings resolved: selected marks, third-distribution visibility, keyboard continuity and phone caption spacing. This is an established-world extension and a scoped resolution verdict, not a new concept approval or whole-app visual approval. Earlier review records remain historical evidence. No builds or tests were run by this documentation pass.

Not canonized for profiling: demonstration counts, viewBox coordinates, chart subset limits, one-off origin-strip colors and the example-entry button's isolated radius are implementation details, not reusable tokens. Existing glyph-icon debt and previously recorded below-target layout findings remain defects rather than house style.

Final profiling refinement evidence: the updated `src/profiling.css` removes the SVG height cap, and `src/profiling-drawings.ts` adds the elapsed-time axis and named cursor, consistent dependency names and caption ladder, more prerequisite-row spacing, and discrete three-band materials for housings and dependency trays. Statistical marks remain flat. The same final capture paths were replaced; this pass inspected the replacement `desktop-4-r2.png` and `phone-0-r2.png`. The parent reports a final finish verdict resolving this batch while retaining the prior selection fixes. That scoped verdict does not establish new whole-app approval. No tests or build were run by this documentation pass.

Final profiling clarity reconciliation: current `src/profiling.css` and `src/profiling-panel.ts` add the large-desktop SVG label override, visible chart keys and distinct inactive-navigation hover. The parent reports all three findings resolved in the final finish review; the same screenshot paths were replaced. Prior selection fixes and historical review limits remain recorded. This pass reconciled source only and did not rerun builds or tests.
