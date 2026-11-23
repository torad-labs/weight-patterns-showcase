import {
  stepRole,
  type Capture,
  type MechanismId,
  type RecordedStep,
} from "./replay-model";
import { workPhase } from "./flow-motion";

export type EquipmentKind =
  | "gather"
  | "weights"
  | "press"
  | "balance"
  | "rotor"
  | "score"
  | "cam"
  | "merge"
  | "latch"
  | "rack"
  | "store";
export type Equipment = {
  id: string;
  name: string;
  kind: EquipmentKind;
  does: string;
  input: string;
  uses: string;
  output: string;
  source: string;
  stage: number;
  lane: number;
  role?: string;
  shared: boolean;
  children: Equipment[];
  from?: string[];
};
export function equipmentPathForRole(parts: Equipment[], role: string): Equipment[] {
  const direct = parts.find(p => p.role === role);
  if (direct) return [direct];
  for (const part of parts) {
    const nested = equipmentPathForRole(part.children, role);
    if (nested.length) return [part, ...nested];
  }
  return [];
}
type Spec = [
  name: string,
  kind: EquipmentKind,
  input: string,
  uses: string,
  output: string,
  does: string,
  source: string,
  role?: string,
];
function part(
  id: string,
  s: Spec,
  stage: number,
  lane = 0,
  shared = false,
): Equipment {
  const [name, kind, input, uses, output, does, source, role] = s;
  return {
    id,
    name,
    kind,
    input,
    uses,
    output,
    does,
    source,
    role,
    stage,
    lane,
    shared,
    children: [],
  };
}
const detail: Partial<Record<EquipmentKind, Spec[]>> = {
  press: [
    [
      "Operand tray",
      "rack",
      "Incoming channels",
      "Shared-memory staging",
      "Aligned input tile",
      "A slotted tray arranges the input channels where the press can reach them.",
      "stage",
    ],
    [
      "Weight drawer",
      "weights",
      "Packed weights",
      "Block scales",
      "Reusable weight tiles",
      "Amber balls stay in their drawer. The press reads weights; it does not consume them.",
      "w4a4",
    ],
    [
      "Scale gearing",
      "cam",
      "Input and weight codes",
      "Their block scales",
      "Scaled operands",
      "Coupled gears represent the scales that give narrow codes their numerical range.",
      "scale",
    ],
    [
      "Product press",
      "press",
      "Two operand tiles",
      "Tensor-core multiply / accumulate",
      "Partial products summed into outputs",
      "Intersecting bars meet beneath a platen: each output combines input channels with its row of weights.",
      "matvec",
    ],
    [
      "Output cradle",
      "store",
      "Accumulated outputs",
      "Narrow codes and scales",
      "Stored output channels",
      "A receiving cradle catches the encoded result for the next consumer. It does not show actual numerical values.",
      "stage",
    ],
  ],
  balance: [
    [
      "Square levers",
      "press",
      "Input channels",
      "Each value multiplied by itself",
      "Squared magnitudes",
      "Each lever couples a channel to itself to measure magnitude without its sign.",
      "sumsq",
    ],
    [
      "Collecting beam",
      "merge",
      "Squared magnitudes",
      "Reduction across channels",
      "Sum of squares",
      "Several pans feed one shared beam, representing the reduction.",
      "sumsq",
    ],
    [
      "RMS cam",
      "cam",
      "Sum of squares",
      "Channel count and epsilon",
      "Inverse RMS factor",
      "A shaped cam represents mean, epsilon and reciprocal square root. This is a numerical function, not a physical fluid law.",
      "rsqrt",
    ],
    [
      "Gain jaws",
      "balance",
      "Original channels",
      "Inverse RMS and learned gain where present",
      "Normalized channels",
      "The common jaw adjustment rescales all channels; the individual gain settings apply where this normalization uses them.",
      "norm",
    ],
  ],
  rotor: [
    [
      "Pair picker",
      "gather",
      "Coordinate channels",
      "Pairing indices",
      "Coordinate pairs",
      "Two channels are brought together as the arms of one rotor.",
      "rope_partner",
    ],
    [
      "Angle drum",
      "cam",
      "Token position",
      "Stored sine / cosine tables",
      "Rotation coefficients",
      "The position selects a setting on the drum. It does not rotate or consume model weights.",
      "rope_table",
    ],
    [
      "Coupled turntable",
      "rotor",
      "Coordinate pairs",
      "Sine and cosine coefficients",
      "Rotated coordinate pairs",
      "Both arms turn together, illustrating a change of basis that preserves the pair length.",
      "rope_apply",
    ],
    [
      "Channel return",
      "store",
      "Rotated coordinates",
      "Partial-rotation selection",
      "Stored Q or K channels",
      "Rotated and unrotated channels return to their correct slots.",
      "rope_partial",
    ],
  ],
  score: [
    [
      "Query comb",
      "gather",
      "Current Q channels",
      "Head and channel indices",
      "Aligned query",
      "The comb presents one query against the stored key positions.",
      "attention",
    ],
    [
      "Key drum",
      "rack",
      "Cached keys",
      "Allowed context positions",
      "Key tiles",
      "Only eligible positions are presented to the comparison press.",
      "attention",
    ],
    [
      "Comparison press",
      "press",
      "Q and K tiles",
      "Dot product and score scale",
      "Attention scores",
      "Matching channels contribute to a score for each allowed context position.",
      "score",
    ],
    [
      "Score shelf",
      "store",
      "Attention scores",
      "Temporary score storage",
      "Stashed scores",
      "The score shelf holds results for the normalization and value-reading equipment.",
      "stash",
    ],
  ],
  merge: [
    [
      "Input pans",
      "gather",
      "Incoming contributions",
      "Their scales and positions",
      "Aligned contributions",
      "The incoming lanes arrive in separate pans. They keep their identity until the join.",
      "gather",
    ],
    [
      "Combining beam",
      "merge",
      "Aligned contributions",
      "Addition / reduction",
      "Combined channels",
      "The common beam combines contributions. For a residual join, the preserved incoming stream is one of them.",
      "pair_add",
    ],
    [
      "Two-term cradle",
      "store",
      "Combined channels",
      "Hi / lo carrier where specified",
      "Stored combined result",
      "The paired cradle represents the retained main and correction terms used by residual accumulation.",
      "two_term",
    ],
  ],
  latch: [
    [
      "Ready counter",
      "rack",
      "Published readiness signals",
      "The required generation / count",
      "Observed readiness",
      "A bank of tabs represents the condition the reader needs to observe.",
      "counter",
    ],
    [
      "Stop arm",
      "latch",
      "Reader requesting data",
      "Required readiness condition",
      "Released reader",
      "The arm stays down while the reader waits, then lifts when the recorded wait ends.",
      "wait",
    ],
  ],
};

function addDetails(p: Equipment) {
  const specs =
    p.kind === "press" && !/weights/i.test(p.uses)
      ? ([
          [
            "First operand tray",
            "rack",
            p.input,
            "Channel layout",
            "First operand",
            "The first operand is presented in aligned channel slots.",
            "stage",
          ],
          [
            "Second operand tray",
            "rack",
            p.uses,
            "Corresponding channel positions",
            "Second operand",
            "The other operand supplies the matching value or coefficient; it need not be a learned weight.",
            "gather",
          ],
          [
            "Product coupler",
            "press",
            "Aligned operands",
            "Multiplication",
            "Channel products",
            "A pair of coupled bars represents the multiplication specified by the enclosing operation.",
            "mul",
          ],
          [
            "Result cradle",
            "store",
            "Channel products",
            "The enclosing operation’s output storage",
            "Result channels",
            "The result returns to the channel slots used by its consumer.",
            "stage",
          ],
        ] as Spec[])
      : (detail[p.kind] ?? []);
  p.children = specs.map((s, i) => {
    const child = part(`${p.id}/${i}`, s, i, 0, Boolean(p.role));
    child.role = p.role;
    return child;
  });
  if (p.kind === "press" || p.kind === "score") {
    p.children[0].stage = 0;
    p.children[0].lane = -0.65;
    p.children[1].stage = 0;
    p.children[1].lane = 0.65;
    p.children.slice(2).forEach((child, i) => (child.stage = i + 1));
  }
  return p;
}

export function workshopParts(
  c: Capture,
  layer: number,
  mechanism: MechanismId,
): Equipment[] {
  const l = c.layers[layer],
    owned = l.owner === layer,
    owner = `layer ${l.owner + 1}`,
    width = l.ffnWidth.toLocaleString();
  let specs: Spec[] = [];
  switch (mechanism) {
    case "norm":
      specs = [
        [
          "Residual intake",
          "gather",
          "1536 hidden channels",
          "Residual hi / lo planes",
          "A channel row",
          "The intake brings the preserved channels into one row for normalization.",
          "gather",
        ],
        [
          "Magnitude balance",
          "balance",
          "Hidden channel row",
          "Sum of squares, channel count, epsilon",
          "Inverse RMS",
          "A collecting balance measures the row’s magnitude and sets a common adjustment.",
          "rms",
        ],
        [
          "Learned gain jaws",
          "press",
          "Original hidden channels",
          "Inverse RMS and normalization gains",
          "Normalized channels",
          "The jaws apply the common scale and learned per-channel gain. Input normalization has no separate interval in this tape.",
          "gamma",
        ],
        [
          "Operand staging",
          "rack",
          "Normalized channels",
          "Shared-memory tile and scales",
          "Projection-ready input",
          "The channels are arranged into the narrow operand tile used by projection.",
          "stage",
        ],
      ];
      break;
    case "qkv":
      specs = [
        [
          "Input loader",
          "gather",
          "Normalized hidden channels",
          "Shared-memory operand tile",
          "Ready projection input",
          "The fork brings the input tile to the projection presses.",
          "x_tile",
        ],
        [
          "Q press",
          "press",
          "Normalized hidden channels",
          "Q weights and scales",
          `${c.queryHeads} query heads`,
          "Every query head is a different weighted projection of the same input.",
          "matvec",
          "q:0",
        ],
        ...(owned
          ? ([
              [
                "K press",
                "press",
                "Normalized hidden channels",
                "K weights and scales",
                "One key head",
                "A separate weighted press forms the key channels. Its individual interval is not recorded.",
                "matvec",
                "mixer:0",
              ],
              [
                "V press",
                "press",
                "Normalized hidden channels",
                "V weights and scales",
                "One value head",
                "Another weighted press forms the values that attention will read. Its individual interval is not recorded.",
                "matvec",
                "mixer:0",
              ],
            ] as Spec[])
          : []),
        [
          "Q collection",
          "gather",
          "Raw Q channels",
          "Projection output slots",
          "A complete Q row",
          "The collector gathers the query channels before head normalization and rotation.",
          "gather",
          "qtail:0",
        ],
        [
          "Q sizing jaws",
          "balance",
          "Q row",
          "Head normalization gain and epsilon",
          "Normalized Q",
          "The jaws normalize the query head before it enters the paired rotors.",
          "rms",
          "qtail:1",
        ],
        [
          "Q position rotors",
          "rotor",
          "Normalized Q",
          "Position and sine / cosine tables",
          "Rotated Q",
          "The query pairs turn before they are stored for attention.",
          "rope",
          "qtail:3",
        ],
        [
          "Q / K / V ready gate",
          "latch",
          "Readers waiting for projection",
          "Published projection completion",
          "Ready attention inputs",
          "The latch represents the recorded wait for Q/K/V readiness. Shared layers obtain K/V from their owner.",
          "wait",
          "mixer:1",
        ],
      ];
      break;
    case "rope":
      specs = [
        [
          "Q gather",
          "gather",
          "Raw Q row",
          "Per-head channel slots",
          "Collected Q",
          "Channels are gathered into the head’s working row.",
          "gather",
          "qtail:0",
        ],
        [
          "Head normalization",
          "balance",
          "Q row",
          "RMS normalization",
          "Normalized Q",
          "The magnitude balance normalizes the head before rotation.",
          "rms",
          "qtail:1",
        ],
        [
          "Stage & rendezvous",
          "latch",
          "Normalized Q",
          "Shared stage and barrier",
          "A ready row",
          "The marked interval contains staging and synchronization together.",
          "stage",
          "qtail:2",
        ],
        [
          "Position rotors",
          "rotor",
          "Normalized Q",
          "Token position, sine / cosine tables",
          "Rotated Q",
          "Paired coordinates turn together. This span also includes storing Q; its internals are not independently timed.",
          "rope",
          "qtail:3",
        ],
        ...(owned
          ? [
              [
                "K rotors",
                "rotor",
                "Normalized K",
                "Token position, sine / cosine tables",
                "Rotated K",
                "Owned K is normalized and rotated before storage; this tape has no separate K-tail timestamps.",
                "rope",
              ] as Spec,
            ]
          : []),
      ];
      break;
    case "attention": {
      const passes = l.kind === "F" ? 2 : 1;
      for (let pass = 0; pass < passes; pass++) {
        const prefix = `attention-pass${pass}:`,
          name = passes > 1 ? ` · pass ${pass + 1}` : "";
        specs.push(
          [
            `Score press${name}`,
            "score",
            "Q + cached K",
            `${c.queryHeads} query heads; ${l.window.toLocaleString()}-position capacity`,
            "Stashed attention scores",
            "The press compares queries with allowed keys. This measured entry span includes earlier tiles and the final tile arrival.",
            "score",
            prefix + "0",
          ],
          [
            `Stash gate${name}`,
            "latch",
            "Score-producing warps",
            "Stash barrier",
            "Scores ready to read",
            "Other warps arrive and rendezvous before the score shelf can be consumed.",
            "stash",
            prefix + "1",
          ],
          [
            `Probability / V mill${name}`,
            "cam",
            "Stashed scores + cached V",
            "Max, exponential, sums",
            "Weighted value contributions",
            "The mill turns scores into relative weights and applies them to V. Open it to see each operation.",
            "attend",
            prefix + "2",
          ],
          [
            `Tile combining beam${name}`,
            "merge",
            "Attention tile contributions",
            "Running maxima and sums",
            "Combined tile result",
            "Partial attention tiles are rescaled and combined consistently.",
            "tile_combine",
            prefix + "3",
          ],
          [
            `Channel outlet${name}`,
            "store",
            "Combined attention result",
            "Normalization and output slots",
            "Attention output channels",
            "The final division and store finish this channel pass.",
            "normalize_guarded",
            prefix + "4",
          ],
        );
      }
      specs.push([
        "Attention-ready gate",
        "latch",
        "Output projection reader",
        "Published attention output",
        "Released output projection",
        "The output projection waits for the attention channels it needs.",
        "wait",
        "mixer:3",
      ]);
      break;
    }
    case "cache":
      specs = owned
        ? [
            [
              "Key intake",
              "rotor",
              "Normalized key channels",
              "Position rotation",
              "Keys for this position",
              "The key coordinate pairs turn before entering the storage drum.",
              "rope",
            ],
            [
              "K storage drum",
              "rack",
              "Rotated K",
              "Position-indexed cache slots",
              "Retained keys",
              "Key channels are retained for later tokens to compare against. Contents and occupancy were not recorded.",
              "kv",
            ],
            [
              "V storage drum",
              "rack",
              "V channels",
              "Position-indexed cache slots",
              "Retained values",
              "Value channels are kept in a separate drum alongside the keys.",
              "kv",
            ],
            [
              "Context selector",
              "cam",
              "Current token position",
              l.kind === "F"
                ? "Full-context capacity"
                : "512-position sliding window",
              "Eligible K / V positions",
              "A selector opens the positions this layer is allowed to read; it never opens future positions.",
              "attention",
            ],
            [
              "Read forks",
              "gather",
              "Eligible K and V",
              "Eight query heads sharing one K/V head",
              "K to score; V to weighted sum",
              "Independent read paths feed the comparison press and value mill. Reading leaves the stored data intact.",
              "gather",
            ],
          ]
        : [
            [
              "Owner coupling",
              "rack",
              `K / V retained by ${owner}`,
              "Shared-cache connection",
              "Available K and V",
              "This layer connects to the earlier owner. It does not run another K/V projection or refill its own drums.",
              "kv",
            ],
            [
              "Context selector",
              "cam",
              "Current position",
              l.kind === "F"
                ? "Full-context capacity"
                : "512-position sliding window",
              "Eligible owner positions",
              "Only the positions allowed for this attention layer are selected.",
              "attention",
            ],
            [
              "Read forks",
              "gather",
              `Keys and values from ${owner}`,
              "Eight query heads sharing one K/V head",
              "K to score; V to weighted sum",
              "Read copies travel to the attention machinery; the owner keeps its stored K/V.",
              "gather",
            ],
          ];
      break;
    case "output":
      specs = [
        [
          "Head collector",
          "gather",
          "Eight attention-head outputs",
          "Head/channel arrangement",
          "Projection input",
          "The collector arranges the head channels for the output projection.",
          "gather",
        ],
        [
          "Output press",
          "press",
          "Attention output channels",
          "O weights; 2 split-K contributions",
          "Two partial contributions",
          "Each partition contributes to the output; the next fold must wait for the needed partials.",
          "matvec",
          "main:0",
        ],
        [
          "Partial shelves",
          "rack",
          "Output partials",
          "Two contribution slots",
          "Published partials",
          "Separate shelves retain each split-K contribution. Their store timing is part of the projection interval.",
          "kslice",
          "main:0",
        ],
        [
          "Output-ready gate",
          "latch",
          "Consumer waiting for partials",
          "Published output counters",
          "Released output fold",
          "The gate lifts only after the recorded partial wait.",
          "wait",
          "main:1",
        ],
      ];
      break;
    case "residual":
      specs = [
        [
          "Partial collector",
          "gather",
          "O projection partials",
          "Split-K gather / fold",
          "Collected output",
          "The pans collect the projection’s partial results.",
          "gather",
          "ew:0",
        ],
        [
          "Output sizing jaws",
          "balance",
          "Collected output",
          "Post-attention normalization",
          "Normalized branch",
          "The branch is normalized before it joins the preserved residual.",
          "norm",
          "ew:1",
        ],
        [
          "Residual joining beam",
          "merge",
          "Normalized branch + incoming residual",
          "Preserved hi / lo channels",
          "Updated residual",
          "A bypass carries the incoming information intact to this joining station.",
          "two_term",
          "ew:2",
        ],
        [
          "Cooperative staging",
          "rack",
          "Updated residual",
          "Shared stage; narrow codes and scales",
          "Staged result",
          "Workers place the updated result into slots the next consumer can read.",
          "stage",
          "ew:3",
        ],
        [
          "Publication latch",
          "store",
          "Staged result",
          "Readiness publication",
          "Visible result",
          "Publication follows the store; it signals availability rather than changing the tensor.",
          "publish",
          "ew:4",
        ],
        [
          "Region exit",
          "gather",
          "Published result",
          "Remaining region control",
          "Completed consumer span",
          "This interval covers the remaining exit from the fused consumer.",
          "region",
          "ew:5",
        ],
        [
          "Next-input gate",
          "latch",
          "FFN input reader",
          "Normalization-input readiness",
          "Released FFN input",
          "The next consumer waits for the inputs it requires.",
          "wait",
          "main:5",
        ],
      ];
      break;
    case "gateup":
      specs = [
        [
          "Input sizing & stage",
          "balance",
          "Post-attention residual",
          "Partial fold, RMS factor, shared tile",
          "FFN input",
          "This fused entry stages the input and derives its normalization scalar.",
          "norm",
          "gate-up:0",
        ],
        [
          "Gate press",
          "press",
          "FFN input",
          `Gate weights; ${width} output channels`,
          "Gate branch",
          "One press expands the input into the gate branch. The two presses share the recorded weight-stream interval.",
          "gate_up",
          "gate-up:1",
        ],
        [
          "Up press",
          "press",
          "The same FFN input",
          `Up weights; ${width} output channels`,
          "Up branch",
          "The parallel press forms a second expanded branch with different weights.",
          "gate_up",
          "gate-up:1",
        ],
        [
          "GELU metering cam",
          "cam",
          "Gate branch",
          "GELU function",
          "Activated gate",
          "A shaped cam represents GELU acting element by element; it is not a hard on/off switch.",
          "gelu",
          "gate-up:2",
        ],
        [
          "Branch multiplier",
          "press",
          "Activated gate + up branch",
          "Elementwise multiplication",
          "Gated FFN activation",
          "The two branches meet and multiply channel by channel. This is not another learned projection.",
          "geglu",
          "gate-up:2",
        ],
        [
          "Activation outlet",
          "store",
          "Gated FFN activation",
          "Narrow stage and C1 publication",
          "Visible FFN intermediate",
          "The intermediate is stored and its readiness is published.",
          "publish",
          "gate-up:2",
        ],
        [
          "Activation-ready gate",
          "latch",
          "Down-projection reader",
          "C1 readiness",
          "Released down input",
          "The down projection waits for the activation it will consume.",
          "wait",
          "main:9",
        ],
      ];
      break;
    case "down":
      specs = [
        [
          "Activation loader",
          "gather",
          `${width} FFN channels`,
          "Operand tiles",
          "Down-projection input",
          "The fork gathers the expanded activation for compression.",
          "stage",
        ],
        [
          "Compression press",
          "press",
          "Expanded FFN activation",
          "Down weights; 6 split-K contributions",
          "1536-channel partial results",
          "Each weighted partition contributes to the narrower hidden-width result.",
          "matvec",
          "main:10",
        ],
        [
          "Six partial cups",
          "rack",
          "Down partial results",
          "Six split-K slots",
          "Published contributions",
          "The cups keep the separate contributions until the fold can read them.",
          "kslice",
          "main:10",
        ],
        [
          "Partial-ready gate",
          "latch",
          "Residual consumer",
          "Published down-partial counters",
          "Released fold",
          "The fold waits for its required contributions.",
          "wait",
          "main:11",
        ],
        [
          "Down combining beam",
          "merge",
          "Six partial contributions",
          "Split-K fold",
          "Combined down result",
          "The six partials are combined within the fused down-consumer span.",
          "kslice",
          "main:12",
        ],
        [
          "Post-down sizing jaws",
          "balance",
          "Combined down result",
          "Post-FFN normalization",
          "Normalized down branch",
          "The branch is normalized before the residual addition. Its timing is not separated from the fused consumer.",
          "norm",
          "main:12",
        ],
        [
          "Residual joining beam",
          "merge",
          "Normalized branch + preserved residual",
          "Hi / lo residual carrier",
          "Updated hidden channels",
          "The original FFN input rejoins the transformed branch. The PLE tail continues from this result.",
          "two_term",
          "main:12",
        ],
      ];
      break;
    case "ple":
      specs = [
        [
          "Prepared side cartridge",
          "rack",
          "256-channel per-layer input",
          "Projected/normalized input + token table",
          "Prepared side input",
          "The cartridge was prepared from the input projection and token-indexed table. Its preparation has no timing in this layer tape.",
          "embed",
        ],
        [
          "Side gate press",
          "press",
          "Updated hidden channels",
          "PLE gate weights",
          "256-channel gate",
          "The hidden state determines how the prepared side input is gated.",
          "matvec",
          "main:12",
        ],
        [
          "Side GELU / multiply",
          "cam",
          "Gate + prepared side input",
          "GELU and channelwise multiplication",
          "Gated side activation",
          "The gate acts on the prepared side input through GEGLU.",
          "geglu",
          "main:12",
        ],
        [
          "Side output press",
          "press",
          "Gated side activation",
          "PLE output weights",
          "Hidden-width side branch",
          "The side branch is projected back into the hidden channels.",
          "matvec",
          "main:12",
        ],
        [
          "Side sizing jaws",
          "balance",
          "Projected side branch",
          "Post-PLE normalization",
          "Normalized side branch",
          "A magnitude balance applies the side branch normalization.",
          "norm",
          "main:12",
        ],
        [
          "Scaled residual join",
          "merge",
          "Side branch + retained hidden state",
          "Layer scalar; hi / lo residual",
          "Layer result",
          "The final join incorporates the side branch and applies the layer scalar before next-layer staging.",
          "residual_add_scaled",
          "main:12",
        ],
      ];
      break;
    case "latch":
      specs = [
        [
          "Result storage",
          "store",
          "Updated residual hi / lo",
          "Codes, scales and next-layer stage",
          "Stored next-layer input",
          "The data must be stored before it is advertised as ready. Store and publication share the consumer’s interval.",
          "stage",
          "main:12",
        ],
        [
          "Readiness publication",
          "rack",
          "Completed writes",
          "Readiness counters",
          "Published completion",
          "A readiness tab is raised after the result is available. Counters coordinate the persistent workers.",
          "publish",
          "main:12",
        ],
        [
          "Next-layer stop arm",
          "latch",
          "Next-layer reader",
          "Required residual generation",
          "Released reader",
          "The recorded residual wait holds the reader here. Ending the wait releases it to the next layer.",
          "wait",
          "main:15",
        ],
      ];
      break;
    case "weights":
      specs = [
        [
          "Weight reservoir",
          "weights",
          "Reusable packed model weights",
          "Weight blocks and their scales",
          "Readable weight tiles",
          "Amber balls stand for stored weights. The drawer remains full when an operation reads from it.",
          "w4a4",
        ],
        [
          "Tile loader",
          "gather",
          "Weight tile request",
          "TMA tensor map and addresses",
          "Transferred tile",
          "The loader represents transfer from stored weight memory into the shared staging ring. Transfer timing is absent from this recording.",
          "tma",
        ],
        [
          "Staging ring",
          "rack",
          "Loaded weight tiles",
          "Shared-memory slots",
          "Available operand tiles",
          "Slots let a loader prepare data for consumers. Actual occupancy is unknown.",
          "stage",
        ],
        [
          "Scale gearing",
          "cam",
          "Packed codes",
          "Associated block scales",
          "Numerical operands",
          "Codes and scales travel together. The scale gearing explains their interpretation without inventing values.",
          "scale",
        ],
        [
          "Consumer forks",
          "gather",
          "Ready operand tiles",
          "Tensor-core operand layout",
          "Weights at the product press",
          "Consumers read the staged operands. The original reservoir is retained.",
          "matvec",
        ],
      ];
      break;
  }
  const parts = specs.map((s, i) =>
    addDetails(part(`${mechanism}-${i}`, s, i)),
  );
  // Parallel projections share a stage and split the incoming pipe before rejoining.
  if (mechanism === "qkv") {
    const projections = parts.filter((p) => /^[QKV] press$/.test(p.name));
    projections.forEach((p, i) => {
      p.stage = 1;
      p.lane = i - (projections.length - 1) / 2;
    });
    parts.slice(1 + projections.length).forEach((p, i) => (p.stage = 2 + i));
  }
  if (mechanism === "gateup") {
    parts[1].stage = 1;
    parts[1].lane = -0.65;
    parts[2].stage = 1;
    parts[2].lane = 0.65;
    parts.slice(3).forEach((p, i) => (p.stage = i + 2));
  }
  if (mechanism === "gateup") {
    parts[3].from = [parts[1].id];
    parts[4].from = [parts[2].id, parts[3].id];
  }
  if (mechanism === "qkv") {
    const query = parts.find((p) => p.name === "Q press")!,
      gather = parts.find((p) => p.name === "Q collection")!,
      ready = parts.at(-1)!;
    gather.from = [query.id];
    ready.from = [
      parts.at(-2)!.id,
      ...parts
        .filter((p) => p.name === "K press" || p.name === "V press")
        .map((p) => p.id),
    ];
  }
  if (mechanism === "cache" && owned) {
    parts[1].stage = 1;
    parts[1].lane = -0.65;
    parts[2].stage = 1;
    parts[2].lane = 0.65;
    parts[3].stage = 2;
    parts[4].stage = 3;
  }
  if (mechanism === "cache" && owned) parts[2].from = [];
  if (mechanism === "ple") {
    parts[1].from = [];
    parts[2].from = [parts[0].id, parts[1].id];
  }
  if (mechanism === "norm") parts[2].from = [parts[0].id, parts[1].id];
  for (const p of parts) {
    if (
      mechanism === "ple" ||
      (mechanism === "down" && p.role === "main:12") ||
      (["main:0", "main:10", "mixer:0"].includes(p.role ?? "") &&
        p.kind === "rack") ||
      p.name === "K press" ||
      p.name === "V press" ||
      p.role === "gate-up:2"
    )
      p.shared = Boolean(p.role);
    if (p.name.startsWith("Probability / V mill")) {
      const specs: Spec[] = [
        [
          "Maximum selector",
          "cam",
          "Stashed attention scores",
          "Maximum reduction",
          "Largest score",
          "The tallest follower sets a common score reference.",
          "fixed_max",
        ],
        [
          "Exponential cam",
          "cam",
          "Scores minus maximum",
          "Exponential function",
          "Positive relative weights",
          "A nonlinear cam converts relative scores into positive weights.",
          "attend",
        ],
        [
          "Value product press",
          "press",
          "Relative weights + cached V",
          "Weighted value products",
          "Value numerator",
          "The press weights each value by the corresponding score-derived factor.",
          "attend",
        ],
        [
          "Weight collecting beam",
          "merge",
          "Relative weights",
          "Sum across allowed positions",
          "Normalization denominator",
          "A second lane sums the relative weights used to normalize the value result.",
          "attend",
        ],
        [
          "Gathering cradle",
          "store",
          "Value numerator + weight sum",
          "Temporary per-tile storage",
          "Tile contributions",
          "Both quantities are retained for tile combination and channel finalization.",
          "attend",
        ],
      ];
      p.children = specs.map((s, i) => {
        const child = addDetails(part(`${p.id}/${i}`, s, i, 0, true));
        child.role = p.role;
        for (const leaf of child.children) {
          leaf.role = p.role;
          leaf.shared = true;
        }
        return child;
      });
      p.children[2].stage = 2;
      p.children[2].lane = -0.65;
      p.children[3].stage = 2;
      p.children[3].lane = 0.65;
      p.children[4].stage = 3;
    }
    if (p.name === "GELU metering cam" || p.name === "Side GELU / multiply") {
      const specs: Spec[] = [
        [
          "Cubic coupling",
          "press",
          "Gate value g",
          "g multiplied by itself twice",
          "g cubed",
          "The coupler forms the cubic term used by the tanh approximation.",
          "gelu",
        ],
        [
          "Affine gearing",
          "cam",
          "g and g cubed",
          "GELU coefficients",
          "Scaled tanh input",
          "The gearing combines g with the scaled cubic contribution.",
          "gelu_c",
        ],
        [
          "Tanh cam",
          "cam",
          "Scaled input",
          "Hyperbolic tangent",
          "Smooth bounded response",
          "The cam has a smooth response, rather than a binary open/closed threshold.",
          "tanh",
        ],
        [
          "Half-gain coupler",
          "press",
          "g and the tanh response",
          "0.5 × g × (1 + response)",
          "GELU(g)",
          "The final coupling completes the tanh-approximate GELU.",
          "gelu_half",
        ],
      ];
      if (p.name === "Side GELU / multiply")
        specs.push([
          "Side-input multiplier",
          "press",
          "GELU(g) and prepared side input",
          "Channelwise multiplication",
          "Gated side activation",
          "The activated gate multiplies the prepared side input.",
          "geglu",
        ]);
      p.children = specs.map((s, i) => {
        const child = part(`${p.id}/${i}`, s, i, 0, true);
        child.role = p.role;
        return child;
      });
    }
  }
  return parts;
}

export function mechanismForRole(role: string): MechanismId {
  if (
    role.startsWith("attention-pass") ||
    role === "mixer:2" ||
    role === "mixer:3"
  )
    return "attention";
  if (role.startsWith("qtail:") || role === "q:1") return "rope";
  if (role.startsWith("q:") || role.startsWith("mixer:")) return "qkv";
  if (role.startsWith("ew:")) return "residual";
  if (role.startsWith("gate-up:")) return "gateup";
  const n = Number(role.split(":")[1]);
  return n <= 1
    ? "output"
    : n <= 5
      ? "residual"
      : n <= 9
        ? "gateup"
        : n <= 12
          ? "down"
          : "latch";
}

export function equipmentState(
  c: Capture,
  layer: number,
  p: Equipment,
  time: number,
  crank: number | null = null,
) {
  const step = p.role ? stepRole(c, layer, p.role) : undefined;
  const phase = workPhase(step, time),
    active = phase.active;
  const progress = crank ?? phase.progress;
  return {
    step,
    active: crank === null ? active : progress > 0 && progress < 1,
    present: crank === null ? phase.present : progress > 0 && progress < 1,
    progress,
    done: crank === null ? Boolean(step && time >= step.end) : progress === 1,
    evidence:
      crank !== null
        ? "Hand-cranked illustration"
        : !step
          ? "Source anatomy · no independent timing"
          : p.shared
            ? "Shares enclosing recorded interval"
            : "Recorded interval",
  };
}

// Several pieces can explain one fused recorded interval. Traverse its dependent
// stages inside that interval; pieces at the same stage still run in parallel.
export function equipmentMotionState(
  c: Capture,
  layer: number,
  parts: Equipment[],
  p: Equipment,
  time: number,
) {
  const state = equipmentState(c, layer, p, time);
  if (!p.role) return state;
  const stages = [
    ...new Set(
      parts
        .filter((other) => other.role === p.role)
        .map((other) => other.stage),
    ),
  ].sort((a, b) => a - b);
  const progress = Math.max(
    0,
    Math.min(1, state.progress * stages.length - stages.indexOf(p.stage)),
  );
  const present = state.present && progress > 0 && progress < 1;
  return {
    ...state,
    progress,
    present,
    active: state.active && present,
    done: progress === 1,
  };
}

export function stepForEquipment(
  c: Capture,
  layer: number,
  p: Equipment,
): RecordedStep | undefined {
  return p.role ? stepRole(c, layer, p.role) : undefined;
}

export function equipmentEdges(parts: Equipment[]): [Equipment, Equipment][] {
  const stages = [...new Set(parts.map((p) => p.stage))].sort((a, b) => a - b);
  return parts.flatMap((to) => {
    const previous = stages[stages.indexOf(to.stage) - 1];
    return parts
      .filter((from) =>
        to.from
          ? to.from.includes(from.id)
          : previous !== undefined && from.stage === previous,
      )
      .map((from) => [from, to] as [Equipment, Equipment]);
  });
}
