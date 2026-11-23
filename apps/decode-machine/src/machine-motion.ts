import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import { journeyPartState } from "./connected-journey";
import { packetFlight, workPhase } from "./flow-motion";
import { intervalProgress, stepRole } from "./replay-model";
import type { MechanicalWorld } from "./replay-world";
import { workerActivity } from "./recorded-motion";

/** One compression and release per phase, shared by every level of the machine. */
export function pressStroke(progress: number) {
  if (progress <= 0 || progress >= 1) return 0;
  return Math.sin(progress * Math.PI) ** 2;
}

/** This is the sole writer of machine poses. Carriers have their own routes. */
export function updateMachinery(world: MechanicalWorld, time: number) {
  const story = world.story;
  if (world.view === "stack") {
    for (const layer of world.capture.layers) {
      const machine = world.macros[layer.id];
      const active = story
        ? story.layerActive && story.layerIndex === layer.id
        : layer.journeyStart <= time && time < layer.end;
      const work =
        active &&
        (story !== null ||
          world.recorded.layers[layer.id].some(
            (step) =>
              step.parent === null &&
              step.kind === "work" &&
              workPhase(step, time).active,
          ));
      const progress = story
        ? active
          ? story.layerProgress
          : 0
        : intervalProgress(layer.journeyStart, layer.end, time);
      const done = story ? story.cacheCount(layer.id) > 0 : time >= layer.end;
      machine.piston.position.y = 1.13 - 0.3 * pressStroke(progress);
      machine.lamp.material =
        world.materials[
          active ? (work ? "teal" : "wait") : done ? "green" : "paper"
        ];
    }
  }

  if (world.view === "layer") {
    for (const moving of world.moving) {
      const step = moving.role
        ? stepRole(world.capture, world.selectedLayer, moving.role)
        : undefined;
      const state = story
        ? journeyPartState(story, world.selectedLayer, moving.part)
        : { ...workPhase(step, time), done: !!step && time >= step.end };
      if (moving.motion === "press")
        moving.node.position.y =
          moving.base.y - 0.5 * pressStroke(state.progress);
      if (moving.motion === "rotate")
        moving.node.rotation.y = state.progress * Math.PI * 2;
      if (moving.motion === "gate")
        moving.node.rotation.z = state.done ? -Math.PI / 2 : 0;
    }
  }

  if (world.view === "engine") {
    world.workerStates = world.workers.map((machine, block) => {
      const state = workerActivity(story, world.recorded, block, time);
      machine.piston.position.y = 1.25 - 0.35 * pressStroke(state.progress);
      machine.lamp.material =
        world.materials[
          state.active ? (state.wait ? "wait" : "teal") : "paper"
        ];
      machine.latch.rotation.z = state.active && !state.wait ? -Math.PI / 2 : 0;
      const read = story
        ? { present: state.active, progress: state.progress }
        : state.step
          ? world.recorded.workerWeights(block, state.step.layer, time)
          : { present: false, progress: 0 };
      machine.reads.forEach((copy, index) => {
        const flight = packetFlight(read.progress, index, machine.reads.length);
        copy.setEnabled(read.present && flight.visible);
        copy.position = new Vector3(0.95 - flight.progress * 1.4, 0.9, 0);
      });
      return state;
    });
  }
}
