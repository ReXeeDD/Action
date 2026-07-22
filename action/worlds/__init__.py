"""
worlds — every physical system we learn to predict.

Add a world here and the whole pipeline (data generation, memory model, renderer)
picks it up without further changes.

Available names:
    leaf                 fluttering leaf (deterministic sway, NOT chaotic)
    ball                 thrown/bouncing solid ball, real contacts, any direction
    pendulum1..pendulumN 1-link (integrable) .. n-link chained pendulum (chaotic n>=2)
    nbody2..nbodyN       Kepler 2-body (integrable) .. n-body gravitation (chaotic n>=3)

The families deliberately span integrable -> chaotic, so we can measure a real
prediction horizon and compare it against the leaf's (which we measured to be
non-chaotic).
"""
from __future__ import annotations

from action.worlds.base import MujocoWorld
from action.worlds.ball import BallWorld
from action.worlds.pendulum import ChainPendulum
from action.worlds.nbody import NBodyWorld
from action.worlds.generic import GenericObject, SHAPES

WORLD_NAMES = (["leaf", "ball", "object"]
               + [f"object_{s}" for s in SHAPES]
               + [f"pendulum{i}" for i in (1, 2, 3, 4, 5)]
               + [f"nbody{i}" for i in (2, 3, 4, 5)])


def make_world(name: str, seed: int | None = None):
    """Build a world by name, e.g. 'pendulum3', 'nbody4', 'ball', 'leaf'."""
    name = name.strip().lower()
    if name == "leaf":
        from action.leaf_world import LeafWorld
        return LeafWorld(seed=seed)
    if name == "ball":
        return BallWorld(seed=seed)
    if name == "object":
        return GenericObject(seed=seed)
    if name.startswith("object_"):
        return GenericObject(shape=name[len("object_"):], seed=seed)
    if name.startswith("pendulum"):
        return ChainPendulum(int(name[len("pendulum"):] or 2), seed=seed)
    if name.startswith("nbody"):
        return NBodyWorld(int(name[len("nbody"):] or 3), seed=seed)
    raise ValueError(f"unknown world {name!r}; try one of {WORLD_NAMES}")


__all__ = ["make_world", "WORLD_NAMES", "MujocoWorld",
           "BallWorld", "ChainPendulum", "NBodyWorld", "GenericObject"]
