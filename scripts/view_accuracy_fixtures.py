"""Visualize accuracy fixtures one at a time.

Usage:
  python scripts/view_accuracy_fixtures.py
  python scripts/view_accuracy_fixtures.py --fixture "attack neutral"

Controls:
  Space: play/pause
  N: step one sim tick
  R: reset current fixture
  Left/Right: previous/next fixture
  Esc: quit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim import config as C
from sim.actions import Action
from sim.engine import step_tick
from sim.state import empty_state, precompute_distances


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "levels" / "accuracy"

MAP_UNITS = 700
PADDING = 60
WIN_W = WIN_H = MAP_UNITS + 2 * PADDING
FPS = 60
TICK_SECONDS = 1.0 / C.TICK_HZ

COLOR_BG = (24, 24, 32)
COLOR_PANEL = (18, 18, 24)
COLOR_NEUTRAL = (160, 160, 170)
COLOR_P1 = (70, 140, 240)
COLOR_P2 = (230, 90, 90)
COLOR_TEXT = (240, 240, 245)
COLOR_MUTED = (130, 130, 140)
COLOR_HIGHLIGHT = (255, 230, 90)

OWNER_MAP = {
    "neutral": C.OWNER_NEUTRAL,
    "p1": C.OWNER_P1,
    "p2": C.OWNER_P2,
}

OWNER_NAME = {
    C.OWNER_NEUTRAL: "neutral",
    C.OWNER_P1: "p1",
    C.OWNER_P2: "p2",
}

OWNER_COLOR = {
    C.OWNER_NEUTRAL: COLOR_NEUTRAL,
    C.OWNER_P1: COLOR_P1,
    C.OWNER_P2: COLOR_P2,
}

BUILDING_RADIUS = 28


def _load_fixture(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_internal(entry: dict, real_key: str, internal_key: str, default: int = 0) -> int:
    if internal_key in entry:
        return int(entry[internal_key])
    if real_key in entry:
        return int(round(float(entry[real_key]) * C.SCALE))
    return default


def _build_state(data: dict):
    state = empty_state()
    b = state.buildings
    g = state.unit_groups
    b[:] = 0
    g[:] = 0

    for building in data["buildings"]:
        slot = int(building["slot"])
        b[slot]["alive"] = int(building.get("alive", 1))
        b[slot]["owner"] = OWNER_MAP[building["owner"]]
        b[slot]["type_id"] = int(building.get("type_id", C.TYPE_BASIC))
        b[slot]["garrison"] = _to_internal(building, "garrison_real", "garrison_internal")
        b[slot]["capacity"] = _to_internal(
            building,
            "capacity_real",
            "capacity_internal",
            default=C.DEFAULT_CAPACITY,
        )
        b[slot]["x"] = int(building["x"])
        b[slot]["y"] = int(building["y"])

    precompute_distances(state)

    for group in data.get("unit_groups", []):
        slot = int(group["slot"])
        g[slot]["alive"] = int(group.get("alive", 1))
        g[slot]["owner"] = OWNER_MAP[group["owner"]]
        g[slot]["src_slot"] = int(group["src_slot"])
        g[slot]["tgt_slot"] = int(group["tgt_slot"])
        g[slot]["count"] = _to_internal(group, "count_real", "count_internal")
        g[slot]["progress"] = int(group.get("progress", 0))
        g[slot]["travel_ticks"] = int(group["travel_ticks"])

    state.tick = int(data.get("tick", 0))
    return state


def _actions_for_tick(data: dict, tick: int):
    action_p1 = None
    action_p2 = None

    for action in data.get("actions", []):
        if int(action["tick"]) != tick:
            continue
        decoded = Action(
            kind="send",
            type_idx=int(action["type_idx"]),
            src=int(action["src"]),
            tgt=int(action["tgt"]),
        )
        if action["player"] == "p1":
            action_p1 = decoded
        elif action["player"] == "p2":
            action_p2 = decoded
    return action_p1, action_p2


def _to_screen(x: int, y: int) -> tuple[int, int]:
    return PADDING + x, PADDING + y


class FixtureRun:
    def __init__(self, data: dict):
        self.data = data
        self.reset()

    def reset(self):
        self.state = _build_state(self.data)
        self.prev_groups = self.state.unit_groups.copy()
        self.r1_total = 0.0
        self.r2_total = 0.0
        self.done = False

    def step(self):
        if self.done:
            return
        self.prev_groups = self.state.unit_groups.copy()
        action_p1, action_p2 = _actions_for_tick(self.data, self.state.tick)
        r1, r2, _ = step_tick(self.state, action_p1=action_p1, action_p2=action_p2)
        self.r1_total += r1
        self.r2_total += r2
        if self.state.tick >= int(self.data["steps"]):
            self.done = True


def _draw(screen, font, big_font, run: FixtureRun, tick_progress: float, idx: int, total: int):
    screen.fill(COLOR_BG)
    state = run.state
    b = state.buildings
    g = state.unit_groups

    for i in range(C.MAX_UNIT_GROUP_SLOTS):
        prev_alive = bool(run.prev_groups[i]["alive"])
        cur_alive = bool(g["alive"][i])
        if not prev_alive and not cur_alive:
            continue

        rec = g[i] if cur_alive else run.prev_groups[i]
        src = int(rec["src_slot"])
        tgt = int(rec["tgt_slot"])
        travel = max(1, int(rec["travel_ticks"]))
        prev_prog = int(run.prev_groups[i]["progress"]) / travel if prev_alive else 0.0
        cur_prog = int(g[i]["progress"]) / travel if cur_alive else 1.0
        prog = prev_prog + (cur_prog - prev_prog) * tick_progress
        prog = min(1.0, max(0.0, prog))
        sx, sy = _to_screen(int(b["x"][src]), int(b["y"][src]))
        tx, ty = _to_screen(int(b["x"][tgt]), int(b["y"][tgt]))
        x = int(sx + (tx - sx) * prog)
        y = int(sy + (ty - sy) * prog)
        color = OWNER_COLOR[int(rec["owner"])]
        count_real = int(rec["count"]) / C.SCALE
        pygame.draw.circle(screen, color, (x, y), 10)
        label = font.render(f"{count_real:g}", True, COLOR_TEXT)
        screen.blit(label, label.get_rect(center=(x, y - 18)))

    for i in range(C.MAX_BUILDING_SLOTS):
        if not b["alive"][i]:
            continue
        x, y = _to_screen(int(b["x"][i]), int(b["y"][i]))
        color = OWNER_COLOR[int(b["owner"][i])]
        pygame.draw.circle(screen, color, (x, y), BUILDING_RADIUS)
        pygame.draw.circle(screen, (20, 20, 28), (x, y), BUILDING_RADIUS, 2)
        gar_real = int(b["garrison"][i]) / C.SCALE
        label = big_font.render(f"{gar_real:g}", True, COLOR_TEXT)
        screen.blit(label, label.get_rect(center=(x, y)))
        slot_label = font.render(f"s{i}", True, COLOR_HIGHLIGHT)
        screen.blit(slot_label, (x - 10, y + 34))

    panel = pygame.Rect(8, WIN_H - 170, WIN_W - 16, 162)
    pygame.draw.rect(screen, COLOR_PANEL, panel, border_radius=8)
    pygame.draw.rect(screen, (45, 45, 55), panel, 1, border_radius=8)

    lines = [
        f"{idx + 1}/{total}  {run.data['name']}",
        f"setup: {run.data['setup_summary']}",
        f"expected: {run.data['expected_summary']}",
        f"tick: {state.tick}/{run.data['steps']}  rewards: {run.r1_total:.1f}/{run.r2_total:.1f}",
        "controls: space play/pause, n step, r reset, left/right switch, esc quit",
    ]
    if run.data.get("xfail_reason"):
        lines.append(f"known mismatch: {run.data['xfail_reason']}")

    y = panel.y + 10
    for line in lines:
        surf = font.render(line, True, COLOR_TEXT if not line.startswith("known mismatch") else COLOR_HIGHLIGHT)
        screen.blit(surf, (panel.x + 10, y))
        y += 24


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", help="Fixture name to open directly")
    args = parser.parse_args()

    fixtures = [_load_fixture(path) for path in sorted(FIXTURE_DIR.glob("*.json"))]
    if args.fixture:
        fixtures = [f for f in fixtures if f["name"] == args.fixture]
        if not fixtures:
            raise SystemExit(f"Unknown fixture: {args.fixture!r}")

    pygame.init()
    pygame.display.set_caption("Mushroom Wars v2 — Accuracy Fixtures")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo,Consolas,monospace", 16)
    big_font = pygame.font.SysFont("Menlo,Consolas,monospace", 22, bold=True)

    idx = 0
    run = FixtureRun(fixtures[idx])
    playing = False
    last_tick_wall = time.perf_counter()
    running = True

    while running:
        now = time.perf_counter()
        tick_progress = min(1.0, (now - last_tick_wall) / TICK_SECONDS) if playing and not run.done else 0.0

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    playing = not playing
                    last_tick_wall = time.perf_counter()
                elif ev.key == pygame.K_n:
                    run.step()
                    last_tick_wall = time.perf_counter()
                elif ev.key == pygame.K_r:
                    run.reset()
                    playing = False
                    last_tick_wall = time.perf_counter()
                elif ev.key == pygame.K_RIGHT:
                    idx = (idx + 1) % len(fixtures)
                    run = FixtureRun(fixtures[idx])
                    playing = False
                    last_tick_wall = time.perf_counter()
                elif ev.key == pygame.K_LEFT:
                    idx = (idx - 1) % len(fixtures)
                    run = FixtureRun(fixtures[idx])
                    playing = False
                    last_tick_wall = time.perf_counter()

        if playing and not run.done and (now - last_tick_wall) >= TICK_SECONDS:
            run.step()
            last_tick_wall = now
            tick_progress = 0.0

        _draw(screen, font, big_font, run, tick_progress, idx, len(fixtures))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
