"""
Playable visualization for mushroom-wars-v2.

Human (P1, blue) vs random-legal-action AI (P2, red).
  - Click an owned building, then click a target to send units.
  - Keys 1/2/3/4 select send percentage (25/50/75/100).
  - R restarts, Esc quits.

Sim runs at 1 Hz internal tick (C.TICK_HZ). Renderer runs at 60 FPS with
linear interpolation of in-flight unit groups between src and tgt.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pygame

# Allow running as `python scripts/play.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim import config as C
from sim import levels
from sim.actions import Action, compute_mask, decode
from sim.engine import step_tick


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
MAP_UNITS       = 700                   # levels use 0..700 coordinate space
PADDING         = 60
WIN_W = WIN_H   = MAP_UNITS + 2 * PADDING    # 820 px
FPS             = 60
TICK_SECONDS    = 1.0 / C.TICK_HZ            # 1.0 s at TICK_HZ=1

COLOR_BG        = (24, 24, 32)
COLOR_NEUTRAL   = (160, 160, 170)
COLOR_P1        = (70, 140, 240)
COLOR_P2        = (230, 90, 90)
COLOR_HIGHLIGHT = (255, 230, 90)
COLOR_TEXT      = (240, 240, 245)
COLOR_MUTED     = (130, 130, 140)

OWNER_COLOR = {
    C.OWNER_NEUTRAL: COLOR_NEUTRAL,
    C.OWNER_P1:      COLOR_P1,
    C.OWNER_P2:      COLOR_P2,
}

BUILDING_RADIUS = 28


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------
def to_screen(x: int, y: int) -> tuple[int, int]:
    return PADDING + int(x * MAP_UNITS / MAP_UNITS), PADDING + int(y * MAP_UNITS / MAP_UNITS)


def building_at(state, mx: int, my: int) -> int | None:
    b = state.buildings
    for i in range(C.MAX_BUILDING_SLOTS):
        if not b["alive"][i]:
            continue
        sx, sy = to_screen(int(b["x"][i]), int(b["y"][i]))
        if (mx - sx) ** 2 + (my - sy) ** 2 <= BUILDING_RADIUS ** 2:
            return i
    return None


# ---------------------------------------------------------------------------
# AI: pick a random legal non-noop action; fall back to noop.
# ---------------------------------------------------------------------------
def ai_action(state, player: int, rng: np.random.Generator) -> Action | None:
    mask = compute_mask(state, player)
    legal = np.flatnonzero(mask)
    non_noop = legal[legal != (len(mask) - 1)]
    if non_noop.size == 0:
        return None
    return decode(int(rng.choice(non_noop)))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def draw_state(screen, font, big_font, state, selected_src: int | None,
               send_pct: int, tick_progress: float, status: str,
               prev_groups: np.ndarray | None = None):
    screen.fill(COLOR_BG)

    b = state.buildings
    g = state.unit_groups

    # In-flight unit groups (drawn under buildings).
    for i in range(C.MAX_UNIT_GROUP_SLOTS):
        prev_alive = prev_groups is not None and bool(prev_groups[i]["alive"])
        cur_alive = bool(g["alive"][i])
        if not prev_alive and not cur_alive:
            continue

        rec = g[i] if cur_alive else prev_groups[i]
        src = int(rec["src_slot"])
        tgt = int(rec["tgt_slot"])
        travel = max(1, int(rec["travel_ticks"]))

        prev_prog = int(prev_groups[i]["progress"]) / travel if prev_alive else 0.0
        if cur_alive:
            cur_prog = int(g[i]["progress"]) / travel
        else:
            cur_prog = 1.0
        prog = prev_prog + (cur_prog - prev_prog) * tick_progress
        prog = min(1.0, max(0.0, prog))

        sx, sy = to_screen(int(b["x"][src]), int(b["y"][src]))
        tx, ty = to_screen(int(b["x"][tgt]), int(b["y"][tgt]))
        x = int(sx + (tx - sx) * prog)
        y = int(sy + (ty - sy) * prog)
        color = OWNER_COLOR[int(rec["owner"])]
        count_real = int(rec["count"]) // C.SCALE
        pygame.draw.circle(screen, color, (x, y), 10)
        label = font.render(str(count_real), True, COLOR_TEXT)
        screen.blit(label, label.get_rect(center=(x, y - 18)))

    # Buildings.
    for i in range(C.MAX_BUILDING_SLOTS):
        if not b["alive"][i]:
            continue
        x, y = to_screen(int(b["x"][i]), int(b["y"][i]))
        color = OWNER_COLOR[int(b["owner"][i])]
        pygame.draw.circle(screen, color, (x, y), BUILDING_RADIUS)
        if i == selected_src:
            pygame.draw.circle(screen, COLOR_HIGHLIGHT, (x, y), BUILDING_RADIUS + 4, 3)
        pygame.draw.circle(screen, (20, 20, 28), (x, y), BUILDING_RADIUS, 2)

        gar_real = int(b["garrison"][i]) // C.SCALE
        label = big_font.render(str(gar_real), True, COLOR_TEXT)
        screen.blit(label, label.get_rect(center=(x, y)))

    # HUD.
    hud_y = 6
    lines = [
        f"Tick {state.tick}/{C.GAME_TIMEOUT_TICKS}   Send %: {send_pct}   (1/2/3/4 to change)",
        status,
    ]
    for line in lines:
        surf = font.render(line, True, COLOR_TEXT)
        screen.blit(surf, (10, hud_y))
        hud_y += 20


def draw_end_overlay(screen, big_font, font, state):
    overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    screen.blit(overlay, (0, 0))

    if state.phase == C.PHASE_P1_WINS:
        msg = "You win!"
    elif state.phase == C.PHASE_P2_WINS:
        msg = "AI wins"
    else:
        msg = "Draw"

    text = big_font.render(msg, True, COLOR_TEXT)
    screen.blit(text, text.get_rect(center=(WIN_W // 2, WIN_H // 2 - 20)))
    sub = font.render("Press R to restart, Esc to quit", True, COLOR_MUTED)
    screen.blit(sub, sub.get_rect(center=(WIN_W // 2, WIN_H // 2 + 20)))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    pygame.init()
    pygame.display.set_caption("Mushroom Wars v2 — Play")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo,Consolas,monospace", 14)
    big_font = pygame.font.SysFont("Menlo,Consolas,monospace", 22, bold=True)

    rng = np.random.default_rng()
    state = levels.reset("crossroads_6")

    selected_src: int | None = None
    pct_options = list(C.SEND_PERCENTAGES)  # (25, 50, 75, 100)
    pct_idx = 3                              # default 100%
    pending_p1: Action | None = None
    last_tick_wall = time.perf_counter()
    status = "Click a blue building, then a target."
    prev_groups = state.unit_groups.copy()

    running = True
    while running:
        now = time.perf_counter()
        tick_progress = min(1.0, (now - last_tick_wall) / TICK_SECONDS)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    state = levels.reset("crossroads_6")
                    selected_src = None
                    pending_p1 = None
                    prev_groups = state.unit_groups.copy()
                    last_tick_wall = time.perf_counter()
                    status = "Click a blue building, then a target."
                elif ev.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    pct_idx = {pygame.K_1: 0, pygame.K_2: 1,
                               pygame.K_3: 2, pygame.K_4: 3}[ev.key]
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if state.phase != C.PHASE_PLAYING:
                    continue
                mx, my = ev.pos
                hit = building_at(state, mx, my)
                if hit is None:
                    selected_src = None
                    continue
                b = state.buildings
                if selected_src is None:
                    if int(b["owner"][hit]) == C.OWNER_P1:
                        selected_src = hit
                        status = f"Source slot {hit}. Click a target."
                elif hit == selected_src:
                    selected_src = None
                    status = "Click a blue building, then a target."
                else:
                    pending_p1 = Action(
                        kind="send",
                        type_idx=pct_idx,
                        src=selected_src,
                        tgt=hit,
                    )
                    status = f"Queued: {pct_options[pct_idx]}% from {selected_src} → {hit}"
                    selected_src = None

        # Tick the sim once per TICK_SECONDS of wall time.
        if state.phase == C.PHASE_PLAYING and (now - last_tick_wall) >= TICK_SECONDS:
            action_p2 = None
            # AI decides every DECISION_INTERVAL_TICKS.
            if state.tick % C.DECISION_INTERVAL_TICKS == 0:
                action_p2 = ai_action(state, C.OWNER_P2, rng)
            prev_groups = state.unit_groups.copy()
            step_tick(state, pending_p1, action_p2)
            pending_p1 = None
            last_tick_wall = now
            tick_progress = 0.0

        draw_state(screen, font, big_font, state, selected_src,
                   pct_options[pct_idx], tick_progress, status, prev_groups)
        if state.phase != C.PHASE_PLAYING:
            draw_end_overlay(screen, big_font, font, state)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
