
"""
Forest RL GUI (Pygame) — Integrated with radar_adapter
------------------------------------------------------
- Loads your DQN once via radar_adapter.load_agent(...)
- Uses radar_adapter.policy_action(...) each step
- Maps "SMELL" to "STAY" (GUI doesn't implement smell mechanics yet)
- Default grid set to 10x10 (adjust if your training used a different size)

Controls:
  SPACE -> Play/Pause
  N     -> Step one turn (when paused)
  R     -> Reset episode
  +/-   -> Adjust simulation speed (steps/sec)
  M     -> Toggle animal controller: Model vs Random
  H     -> Toggle hunter policy: Greedy vs Random
  ESC/Q -> Quit
"""

import random
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import pygame

# ---- Model adapter import ----
from radar_adapter import load_agent, policy_action

# ----------------------------
# Config
# ----------------------------
GRID_ROWS = 10   # Adjust to your training grid
GRID_COLS = 10
CELL_SIZE = 42
MARGIN = 3  # gap between cells (visual)

TOP_BAR_H = 60
GRID_PADDING = 24
SIDEBAR_W = 300

GRID_W = GRID_COLS * (CELL_SIZE + MARGIN) + MARGIN
GRID_H = GRID_ROWS * (CELL_SIZE + MARGIN) + MARGIN

W = GRID_PADDING * 2 + GRID_W + SIDEBAR_W
H = TOP_BAR_H + GRID_PADDING * 2 + GRID_H

FPS_DEFAULT = 8       # simulation steps per second
RENDER_FPS = 60        # screen refresh rate (drives smooth animation)
MAX_TURNS = 50

# ----------------------------
# Palette
# ----------------------------
BG_COLOR = (16, 18, 22)
TOP_BAR_COLOR = (22, 25, 30)
PANEL_BG = (26, 29, 35)
PANEL_BORDER = (44, 48, 56)

GRID_CELL_A = (32, 36, 43)
GRID_CELL_B = (36, 40, 48)
GRID_BORDER = (48, 53, 62)

TEXT_PRIMARY = (235, 238, 242)
TEXT_MUTED = (145, 152, 163)
TEXT_FAINT = (95, 101, 112)

ANIMAL_COLOR = (96, 226, 150)
ANIMAL_GLOW = (96, 226, 150, 40)
HUNTER_COLOR = (235, 96, 96)
HUNTER_GLOW = (235, 96, 96, 35)

ACCENT = (240, 196, 80)
WIN_COLOR = (96, 226, 150)
LOSE_COLOR = (235, 96, 96)
OVERLAY_BG = (10, 11, 14, 180)

BADGE_ON_BG = (44, 92, 68)
BADGE_ON_FG = (140, 240, 180)
BADGE_OFF_BG = (58, 46, 30)
BADGE_OFF_FG = (235, 178, 96)

# Actions
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
DELTA = {
    "UP":    (-1, 0),
    "DOWN":  (1, 0),
    "LEFT":  (0, -1),
    "RIGHT": (0, 1),
    "STAY":  (0, 0),
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass
class ForestEnv:
    rows: int = GRID_ROWS
    cols: int = GRID_COLS
    max_turns: int = MAX_TURNS
    animal: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    hunters: List[Tuple[int, int]] = field(default_factory=list)
    turn: int = 0
    done: bool = False
    winner: str = ""          # "animal" ou "hunters"
    warmup_turns: int = 4     # chasseurs aléatoires pendant les 4 premiers tours

    def reset(self):
        """Réinitialise l'épisode avec un spawn sécurisé (distance mini)."""
        self.turn = 0
        self.done = False
        self.winner = ""

        def rand_cell():
            return (random.randint(0, self.rows - 1), random.randint(0, self.cols - 1))

        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        MIN_DIST = 4  # distance de sécurité proie ↔ chasseurs
        tries = 0
        while True:
            tries += 1
            self.animal = rand_cell()
            used = {self.animal}
            self.hunters = []
            ok = True

            for _ in range(3):
                # cherche une case loin de l'animal
                placed = False
                for _ in range(200):
                    h = rand_cell()
                    if h in used:
                        continue
                    if manhattan(h, self.animal) >= MIN_DIST:
                        self.hunters.append(h)
                        used.add(h)
                        placed = True
                        break
                if not placed:
                    ok = False
                    break

            if ok:
                break
            if tries > 200:
                # fallback si la grille est trop petite
                self.hunters = [rand_cell() for _ in range(3)]
                break

        return self._get_state()

    def _move(self, pos: Tuple[int, int], action: str) -> Tuple[int, int]:
        dr, dc = DELTA.get(action, (0, 0))
        r = clamp(pos[0] + dr, 0, self.rows - 1)
        c = clamp(pos[1] + dc, 0, self.cols - 1)
        return (r, c)

    def _greedy_toward(self, hunter: Tuple[int, int], target: Tuple[int, int]) -> str:
        hr, hc = hunter
        tr, tc = target
        d_r = tr - hr
        d_c = tc - hc
        if abs(d_r) > abs(d_c):
            return "DOWN" if d_r > 0 else "UP"
        elif abs(d_c) > 0:
            return "RIGHT" if d_c > 0 else "LEFT"
        else:
            return "STAY"

    def _get_state(self) -> Dict:
        return {
            "grid_size": (self.rows, self.cols),
            "turn": self.turn,
            "max_turns": self.max_turns,
            "animal": self.animal,
            "hunters": list(self.hunters),
            "done": self.done,
            "winner": self.winner,
        }

    def step(self, animal_action: str, hunter_policy: str = "greedy"):
        if self.done:
            return self._get_state()

        # --- Animal moves first
        self.animal = self._move(self.animal, animal_action)

        # Check capture after animal move
        if any(h == self.animal for h in self.hunters):
            self.done = True
            self.winner = "hunters"
            return self._get_state()

        # --- Hunters move (warmup: random pendant les premiers tours)
        new_hunters = []
        for h in self.hunters:
            if self.turn < self.warmup_turns:
                action = random.choice(ACTIONS)
            else:
                if hunter_policy == "greedy":
                    action = self._greedy_toward(h, self.animal)
                else:
                    action = random.choice(ACTIONS)
            new_h = self._move(h, action)
            new_hunters.append(new_h)
        self.hunters = new_hunters

        # Check capture after hunters move
        if any(h == self.animal for h in self.hunters):
            self.done = True
            self.winner = "hunters"
        else:
            self.turn += 1
            if self.turn >= self.max_turns:
                self.done = True
                self.winner = "animal"

        return self._get_state()


# ----------------------------
# Model integration (uses radar_adapter)
# ----------------------------
_MODEL = None

def _get_model(input_size=64, weights_path='prey_dqn_model.pth'):
    """Load once and reuse."""
    global _MODEL
    if _MODEL is None:
        _MODEL = load_agent(weights_path, input_size)
    return _MODEL

def get_action_from_model(state: Dict) -> str:
    """
    Calls your DQN via radar_adapter.policy_action.
    Maps 'SMELL' -> 'STAY' since smell isn't simulated in this GUI (yet).
    """
    agent = _get_model(input_size=27)
    action = policy_action(agent, state, remaining_moves=2, sight_radius=3)
    return "STAY" if action == "SMELL" else action


# ----------------------------
# Small drawing helpers
# ----------------------------

def draw_round_rect(surface, rect, color, radius=10, width=0):
    pygame.draw.rect(surface, color, rect, width=width, border_radius=radius)


def draw_badge(surface, font, center_x, y, text, bg, fg, pad_x=12, pad_y=6):
    label = font.render(text, True, fg)
    w = label.get_width() + pad_x * 2
    h = label.get_height() + pad_y * 2
    rect = pygame.Rect(0, 0, w, h)
    rect.centerx = center_x
    rect.top = y
    draw_round_rect(surface, rect, bg, radius=h // 2)
    surface.blit(label, label.get_rect(center=rect.center))
    return rect.height


def draw_glow_circle(surface, color_rgba, center, radius):
    glow = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(glow, color_rgba, (radius, radius), radius)
    surface.blit(glow, (center[0] - radius, center[1] - radius))


# ----------------------------
# GUI
# ----------------------------
class ForestGUI:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Forest RL — Animal vs Hunters")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("segoeui", 24, bold=True)
        self.font_section = pygame.font.SysFont("segoeui", 15, bold=True)
        self.font_body = pygame.font.SysFont("segoeui", 14)
        self.font_small = pygame.font.SysFont("consolas", 12)
        self.font_badge = pygame.font.SysFont("segoeui", 13, bold=True)
        self.font_banner = pygame.font.SysFont("segoeui", 34, bold=True)

        self.grid_origin = (GRID_PADDING, TOP_BAR_H + GRID_PADDING)

        self.env = ForestEnv()
        self.state = self.env.reset()
        self.prev_state = self.state

        self.paused = True
        self.fps = FPS_DEFAULT
        self.use_model = True
        self.hunter_policy = "greedy"  # or "random"

        self._accumulator = 0.0
        self.step_progress = 1.0

    # ---- coordinate helpers ----
    def cell_pixel(self, rc: Tuple[int, int]) -> Tuple[int, int]:
        r, c = rc
        ox, oy = self.grid_origin
        x = ox + c * (CELL_SIZE + MARGIN)
        y = oy + r * (CELL_SIZE + MARGIN)
        return x, y

    def interp_pixel(self, prev_rc, cur_rc, t) -> Tuple[float, float]:
        px, py = self.cell_pixel(prev_rc)
        cx, cy = self.cell_pixel(cur_rc)
        return lerp(px, cx, t), lerp(py, cy, t)

    # ---- drawing ----
    def draw_top_bar(self):
        pygame.draw.rect(self.screen, TOP_BAR_COLOR, (0, 0, W, TOP_BAR_H))
        pygame.draw.line(self.screen, PANEL_BORDER, (0, TOP_BAR_H), (W, TOP_BAR_H), 1)

        title = self.font_title.render("FOREST RL", True, TEXT_PRIMARY)
        self.screen.blit(title, (GRID_PADDING, TOP_BAR_H // 2 - title.get_height() // 2))

        subtitle = self.font_body.render("prey vs hunters", True, TEXT_FAINT)
        self.screen.blit(subtitle, (GRID_PADDING + title.get_width() + 12,
                                     TOP_BAR_H // 2 - subtitle.get_height() // 2 + 2))

        state_txt = "RUNNING" if not self.paused else "PAUSED"
        state_color = ACCENT if self.paused else ANIMAL_COLOR
        dot_r = 5
        dot_x = W - GRID_PADDING - 120
        pygame.draw.circle(self.screen, state_color, (dot_x, TOP_BAR_H // 2), dot_r)
        state_label = self.font_section.render(state_txt, True, TEXT_PRIMARY)
        self.screen.blit(state_label, (dot_x + 12, TOP_BAR_H // 2 - state_label.get_height() // 2))

    def draw_grid(self):
        ox, oy = self.grid_origin
        panel_rect = pygame.Rect(ox - 8, oy - 8, GRID_W + 16, GRID_H + 16)
        draw_round_rect(self.screen, panel_rect, PANEL_BG, radius=14)
        draw_round_rect(self.screen, panel_rect, PANEL_BORDER, radius=14, width=1)

        for r in range(self.env.rows):
            for c in range(self.env.cols):
                x, y = self.cell_pixel((r, c))
                color = GRID_CELL_A if (r + c) % 2 == 0 else GRID_CELL_B
                draw_round_rect(self.screen, (x, y, CELL_SIZE, CELL_SIZE), color, radius=8)

    def draw_sight_radius(self, animal_px, sight_cells=3):
        radius_px = int(sight_cells * (CELL_SIZE + MARGIN))
        cx = int(animal_px[0] + CELL_SIZE / 2)
        cy = int(animal_px[1] + CELL_SIZE / 2)
        draw_glow_circle(self.screen, (*ANIMAL_COLOR, 18), (cx, cy), radius_px)

    def draw_token(self, px, color, glow_color, inset=7, scale=1.0):
        x, y = px
        size = (CELL_SIZE - 2 * inset) * scale
        cx = x + CELL_SIZE / 2
        cy = y + CELL_SIZE / 2
        draw_glow_circle(self.screen, glow_color, (int(cx), int(cy)), int(CELL_SIZE * 0.9))
        rect = pygame.Rect(0, 0, size, size)
        rect.center = (cx, cy)
        draw_round_rect(self.screen, rect, color, radius=int(size * 0.32))

    def draw_entities(self, t):
        prev_animal = self.prev_state["animal"]
        cur_animal = self.state["animal"]
        animal_px = self.interp_pixel(prev_animal, cur_animal, t)

        self.draw_sight_radius(animal_px)
        self.draw_token(animal_px, ANIMAL_COLOR, ANIMAL_GLOW, inset=7)

        prev_hunters = self.prev_state["hunters"]
        cur_hunters = self.state["hunters"]
        for i, h in enumerate(cur_hunters):
            prev_h = prev_hunters[i] if i < len(prev_hunters) else h
            hx = self.interp_pixel(prev_h, h, t)
            self.draw_token(hx, HUNTER_COLOR, HUNTER_GLOW, inset=10)

    def draw_sidebar(self):
        sx = GRID_PADDING * 2 + GRID_W
        sy = TOP_BAR_H
        sw = SIDEBAR_W
        sh = H - TOP_BAR_H

        pygame.draw.line(self.screen, PANEL_BORDER, (sx - GRID_PADDING, sy), (sx - GRID_PADDING, H), 1)

        pad = 22
        x = sx
        y = sy + pad

        # --- Turn progress ---
        header = self.font_section.render("EPISODE", True, TEXT_MUTED)
        self.screen.blit(header, (x, y))
        y += header.get_height() + 8

        turn_txt = self.font_body.render(
            f"Turn {self.state['turn']} / {self.state['max_turns']}", True, TEXT_PRIMARY)
        self.screen.blit(turn_txt, (x, y))
        y += turn_txt.get_height() + 6

        bar_w = sw - pad * 2
        bar_rect = pygame.Rect(x, y, bar_w, 10)
        draw_round_rect(self.screen, bar_rect, PANEL_BG, radius=5)
        frac = clamp(self.state["turn"] / max(1, self.state["max_turns"]), 0, 1)
        if frac > 0:
            fill_rect = pygame.Rect(x, y, max(10, int(bar_w * frac)), 10)
            draw_round_rect(self.screen, fill_rect, ACCENT, radius=5)
        y += 26

        # --- Speed ---
        speed_txt = self.font_body.render(f"Speed: {self.fps} steps/s", True, TEXT_MUTED)
        self.screen.blit(speed_txt, (x, y))
        y += speed_txt.get_height() + 18

        # --- Mode badges ---
        header2 = self.font_section.render("SETTINGS", True, TEXT_MUTED)
        self.screen.blit(header2, (x, y))
        y += header2.get_height() + 10

        row_label = self.font_body.render("Animal controller", True, TEXT_PRIMARY)
        self.screen.blit(row_label, (x, y))
        bg, fg = (BADGE_ON_BG, BADGE_ON_FG) if self.use_model else (BADGE_OFF_BG, BADGE_OFF_FG)
        badge_h = draw_badge(self.screen, self.font_badge, x + sw - pad * 2 - 40, y - 4,
                              "MODEL" if self.use_model else "RANDOM", bg, fg)
        y += max(row_label.get_height(), badge_h) + 12

        row_label2 = self.font_body.render("Hunter policy", True, TEXT_PRIMARY)
        self.screen.blit(row_label2, (x, y))
        bg2, fg2 = (BADGE_ON_BG, BADGE_ON_FG) if self.hunter_policy == "greedy" else (BADGE_OFF_BG, BADGE_OFF_FG)
        badge_h2 = draw_badge(self.screen, self.font_badge, x + sw - pad * 2 - 40, y - 4,
                               self.hunter_policy.upper(), bg2, fg2)
        y += max(row_label2.get_height(), badge_h2) + 22

        # --- Legend ---
        header3 = self.font_section.render("LEGEND", True, TEXT_MUTED)
        self.screen.blit(header3, (x, y))
        y += header3.get_height() + 10

        pygame.draw.circle(self.screen, ANIMAL_COLOR, (x + 7, y + 7), 7)
        lbl = self.font_body.render("Animal (prey)", True, TEXT_PRIMARY)
        self.screen.blit(lbl, (x + 22, y))
        y += lbl.get_height() + 10

        pygame.draw.circle(self.screen, HUNTER_COLOR, (x + 7, y + 7), 7)
        lbl2 = self.font_body.render("Hunter", True, TEXT_PRIMARY)
        self.screen.blit(lbl2, (x + 22, y))
        y += lbl2.get_height() + 24

        # --- Controls ---
        header4 = self.font_section.render("CONTROLS", True, TEXT_MUTED)
        self.screen.blit(header4, (x, y))
        y += header4.get_height() + 10

        controls = [
            ("SPACE", "play / pause"),
            ("N", "step once"),
            ("R", "reset episode"),
            ("+ / -", "change speed"),
            ("M", "toggle animal AI"),
            ("H", "toggle hunter AI"),
            ("ESC", "quit"),
        ]
        for key, desc in controls:
            key_lbl = self.font_small.render(key, True, ACCENT)
            self.screen.blit(key_lbl, (x, y))
            desc_lbl = self.font_small.render(desc, True, TEXT_FAINT)
            self.screen.blit(desc_lbl, (x + 62, y))
            y += desc_lbl.get_height() + 8

    def draw_game_over(self):
        if not self.state["done"]:
            return
        ox, oy = self.grid_origin
        overlay = pygame.Surface((GRID_W, GRID_H), pygame.SRCALPHA)
        overlay.fill(OVERLAY_BG)
        self.screen.blit(overlay, (ox, oy))

        won = self.state["winner"] == "animal"
        color = WIN_COLOR if won else LOSE_COLOR
        msg = "ANIMAL SURVIVED!" if won else "CAUGHT!"
        sub = "The prey outlasted every hunter." if won else "A hunter closed the distance."

        banner = self.font_banner.render(msg, True, color)
        banner_rect = banner.get_rect(center=(ox + GRID_W // 2, oy + GRID_H // 2 - 18))
        self.screen.blit(banner, banner_rect)

        sub_lbl = self.font_body.render(sub, True, TEXT_MUTED)
        sub_rect = sub_lbl.get_rect(center=(ox + GRID_W // 2, oy + GRID_H // 2 + 16))
        self.screen.blit(sub_lbl, sub_rect)

        hint = self.font_small.render("Press R to play again", True, TEXT_FAINT)
        hint_rect = hint.get_rect(center=(ox + GRID_W // 2, oy + GRID_H // 2 + 44))
        self.screen.blit(hint, hint_rect)

    def draw(self):
        self.screen.fill(BG_COLOR)
        self.draw_top_bar()
        self.draw_grid()
        t = 1.0 if self.paused or self.state["done"] else self.step_progress
        self.draw_entities(t)
        self.draw_game_over()
        self.draw_sidebar()
        pygame.display.flip()

    # ---- simulation ----
    def get_animal_action(self) -> str:
        if self.use_model:
            return get_action_from_model(self.state)
        else:
            return random.choice(ACTIONS)

    def step_once(self):
        if self.state["done"]:
            return
        self.prev_state = self.state
        a_action = self.get_animal_action()
        self.state = self.env.step(a_action, hunter_policy=self.hunter_policy)

    def reset(self):
        self.state = self.env.reset()
        self.prev_state = self.state
        self._accumulator = 0.0
        self.step_progress = 1.0

    def run(self):
        last_ticks = pygame.time.get_ticks()

        while True:
            now_ticks = pygame.time.get_ticks()
            dt = (now_ticks - last_ticks) / 1000.0
            last_ticks = now_ticks

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        pygame.quit()
                        sys.exit(0)
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                        self._accumulator = 0.0
                    elif event.key == pygame.K_n:
                        if self.paused:
                            self.step_once()
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        self.fps = min(60, self.fps + 1)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.fps = max(1, self.fps - 1)
                    elif event.key == pygame.K_m:
                        self.use_model = not self.use_model
                    elif event.key == pygame.K_h:
                        self.hunter_policy = "random" if self.hunter_policy == "greedy" else "greedy"

            step_interval = 1.0 / self.fps
            if not self.paused and not self.state["done"]:
                self._accumulator += dt
                while self._accumulator >= step_interval and not self.state["done"]:
                    self.step_once()
                    self._accumulator -= step_interval
                self.step_progress = clamp(self._accumulator / step_interval, 0.0, 1.0)
            else:
                self._accumulator = 0.0
                self.step_progress = 1.0

            self.draw()
            self.clock.tick(RENDER_FPS)


if __name__ == "__main__":
    ForestGUI().run()
