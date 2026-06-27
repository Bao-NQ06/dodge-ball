"""Minimal pygame renderer. Works headless (rgb_array) or with display (human)."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

# Use SDL2 dummy driver unless user explicitly wants display.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from dodgeball.env.entities import Arena, Agent, Ball  # noqa: E402


class PygameRenderer:
    def __init__(self, arena_w: float, arena_h: float,
                 scale: float = 1.0):
        pygame.init()
        self.w = int(arena_w * scale)
        self.h = int(arena_h * scale)
        self.surface = pygame.Surface((self.w, self.h))
        self.font = pygame.font.SysFont("monospace", 14)
        self.clock = pygame.time.Clock()

    def render(self, agents, ball: Ball, step: int, max_steps: int):
        self.surface.fill((245, 240, 230))
        # center line
        cx = self.w // 2
        pygame.draw.line(self.surface, (180, 180, 180), (cx, 0), (cx, self.h), 2)
        # walls
        pygame.draw.rect(self.surface, (90, 90, 90), self.surface.get_rect(), 3)
        # agents
        for i, ag in enumerate(agents):
            color = (220, 60, 60) if i == 0 else (60, 120, 220)
            pygame.draw.circle(self.surface, color,
                               (int(ag.x), int(ag.y)), int(ag.radius))
        # ball
        pygame.draw.circle(self.surface, (40, 40, 40),
                           (int(ball.x), int(ball.y)), int(ball.radius))
        # text
        text = f"step {step}/{max_steps}"
        self.surface.blit(self.font.render(text, True, (0, 0, 0)), (10, 10))
        # return rgb array
        return pygame.surfarray.array3d(self.surface).swapaxes(0, 1).copy()

    def close(self):
        pygame.quit()