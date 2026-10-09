import math
import tkinter as tk
from tkinter import ttk
import time

import numpy as np


class Controller:
    def __init__(self, stage, net):
        self.stage = stage
        self.net = net
        self.root = tk.Tk()
        self.root.title("Stage Controller")

        # --- Stage canvas ---
        self.canvas_size = 500
        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="white",
        )
        self.canvas.pack(padx=10, pady=10)

        # --- Time control slider ---
        self.time_scale = tk.DoubleVar(value=1.0)
        slider_frame = tk.Frame(self.root)
        slider_frame.pack(fill="x", padx=10, pady=(0, 10))

        tk.Label(slider_frame, text="Time scale:").pack(side="left")
        self.slider = ttk.Scale(
            slider_frame,
            from_=0.0,
            to=5.0,
            orient="horizontal",
            variable=self.time_scale,
            command=self._on_slider,
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=5)

        self.slider_label = tk.Label(slider_frame, text="1.00x", width=6)
        self.slider_label.pack(side="left")

        # --- Keyboard state ---
        self.keys = set()
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.focus_set()

        # --- Timing ---
        self.last_time = time.perf_counter()

    # ------------------------------------------------------------------
    def _on_slider(self, _value):
        self.slider_label.config(text=f"{self.time_scale.get():.2f}x")

    def _on_key_press(self, event):
        self.keys.add(event.keysym.lower())

    def _on_key_release(self, event):
        self.keys.discard(event.keysym.lower())

    # ------------------------------------------------------------------
    def handle_keys(self):
        if "r" in self.keys:
            self.stage.player.x = 0.5
            self.stage.player.y = 0.5

    def _get_input_vector(self):
        """Returns (dx, dy) from WASD input."""
        dx = dy = 0
        if "a" in self.keys:
            dx -= 0.01
        if "d" in self.keys:
            dx += 0.01
        if "w" in self.keys:
            dy -= 0.01
        if "s" in self.keys:
            dy += 0.01
        return dx, dy

    # ------------------------------------------------------------------
    def _draw(self):
        self.canvas.delete("all")
        s = self.canvas_size

        # Player
        p = self.stage.player
        r = p.radius * self.canvas_size
        self.canvas.create_oval(
            p.x * s - r, p.y * s - r,
            p.x * s + r, p.y * s + r,
            fill="blue", outline="",
            )

        # targets
        for o in self.stage.targets:
            r = o.radius * self.canvas_size
            self.canvas.create_oval(
                o.x * s - r, o.y * s - r,
                o.x * s + r, o.y * s + r,
                fill="red", outline="",
                )

    # ------------------------------------------------------------------
    def _loop(self):
        now = time.perf_counter()
        real_delta = now - self.last_time
        self.last_time = now

        self.handle_keys()

        # Scale delta by the time slider; clamp to avoid huge jumps
        scaled_delta = real_delta * self.time_scale.get()
        scaled_delta = min(scaled_delta, 0.1)

        # Apply input and advance physics

        inp = self.stage.getData()
        vec = self.net.forward(inp)
        acts = []
        for i in range(len(vec[-1])//2):
            expected = vec[-1][i*2]
            deviation = vec[-1][i*2+1]
            expected = math.tanh(expected)
            deviation = math.tanh(deviation)/2+0.5

            acts.append(np.random.normal(expected, deviation))

        self.stage.move(tuple(i/10 for i in acts))



        # Simple fixed-step ticks per frame (adjust as needed)
        self.stage.tick(scaled_delta if False else 0.25)

        self.stage.collide()

        self._draw()
        self.root.after(30, self._loop)  # ~60 FPS

    # ------------------------------------------------------------------
    def run(self):
        self._loop()
        self.root.mainloop()

