import math
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

import numpy as np


def _default_weights_path(inp_outp_size):
    """Fixed location for the network weights (next to this file if possible)."""
    name = f"stage_weights_{inp_outp_size[0]}-{inp_outp_size[1]}.npz"
    try:
        return Path(__file__).resolve().parent / name
    except NameError:  # e.g. running from an interactive session
        return Path.cwd() / name


class Controller:
    def __init__(self, stage, net):
        self.stage = stage
        self.net = net
        self.weights_path = _default_weights_path(net.getDimensions())

        self.root = tk.Tk()
        self.root.title("Stage Controller")

        # The play field is a 1x1 square. Keep a margin around it so its
        # borders (and anything slightly outside them) are visible.
        self.margin = 50
        self.field_size = 500
        self.canvas_size = self.field_size + 2 * self.margin

        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="white",
            highlightthickness=0,
        )
        self.canvas.pack(padx=10, pady=10)

        # --- Controls ---
        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=10, pady=(0, 8))

        self.control_mode = tk.StringVar(value="AI")
        self.mode_button = ttk.Button(
            controls, text="Control: AI", command=self._toggle_control_mode
        )
        self.mode_button.pack(side="left", padx=(0, 6))

        ttk.Button(controls, text="Save weights", command=self._save_weights).pack(
            side="left", padx=3
        )
        ttk.Button(controls, text="Load weights", command=self._load_weights).pack(
            side="left", padx=3
        )

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

        # --- Status line (shows where weights are stored) ---
        self.status = tk.Label(
            self.root,
            text=f"Weights file: {self.weights_path}",
            anchor="w",
            fg="#444444",
        )
        self.status.pack(fill="x", padx=10, pady=(0, 8))

        # --- Keyboard state ---
        self.keys = set()
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.focus_set()

        self.last_time = time.perf_counter()



    def _on_slider(self, _value):
        self.slider_label.config(text=f"{self.time_scale.get():.2f}x")

    def _on_key_press(self, event):
        self.keys.add(event.keysym.lower())

    def _on_key_release(self, event):
        self.keys.discard(event.keysym.lower())

    def _toggle_control_mode(self):
        new_mode = "Manual" if self.control_mode.get() == "AI" else "AI"
        self.control_mode.set(new_mode)
        self.mode_button.config(text=f"Control: {new_mode}")
        self.keys.clear()
        self.root.focus_set()

    def _save_weights(self):
        """Save to the fixed weights file (no file chooser)."""
        try:
            # Store each array separately; loading does not require pickle.
            payload = {"weight_count": np.array(len(self.net.weights))}
            for i, weight in enumerate(self.net.weights):
                payload[f"weight_{i}"] = np.asarray(weight)
            for i, bias in enumerate(self.net.biases):
                payload[f"bias_{i}"] = np.asarray(bias)
            payload["bias_count"] = np.array(len(self.net.biases))
            np.savez_compressed(self.weights_path, **payload)
        except (OSError, ValueError, TypeError) as exc:
            messagebox.showerror(
                "Save failed",
                f"Could not save weights to:\n{self.weights_path}\n\n{exc}",
                parent=self.root,
            )
            return

        self.status.config(text=f"Saved weights to {self.weights_path}")

    def _load_weights(self):
        """Load from the fixed weights file (no file chooser)."""
        if not self.weights_path.exists():
            messagebox.showerror(
                "Load failed",
                f"No weights file found at:\n{self.weights_path}",
                parent=self.root,
            )
            return

        try:
            with np.load(self.weights_path, allow_pickle=False) as data:
                weight_count = int(data["weight_count"])
                bias_count = int(data["bias_count"])
                if weight_count != len(self.net.weights) or bias_count != len(self.net.biases):
                    raise ValueError("The file has a different number of network layers.")

                weights = [data[f"weight_{i}"].copy() for i in range(weight_count)]
                biases = [data[f"bias_{i}"].copy() for i in range(bias_count)]

            # Validate everything before changing the live network.
            self.net.applyWeights(weights, biases)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            messagebox.showerror(
                "Load failed",
                f"Could not load weights from:\n{self.weights_path}\n\n{exc}",
                parent=self.root,
            )
            return

        self.status.config(text=f"Loaded weights from {self.weights_path}")

    def handle_keys(self):
        if "r" in self.keys:
            self.stage.player.reset()
            print(self.stage.getData())

    def _get_input_vector(self):
        """Returns (dx, dy) from WASD input."""
        dx = dy = 0
        if "a" in self.keys:
            dx -= 0.07
        if "d" in self.keys:
            dx += 0.07
        if "w" in self.keys:
            dy -= 0.07
        if "s" in self.keys:
            dy += 0.07
        return dx, dy

    def _draw(self):
        self.canvas.delete("all")
        m = self.margin
        f = self.field_size

        # Border of the 1x1 play field.
        self.canvas.create_rectangle(
            m, m, m + f, m + f,
            outline="black", width=2,
                  )

        p = self.stage.player
        r = p.radius * f
        self.canvas.create_oval(
            m + p.x * f - r, m + p.y * f - r,
            m + p.x * f + r, m + p.y * f + r,
            fill="blue", outline="",
            )

        for o in self.stage.targets:
            r = o.radius * f
            self.canvas.create_oval(
                m + o.x * f - r, m + o.y * f - r,
                m + o.x * f + r, m + o.y * f + r,
                fill="red", outline="",
                )

    def _ai_action(self):
        inp = self.stage.getData()
        vec = self.net.forward(inp)
        acts = []
        for i in range(len(vec[-1]) // 2):
            expected_raw = vec[-1][i * 2]
            deviation_raw = vec[-1][i * 2 + 1]
            expected_real = math.tanh(expected_raw)
            deviation_real = math.tanh(deviation_raw) / 2 + 0.5
            acts.append(np.random.normal(expected_real, deviation_real))
        self.stage.move(tuple(value / 6 for value in acts))

    def _loop(self):
        now = time.perf_counter()
        real_delta = now - self.last_time
        self.last_time = now

        self.handle_keys()

        if self.control_mode.get() == "AI":
            self._ai_action()
        else:
            # Manual mode uses WASD. The stage expects an acceleration vector.
            self.stage.move(self._get_input_vector())

        self.stage.tick(0.25)
        self.stage.collide()
        self._draw()

        # Guard against a zero time scale (avoids division by zero).
        delay = max(1, int(30 / max(self.time_scale.get(), 0.01)))
        self.root.after(delay, self._loop)

    def run(self):
        self._loop()
        self.root.mainloop()