"""Tkinter interface for selecting a probe endpoint candidate."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageTk

from .core import FloatArray, display_basis

IntArray = NDArray[np.int32]

WINDOW_WIDTH = 1300
WINDOW_HEIGHT = 800
CANVAS_MARGIN = 100
MIN_CANVAS_SIZE = 300
RENDER_DELAY_MS = 25


class EndpointSelectionWindow:
    """Render a point cloud and let the user confirm one endpoint candidate."""

    def __init__(self, points: FloatArray, candidate_indices: tuple[int, int]) -> None:
        self._points = points
        self._candidate_indices = candidate_indices
        self._center = points.mean(axis=0)
        self._coordinates = (points - self._center) @ display_basis(points)

        self._yaw = 0.0
        self._pitch = 0.0
        self._roll = 0.0
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._drag_start: tuple[int, int] | None = None
        self._pan_start: tuple[int, int] | None = None
        self._render_pending = False
        self._point_image: ImageTk.PhotoImage | None = None
        self._last_rotation = np.eye(3)
        self._last_scale = 1.0
        self.selected_index: int | None = None

        self._root = tk.Tk()
        self._root.title("测杆球头前端标定")
        self._root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self._info = tk.StringVar(
            value="右键旋转、中键平移、滚轮缩放；左键点击球头最前端候选点。"
        )
        tk.Label(
            self._root,
            textvariable=self._info,
            font=("Microsoft YaHei", 12),
        ).pack(pady=8)
        self._build_controls()

        self._canvas = tk.Canvas(self._root, bg="white")
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        self._bind_events()
        self._root.after(150, self._render)

    def run(self) -> int:
        """Block until an endpoint is confirmed and return its point index."""
        self._root.mainloop()
        if self.selected_index is None:
            raise RuntimeError("用户取消了端点选择。")
        return self.selected_index

    def show_saved_result(self, output_name: str) -> None:
        """Keep the selected point visible after the result is saved."""
        point = self._points[self.selected_index]
        self._info.set(
            "已采用原始扫描点作为接触点："
            f"({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})；"
            f"结果已保存至 {output_name}。"
        )
        self._schedule_render()
        self._root.mainloop()

    def _build_controls(self) -> None:
        controls = tk.Frame(self._root)
        controls.pack(pady=(0, 5))
        buttons = (
            ("杆身侧视 1", lambda: self._set_view(0.0, 0.0, 0.0)),
            ("杆身侧视 2", lambda: self._set_view(0.0, 0.0, np.pi / 2)),
            ("重置视角", lambda: self._set_view(0.0, 0.0, 0.0)),
            ("重新选点", self._reset_selection),
        )
        for text, command in buttons:
            tk.Button(
                controls,
                text=text,
                command=command,
                font=("Microsoft YaHei", 10),
                width=13,
            ).pack(side=tk.LEFT, padx=4)

    def _bind_events(self) -> None:
        self._canvas.bind("<Button-1>", self._on_click)
        self._canvas.bind("<ButtonPress-3>", self._on_drag_start)
        self._canvas.bind("<B3-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-3>", self._on_drag_end)
        self._canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self._canvas.bind("<B2-Motion>", self._on_pan)
        self._canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        self._canvas.bind("<MouseWheel>", self._on_wheel)
        self._root.bind("<Configure>", lambda _event: self._schedule_render())

    def _rotation_matrix(self) -> FloatArray:
        cy, sy = np.cos(self._yaw), np.sin(self._yaw)
        cp, sp = np.cos(self._pitch), np.sin(self._pitch)
        cr, sr = np.cos(self._roll), np.sin(self._roll)
        yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        roll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
        return roll @ pitch @ yaw

    def _schedule_render(self) -> None:
        if not self._render_pending:
            self._render_pending = True
            self._root.after(RENDER_DELAY_MS, self._render)

    def _render(self) -> None:
        self._render_pending = False
        if not self._canvas.winfo_exists():
            return

        width = max(self._canvas.winfo_width(), MIN_CANVAS_SIZE)
        height = max(self._canvas.winfo_height(), MIN_CANVAS_SIZE)
        rotation = self._rotation_matrix()
        rotated = self._coordinates @ rotation.T
        scale = self._fit_scale(rotated, width, height) * self._zoom
        self._last_rotation = rotation
        self._last_scale = scale

        x_pixels, y_pixels = self._screen_pixels(rotated, width, height, scale)
        self._draw_point_cloud(x_pixels, y_pixels, width, height)
        self._draw_candidates(rotation, width, height, scale)
        self._canvas.create_text(
            12,
            12,
            text="右键拖动：旋转　中键拖动：平移　滚轮：缩放　左键：选择端点",
            anchor="nw",
            fill="#222",
            font=("Microsoft YaHei", 10),
        )

    def _fit_scale(self, rotated: FloatArray, width: int, height: int) -> float:
        span_x = max(float(np.ptp(rotated[:, 0])), 1e-9)
        span_y = max(float(np.ptp(rotated[:, 1])), 1e-9)
        return min((width - CANVAS_MARGIN) / span_x, (height - CANVAS_MARGIN) / span_y)

    def _screen_pixels(
        self, rotated: FloatArray, width: int, height: int, scale: float
    ) -> tuple[IntArray, IntArray]:
        screen_x = width / 2.0 + self._pan_x + rotated[:, 0] * scale
        screen_y = height / 2.0 + self._pan_y - rotated[:, 1] * scale
        x_pixels = np.clip(np.rint(screen_x).astype(np.int32), 0, width - 1)
        y_pixels = np.clip(np.rint(screen_y).astype(np.int32), 0, height - 1)
        return x_pixels, y_pixels

    def _draw_point_cloud(
        self, x_pixels: IntArray, y_pixels: IntArray, width: int, height: int
    ) -> None:
        density = np.zeros((height, width), dtype=np.uint32)
        np.add.at(density, (y_pixels, x_pixels), 1)
        occupied = density > 0
        rgb = np.full((height, width, 3), 255, dtype=np.uint8)
        rgb[occupied] = np.array([90, 135, 178], dtype=np.uint8)
        self._point_image = ImageTk.PhotoImage(Image.fromarray(rgb, mode="RGB"))
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, image=self._point_image, anchor="nw")

    def _draw_candidates(
        self, rotation: FloatArray, width: int, height: int, scale: float
    ) -> None:
        candidates = self._coordinates[list(self._candidate_indices)] @ rotation.T
        candidate_x = width / 2.0 + self._pan_x + candidates[:, 0] * scale
        candidate_y = height / 2.0 + self._pan_y - candidates[:, 1] * scale
        for order, (point_index, x_pos, y_pos) in enumerate(
            zip(self._candidate_indices, candidate_x, candidate_y, strict=True), start=1
        ):
            is_selected = point_index == self.selected_index
            radius = 14 if is_selected else 10
            fill = "#ff0000" if is_selected else "#ffb000"
            outline = "#770000" if is_selected else "#7a4b00"
            self._canvas.create_oval(
                x_pos - radius,
                y_pos - radius,
                x_pos + radius,
                y_pos + radius,
                fill=fill,
                outline=outline,
                width=3,
            )
            label = "接触点" if is_selected else f"端点候选 {order}"
            self._canvas.create_text(
                x_pos + radius + 4,
                y_pos - radius - 3,
                text=label,
                anchor="sw",
                fill="#b00000" if is_selected else "#754500",
                font=("Microsoft YaHei", 11, "bold"),
            )

    def _candidate_screen_positions(self) -> FloatArray:
        width = max(self._canvas.winfo_width(), MIN_CANVAS_SIZE)
        height = max(self._canvas.winfo_height(), MIN_CANVAS_SIZE)
        rotated = self._coordinates[list(self._candidate_indices)] @ self._last_rotation.T
        return np.column_stack(
            (
                width / 2.0 + self._pan_x + rotated[:, 0] * self._last_scale,
                height / 2.0 + self._pan_y - rotated[:, 1] * self._last_scale,
            )
        )

    def _on_click(self, event: tk.Event) -> None:
        if self.selected_index is not None:
            return
        screen_positions = self._candidate_screen_positions()
        distances = np.linalg.norm(screen_positions - np.array([event.x, event.y]), axis=1)
        choice = int(np.argmin(distances))
        point_index = self._candidate_indices[choice]
        point = self._points[point_index]
        message = (
            "将这个原始扫描点作为接触点吗？\n\n"
            f"X = {point[0]:.8f}\nY = {point[1]:.8f}\nZ = {point[2]:.8f}"
        )
        if messagebox.askyesno("确认最远扫描点", message):
            self.selected_index = point_index
            self._root.quit()

    def _reset_selection(self) -> None:
        self.selected_index = None
        self._info.set("右键旋转、中键平移、滚轮缩放；左键点击球头最前端候选点。")
        self._schedule_render()

    def _on_drag_start(self, event: tk.Event) -> None:
        self._drag_start = (event.x, event.y)

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        self._yaw += (event.x - self._drag_start[0]) * 0.008
        self._pitch += (event.y - self._drag_start[1]) * 0.008
        self._drag_start = (event.x, event.y)
        self._schedule_render()

    def _on_drag_end(self, _event: tk.Event) -> None:
        self._drag_start = None

    def _on_pan_start(self, event: tk.Event) -> None:
        self._pan_start = (event.x, event.y)

    def _on_pan(self, event: tk.Event) -> None:
        if self._pan_start is None:
            return
        self._pan_x += event.x - self._pan_start[0]
        self._pan_y += event.y - self._pan_start[1]
        self._pan_start = (event.x, event.y)
        self._schedule_render()

    def _on_pan_end(self, _event: tk.Event) -> None:
        self._pan_start = None

    def _on_wheel(self, event: tk.Event) -> None:
        self._zoom *= 1.12 if event.delta > 0 else 1.0 / 1.12
        self._zoom = float(np.clip(self._zoom, 0.25, 10.0))
        self._schedule_render()

    def _set_view(self, yaw: float, pitch: float, roll: float) -> None:
        self._yaw, self._pitch, self._roll = yaw, pitch, roll
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._schedule_render()

