"""Core point-cloud operations used by the calibration interface."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

MINIMUM_POINT_COUNT = 3


@dataclass(frozen=True)
class CalibrationResult:
    """Serializable result of a manually confirmed endpoint calibration."""

    input_file: str
    point_count: int
    contact_point: list[float]
    other_endpoint_candidate: list[float]
    distance_between_endpoint_candidates: float
    mode: str = "direct_raw_scanned_endpoint"
    contact_point_is_original_scan_point: bool = True
    notes: str = (
        "接触点直接取自原始扫描数据，未进行杆轴线、圆柱或球面拟合。"
        "两个端点候选由欧氏距离双向搜索生成，最终端点由人工确认。"
    )

    @classmethod
    def from_selection(
        cls,
        input_path: Path,
        points: FloatArray,
        selected_index: int,
        other_index: int,
    ) -> "CalibrationResult":
        """Build a result from two validated point indices."""
        if selected_index == other_index:
            raise ValueError("接触点和另一端候选点不能相同。")
        for index in (selected_index, other_index):
            if not 0 <= index < len(points):
                raise IndexError(f"点索引超出范围: {index}")

        contact_point = points[selected_index]
        other_point = points[other_index]
        return cls(
            input_file=str(input_path),
            point_count=len(points),
            contact_point=contact_point.tolist(),
            other_endpoint_candidate=other_point.tolist(),
            distance_between_endpoint_candidates=float(
                np.linalg.norm(contact_point - other_point)
            ),
        )

    def write_json(self, output_path: Path) -> None:
        """Write the result as readable UTF-8 JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def load_points(path: Path) -> FloatArray:
    """Load finite XYZ coordinates from a whitespace-delimited text file.

    Extra columns are ignored. Rows containing NaN or infinity are removed.
    """
    if not path.is_file():
        raise FileNotFoundError(f"点云文件不存在: {path}")

    try:
        loaded = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except ValueError as error:
        raise ValueError(f"无法解析点云文件 {path}: {error}") from error

    if loaded.shape[1] < 3:
        raise ValueError("点云数据每行必须至少包含 X、Y、Z 三个数值。")

    points = np.ascontiguousarray(loaded[:, :3])
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < MINIMUM_POINT_COUNT:
        raise ValueError(
            f"至少需要 {MINIMUM_POINT_COUNT} 个有效点，当前只有 {len(points)} 个。"
        )
    return points


def endpoint_candidates(points: FloatArray) -> tuple[int, int]:
    """Return original point indices approximating opposite cloud endpoints.

    The two-sweep farthest-point search is linear in the number of points and
    preserves the exact input coordinates for the final manual selection.
    """
    _validate_point_array(points)
    center = points.mean(axis=0)
    first = _farthest_index(points, center)
    second = _farthest_index(points, points[first])
    first = _farthest_index(points, points[second])
    if first == second:
        raise ValueError("点云没有足够的空间范围，无法生成两个端点候选。")
    return first, second


def display_basis(points: FloatArray) -> FloatArray:
    """Return an orthonormal basis whose first axis follows the point cloud."""
    _validate_point_array(points)
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered
    _, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, -1]
    direction /= np.linalg.norm(direction)

    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(direction @ helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    second = np.cross(direction, helper)
    second /= np.linalg.norm(second)
    third = np.cross(direction, second)
    return np.column_stack((direction, second, third))


def _farthest_index(points: FloatArray, origin: FloatArray) -> int:
    squared_distances = np.einsum("ij,ij->i", points - origin, points - origin)
    return int(np.argmax(squared_distances))


def _validate_point_array(points: FloatArray) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("点云数组形状必须为 (N, 3)。")
    if len(points) < MINIMUM_POINT_COUNT:
        raise ValueError(f"点云至少需要 {MINIMUM_POINT_COUNT} 个点。")
    if not np.isfinite(points).all():
        raise ValueError("点云数组不能包含 NaN 或无穷值。")
