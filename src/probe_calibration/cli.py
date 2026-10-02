"""Command-line entry point for interactive probe calibration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import CalibrationResult, endpoint_candidates, load_points
from .gui import EndpointSelectionWindow

DEFAULT_OUTPUT_NAME = "标定结果.json"
DEFAULT_INPUT_NAMES = ("测杆.extract", "测杆.extract.txt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从原始点云选择测杆球头端点")
    parser.add_argument("input", nargs="?", type=Path, help="每行至少三列的 XYZ 点云文本文件")
    parser.add_argument("--output", type=Path, help="输出 JSON 文件路径")
    return parser


def find_default_input(search_directory: Path) -> Path:
    """Find a conventionally named point-cloud file near the launcher."""
    for directory in (search_directory, search_directory.parent):
        for name in DEFAULT_INPUT_NAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    names = " 或 ".join(DEFAULT_INPUT_NAMES)
    raise FileNotFoundError(f"未找到 {names}；请通过命令行指定输入文件。")


def run(input_path: Path, output_path: Path) -> CalibrationResult:
    """Load points, collect the manual selection, and save a result."""
    resolved_input = input_path.expanduser().resolve()
    resolved_output = output_path.expanduser().resolve()
    points = load_points(resolved_input)
    candidates = endpoint_candidates(points)

    window = EndpointSelectionWindow(points, candidates)
    selected_index = window.run()
    other_index = candidates[1] if selected_index == candidates[0] else candidates[0]
    result = CalibrationResult.from_selection(
        resolved_input, points, selected_index, other_index
    )
    result.write_json(resolved_output)

    print("标定完成")
    print("接触点:", result.contact_point)
    print("结果文件:", resolved_output)
    window.show_saved_result(resolved_output.name)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_directory = Path.cwd()
    try:
        input_path = args.input or find_default_input(project_directory)
        output_path = args.output or project_directory / DEFAULT_OUTPUT_NAME
        run(input_path, output_path)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

