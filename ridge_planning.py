"""
通用垄线规划与评价软件

本文件面向软件著作权材料整理，保留“通用模型”的表达方式：
1. 支持凸四边形地块输入与南北/东西两类主作业方向对比。
2. 内置可扩展案例库，用于软件演示、回归测试和指标体系验证。
3. 集成地块规整化、垄线生成、作业路径组织、评价指标计算和报告导出。
4. 不绑定单一示范区，案例参数仅作为通用农田形态样例。

运行示例：
    python ridge_planning.py --list-cases
    python ridge_planning.py --case standard_rectangle --report
    python ridge_planning.py --case bayannur_demo --plot
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception:  # pragma: no cover
    tk = None
    ttk = None
    messagebox = None
    filedialog = None

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise RuntimeError("本软件需要安装 numpy。") from exc

try:
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:  # pragma: no cover
    plt = None
    patches = None
    FigureCanvasTkAgg = None

if plt is not None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


EPS = 1e-9
MU_M2 = 666.6666667
DEFAULT_OUTPUT_DIR = os.path.join("5.10", "outputs")


@dataclass
class RidgeSpec:
    """垄型参数。"""

    top_width_m: float = 0.75
    bottom_width_m: float = 0.80
    ditch_width_m: float = 0.40
    pitch_m: float = 1.20
    row_spacing_m: float = 0.45
    plant_spacing_m: float = 0.35
    seedlings_per_hole: int = 2

    def validate(self) -> None:
        values = {
            "top_width_m": self.top_width_m,
            "bottom_width_m": self.bottom_width_m,
            "ditch_width_m": self.ditch_width_m,
            "pitch_m": self.pitch_m,
            "row_spacing_m": self.row_spacing_m,
            "plant_spacing_m": self.plant_spacing_m,
        }
        for name, value in values.items():
            if value <= 0:
                raise ValueError(f"{name} 必须为正数。")
        if self.bottom_width_m > self.pitch_m:
            raise ValueError("垄底宽不能大于垄距。")


@dataclass
class MachineSpec:
    """作业机具参数。"""

    model: str = "通用移栽/起垄装备"
    working_width_m: float = 1.20
    turn_distance_m: float = 2.40
    turn_time_min: float = 2.0
    min_headland_m: float = 3.60
    max_slope_deg: float = 8.0
    speed_m_min: float = 30.0
    fuel_l_per_h: float = 5.0

    def headland_depth(self, ridge: RidgeSpec) -> float:
        return max(self.min_headland_m, 2.0 * self.turn_distance_m, 3.0 * ridge.pitch_m)


@dataclass
class Obstacle:
    """地块内部障碍物或禁入区，可用于后续扩展。"""

    obstacle_id: str
    name: str
    polygon: List[Tuple[float, float]]
    buffer_m: float = 0.5
    obstacle_type: str = "general"

    def center(self) -> Tuple[float, float]:
        if not self.polygon:
            return (0.0, 0.0)
        return (
            sum(p[0] for p in self.polygon) / len(self.polygon),
            sum(p[1] for p in self.polygon) / len(self.polygon),
        )


@dataclass
class FieldCase:
    """案例库条目。"""

    case_id: str
    name: str
    vertices: List[Tuple[float, float]]
    ridge_spec: RidgeSpec = field(default_factory=RidgeSpec)
    machine_spec: MachineSpec = field(default_factory=MachineSpec)
    obstacles: List[Obstacle] = field(default_factory=list)
    preferred_direction: Optional[str] = None
    crop_name: str = "露地蔬菜"
    note: str = ""

    def vertices_array(self) -> np.ndarray:
        return np.array(self.vertices, dtype=float)


@dataclass
class RidgeSegment:
    """单条垄线及其几何对象。"""

    order: int
    phase: str
    centerline: Tuple[np.ndarray, np.ndarray]
    polygon: np.ndarray
    length_m: float
    role: str
    path_points: List[np.ndarray] = field(default_factory=list)
    group: int = 0

    @property
    def start(self) -> np.ndarray:
        return self.path_points[0] if self.path_points else self.centerline[0]

    @property
    def end(self) -> np.ndarray:
        return self.path_points[-1] if self.path_points else self.centerline[1]


@dataclass
class MetricDefinition:
    """评价指标定义。"""

    code: str
    name: str
    weight: float
    target: str
    description: str


@dataclass
class MetricValue:
    """单项指标计算结果。"""

    code: str
    name: str
    raw_value: float
    normalized_score: float
    weight: float
    weighted_score: float
    unit: str = ""


@dataclass
class PlanMetrics:
    """规划方案综合指标。"""

    original_area_m2: float
    regularized_area_m2: float
    core_area_m2: float
    headland_area_m2: float
    cut_area_m2: float
    ridge_count: int
    headland_ridge_count: int
    total_ridge_length_m: float
    path_length_m: float
    turn_count: int
    estimated_time_min: float
    estimated_fuel_l: float
    estimated_seedling_count: int
    metric_values: List[MetricValue]
    total_score: float
    grade: str


@dataclass
class RidgePlan:
    """完整规划结果。"""

    case_id: str
    scheme_id: str
    display_name: str
    target_direction: str
    base_edge_index: int
    regularized_polygon: np.ndarray
    core_polygon: np.ndarray
    headland_polygons: List[np.ndarray]
    ridges: List[RidgeSegment]
    phase_paths: Dict[str, List[np.ndarray]]
    metrics: PlanMetrics
    headland_depth_m: float = 0.0
    warning_messages: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def is_valid(self) -> bool:
        return self.error is None

    @property
    def max_frame(self) -> int:
        return max(1, len(self.ridges) * 18)

    def animation_state(self, frame: int) -> Tuple[int, float]:
        if not self.ridges or frame <= 0:
            return -1, 0.0
        frame = min(frame, self.max_frame)
        idx = min(len(self.ridges) - 1, (frame - 1) // 18)
        sub = (frame - 1) % 18 + 1
        return idx, sub / 18.0


class GeometryToolkit:
    """二维地块几何计算工具。"""

    @staticmethod
    def area(poly: np.ndarray) -> float:
        if poly is None or len(poly) < 3:
            return 0.0
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @staticmethod
    def signed_area(poly: np.ndarray) -> float:
        if poly is None or len(poly) < 3:
            return 0.0
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    @staticmethod
    def centroid(poly: np.ndarray) -> np.ndarray:
        if poly is None or len(poly) == 0:
            return np.array([0.0, 0.0])
        return np.mean(poly, axis=0)

    @staticmethod
    def edge_lengths(poly: np.ndarray) -> List[float]:
        return [float(np.linalg.norm(poly[(i + 1) % len(poly)] - poly[i])) for i in range(len(poly))]

    @staticmethod
    def convex_quad(poly: np.ndarray) -> bool:
        if poly.shape != (4, 2):
            return False
        signs: List[bool] = []
        for i in range(4):
            a = poly[i]
            b = poly[(i + 1) % 4]
            c = poly[(i + 2) % 4]
            cross = float(np.cross(b - a, c - b))
            if abs(cross) > EPS:
                signs.append(cross > 0)
        return bool(signs) and all(item == signs[0] for item in signs)

    @staticmethod
    def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
        def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
            return float(np.cross(q - p, r - p))

        o1 = orient(a, b, c)
        o2 = orient(a, b, d)
        o3 = orient(c, d, a)
        o4 = orient(c, d, b)
        return (o1 * o2 < -EPS) and (o3 * o4 < -EPS)

    @staticmethod
    def self_crossing_quad(poly: np.ndarray) -> bool:
        return GeometryToolkit.segments_intersect(poly[0], poly[1], poly[2], poly[3]) or GeometryToolkit.segments_intersect(poly[1], poly[2], poly[3], poly[0])

    @staticmethod
    def validate_quad(poly: np.ndarray) -> Tuple[bool, str]:
        if poly.shape != (4, 2):
            return False, "当前版本要求输入 4 个二维顶点。"
        if GeometryToolkit.area(poly) < 1e-6:
            return False, "地块面积过小或顶点共线。"
        if GeometryToolkit.self_crossing_quad(poly):
            return False, "顶点顺序导致边界交叉，请按顺时针或逆时针输入。"
        if not GeometryToolkit.convex_quad(poly):
            return False, "当前通用模型要求输入凸四边形。"
        return True, "OK"

    @staticmethod
    def clip_polygon_by_line(poly: np.ndarray, line_point: np.ndarray, line_vec: np.ndarray, keep_left: bool = True) -> np.ndarray:
        if poly is None or len(poly) == 0:
            return np.empty((0, 2))
        v = line_vec / (np.linalg.norm(line_vec) + EPS)
        normal = np.array([-v[1], v[0]])

        def inside(p: np.ndarray) -> bool:
            dot = float(np.dot(p - line_point, normal))
            return dot >= -1e-8 if keep_left else dot <= 1e-8

        def intersect(p1: np.ndarray, p2: np.ndarray) -> Optional[np.ndarray]:
            d = p2 - p1
            den = float(np.cross(d, v))
            if abs(den) < EPS:
                return None
            t = float(np.cross(line_point - p1, v) / den)
            if -1e-8 <= t <= 1.0 + 1e-8:
                return p1 + t * d
            return None

        out: List[np.ndarray] = []
        for i in range(len(poly)):
            curr = poly[i]
            prev = poly[(i - 1) % len(poly)]
            curr_in = inside(curr)
            prev_in = inside(prev)
            if curr_in:
                if not prev_in:
                    p = intersect(prev, curr)
                    if p is not None:
                        out.append(p)
                out.append(curr)
            elif prev_in:
                p = intersect(prev, curr)
                if p is not None:
                    out.append(p)
        if not out:
            return np.empty((0, 2))
        return np.array(out, dtype=float)

    @staticmethod
    def rotate_to_edge(poly: np.ndarray, base_edge_index: int) -> Tuple[np.ndarray, float, np.ndarray]:
        origin = poly[base_edge_index].copy()
        edge = poly[(base_edge_index + 1) % len(poly)] - origin
        angle = math.atan2(edge[1], edge[0])
        c = math.cos(-angle)
        s = math.sin(-angle)
        rot = np.array([[c, -s], [s, c]])
        return (poly - origin) @ rot.T, angle, origin

    @staticmethod
    def to_global(local: np.ndarray, angle: float, origin: np.ndarray) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + origin

    @staticmethod
    def polyline_length(points: Sequence[np.ndarray]) -> float:
        total = 0.0
        for a, b in zip(points[:-1], points[1:]):
            if np.any(np.isnan(a)) or np.any(np.isnan(b)):
                continue
            total += float(np.linalg.norm(b - a))
        return total


def point_to_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    length2 = float(np.dot(d, d))
    if length2 <= EPS:
        return float(np.linalg.norm(point - a))
    t = max(0.0, min(1.0, float(np.dot(point - a, d) / length2)))
    projection = a + d * t
    return float(np.linalg.norm(point - projection))


def segment_distance(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    if GeometryToolkit.segments_intersect(a, b, c, d):
        return 0.0
    return min(
        point_to_segment_distance(a, c, d),
        point_to_segment_distance(b, c, d),
        point_to_segment_distance(c, a, b),
        point_to_segment_distance(d, a, b),
    )


class CaseLibrary:
    """软件内置案例库，用于演示和测试。"""

    def __init__(self) -> None:
        self._cases: Dict[str, FieldCase] = {}
        self._load_builtin_cases()

    def _load_builtin_cases(self) -> None:
        self.add(FieldCase(
            case_id="standard_rectangle",
            name="标准矩形露地蔬菜地块",
            vertices=[(0.0, 0.0), (120.0, 0.0), (120.0, 80.0), (0.0, 80.0)],
            preferred_direction="NS",
            crop_name="辣椒/甘蓝等露地蔬菜",
            note="用于验证基础垄距、地头区和路径长度计算。",
        ))
        self.add(FieldCase(
            case_id="trapezoid_demo",
            name="梯形边界示范地块",
            vertices=[(0.0, 0.0), (146.0, 8.0), (132.0, 96.0), (8.0, 88.0)],
            preferred_direction="NS",
            crop_name="露地蔬菜",
            note="用于验证斜边规整化与南北/东西方向方案对比。",
        ))
        self.add(FieldCase(
            case_id="wide_quad_demo",
            name="宽幅四边形农机协同地块",
            vertices=[(0.0, 0.0), (210.0, -5.0), (220.0, 135.0), (12.0, 126.0)],
            ridge_spec=RidgeSpec(top_width_m=0.72, bottom_width_m=0.82, ditch_width_m=0.38, pitch_m=1.20),
            machine_spec=MachineSpec(model="通用双行作业装备", turn_distance_m=2.60, speed_m_min=32.0),
            preferred_direction="NS",
            note="用于验证大面积地块的垄数、时间和燃油估算。",
        ))
        self.add(FieldCase(
            case_id="bayannur_demo",
            name="巴彦淖尔井灌区风格长条田",
            vertices=[(0.0, 0.0), (374.5, 0.0), (374.5, 558.0), (0.0, 558.0)],
            ridge_spec=RidgeSpec(top_width_m=0.75, bottom_width_m=0.80, ditch_width_m=0.40, pitch_m=1.20),
            machine_spec=MachineSpec(model="2ZB-2M 类双行移栽机", turn_distance_m=2.40, min_headland_m=5.0, speed_m_min=28.0),
            preferred_direction="NS",
            crop_name="辣椒",
            note="该案例抽象自巴彦淖尔井灌区风格长条田，仅作为通用模型测试用例，不内置具体地块细节。",
        ))

    def add(self, case: FieldCase) -> None:
        ok, msg = GeometryToolkit.validate_quad(case.vertices_array())
        if not ok:
            raise ValueError(f"案例 {case.case_id} 无效：{msg}")
        case.ridge_spec.validate()
        self._cases[case.case_id] = case

    def get(self, case_id: str) -> FieldCase:
        if case_id not in self._cases:
            raise KeyError(f"未找到案例：{case_id}")
        return self._cases[case_id]

    def list_cases(self) -> List[FieldCase]:
        return list(self._cases.values())

    def load_json(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("cases", [])
        for item in items:
            ridge = RidgeSpec(**item.get("ridge_spec", {}))
            machine = MachineSpec(**item.get("machine_spec", {}))
            obstacles = [Obstacle(**obj) for obj in item.get("obstacles", [])]
            self.add(FieldCase(
                case_id=item["case_id"],
                name=item.get("name", item["case_id"]),
                vertices=[tuple(v) for v in item["vertices"]],
                ridge_spec=ridge,
                machine_spec=machine,
                obstacles=obstacles,
                preferred_direction=item.get("preferred_direction"),
                crop_name=item.get("crop_name", "露地蔬菜"),
                note=item.get("note", ""),
            ))


class FieldRegularizer:
    """地块规整化模块。"""

    def __init__(self, ridge: RidgeSpec, machine: MachineSpec):
        self.ridge = ridge
        self.machine = machine

    def choose_base_edge(self, vertices: np.ndarray, target: str) -> int:
        target_vec = np.array([0.0, 1.0]) if target == "NS" else np.array([1.0, 0.0])
        scores = []
        for i in range(4):
            edge_vec = vertices[(i + 1) % 4] - vertices[i]
            unit = edge_vec / (np.linalg.norm(edge_vec) + EPS)
            scores.append(abs(float(np.dot(unit, target_vec))))
        return int(np.argmax(scores))

    def regularize(self, vertices: np.ndarray, target: str) -> Tuple[Optional[Dict], Optional[str]]:
        base_idx = self.choose_base_edge(vertices, target)
        rotated, angle, origin = GeometryToolkit.rotate_to_edge(vertices, base_idx)
        idx0 = base_idx
        idx1 = (base_idx + 1) % 4
        idx2 = (base_idx + 2) % 4
        idx3 = (base_idx + 3) % 4
        opposite_y = [rotated[idx2][1], rotated[idx3][1]]
        is_above = sum(opposite_y) / 2.0 > 0

        if is_above:
            y_cut = min(opposite_y)
            reg = GeometryToolkit.clip_polygon_by_line(rotated, np.array([0.0, y_cut]), np.array([1.0, 0.0]), keep_left=False)
            reg = GeometryToolkit.clip_polygon_by_line(reg, np.array([0.0, 0.0]), np.array([1.0, 0.0]), keep_left=True)
        else:
            y_cut = max(opposite_y)
            reg = GeometryToolkit.clip_polygon_by_line(rotated, np.array([0.0, y_cut]), np.array([1.0, 0.0]), keep_left=True)
            reg = GeometryToolkit.clip_polygon_by_line(reg, np.array([0.0, 0.0]), np.array([1.0, 0.0]), keep_left=False)

        if len(reg) < 3 or GeometryToolkit.area(reg) < 1e-6:
            return None, "规整化失败：地块无法形成有效作业区。"

        center = GeometryToolkit.centroid(reg)
        headland = self.machine.headland_depth(self.ridge)
        left_point = rotated[idx0]
        left_vec = rotated[idx3] - rotated[idx0]
        right_point = rotated[idx1]
        right_vec = rotated[idx2] - rotated[idx1]

        def offset_line_inward(pt: np.ndarray, vec: np.ndarray, dist: float) -> Tuple[np.ndarray, np.ndarray]:
            unit = vec / (np.linalg.norm(vec) + EPS)
            n1 = np.array([-unit[1], unit[0]])
            n2 = np.array([unit[1], -unit[0]])
            probe = pt + 0.5 * vec
            inward = n1 if np.linalg.norm((probe + 0.1 * n1) - center) < np.linalg.norm((probe + 0.1 * n2) - center) else n2
            return pt + inward * dist, vec

        left_offset_p, left_offset_v = offset_line_inward(left_point, left_vec, headland)
        right_offset_p, right_offset_v = offset_line_inward(right_point, right_vec, headland)

        def clip_to_side(poly: np.ndarray, line_p: np.ndarray, line_v: np.ndarray, keep_center: bool) -> np.ndarray:
            normal = np.array([-line_v[1], line_v[0]])
            center_is_left = float(np.dot(center - line_p, normal)) >= 0
            return GeometryToolkit.clip_polygon_by_line(poly, line_p, line_v, keep_left=center_is_left if keep_center else not center_is_left)

        core = clip_to_side(reg, left_offset_p, left_offset_v, True)
        core = clip_to_side(core, right_offset_p, right_offset_v, True)
        head1 = clip_to_side(reg, left_offset_p, left_offset_v, False)
        head2 = clip_to_side(reg, right_offset_p, right_offset_v, False)

        if len(core) < 3 or GeometryToolkit.area(core) < 1e-6:
            return None, "核心作业区不足，无法生成垄线。"

        return {
            "base_edge_index": base_idx,
            "angle": angle,
            "origin": origin,
            "reg_local": reg,
            "core_local": core,
            "head1_local": head1,
            "head2_local": head2,
            "reg_poly": GeometryToolkit.to_global(reg, angle, origin),
            "core_poly": GeometryToolkit.to_global(core, angle, origin),
            "head1_poly": GeometryToolkit.to_global(head1, angle, origin),
            "head2_poly": GeometryToolkit.to_global(head2, angle, origin),
            "head1_vec": left_vec,
            "head2_vec": right_vec,
            "headland_m": headland,
        }, None


class RidgeLineGenerator:
    """垄线生成模块。"""

    def __init__(self, ridge: RidgeSpec):
        self.ridge = ridge

    def _line_segments_in_polygon(self, poly: np.ndarray, angle_vec: np.ndarray, spacing: float) -> List[Tuple[np.ndarray, np.ndarray]]:
        if len(poly) < 3:
            return []
        theta = math.atan2(angle_vec[1], angle_vec[0])
        c = math.cos(-theta)
        s = math.sin(-theta)
        rotation = np.array([[c, -s], [s, c]])
        inverse = np.array([[c, s], [-s, c]])
        rotated = poly @ rotation.T
        min_y = float(np.min(rotated[:, 1]))
        max_y = float(np.max(rotated[:, 1]))
        height = max_y - min_y
        count = int(math.floor(height / spacing + 1e-9))
        if count <= 0:
            return []
        occupied = (count - 1) * spacing
        margin = max(0.0, (height - occupied) / 2.0)
        edges = [(rotated[i], rotated[(i + 1) % len(rotated)]) for i in range(len(rotated))]
        lines: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(count):
            y = min_y + margin + i * spacing
            xs: List[float] = []
            for p1, p2 in edges:
                if (p1[1] <= y <= p2[1]) or (p2[1] <= y <= p1[1]):
                    if abs(p1[1] - p2[1]) > EPS:
                        t = (y - p1[1]) / (p2[1] - p1[1])
                        xs.append(float(p1[0] + t * (p2[0] - p1[0])))
            xs = sorted(xs)
            if len(xs) >= 2:
                start = np.array([xs[0], y]) @ inverse.T
                end = np.array([xs[-1], y]) @ inverse.T
                if np.linalg.norm(end - start) > self.ridge.pitch_m:
                    lines.append((start, end))
        return lines

    def _strip_polygon(self, start: np.ndarray, end: np.ndarray, width: float) -> np.ndarray:
        direction = end - start
        normal = np.array([-direction[1], direction[0]]) / (np.linalg.norm(direction) + EPS) * (width / 2.0)
        return np.array([start + normal, end + normal, end - normal, start - normal], dtype=float)

    def _make_ridges(self, lines: List[Tuple[np.ndarray, np.ndarray]], phase: str, start_order: int, role: str) -> List[RidgeSegment]:
        ridges: List[RidgeSegment] = []
        for i, (a, b) in enumerate(lines):
            start, end = (a, b) if i % 2 == 0 else (b, a)
            ridges.append(RidgeSegment(
                order=start_order + i,
                phase=phase,
                centerline=(start, end),
                polygon=self._strip_polygon(start, end, self.ridge.bottom_width_m),
                length_m=float(np.linalg.norm(end - start)),
                role=role,
                path_points=[start, end],
                group=0,
            ))
        return ridges

    def _line_removed_by_barrier(self, line: Tuple[np.ndarray, np.ndarray], obstacles: List[Obstacle]) -> bool:
        a, b = line
        for obstacle in obstacles:
            if obstacle.obstacle_type == "line" and len(obstacle.polygon) >= 2:
                p = np.array(obstacle.polygon[0], dtype=float)
                q = np.array(obstacle.polygon[1], dtype=float)
                if segment_distance(a, b, p, q) <= max(self.ridge.pitch_m * 0.55, obstacle.buffer_m):
                    return True
        return False

    def _line_with_point_detour(self, line: Tuple[np.ndarray, np.ndarray], obstacles: List[Obstacle]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], float]:
        a, b = line
        path = [a]
        direction = b - a
        length = float(np.linalg.norm(direction))
        if length <= EPS:
            return a, b, [a, b], 0.0
        unit = direction / length
        normal = np.array([-unit[1], unit[0]])
        affected: List[Tuple[float, np.ndarray, float]] = []
        for obstacle in obstacles:
            if obstacle.obstacle_type != "point" or not obstacle.polygon:
                continue
            center = np.array(obstacle.polygon[0], dtype=float)
            radius = max(0.1, obstacle.buffer_m)
            dist = point_to_segment_distance(center, a, b)
            if dist > radius:
                continue
            t = float(np.dot(center - a, direction) / max(length * length, EPS))
            if 0.02 < t < 0.98:
                affected.append((t, center, radius))
        for t, center, radius in sorted(affected, key=lambda item: item[0]):
            clearance = max(radius + self.ridge.bottom_width_m, radius * 1.25)
            before = max(0.0, t - clearance / length)
            after = min(1.0, t + clearance / length)
            side = normal
            if float(np.dot(center - (a + direction * t), normal)) > 0:
                side = -normal
            path.append(a + direction * before)
            path.append(center + side * clearance)
            path.append(a + direction * after)
        path.append(b)
        cleaned: List[np.ndarray] = []
        for point in path:
            if not cleaned or np.linalg.norm(point - cleaned[-1]) > 1e-6:
                cleaned.append(point)
        path_length = GeometryToolkit.polyline_length(cleaned)
        return a, b, cleaned, path_length

    def _split_line_by_obstacles(self, line: Tuple[np.ndarray, np.ndarray], obstacles: List[Obstacle]) -> List[Tuple[np.ndarray, np.ndarray]]:
        a, b = line
        length = float(np.linalg.norm(b - a))
        if length <= EPS or not obstacles:
            return [line]
        blocked: List[Tuple[float, float]] = []
        for obstacle in obstacles:
            if obstacle.obstacle_type == "line" and len(obstacle.polygon) >= 2:
                p = np.array(obstacle.polygon[0], dtype=float)
                q = np.array(obstacle.polygon[1], dtype=float)
                blocked.extend(self._line_block_interval(a, b, p, q, max(0.1, obstacle.buffer_m)))
            elif len(obstacle.polygon) >= 3:
                center = np.array(obstacle.center(), dtype=float)
                radius = max(obstacle.buffer_m, max(float(np.linalg.norm(np.array(pt) - center)) for pt in obstacle.polygon))
                blocked.extend(self._point_block_interval(a, b, center, radius))
        free = self._subtract_param_intervals((0.0, 1.0), blocked)
        segments: List[Tuple[np.ndarray, np.ndarray]] = []
        for t0, t1 in free:
            p0 = a + (b - a) * t0
            p1 = a + (b - a) * t1
            if np.linalg.norm(p1 - p0) >= max(self.ridge.pitch_m, self.ridge.bottom_width_m * 2.0):
                segments.append((p0, p1))
        return segments

    @staticmethod
    def _point_block_interval(a: np.ndarray, b: np.ndarray, center: np.ndarray, radius: float) -> List[Tuple[float, float]]:
        d = b - a
        length2 = float(np.dot(d, d))
        if length2 <= EPS:
            return []
        t = float(np.dot(center - a, d) / length2)
        closest = a + d * max(0.0, min(1.0, t))
        dist = float(np.linalg.norm(center - closest))
        if dist > radius:
            return []
        half = math.sqrt(max(0.0, radius * radius - dist * dist)) / math.sqrt(length2)
        return [(max(0.0, t - half), min(1.0, t + half))]

    @staticmethod
    def _line_block_interval(a: np.ndarray, b: np.ndarray, p: np.ndarray, q: np.ndarray, buffer_m: float) -> List[Tuple[float, float]]:
        d = b - a
        length = float(np.linalg.norm(d))
        if length <= EPS:
            return []
        min_dist = segment_distance(a, b, p, q)
        if min_dist > buffer_m:
            return []
        unit = d / length
        projections = [float(np.dot(p - a, unit) / length), float(np.dot(q - a, unit) / length)]
        lo = max(0.0, min(projections) - buffer_m / length)
        hi = min(1.0, max(projections) + buffer_m / length)
        return [(lo, hi)] if hi > lo else []

    @staticmethod
    def _subtract_param_intervals(base: Tuple[float, float], blocked: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not blocked:
            return [base]
        merged: List[Tuple[float, float]] = []
        for start, end in sorted((max(base[0], s), min(base[1], e)) for s, e in blocked if e > s):
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        free = [base]
        for block_start, block_end in merged:
            next_free: List[Tuple[float, float]] = []
            for free_start, free_end in free:
                if block_end <= free_start or block_start >= free_end:
                    next_free.append((free_start, free_end))
                else:
                    if block_start > free_start:
                        next_free.append((free_start, block_start))
                    if block_end < free_end:
                        next_free.append((block_end, free_end))
            free = next_free
        return free

    def _apply_obstacle_splitting(self, lines: List[Tuple[np.ndarray, np.ndarray]], obstacles: List[Obstacle]) -> List[Tuple[np.ndarray, np.ndarray]]:
        if not obstacles:
            return lines
        split: List[Tuple[np.ndarray, np.ndarray]] = []
        for line in lines:
            if self._line_removed_by_barrier(line, obstacles):
                continue
            split.extend(self._split_line_by_obstacles(line, obstacles))
        return split

    def _apply_point_detours(self, ridges: List[RidgeSegment], obstacles: List[Obstacle]) -> None:
        if not obstacles:
            return
        for ridge in ridges:
            start, end, path, path_length = self._line_with_point_detour(ridge.centerline, obstacles)
            ridge.centerline = (start, end)
            ridge.path_points = path
            ridge.length_m = path_length

    def _assign_split_groups(self, ridges: List[RidgeSegment], obstacles: List[Obstacle]) -> None:
        line_obstacles = [ob for ob in obstacles if ob.obstacle_type == "line" and len(ob.polygon) >= 2]
        if not line_obstacles:
            return
        for ridge in ridges:
            group = 0
            midpoint = (ridge.centerline[0] + ridge.centerline[1]) / 2.0
            for idx, obstacle in enumerate(line_obstacles):
                p = np.array(obstacle.polygon[0], dtype=float)
                q = np.array(obstacle.polygon[1], dtype=float)
                side = float(np.cross(q - p, midpoint - p))
                bit = 1 if side >= 0 else 0
                group |= bit << idx
            ridge.group = group

    def generate(self, reg: Dict, obstacles: Optional[List[Obstacle]] = None) -> List[RidgeSegment]:
        obstacles = obstacles or []
        core_lines = self._line_segments_in_polygon(reg["core_local"], np.array([1.0, 0.0]), self.ridge.pitch_m)
        core_lines = sorted(core_lines, key=lambda line: float((line[0][1] + line[1][1]) / 2.0))
        head1_lines = self._line_segments_in_polygon(reg["head1_local"], reg["head1_vec"], self.ridge.pitch_m)
        head2_lines = self._line_segments_in_polygon(reg["head2_local"], reg["head2_vec"], self.ridge.pitch_m)

        angle = reg["angle"]
        origin = reg["origin"]

        def line_to_global(line: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
            pair = GeometryToolkit.to_global(np.array([line[0], line[1]]), angle, origin)
            return pair[0], pair[1]

        core_global = [line_to_global(line) for line in core_lines]
        head1_global = [line_to_global(line) for line in head1_lines]
        head2_global = [line_to_global(line) for line in head2_lines]
        core_global = self._apply_obstacle_splitting(core_global, obstacles)
        head1_global = self._apply_obstacle_splitting(head1_global, obstacles)
        head2_global = self._apply_obstacle_splitting(head2_global, obstacles)

        ridges: List[RidgeSegment] = []
        if not core_global:
            return ridges
        core_body = core_global[:-1] if len(core_global) >= 2 else core_global
        ridges.extend(self._make_ridges(core_body, "A", 1, "核心区连续作业"))
        order = len(ridges) + 1
        ridges.extend(self._make_ridges(head1_global, "B", order, "第一侧地头处理"))
        order = len(ridges) + 1
        if len(core_global) >= 2:
            ridges.extend(self._make_ridges([core_global[-1]], "C", order, "核心区收尾垄"))
            order += 1
        ridges.extend(self._make_ridges(head2_global, "D", order, "第二侧地头处理"))
        self._assign_split_groups(ridges, obstacles)
        self._apply_point_detours(ridges, obstacles)
        ridges.sort(key=lambda r: (r.group, r.order))
        return ridges


class PathOrganizer:
    """按阶段组织作业路径。"""

    PHASES = ("A", "B", "C", "D")

    @staticmethod
    def phase_paths(ridges: List[RidgeSegment]) -> Dict[str, List[np.ndarray]]:
        paths: Dict[str, List[np.ndarray]] = {phase: [] for phase in PathOrganizer.PHASES}
        last_group_by_phase: Dict[str, Optional[int]] = {phase: None for phase in PathOrganizer.PHASES}
        for ridge in ridges:
            pts = paths[ridge.phase]
            ridge_points = ridge.path_points if ridge.path_points else [ridge.start, ridge.end]
            if pts and last_group_by_phase[ridge.phase] is not None and ridge.group != last_group_by_phase[ridge.phase]:
                pts.append(np.array([np.nan, np.nan]))
            if not pts:
                pts.append(ridge_points[0])
            elif np.any(np.isnan(pts[-1])) or np.linalg.norm(pts[-1] - ridge_points[0]) > 1e-8:
                pts.append(ridge_points[0])
            pts.extend(ridge_points[1:])
            last_group_by_phase[ridge.phase] = ridge.group
        return paths

    @staticmethod
    def total_path_length(paths: Dict[str, List[np.ndarray]]) -> float:
        return sum(GeometryToolkit.polyline_length(points) for points in paths.values())

    @staticmethod
    def path_turn_count(ridges: List[RidgeSegment]) -> int:
        return max(0, len(ridges) - 1)


class EvaluationMetricSystem:
    """垄线规划评价指标体系。"""

    def __init__(self) -> None:
        self.definitions = [
            MetricDefinition("M01", "土地规整利用率", 0.22, "max", "规整化后可利用面积占原始面积比例。"),
            MetricDefinition("M02", "核心作业区比例", 0.18, "max", "核心垄线作业区占原始面积比例。"),
            MetricDefinition("M03", "裁切损失控制", 0.12, "min", "规整化裁切面积占原始面积比例，越低越好。"),
            MetricDefinition("M04", "垄线有效长度", 0.12, "max", "核心区有效垄线总长度相对面积的匹配程度。"),
            MetricDefinition("M05", "路径紧凑性", 0.12, "max", "有效垄线长度与总行驶路径长度的比例。"),
            MetricDefinition("M06", "掉头次数控制", 0.08, "min", "掉头次数相对垄线数量的控制水平。"),
            MetricDefinition("M07", "地头区适配性", 0.08, "max", "地头区面积与机具掉头需求的匹配程度。"),
            MetricDefinition("M08", "农艺参数匹配", 0.08, "max", "垄型、垄距、株行距与露地蔬菜作业要求的匹配程度。"),
        ]

    def evaluate(self, case: FieldCase, reg: Dict, ridges: List[RidgeSegment], paths: Dict[str, List[np.ndarray]]) -> PlanMetrics:
        vertices = case.vertices_array()
        original_area = GeometryToolkit.area(vertices)
        regularized_area = GeometryToolkit.area(reg["reg_poly"])
        core_area = GeometryToolkit.area(reg["core_poly"])
        headland_area = GeometryToolkit.area(reg["head1_poly"]) + GeometryToolkit.area(reg["head2_poly"])
        cut_area = max(0.0, original_area - regularized_area)
        core_ridges = [ridge for ridge in ridges if ridge.phase in ("A", "C")]
        headland_ridges = [ridge for ridge in ridges if ridge.phase in ("B", "D")]
        ridge_count = len(core_ridges)
        headland_ridge_count = len(headland_ridges)
        total_ridge_length = sum(r.length_m for r in core_ridges)
        path_length = PathOrganizer.total_path_length(paths)
        turn_count = PathOrganizer.path_turn_count(ridges)
        estimated_time = self._estimate_time(case.machine_spec, path_length, turn_count)
        estimated_fuel = estimated_time / 60.0 * case.machine_spec.fuel_l_per_h
        estimated_seedling_count = self._estimate_seedlings(case.ridge_spec, total_ridge_length)

        values = self._metric_values(
            case=case,
            original_area=original_area,
            regularized_area=regularized_area,
            core_area=core_area,
            headland_area=headland_area,
            cut_area=cut_area,
            ridge_count=ridge_count,
            headland_ridge_count=headland_ridge_count,
            total_ridge_length=total_ridge_length,
            path_length=path_length,
            turn_count=turn_count,
        )
        total_score = sum(item.weighted_score for item in values)
        return PlanMetrics(
            original_area_m2=original_area,
            regularized_area_m2=regularized_area,
            core_area_m2=core_area,
            headland_area_m2=headland_area,
            cut_area_m2=cut_area,
            ridge_count=ridge_count,
            headland_ridge_count=headland_ridge_count,
            total_ridge_length_m=total_ridge_length,
            path_length_m=path_length,
            turn_count=turn_count,
            estimated_time_min=estimated_time,
            estimated_fuel_l=estimated_fuel,
            estimated_seedling_count=estimated_seedling_count,
            metric_values=values,
            total_score=total_score,
            grade=self._grade(total_score),
        )

    def _metric_values(
        self,
        case: FieldCase,
        original_area: float,
        regularized_area: float,
        core_area: float,
        headland_area: float,
        cut_area: float,
        ridge_count: int,
        headland_ridge_count: int,
        total_ridge_length: float,
        path_length: float,
        turn_count: int,
    ) -> List[MetricValue]:
        utilization = regularized_area / max(original_area, EPS)
        core_ratio = core_area / max(original_area, EPS)
        cut_ratio = cut_area / max(original_area, EPS)
        effective_length_density = total_ridge_length / max(original_area / 100.0, EPS)
        target_density = 100.0 / max(case.ridge_spec.pitch_m, EPS)
        length_score = max(0.0, min(100.0, 100.0 - abs(effective_length_density - target_density) / max(target_density, EPS) * 100.0))
        path_compactness = total_ridge_length / max(path_length, EPS)
        turn_ratio = turn_count / max(ridge_count + headland_ridge_count, 1)
        headland_need = 2.0 * case.machine_spec.headland_depth(case.ridge_spec) * math.sqrt(max(original_area, EPS))
        headland_score = min(100.0, headland_area / max(headland_need, EPS) * 100.0)
        agronomic_score = self._agronomic_score(case.ridge_spec)

        raw_by_code = {
            "M01": (utilization, utilization * 100.0, "%"),
            "M02": (core_ratio, core_ratio * 100.0, "%"),
            "M03": (cut_ratio, max(0.0, 100.0 - cut_ratio * 200.0), "%"),
            "M04": (effective_length_density, length_score, "m/100m2"),
            "M05": (path_compactness, min(100.0, path_compactness * 100.0), "%"),
            "M06": (turn_ratio, max(0.0, 100.0 - turn_ratio * 35.0), ""),
            "M07": (headland_area, headland_score, "m2"),
            "M08": (agronomic_score, agronomic_score, "分"),
        }
        values: List[MetricValue] = []
        for definition in self.definitions:
            raw, score, unit = raw_by_code[definition.code]
            values.append(MetricValue(
                code=definition.code,
                name=definition.name,
                raw_value=float(raw),
                normalized_score=float(max(0.0, min(100.0, score))),
                weight=definition.weight,
                weighted_score=float(max(0.0, min(100.0, score)) * definition.weight),
                unit=unit,
            ))
        return values

    @staticmethod
    def _estimate_time(machine: MachineSpec, path_length: float, turn_count: int) -> float:
        travel = path_length / max(machine.speed_m_min, EPS)
        turn = turn_count * machine.turn_time_min
        return travel + turn

    @staticmethod
    def _estimate_seedlings(ridge: RidgeSpec, total_ridge_length: float) -> int:
        holes_per_line = total_ridge_length / max(ridge.plant_spacing_m, EPS)
        row_factor = max(1, int(round(ridge.top_width_m / max(ridge.row_spacing_m, EPS))))
        return int(round(holes_per_line * row_factor * ridge.seedlings_per_hole))

    @staticmethod
    def _agronomic_score(ridge: RidgeSpec) -> float:
        score = 100.0
        if not (0.65 <= ridge.top_width_m <= 0.90):
            score -= 15.0
        if not (0.70 <= ridge.bottom_width_m <= 1.05):
            score -= 12.0
        if not (1.0 <= ridge.pitch_m <= 1.4):
            score -= 18.0
        if ridge.ditch_width_m < 0.25:
            score -= 10.0
        if ridge.plant_spacing_m < 0.20 or ridge.plant_spacing_m > 0.55:
            score -= 10.0
        return max(0.0, score)

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 90:
            return "优秀"
        if score >= 80:
            return "良好"
        if score >= 70:
            return "中等"
        if score >= 60:
            return "可用"
        return "需调整"


class GenericRidgePlanningEngine:
    """通用垄线规划主引擎。"""

    def __init__(self, case: FieldCase):
        case.ridge_spec.validate()
        ok, msg = GeometryToolkit.validate_quad(case.vertices_array())
        if not ok:
            raise ValueError(msg)
        self.case = case
        self.regularizer = FieldRegularizer(case.ridge_spec, case.machine_spec)
        self.line_generator = RidgeLineGenerator(case.ridge_spec)
        self.evaluator = EvaluationMetricSystem()

    def generate_one(self, target_direction: str) -> RidgePlan:
        target_direction = target_direction.upper()
        display_name = "南北向主作业方案" if target_direction == "NS" else "东西向主作业方案"
        scheme_id = f"{self.case.case_id}_{target_direction}"
        reg, err = self.regularizer.regularize(self.case.vertices_array(), target_direction)
        if err or reg is None:
            empty_metrics = self._empty_metrics()
            return RidgePlan(
                case_id=self.case.case_id,
                scheme_id=scheme_id,
                display_name=display_name,
                target_direction=target_direction,
                base_edge_index=-1,
                regularized_polygon=np.empty((0, 2)),
                core_polygon=np.empty((0, 2)),
                headland_polygons=[],
                ridges=[],
                phase_paths={},
                metrics=empty_metrics,
                error=err or "未知错误",
            )
        ridges = self.line_generator.generate(reg, self.case.obstacles)
        paths = PathOrganizer.phase_paths(ridges)
        metrics = self.evaluator.evaluate(self.case, reg, ridges, paths)
        warnings = self._build_warnings(reg, metrics)
        return RidgePlan(
            case_id=self.case.case_id,
            scheme_id=scheme_id,
            display_name=display_name,
            target_direction=target_direction,
            base_edge_index=reg["base_edge_index"],
            regularized_polygon=reg["reg_poly"],
            core_polygon=reg["core_poly"],
            headland_polygons=[reg["head1_poly"], reg["head2_poly"]],
            ridges=ridges,
            phase_paths=paths,
            metrics=metrics,
            headland_depth_m=float(reg["headland_m"]),
            warning_messages=warnings,
        )

    def generate_all(self) -> Dict[str, RidgePlan]:
        return {"NS": self.generate_one("NS"), "EW": self.generate_one("EW")}

    def recommend(self, plans: Dict[str, RidgePlan]) -> RidgePlan:
        valid = [plan for plan in plans.values() if plan.is_valid()]
        if not valid:
            return next(iter(plans.values()))
        best_score = max(plan.metrics.total_score for plan in valid)
        best = [
            plan for plan in valid
            if math.isclose(plan.metrics.total_score, best_score, rel_tol=0.0, abs_tol=1e-9)
        ]
        if self.case.preferred_direction:
            for plan in best:
                if plan.target_direction == self.case.preferred_direction:
                    return plan
        return best[0]

    def _build_warnings(self, reg: Dict, metrics: PlanMetrics) -> List[str]:
        warnings: List[str] = []
        cut_ratio = metrics.cut_area_m2 / max(metrics.original_area_m2, EPS)
        if cut_ratio > 0.15:
            warnings.append("规整化裁切比例偏高，建议复核地块边界或调整作业方向。")
        if metrics.ridge_count <= 0:
            warnings.append("核心区垄线数量为 0，建议增大地块宽度或减小垄距。")
        if metrics.headland_area_m2 < 2.0 * self.case.machine_spec.headland_depth(self.case.ridge_spec) * self.case.ridge_spec.pitch_m:
            warnings.append("地头区域偏小，需结合机具实际掉头能力复核。")
        if self.case.obstacles:
            warnings.append(f"已启用障碍物避让：共 {len(self.case.obstacles)} 个点状/条状障碍物参与绕行、切分或整垄删除。")
        return warnings

    @staticmethod
    def _empty_metrics() -> PlanMetrics:
        return PlanMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], 0, "无效")


class ReportExporter:
    """规划结果导出模块。"""

    @staticmethod
    def headland_check(case: FieldCase, plan: RidgePlan) -> Dict[str, object]:
        required_m = case.machine_spec.headland_depth(case.ridge_spec)
        actual_m = plan.headland_depth_m
        areas = [GeometryToolkit.area(poly) for poly in plan.headland_polygons]
        passed = (
            plan.is_valid()
            and len(areas) == 2
            and all(area > EPS for area in areas)
            and actual_m + EPS >= required_m
        )
        return {
            "actual_depth_m": actual_m,
            "required_depth_m": required_m,
            "headland_areas_m2": areas,
            "passed": passed,
            "status": "通过（模型判据）" if passed else "未通过，需复核",
            "criterion": "规划地头深度 >= max(最小地头深度, 2×掉头距离, 3×垄距)，且两端地头区面积有效",
        }

    @staticmethod
    def plan_to_dict(case: FieldCase, plan: RidgePlan) -> Dict:
        return {
            "case_id": plan.case_id,
            "scheme_id": plan.scheme_id,
            "display_name": plan.display_name,
            "target_direction": plan.target_direction,
            "base_edge_index": plan.base_edge_index,
            "metrics": {
                "original_area_m2": plan.metrics.original_area_m2,
                "regularized_area_m2": plan.metrics.regularized_area_m2,
                "core_area_m2": plan.metrics.core_area_m2,
                "headland_area_m2": plan.metrics.headland_area_m2,
                "cut_area_m2": plan.metrics.cut_area_m2,
                "ridge_count": plan.metrics.ridge_count,
                "headland_ridge_count": plan.metrics.headland_ridge_count,
                "total_ridge_length_m": plan.metrics.total_ridge_length_m,
                "path_length_m": plan.metrics.path_length_m,
                "turn_count": plan.metrics.turn_count,
                "estimated_time_min": plan.metrics.estimated_time_min,
                "estimated_fuel_l": plan.metrics.estimated_fuel_l,
                "estimated_seedling_count": plan.metrics.estimated_seedling_count,
                "total_score": plan.metrics.total_score,
                "grade": plan.metrics.grade,
            },
            "headland_check": ReportExporter.headland_check(case, plan),
            "metric_values": [asdict(item) for item in plan.metrics.metric_values],
            "warnings": plan.warning_messages,
            "error": plan.error,
        }

    @staticmethod
    def build_text_report(case: FieldCase, plans: Dict[str, RidgePlan], recommended: RidgePlan) -> str:
        lines = [
            "通用垄线规划模型计算报告",
            "=" * 32,
            f"案例编号：{case.case_id}",
            f"案例名称：{case.name}",
            f"作物类型：{case.crop_name}",
            f"地块面积：{GeometryToolkit.area(case.vertices_array()):.2f} m2（约 {GeometryToolkit.area(case.vertices_array()) / MU_M2:.2f} 亩）",
            f"垄型参数：垄面 {case.ridge_spec.top_width_m:.2f} m，垄底 {case.ridge_spec.bottom_width_m:.2f} m，垄沟 {case.ridge_spec.ditch_width_m:.2f} m，垄距 {case.ridge_spec.pitch_m:.2f} m",
            f"机具参数：{case.machine_spec.model}，掉头距离 {case.machine_spec.turn_distance_m:.2f} m，速度 {case.machine_spec.speed_m_min:.1f} m/min，平均油耗 {case.machine_spec.fuel_l_per_h:.1f} L/h",
            f"障碍物数量：{len(case.obstacles)}",
            "",
            "方案对比：",
        ]
        for key in ("NS", "EW"):
            plan = plans[key]
            if plan.error:
                lines.append(f"- {plan.display_name}: 规划失败，原因：{plan.error}")
                continue
            m = plan.metrics
            headland = ReportExporter.headland_check(case, plan)
            recommended_mark = "，推荐" if plan.scheme_id == recommended.scheme_id else ""
            lines.append(
                f"- {plan.display_name}: 综合评分 {m.total_score:.2f}（{m.grade}），"
                f"核心垄数 {m.ridge_count}，有效垄线总长 {m.total_ridge_length_m:.1f} m，"
                f"总路径长度 {m.path_length_m:.1f} m，掉头次数 {m.turn_count} 次，"
                f"预计作业时间 {m.estimated_time_min:.1f} min，预计燃油消耗 {m.estimated_fuel_l:.1f} L，"
                f"地头区核查 {headland['status']}（规划 {headland['actual_depth_m']:.1f} m，"
                f"要求 {headland['required_depth_m']:.1f} m）{recommended_mark}"
            )
        lines.extend([
            "",
            f"推荐方案：{recommended.display_name}",
            "",
            "地头区核查判据：规划地头深度 >= max(最小地头深度, 2×掉头距离, 3×垄距)，且两端地头区面积有效。",
            "该结果仅为模型几何与参数核查，不替代现场安全确认。",
            "",
            "推荐方案指标体系：",
        ])
        for item in recommended.metrics.metric_values:
            lines.append(
                f"- {item.code} {item.name}: 原始值 {item.raw_value:.4f}{item.unit}，"
                f"标准分 {item.normalized_score:.2f}，权重 {item.weight:.2f}，加权 {item.weighted_score:.2f}"
            )
        if recommended.warning_messages:
            lines.append("")
            lines.append("风险提示：")
            for warning in recommended.warning_messages:
                lines.append(f"- {warning}")
        if case.note:
            lines.extend(["", f"备注：{case.note}"])
        return "\n".join(lines)

    @staticmethod
    def export_json(path: str, case: FieldCase, plans: Dict[str, RidgePlan], recommended: RidgePlan) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "case": {
                "case_id": case.case_id,
                "name": case.name,
                "vertices": case.vertices,
                "ridge_spec": asdict(case.ridge_spec),
                "machine_spec": asdict(case.machine_spec),
                "obstacles": [asdict(obstacle) for obstacle in case.obstacles],
                "preferred_direction": case.preferred_direction,
                "crop_name": case.crop_name,
                "note": case.note,
            },
            "plans": {key: ReportExporter.plan_to_dict(case, plan) for key, plan in plans.items()},
            "recommended_scheme": recommended.scheme_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def export_csv(path: str, case: FieldCase, plans: Dict[str, RidgePlan], recommended: RidgePlan) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        headers = [
            "scheme_id", "display_name", "score", "grade", "ridge_count",
            "headland_ridge_count", "regularized_area_m2", "core_area_m2",
            "cut_area_m2", "total_ridge_length_m", "path_length_m", "turn_count",
            "estimated_time_min", "estimated_fuel_l", "headland_actual_depth_m",
            "headland_required_depth_m", "headland_check_status", "is_recommended",
        ]
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for plan in plans.values():
                headland = ReportExporter.headland_check(case, plan)
                writer.writerow({
                    "scheme_id": plan.scheme_id,
                    "display_name": plan.display_name,
                    "score": f"{plan.metrics.total_score:.2f}",
                    "grade": plan.metrics.grade,
                    "ridge_count": plan.metrics.ridge_count,
                    "headland_ridge_count": plan.metrics.headland_ridge_count,
                    "regularized_area_m2": f"{plan.metrics.regularized_area_m2:.2f}",
                    "core_area_m2": f"{plan.metrics.core_area_m2:.2f}",
                    "cut_area_m2": f"{plan.metrics.cut_area_m2:.2f}",
                    "total_ridge_length_m": f"{plan.metrics.total_ridge_length_m:.2f}",
                    "path_length_m": f"{plan.metrics.path_length_m:.2f}",
                    "turn_count": plan.metrics.turn_count,
                    "estimated_time_min": f"{plan.metrics.estimated_time_min:.2f}",
                    "estimated_fuel_l": f"{plan.metrics.estimated_fuel_l:.2f}",
                    "headland_actual_depth_m": f"{headland['actual_depth_m']:.2f}",
                    "headland_required_depth_m": f"{headland['required_depth_m']:.2f}",
                    "headland_check_status": headland["status"],
                    "is_recommended": "是" if plan.scheme_id == recommended.scheme_id else "否",
                })


class PlanVisualizer:
    """规划结果可视化模块。"""

    COLORS = {
        "field": "#ebe7d8",
        "regularized": "#d8e5dc",
        "core": "#c8ddcf",
        "headland": "#e2cd98",
        "ridge": "#bd5d42",
        "path": "#263832",
        "outline": "#52645d",
        "A": "#b84f3b",
        "B": "#735b8e",
        "C": "#2f6b4f",
        "D": "#2e6f95",
    }

    def __init__(self, case: FieldCase):
        self.case = case

    def setup_axes(self, ax, vertices: np.ndarray) -> None:
        all_points = [vertices]
        if len(vertices) > 0:
            xy = np.vstack(all_points)
            min_x, min_y = np.min(xy, axis=0)
            max_x, max_y = np.max(xy, axis=0)
            pad = max(max_x - min_x, max_y - min_y, 1.0) * 0.08
            ax.set_xlim(min_x - pad, max_x + pad)
            ax.set_ylim(min_y - pad, max_y + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#fbfcf8")
        ax.grid(True, color="#ced7d1", alpha=0.55, linewidth=0.7)
        ax.set_xlabel("X / m", color="#41524b")
        ax.set_ylabel("Y / m", color="#41524b")
        ax.tick_params(colors="#5d6c66", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#9eaaa4")
            spine.set_linewidth(0.8)

    def draw_input(self, ax) -> None:
        vertices = self.case.vertices_array()
        self.setup_axes(ax, vertices)
        self._draw_polygon(ax, vertices, self.COLORS["field"], self.COLORS["outline"], 0.70, "原始地块")
        self.draw_obstacles(ax)
        for i, p in enumerate(vertices):
            ax.scatter([p[0]], [p[1]], s=36, color="#222222", zorder=5)
            ax.text(p[0], p[1], f" P{i + 1}", fontsize=10, va="bottom")
        ax.set_title("步骤一：原始地块与顶点坐标", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")

    def draw_regularization(self, ax, plan: RidgePlan) -> None:
        vertices = self.case.vertices_array()
        self.setup_axes(ax, vertices)
        self._draw_polygon(ax, vertices, self.COLORS["field"], self.COLORS["outline"], 0.35, "原始地块")
        self.draw_obstacles(ax)
        if not plan.is_valid():
            ax.text(0.5, 0.5, plan.error or "方案无效", transform=ax.transAxes, ha="center", va="center", color="red", fontsize=13)
            return
        self._draw_polygon(ax, plan.regularized_polygon, self.COLORS["regularized"], "#517a38", 0.58, "规整化作业区")
        for i, headland in enumerate(plan.headland_polygons):
            self._draw_polygon(ax, headland, self.COLORS["headland"], "#a8842b", 0.55, "地头区" if i == 0 else None)
        self._draw_polygon(ax, plan.core_polygon, self.COLORS["core"], "#1f7a42", 0.60, "核心区")
        ax.set_title(f"步骤二：{plan.display_name}规整化与分区", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")

    def draw_ridges(self, ax, plan: RidgePlan, frame: int = 10**9, show_path: bool = False) -> None:
        vertices = self.case.vertices_array()
        self.setup_axes(ax, vertices)
        self._draw_polygon(ax, vertices, self.COLORS["field"], self.COLORS["outline"], 0.24, "原始地块")
        self.draw_obstacles(ax)
        if not plan.is_valid():
            ax.text(0.5, 0.5, plan.error or "方案无效", transform=ax.transAxes, ha="center", va="center", color="red", fontsize=13)
            return
        self._draw_polygon(ax, plan.regularized_polygon, self.COLORS["regularized"], "#5b7d42", 0.35, "规整区")
        self._draw_polygon(ax, plan.core_polygon, self.COLORS["core"], "#1f7a42", 0.28, "核心区")
        active_idx, fraction = plan.animation_state(frame)
        full_count = len(plan.ridges) if frame >= plan.max_frame else max(0, active_idx)
        for i, ridge in enumerate(plan.ridges):
            if i < full_count:
                self._draw_polygon(ax, ridge.polygon, self.COLORS.get(ridge.phase, self.COLORS["ridge"]), "none", 0.72, None)
                self._draw_ridge_center_path(ax, ridge, color="#0b4f2f", alpha=0.75)
            elif i == active_idx:
                partial = self._partial_strip(ridge, fraction, self.case.ridge_spec.bottom_width_m)
                self._draw_polygon(ax, partial, "#f08c00", "#8a4d00", 0.90, "当前生成垄")
                self._draw_ridge_center_path(ax, ridge, color="#8a4d00", alpha=0.95)
            else:
                self._draw_polygon(ax, ridge.polygon, "#d8c7aa", "none", 0.18, None)
        if show_path:
            for phase, points in plan.phase_paths.items():
                if len(points) >= 2:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    ax.plot(xs, ys, color=self.COLORS.get(phase, self.COLORS["path"]), linewidth=1.25, alpha=0.9)
                    ax.scatter([xs[0]], [ys[0]], s=18, color=self.COLORS.get(phase, self.COLORS["path"]))
        title = "步骤四：作业路径规划" if show_path else "步骤三：垄线动态生成"
        ax.set_title(f"{title} - {plan.display_name}", fontsize=14, fontweight="bold")

    @staticmethod
    def _draw_ridge_center_path(ax, ridge: RidgeSegment, color: str, alpha: float) -> None:
        if not ridge.path_points or len(ridge.path_points) <= 2:
            return
        xs = [p[0] for p in ridge.path_points]
        ys = [p[1] for p in ridge.path_points]
        ax.plot(xs, ys, color=color, linewidth=1.4, alpha=alpha, linestyle="-")

    def draw_obstacles(self, ax) -> None:
        for obstacle in self.case.obstacles:
            if obstacle.obstacle_type == "point" and obstacle.polygon:
                center = obstacle.polygon[0]
                circle = patches.Circle(center, radius=obstacle.buffer_m, facecolor="#f4b4a8", edgecolor="#b42318", linewidth=1.4, alpha=0.62, label="障碍物")
                ax.add_patch(circle)
                ax.text(center[0], center[1], obstacle.obstacle_id, ha="center", va="center", fontsize=8, color="#7a1610", weight="bold")
            elif obstacle.obstacle_type == "line" and len(obstacle.polygon) >= 2:
                p, q = obstacle.polygon[0], obstacle.polygon[1]
                ax.plot([p[0], q[0]], [p[1], q[1]], color="#b42318", linewidth=max(2.0, obstacle.buffer_m), alpha=0.62, solid_capstyle="round", label="条状障碍")
                ax.text((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0, obstacle.obstacle_id, fontsize=8, color="#7a1610", weight="bold")

    def draw_comparison(self, ax_ns, ax_ew, ax_table, plans: Dict[str, RidgePlan], recommended: RidgePlan) -> None:
        self.draw_ridges(ax_ns, plans["NS"], plans["NS"].max_frame, show_path=True)
        ax_ns.set_title("方案一：南北向", fontsize=12, fontweight="bold")
        self.draw_ridges(ax_ew, plans["EW"], plans["EW"].max_frame, show_path=True)
        ax_ew.set_title("方案二：东西向", fontsize=12, fontweight="bold")
        ax_table.axis("off")
        ns = plans["NS"]
        ew = plans["EW"]
        n = ns.metrics
        e = ew.metrics
        ns_headland = ReportExporter.headland_check(self.case, ns)
        ew_headland = ReportExporter.headland_check(self.case, ew)

        def higher_text(ns_value: float, ew_value: float, unit: str) -> str:
            if math.isclose(ns_value, ew_value, rel_tol=0.0, abs_tol=1e-9):
                return "两方案相同。"
            label = "南北向" if ns_value > ew_value else "东西向"
            return f"{label}高 {abs(ns_value - ew_value):.1f}{unit}。"

        def lower_text(ns_value: float, ew_value: float, unit: str) -> str:
            if math.isclose(ns_value, ew_value, rel_tol=0.0, abs_tol=1e-9):
                return "两方案相同。"
            label = "南北向" if ns_value < ew_value else "东西向"
            return f"{label}少 {abs(ns_value - ew_value):.1f}{unit}。"

        score_label = "南北向" if recommended.target_direction == "NS" else "东西向"
        time_delta = abs(n.estimated_time_min - e.estimated_time_min)
        safety_judgement = (
            "均通过模型判据；现场仍需确认。"
            if ns_headland["passed"] and ew_headland["passed"]
            else "至少一方案未通过模型判据，需复核。"
        )
        rows = [
            ["综合评分", f"{n.total_score:.2f}（{n.grade}）", f"{e.total_score:.2f}（{e.grade}）",
             f"{score_label}高 {abs(n.total_score - e.total_score):.2f} 分，推荐{score_label}。"],
            ["核心区垄数", f"{n.ridge_count} 条", f"{e.ridge_count} 条",
             higher_text(n.ridge_count, e.ridge_count, " 条")],
            ["有效垄线总长", f"{n.total_ridge_length_m:.1f} m", f"{e.total_ridge_length_m:.1f} m",
             higher_text(n.total_ridge_length_m, e.total_ridge_length_m, " m")],
            ["总路径长度", f"{n.path_length_m:.1f} m", f"{e.path_length_m:.1f} m",
             lower_text(n.path_length_m, e.path_length_m, " m")],
            ["掉头次数", f"{n.turn_count} 次", f"{e.turn_count} 次",
             lower_text(n.turn_count, e.turn_count, " 次")],
            ["预计作业时间", f"{n.estimated_time_min:.1f} min", f"{e.estimated_time_min:.1f} min",
             f"{lower_text(n.estimated_time_min, e.estimated_time_min, ' min')[:-1]}（{time_delta / 60.0:.1f} h）。"],
            ["预计燃油消耗", f"{n.estimated_fuel_l:.1f} L", f"{e.estimated_fuel_l:.1f} L",
             lower_text(n.estimated_fuel_l, e.estimated_fuel_l, " L")],
            ["地头区核查（模型）",
             f"{ns_headland['status']}\n{ns_headland['actual_depth_m']:.1f}≥{ns_headland['required_depth_m']:.1f} m",
             f"{ew_headland['status']}\n{ew_headland['actual_depth_m']:.1f}≥{ew_headland['required_depth_m']:.1f} m",
             safety_judgement],
        ]
        table = ax_table.table(
            cellText=rows,
            colLabels=["指标", "南北向", "东西向", "判断"],
            cellLoc="center",
            loc="center",
            colWidths=[0.16, 0.18, 0.18, 0.48],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.2)
        for (row, _col), cell in table.get_celld().items():
            cell.set_height(0.105 if row else 0.11)
            cell.set_edgecolor("#9eaaa4")
            cell.set_linewidth(0.7)
            if row == 0:
                cell.set_facecolor("#234a3a")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            else:
                cell.set_facecolor("#f7f8f4" if row % 2 else "#eaf0eb")
        ax_table.set_title(
            f"步骤五：完整方案指标对比（推荐：{recommended.display_name}）",
            fontsize=12,
            fontweight="bold",
        )

    def save_comparison(self, plans: Dict[str, RidgePlan], recommended: RidgePlan, path: str) -> None:
        if plt is None or patches is None:
            raise RuntimeError("未安装 matplotlib，无法导出方案对比图。")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig = plt.figure(figsize=(14, 10), facecolor="#fbfcf8")
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08])
        ax_ns = fig.add_subplot(gs[0, 0])
        ax_ew = fig.add_subplot(gs[0, 1])
        ax_table = fig.add_subplot(gs[1, :])
        self.draw_comparison(ax_ns, ax_ew, ax_table, plans, recommended)
        fig.tight_layout(pad=1.8)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def save_plan(self, plan: RidgePlan, path: str) -> None:
        if plt is None or patches is None:
            raise RuntimeError("未安装 matplotlib，无法导出图片。")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig, ax = plt.subplots(figsize=(11, 8), facecolor="#f7f3e8")
        self._draw_polygon(ax, self.case.vertices_array(), self.COLORS["field"], self.COLORS["outline"], 0.45, "原始地块")
        if plan.is_valid():
            self._draw_polygon(ax, plan.regularized_polygon, self.COLORS["regularized"], self.COLORS["outline"], 0.55, "规整化地块")
            for head in plan.headland_polygons:
                self._draw_polygon(ax, head, self.COLORS["headland"], "#b58d2a", 0.45, "地头区")
            self._draw_polygon(ax, plan.core_polygon, self.COLORS["core"], "#247a45", 0.50, "核心区")
            for ridge in plan.ridges:
                self._draw_polygon(ax, ridge.polygon, self.COLORS.get(ridge.phase, self.COLORS["ridge"]), "none", 0.75, None)
            for points in plan.phase_paths.values():
                if len(points) >= 2:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    ax.plot(xs, ys, color=self.COLORS["path"], linewidth=1.0, alpha=0.85)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{self.case.name} - {plan.display_name}", fontsize=14)
        ax.set_xlabel("X / m")
        ax.set_ylabel("Y / m")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    @staticmethod
    def _draw_polygon(ax, poly: np.ndarray, facecolor: str, edgecolor: str, alpha: float, label: Optional[str]) -> None:
        if poly is None or len(poly) < 3:
            return
        patch = patches.Polygon(poly, closed=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.1, alpha=alpha, label=label)
        ax.add_patch(patch)

    @staticmethod
    def _partial_strip(ridge: RidgeSegment, fraction: float, width: float) -> np.ndarray:
        fraction = max(0.0, min(1.0, fraction))
        current = ridge.start + (ridge.end - ridge.start) * fraction
        direction = current - ridge.start
        if np.linalg.norm(direction) < EPS:
            direction = ridge.end - ridge.start
        normal = np.array([-direction[1], direction[0]]) / (np.linalg.norm(direction) + EPS) * (width / 2.0)
        return np.array([ridge.start + normal, current + normal, current - normal, ridge.start - normal], dtype=float)


class RidgePlanningDesktopApp:
    """默认启动的图形化垄线规划软件。"""

    def __init__(self) -> None:
        if tk is None or ttk is None or FigureCanvasTkAgg is None or plt is None:
            raise RuntimeError("当前环境缺少 Tkinter 或 matplotlib，无法启动图形界面。")
        self.library = CaseLibrary()
        self.current_case = self.library.get("standard_rectangle")
        self.engine = GenericRidgePlanningEngine(self.current_case)
        self.plans: Dict[str, RidgePlan] = {}
        self.recommended: Optional[RidgePlan] = None
        self.visualizer = PlanVisualizer(self.current_case)
        self.scheme_key = "NS"
        self.view_step = 1
        self.frame = 0
        self.playing = False
        self.after_id = None
        self.obstacles: List[Obstacle] = list(self.current_case.obstacles)
        self.pending_line_start: Optional[Tuple[float, float]] = None

        self.root = tk.Tk()
        self.root.title("露地蔬菜垄线规划工作台")
        self.root.geometry("1600x980")
        self.root.minsize(1280, 800)
        self._build_ui()
        self.load_case_to_form(self.current_case)
        self.run_planning()

    def _configure_styles(self) -> None:
        self.palette = {
            "ink": "#234a3a",
            "ink_dark": "#17372b",
            "paper": "#f4f6f1",
            "surface": "#fbfcf8",
            "line": "#c8d2cc",
            "muted": "#64736d",
            "blue": "#2e6f95",
            "gold": "#c6a15b",
            "clay": "#b65c3b",
        }
        self.root.configure(bg=self.palette["paper"])
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", font=("Microsoft YaHei UI", 10), background=self.palette["paper"], foreground="#25332e")
        style.configure("Workspace.TFrame", background=self.palette["paper"])
        style.configure("Sidebar.TFrame", background="#eef2ed")
        style.configure("Surface.TFrame", background=self.palette["surface"])
        style.configure("Section.TLabelframe", background="#eef2ed", bordercolor=self.palette["line"], relief="solid", borderwidth=1)
        style.configure("Section.TLabelframe.Label", background="#eef2ed", foreground=self.palette["ink"], font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("TLabel", background=self.palette["paper"], foreground="#25332e")
        style.configure("Muted.TLabel", background=self.palette["paper"], foreground=self.palette["muted"], font=("Microsoft YaHei UI", 9))
        style.configure("Sidebar.TLabel", background="#eef2ed", foreground="#25332e")
        style.configure("Field.TEntry", fieldbackground="#ffffff", foreground="#24312c", bordercolor="#aebbb4", lightcolor="#aebbb4", darkcolor="#aebbb4", padding=(7, 5))
        style.configure("TCombobox", fieldbackground="#ffffff", background="#ffffff", foreground="#24312c", padding=(7, 5), arrowsize=14)
        style.configure("Primary.TButton", background=self.palette["ink"], foreground="#ffffff", bordercolor=self.palette["ink"], padding=(12, 9), font=("Microsoft YaHei UI", 10, "bold"))
        style.map("Primary.TButton", background=[("active", "#2f604b"), ("pressed", self.palette["ink_dark"])])
        style.configure("Secondary.TButton", background="#ffffff", foreground=self.palette["ink"], bordercolor="#aebbb4", padding=(9, 6))
        style.map("Secondary.TButton", background=[("active", "#e2ebe5")])
        style.configure("Danger.TButton", background="#fff7f4", foreground="#8e3f2e", bordercolor="#d7a393", padding=(8, 6))
        style.configure("Step.TButton", background="#edf1ed", foreground="#53635c", bordercolor="#ccd5cf", padding=(11, 7))
        style.configure("StepActive.TButton", background=self.palette["blue"], foreground="#ffffff", bordercolor=self.palette["blue"], padding=(11, 7), font=("Microsoft YaHei UI", 9, "bold"))
        style.map("StepActive.TButton", background=[("active", "#3a7fa5")])
        style.configure("TCheckbutton", background="#eef2ed")
        style.configure("TRadiobutton", background="#eef2ed", foreground="#2c3a35")
        style.configure("Vertical.TScrollbar", background="#dbe3de", troughcolor="#eef2ed", arrowsize=12)

    def _section(self, parent, title: str):
        frame = ttk.LabelFrame(parent, text=title, padding=(12, 10), style="Section.TLabelframe")
        frame.pack(fill="x", pady=(0, 10))
        return frame

    def _field_row(self, parent, label: str, variable: tk.StringVar, row: int, width: int = 8, column: int = 0) -> None:
        ttk.Label(parent, text=label, style="Sidebar.TLabel").grid(row=row, column=column, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=variable, width=width, style="Field.TEntry").grid(
            row=row, column=column + 1, sticky="ew", padx=(6, 10 if column == 0 else 0), pady=3
        )

    def _metric_tile(self, parent, label: str, variable: tk.StringVar, column: int, accent: str) -> None:
        tile = tk.Frame(parent, bg="#f8faf7", highlightbackground="#cbd5cf", highlightthickness=1)
        tile.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 6, 0))
        tk.Frame(tile, bg=accent, width=4).pack(side="left", fill="y")
        body = tk.Frame(tile, bg="#f8faf7")
        body.pack(side="left", fill="both", expand=True, padx=10, pady=7)
        tk.Label(body, text=label, bg="#f8faf7", fg="#68766f", font=("Microsoft YaHei UI", 8)).pack(anchor="w")
        tk.Label(body, textvariable=variable, bg="#f8faf7", fg="#24312c", font=("Consolas", 12, "bold")).pack(anchor="w")

    def _build_ui(self) -> None:
        self._configure_styles()
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        header = tk.Frame(self.root, bg=self.palette["ink"], height=72)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(0, weight=1)
        brand = tk.Frame(header, bg=self.palette["ink"])
        brand.grid(row=0, column=0, sticky="w", padx=22, pady=11)
        tk.Label(brand, text="露地蔬菜垄线规划工作台", bg=self.palette["ink"], fg="#ffffff", font=("Microsoft YaHei UI", 18, "bold")).pack(anchor="w")
        tk.Label(brand, text="RIDGE LAYOUT / FIELD OPERATIONS", bg=self.palette["ink"], fg="#b9d0c5", font=("Consolas", 8)).pack(anchor="w")
        self.header_case_var = tk.StringVar(value="案例 --")
        tk.Label(header, textvariable=self.header_case_var, bg=self.palette["ink"], fg="#dce8e1", font=("Microsoft YaHei UI", 10)).grid(row=0, column=1, sticky="e", padx=22)

        workspace = ttk.Frame(self.root, style="Workspace.TFrame", padding=(12, 12, 12, 8))
        workspace.grid(row=1, column=0, sticky="nsew")
        workspace.grid_rowconfigure(0, weight=1)
        workspace.grid_columnconfigure(1, weight=1)

        left_shell = ttk.Frame(workspace, width=400, style="Sidebar.TFrame")
        left_shell.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left_shell.grid_propagate(False)
        left_shell.grid_rowconfigure(2, weight=1)
        left_shell.grid_columnconfigure(0, weight=1)

        right = ttk.Frame(workspace, style="Surface.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        intro = ttk.Frame(left_shell, style="Sidebar.TFrame", padding=(14, 10, 14, 6))
        intro.grid(row=0, column=0, sticky="ew")
        ttk.Label(intro, text="规划参数", style="Sidebar.TLabel", font=("Microsoft YaHei UI", 13, "bold")).pack(anchor="w")
        ttk.Label(intro, text="按步骤完成输入，最后统一计算", style="Sidebar.TLabel", foreground=self.palette["muted"]).pack(anchor="w")

        wizard_header = tk.Frame(left_shell, bg="#dfe7e2", highlightbackground="#c5d0ca", highlightthickness=1)
        wizard_header.grid(row=1, column=0, sticky="ew", padx=12, pady=(4, 8))
        self.input_step_var = tk.StringVar(value="步骤 1 / 4")
        self.input_title_var = tk.StringVar(value="地块与预设")
        tk.Label(wizard_header, textvariable=self.input_step_var, bg="#dfe7e2", fg="#66756e", font=("Consolas", 9, "bold")).pack(anchor="w", padx=12, pady=(8, 0))
        tk.Label(wizard_header, textvariable=self.input_title_var, bg="#dfe7e2", fg=self.palette["ink"], font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w", padx=12, pady=(0, 6))
        track = tk.Frame(wizard_header, bg="#dfe7e2")
        track.pack(fill="x", padx=12, pady=(0, 9))
        self.input_step_marks = []
        for index, label in enumerate(("地块", "参数", "障碍", "输出"), start=1):
            mark = tk.Label(track, text=f"{index} {label}", bg=self.palette["ink"] if index == 1 else "#c8d2cc", fg="#ffffff" if index == 1 else "#596860", font=("Microsoft YaHei UI", 8, "bold"), padx=7, pady=3)
            mark.pack(side="left", expand=True, fill="x", padx=(0 if index == 1 else 3, 0))
            self.input_step_marks.append(mark)

        page_host = ttk.Frame(left_shell, style="Sidebar.TFrame", padding=(12, 0, 12, 0))
        page_host.grid(row=2, column=0, sticky="nsew")
        page_host.grid_rowconfigure(0, weight=1)
        page_host.grid_columnconfigure(0, weight=1)
        self.input_pages = []
        for _ in range(4):
            page = ttk.Frame(page_host, style="Sidebar.TFrame")
            page.grid(row=0, column=0, sticky="nsew")
            self.input_pages.append(page)

        page_case = self.input_pages[0]
        case_frame = self._section(page_case, "预设案例")
        self.case_var = tk.StringVar(value=self.current_case.case_id)
        case_values = [f"{case.case_id} | {case.name}" for case in self.library.list_cases()]
        self.case_combo = ttk.Combobox(case_frame, textvariable=self.case_var, values=case_values, state="readonly")
        self.case_combo.pack(fill="x", pady=(0, 6))
        ttk.Button(case_frame, text="应用预设参数", command=self.on_case_selected, style="Secondary.TButton").pack(fill="x")
        ttk.Label(case_frame, text="预设会填入地块、垄型和机具参数，仍可继续修改。", style="Sidebar.TLabel", foreground=self.palette["muted"], wraplength=330).pack(anchor="w", pady=(7, 0))

        pts_frame = self._section(page_case, "地块顶点坐标  /  m")
        ttk.Label(pts_frame, text="点位", style="Sidebar.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(pts_frame, text="X", style="Sidebar.TLabel").grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(pts_frame, text="Y", style="Sidebar.TLabel").grid(row=0, column=2, sticky="w", padx=(8, 0))
        self.point_vars: List[Tuple[tk.StringVar, tk.StringVar]] = []
        for i in range(4):
            xv = tk.StringVar()
            yv = tk.StringVar()
            ttk.Label(pts_frame, text=f"P{i + 1}", style="Sidebar.TLabel", font=("Consolas", 10, "bold")).grid(row=i + 1, column=0, sticky="w", pady=3)
            ttk.Entry(pts_frame, textvariable=xv, width=12, style="Field.TEntry").grid(row=i + 1, column=1, sticky="ew", padx=(8, 4), pady=3)
            ttk.Entry(pts_frame, textvariable=yv, width=12, style="Field.TEntry").grid(row=i + 1, column=2, sticky="ew", padx=(4, 0), pady=3)
            self.point_vars.append((xv, yv))
        pts_frame.columnconfigure(1, weight=1)
        pts_frame.columnconfigure(2, weight=1)

        page_params = self.input_pages[1]
        param_frame = self._section(page_params, "垄型与机具参数")
        param_frame.columnconfigure(1, weight=1)
        param_frame.columnconfigure(3, weight=1)
        self.top_var = tk.StringVar()
        self.bottom_var = tk.StringVar()
        self.ditch_var = tk.StringVar()
        self.pitch_var = tk.StringVar()
        self.row_var = tk.StringVar()
        self.plant_var = tk.StringVar()
        self.turn_var = tk.StringVar()
        self.speed_var_text = tk.StringVar()
        # 平均油耗为显式可编辑参数，预设默认值为 5.0 L/h。
        self.fuel_var = tk.StringVar(value="5")
        self.min_headland_var = tk.StringVar()
        for field_index, (label, var) in enumerate([
            ("垄面/m", self.top_var),
            ("垄底/m", self.bottom_var),
            ("垄沟/m", self.ditch_var),
            ("垄距/m", self.pitch_var),
            ("行距/m", self.row_var),
            ("株距/m", self.plant_var),
            ("掉头/m", self.turn_var),
            ("速度/(m/min)", self.speed_var_text),
            ("油耗/(L/h)", self.fuel_var),
            ("地头/m", self.min_headland_var),
        ]):
            self._field_row(
                param_frame, label, var, row=field_index % 5,
                column=0 if field_index < 5 else 2,
            )

        ttk.Label(page_params, text="预计燃油消耗 = 预计作业时间 / 60 × 平均油耗。输入完成后进入下一步。", style="Sidebar.TLabel", foreground=self.palette["muted"], wraplength=340).pack(anchor="w", padx=2)

        page_obstacles = self.input_pages[2]
        obstacle_frame = self._section(page_obstacles, "障碍物避让")
        self.obstacle_mode_var = tk.StringVar(value="off")
        for text, value in [("关闭绘制", "off"), ("点状障碍物", "point"), ("纵向条状障碍", "line")]:
            ttk.Radiobutton(obstacle_frame, text=text, variable=self.obstacle_mode_var, value=value).pack(anchor="w", pady=1)
        self.point_obstacle_radius_var = tk.StringVar(value="3.0")
        self.line_obstacle_buffer_var = tk.StringVar(value="2.0")
        self.point_x_var = tk.StringVar()
        self.point_y_var = tk.StringVar()
        self.line_x1_var = tk.StringVar()
        self.line_y1_var = tk.StringVar()
        self.line_x2_var = tk.StringVar()
        self.line_y2_var = tk.StringVar()
        for label, var in [("点障碍半径/m", self.point_obstacle_radius_var), ("条带半宽/m", self.line_obstacle_buffer_var)]:
            row = ttk.Frame(obstacle_frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=14, style="Sidebar.TLabel").pack(side="left")
            ttk.Entry(row, textvariable=var, width=10, style="Field.TEntry").pack(side="left")
        coord_frame = ttk.Frame(obstacle_frame)
        coord_frame.pack(fill="x", pady=(4, 0))
        ttk.Label(coord_frame, text="点坐标", width=8).grid(row=0, column=0, sticky="w")
        ttk.Entry(coord_frame, textvariable=self.point_x_var, width=8).grid(row=0, column=1, padx=1)
        ttk.Entry(coord_frame, textvariable=self.point_y_var, width=8).grid(row=0, column=2, padx=1)
        ttk.Button(coord_frame, text="添加点", command=self.add_point_obstacle_from_form).grid(row=0, column=3, padx=(4, 0), sticky="ew")
        ttk.Label(coord_frame, text="线起点", width=8).grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Entry(coord_frame, textvariable=self.line_x1_var, width=8).grid(row=1, column=1, padx=1, pady=(2, 0))
        ttk.Entry(coord_frame, textvariable=self.line_y1_var, width=8).grid(row=1, column=2, padx=1, pady=(2, 0))
        ttk.Label(coord_frame, text="线终点", width=8).grid(row=2, column=0, sticky="w", pady=(2, 0))
        ttk.Entry(coord_frame, textvariable=self.line_x2_var, width=8).grid(row=2, column=1, padx=1, pady=(2, 0))
        ttk.Entry(coord_frame, textvariable=self.line_y2_var, width=8).grid(row=2, column=2, padx=1, pady=(2, 0))
        ttk.Button(coord_frame, text="添加线", command=self.add_line_obstacle_from_form).grid(row=1, column=3, rowspan=2, padx=(4, 0), sticky="nsew")
        coord_frame.columnconfigure(3, weight=1)
        row = ttk.Frame(obstacle_frame)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="撤销", command=self.undo_obstacle, style="Secondary.TButton").pack(side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(row, text="清空", command=self.clear_obstacles, style="Danger.TButton").pack(side="left", expand=True, fill="x", padx=(2, 0))

        ttk.Label(page_obstacles, text="也可以直接在右侧地块图中点击添加点状或条状障碍。", style="Sidebar.TLabel", foreground=self.palette["muted"], wraplength=340).pack(anchor="w", padx=2)

        page_output = self.input_pages[3]
        scheme_frame = self._section(page_output, "作业方向")
        self.scheme_var = tk.StringVar(value="NS")
        ttk.Radiobutton(scheme_frame, text="方案一：南北向主作业", variable=self.scheme_var, value="NS", command=self.on_scheme).pack(anchor="w")
        ttk.Radiobutton(scheme_frame, text="方案二：东西向主作业", variable=self.scheme_var, value="EW", command=self.on_scheme).pack(anchor="w")

        anim_frame = self._section(page_output, "动态播放")
        row = ttk.Frame(anim_frame)
        row.pack(fill="x")
        ttk.Button(row, text="播放", command=self.play, style="Secondary.TButton").pack(side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(row, text="暂停", command=self.pause, style="Secondary.TButton").pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(row, text="重播", command=self.replay, style="Secondary.TButton").pack(side="left", expand=True, fill="x", padx=(2, 0))
        row2 = ttk.Frame(anim_frame)
        row2.pack(fill="x", pady=(5, 0))
        ttk.Button(row2, text="上一步", command=self.prev_frame).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(row2, text="下一步", command=self.next_frame).pack(side="left", expand=True, fill="x", padx=(2, 0))
        self.play_speed_var = tk.DoubleVar(value=35.0)
        ttk.Scale(anim_frame, from_=5, to=90, variable=self.play_speed_var, orient="horizontal").pack(fill="x", pady=(6, 0))
        self.progress_var = tk.StringVar(value="")
        ttk.Label(anim_frame, textvariable=self.progress_var).pack(anchor="center", pady=(4, 0))

        export_frame = self._section(page_output, "输出")
        ttk.Button(export_frame, text="导出报告、表格和规划图", command=self.export_outputs, style="Secondary.TButton").pack(fill="x")

        wizard_nav = ttk.Frame(left_shell, style="Sidebar.TFrame", padding=(12, 8, 12, 12))
        wizard_nav.grid(row=3, column=0, sticky="ew")
        wizard_nav.grid_columnconfigure(0, weight=1)
        wizard_nav.grid_columnconfigure(1, weight=2)
        self.input_back_button = ttk.Button(wizard_nav, text="上一步", command=self._previous_input_step, style="Secondary.TButton")
        self.input_back_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.input_next_button = ttk.Button(wizard_nav, text="下一步", command=self._next_input_step, style="Primary.TButton")
        self.input_next_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        self.input_step = 1
        self._show_input_step(1)

        metrics = tk.Frame(right, bg="#eef3ef")
        metrics.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
        for i in range(4):
            metrics.grid_columnconfigure(i, weight=1)
        self.metric_score_var = tk.StringVar(value="--")
        self.metric_grade_var = tk.StringVar(value="--")
        self.metric_ridges_var = tk.StringVar(value="--")
        self.metric_time_var = tk.StringVar(value="--")
        self._metric_tile(metrics, "综合评分", self.metric_score_var, 0, self.palette["ink"])
        self._metric_tile(metrics, "方案等级", self.metric_grade_var, 1, self.palette["gold"])
        self._metric_tile(metrics, "核心垄数", self.metric_ridges_var, 2, self.palette["blue"])
        self._metric_tile(metrics, "预计耗时", self.metric_time_var, 3, self.palette["clay"])

        step_bar = ttk.Frame(right, style="Surface.TFrame", padding=(12, 0, 12, 8))
        step_bar.grid(row=1, column=0, sticky="ew")
        self.step_buttons = {}
        for i, (text, step) in enumerate([("地块", 1), ("规整化", 2), ("垄线生成", 3), ("路径组织", 4), ("方案对比", 5)]):
            step_bar.columnconfigure(i, weight=1)
            button = ttk.Button(step_bar, text=text, command=lambda s=step: self.set_step(s), style="StepActive.TButton" if step == 1 else "Step.TButton")
            button.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 4, 0))
            self.step_buttons[step] = button

        right_pane = ttk.Panedwindow(right, orient="vertical")
        right_pane.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 12))
        chart_frame = ttk.Frame(right_pane, style="Surface.TFrame")
        detail_frame = ttk.Frame(right_pane, style="Surface.TFrame", height=190)
        right_pane.add(chart_frame, weight=5)
        right_pane.add(detail_frame, weight=1)

        self.fig = plt.Figure(figsize=(12, 8.3), facecolor="#fbfcf8")
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        chart_widget = self.canvas.get_tk_widget()
        chart_widget.configure(bg="#fbfcf8", highlightthickness=1, highlightbackground="#c8d2cc")
        chart_widget.pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)

        detail_header = tk.Frame(detail_frame, bg=self.palette["ink"])
        detail_header.pack(fill="x")
        tk.Label(detail_header, text="运行摘要", bg=self.palette["ink"], fg="#ffffff", font=("Microsoft YaHei UI", 10, "bold")).pack(side="left", padx=12, pady=6)
        self.recommendation_var = tk.StringVar(value="等待计算")
        tk.Label(detail_header, textvariable=self.recommendation_var, bg=self.palette["ink"], fg="#cde0d6", font=("Microsoft YaHei UI", 9)).pack(side="right", padx=12)
        detail_body = tk.Frame(detail_frame, bg="#f8faf7")
        detail_body.pack(fill="both", expand=True)
        detail_scroll = ttk.Scrollbar(detail_body, orient="vertical")
        detail_scroll.pack(side="right", fill="y")
        self.info = tk.Text(detail_body, height=8, wrap="word", bg="#f8faf7", fg="#2d3b36", relief="flat", padx=12, pady=8, font=("Microsoft YaHei UI", 9), yscrollcommand=detail_scroll.set)
        self.info.pack(side="left", fill="both", expand=True)
        detail_scroll.configure(command=self.info.yview)

        self.status_var = tk.StringVar(value="准备就绪")
        status = tk.Frame(self.root, bg="#dfe7e2", height=28)
        status.grid(row=2, column=0, sticky="ew")
        status.grid_propagate(False)
        tk.Label(status, textvariable=self.status_var, bg="#dfe7e2", fg="#46564f", font=("Microsoft YaHei UI", 9)).pack(side="left", padx=14)
        tk.Label(status, text="坐标单位 m  |  面积 m²  |  时间 min", bg="#dfe7e2", fg="#68766f", font=("Consolas", 8)).pack(side="right", padx=14)

    def _show_input_step(self, step: int) -> None:
        step = max(1, min(4, step))
        self.input_step = step
        titles = {1: "地块与预设", 2: "垄型与机具", 3: "障碍物设置", 4: "方向、播放与输出"}
        self.input_step_var.set(f"步骤 {step} / 4")
        self.input_title_var.set(titles[step])
        self.input_pages[step - 1].tkraise()
        for index, mark in enumerate(self.input_step_marks, start=1):
            active = index == step
            complete = index < step
            mark.configure(
                bg=self.palette["ink"] if active else ("#7f9c8d" if complete else "#c8d2cc"),
                fg="#ffffff" if active or complete else "#596860",
            )
        self.input_back_button.configure(state="disabled" if step == 1 else "normal")
        self.input_next_button.configure(text="计算并查看对比" if step == 4 else "下一步")
        if hasattr(self, "status_var"):
            self.status_var.set(f"输入向导：{titles[step]}")

    def _previous_input_step(self) -> None:
        self._show_input_step(self.input_step - 1)

    def _next_input_step(self) -> None:
        if self.input_step < 4:
            self._show_input_step(self.input_step + 1)
            return
        self.run_planning()
        if self.plans and self.recommended is not None:
            self.set_step(5)

    def load_case_to_form(self, case: FieldCase) -> None:
        self.case_var.set(f"{case.case_id} | {case.name}")
        self.header_case_var.set(f"{case.name}  /  {case.crop_name}")
        self.obstacles = list(case.obstacles)
        self.pending_line_start = None
        for (xv, yv), (x, y) in zip(self.point_vars, case.vertices):
            xv.set(f"{x:g}")
            yv.set(f"{y:g}")
        self.top_var.set(f"{case.ridge_spec.top_width_m:g}")
        self.bottom_var.set(f"{case.ridge_spec.bottom_width_m:g}")
        self.ditch_var.set(f"{case.ridge_spec.ditch_width_m:g}")
        self.pitch_var.set(f"{case.ridge_spec.pitch_m:g}")
        self.row_var.set(f"{case.ridge_spec.row_spacing_m:g}")
        self.plant_var.set(f"{case.ridge_spec.plant_spacing_m:g}")
        self.turn_var.set(f"{case.machine_spec.turn_distance_m:g}")
        self.speed_var_text.set(f"{case.machine_spec.speed_m_min:g}")
        self.fuel_var.set(f"{case.machine_spec.fuel_l_per_h:g}")
        self.min_headland_var.set(f"{case.machine_spec.min_headland_m:g}")

    def on_case_selected(self, *_args) -> None:
        case_id = self.case_var.get().split("|", 1)[0].strip()
        if not case_id:
            return
        self.current_case = self.library.get(case_id)
        self.load_case_to_form(self.current_case)
        # 应用预设后立即重算，避免右侧继续显示上一个案例的旧规划结果。
        self.run_planning()
        self.status_var.set(f"已应用预设：{self.current_case.name}。可继续修改，完成后在第四步统一计算。")

    def read_case_from_form(self) -> FieldCase:
        vertices = [(float(x.get()), float(y.get())) for x, y in self.point_vars]
        ridge = RidgeSpec(
            top_width_m=float(self.top_var.get()),
            bottom_width_m=float(self.bottom_var.get()),
            ditch_width_m=float(self.ditch_var.get()),
            pitch_m=float(self.pitch_var.get()),
            row_spacing_m=float(self.row_var.get()),
            plant_spacing_m=float(self.plant_var.get()),
        )
        fuel_l_per_h = float(self.fuel_var.get())
        if fuel_l_per_h < 0:
            raise ValueError("机具平均油耗不能为负数。")
        machine = MachineSpec(
            model=self.current_case.machine_spec.model,
            turn_distance_m=float(self.turn_var.get()),
            min_headland_m=float(self.min_headland_var.get()),
            speed_m_min=float(self.speed_var_text.get()),
            turn_time_min=self.current_case.machine_spec.turn_time_min,
            fuel_l_per_h=fuel_l_per_h,
        )
        return FieldCase(
            case_id=self.current_case.case_id,
            name=self.current_case.name,
            vertices=vertices,
            ridge_spec=ridge,
            machine_spec=machine,
            obstacles=list(self.obstacles),
            preferred_direction=self.current_case.preferred_direction,
            crop_name=self.current_case.crop_name,
            note=self.current_case.note,
        )

    def on_plot_click(self, event) -> None:
        mode = self.obstacle_mode_var.get()
        if mode == "off" or event.xdata is None or event.ydata is None:
            return
        point = (float(event.xdata), float(event.ydata))
        self._fill_obstacle_coordinate_form(mode, point)
        try:
            if mode == "point":
                self.add_point_obstacle(point)
            elif mode == "line":
                if self.pending_line_start is None:
                    self.pending_line_start = point
                    self.refresh()
                    self.update_info(extra=f"\n\n条状障碍物：已记录起点 ({point[0]:.2f}, {point[1]:.2f})，请点击终点或在坐标框中填写两点后点击“添加线”。")
                else:
                    self.add_line_obstacle(self.pending_line_start, point)
        except Exception as exc:
            if messagebox:
                messagebox.showerror("障碍物参数错误", str(exc))

    def _fill_obstacle_coordinate_form(self, mode: str, point: Tuple[float, float]) -> None:
        if mode == "point":
            self.point_x_var.set(f"{point[0]:.2f}")
            self.point_y_var.set(f"{point[1]:.2f}")
        elif mode == "line":
            if self.pending_line_start is None:
                self.line_x1_var.set(f"{point[0]:.2f}")
                self.line_y1_var.set(f"{point[1]:.2f}")
            else:
                self.line_x2_var.set(f"{point[0]:.2f}")
                self.line_y2_var.set(f"{point[1]:.2f}")

    def add_point_obstacle_from_form(self) -> None:
        try:
            point = (float(self.point_x_var.get()), float(self.point_y_var.get()))
            self.add_point_obstacle(point)
        except Exception as exc:
            if messagebox:
                messagebox.showerror("点状障碍物参数错误", str(exc))

    def add_line_obstacle_from_form(self) -> None:
        try:
            p1 = (float(self.line_x1_var.get()), float(self.line_y1_var.get()))
            p2 = (float(self.line_x2_var.get()), float(self.line_y2_var.get()))
            self.add_line_obstacle(p1, p2)
        except Exception as exc:
            if messagebox:
                messagebox.showerror("条状障碍物参数错误", str(exc))

    def add_point_obstacle(self, point: Tuple[float, float]) -> None:
        radius = float(self.point_obstacle_radius_var.get())
        self.obstacles.append(Obstacle(
            obstacle_id=f"P{len(self.obstacles) + 1}",
            name="点状障碍物",
            polygon=[point],
            buffer_m=max(0.1, radius),
            obstacle_type="point",
        ))
        self.pending_line_start = None
        self.run_planning()

    def add_line_obstacle(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> None:
        if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < 1e-6:
            raise ValueError("条状障碍物两点不能重合。")
        buffer_m = float(self.line_obstacle_buffer_var.get())
        self.obstacles.append(Obstacle(
            obstacle_id=f"L{len(self.obstacles) + 1}",
            name="条状分割障碍",
            polygon=[p1, p2],
            buffer_m=max(0.1, buffer_m),
            obstacle_type="line",
        ))
        self.pending_line_start = None
        self.run_planning()

    def undo_obstacle(self) -> None:
        self.pending_line_start = None
        if self.obstacles:
            self.obstacles.pop()
        self.run_planning()

    def clear_obstacles(self) -> None:
        self.pending_line_start = None
        self.obstacles = []
        self.run_planning()

    def run_planning(self) -> None:
        try:
            self.status_var.set("正在计算地块方案...")
            self.root.update_idletasks()
            self.current_case = self.read_case_from_form()
            self.engine = GenericRidgePlanningEngine(self.current_case)
            self.plans = self.engine.generate_all()
            self.recommended = self.engine.recommend(self.plans)
            self.visualizer = PlanVisualizer(self.current_case)
            self.frame = 0
            self.refresh()
        except Exception as exc:
            self.status_var.set(f"输入需要检查：{exc}")
            if messagebox:
                messagebox.showerror("输入错误", str(exc))
            else:
                print(f"输入错误：{exc}")

    @property
    def current_plan(self) -> RidgePlan:
        return self.plans[self.scheme_key]

    def on_scheme(self) -> None:
        self.scheme_key = self.scheme_var.get()
        self.frame = 0
        self.refresh()

    def set_step(self, step: int) -> None:
        self.view_step = step
        if step == 5 and self.recommended is not None:
            self.scheme_key = self.recommended.target_direction
            self.scheme_var.set(self.scheme_key)
        for key, button in self.step_buttons.items():
            button.configure(style="StepActive.TButton" if key == step else "Step.TButton")
        if step in (3, 4) and self.frame == 0:
            self.frame = 1
        self.refresh()

    def play(self) -> None:
        self.playing = True
        if self.view_step not in (3, 4):
            self.view_step = 3
        self._tick()

    def pause(self) -> None:
        self.playing = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def replay(self) -> None:
        self.frame = 0
        self.playing = True
        if self.view_step not in (3, 4):
            self.view_step = 3
        self._tick()

    def prev_frame(self) -> None:
        self.pause()
        self.frame = max(0, self.frame - 1)
        if self.view_step not in (3, 4):
            self.view_step = 3
        self.refresh()

    def next_frame(self) -> None:
        self.pause()
        self.frame = min(self.current_plan.max_frame, self.frame + 1)
        if self.view_step not in (3, 4):
            self.view_step = 3
        self.refresh()

    def _tick(self) -> None:
        if not self.playing:
            return
        self.frame += 1
        if self.frame > self.current_plan.max_frame:
            self.playing = False
            self.frame = self.current_plan.max_frame
        self.refresh()
        if self.playing:
            delay = max(35, int(1000 / max(1.0, self.play_speed_var.get())))
            self.after_id = self.root.after(delay, self._tick)

    def refresh(self) -> None:
        if not self.plans or self.recommended is None:
            return
        self.fig.clf()
        if self.view_step == 1:
            ax = self.fig.add_subplot(111)
            self.visualizer.draw_input(ax)
            self._draw_pending_line_start(ax)
        elif self.view_step == 2:
            ax = self.fig.add_subplot(111)
            self.visualizer.draw_regularization(ax, self.current_plan)
            self._draw_pending_line_start(ax)
        elif self.view_step == 3:
            ax = self.fig.add_subplot(111)
            self.visualizer.draw_ridges(ax, self.current_plan, self.frame, show_path=False)
            self._draw_pending_line_start(ax)
        elif self.view_step == 4:
            ax = self.fig.add_subplot(111)
            self.visualizer.draw_ridges(ax, self.current_plan, self.frame, show_path=True)
            self._draw_pending_line_start(ax)
        else:
            gs = self.fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08])
            ax1 = self.fig.add_subplot(gs[0, 0])
            ax2 = self.fig.add_subplot(gs[0, 1])
            ax3 = self.fig.add_subplot(gs[1, :])
            self.visualizer.draw_comparison(ax1, ax2, ax3, self.plans, self.recommended)
        self.fig.patch.set_facecolor("#fbfcf8")
        self.fig.tight_layout(pad=1.6)
        self.canvas.draw_idle()
        self.update_info()

    def _draw_pending_line_start(self, ax) -> None:
        if self.pending_line_start is None:
            return
        x, y = self.pending_line_start
        ax.scatter([x], [y], s=70, color="#b42318", marker="x", linewidths=2.0, zorder=20)
        ax.text(x, y, " 条障碍起点", color="#b42318", fontsize=9, weight="bold")

    def update_info(self, extra: str = "") -> None:
        plan = self.current_plan
        idx, frac = plan.animation_state(self.frame)
        if idx >= 0:
            self.progress_var.set(f"第 {idx + 1}/{len(plan.ridges)} 条 | {frac * 100:.0f}% | 帧 {self.frame}/{plan.max_frame}")
        else:
            self.progress_var.set(f"未开始 | 帧 0/{plan.max_frame}")
        metrics = plan.metrics
        self.metric_score_var.set(f"{metrics.total_score:.2f}")
        self.metric_grade_var.set(metrics.grade)
        self.metric_ridges_var.set(f"{metrics.ridge_count} 条")
        self.metric_time_var.set(f"{metrics.estimated_time_min:.1f} min")
        recommendation = self.recommended or plan
        self.recommendation_var.set(f"推荐：{recommendation.display_name}")
        step_name = {1: "地块", 2: "规整化", 3: "垄线生成", 4: "路径组织", 5: "方案对比"}.get(self.view_step, "规划")
        self.status_var.set(
            f"{step_name}  |  当前 {plan.display_name}  |  障碍物 {len(self.current_case.obstacles)} 个  |  {self.progress_var.get()}"
        )
        text = ReportExporter.build_text_report(self.current_case, self.plans, self.recommended or plan)
        text += f"\n\n障碍物避让：当前 {len(self.current_case.obstacles)} 个障碍物参与计算。"
        if self.current_case.obstacles:
            for obstacle in self.current_case.obstacles:
                coord = "; ".join(f"({x:.2f}, {y:.2f})" for x, y in obstacle.polygon)
                text += f"\n- {obstacle.obstacle_id} {obstacle.name}，类型 {obstacle.obstacle_type}，坐标 {coord}，缓冲 {obstacle.buffer_m:.1f} m"
        text += "\n\n指标权重：\n"
        for definition in EvaluationMetricSystem().definitions:
            text += f"{definition.code} {definition.name}，权重 {definition.weight:.2f}：{definition.description}\n"
        text += extra
        self.info.delete("1.0", "end")
        self.info.insert("1.0", text)

    def export_outputs(self) -> None:
        if self.recommended is None:
            return
        output_dir = DEFAULT_OUTPUT_DIR
        if filedialog:
            selected = filedialog.askdirectory(title="选择输出目录", initialdir=os.path.abspath(DEFAULT_OUTPUT_DIR))
            if selected:
                output_dir = selected
        os.makedirs(output_dir, exist_ok=True)
        run_case(self.current_case, output_dir, write_report=True, write_plot=True)
        if messagebox:
            messagebox.showinfo(
                "完整结果已导出",
                "已导出：详细 TXT 报告、完整 CSV 对比表、可追溯 JSON、"
                "推荐规划图和完整方案对比图。\n"
                f"目录：{output_dir}",
            )

    def run(self) -> None:
        self.root.mainloop()


def run_case(case: FieldCase, output_dir: str, write_report: bool, write_plot: bool) -> Tuple[Dict[str, RidgePlan], RidgePlan, str]:
    engine = GenericRidgePlanningEngine(case)
    plans = engine.generate_all()
    recommended = engine.recommend(plans)
    text_report = ReportExporter.build_text_report(case, plans, recommended)
    if write_report:
        os.makedirs(output_dir, exist_ok=True)
        txt_path = os.path.join(output_dir, f"{case.case_id}_planning_report.txt")
        json_path = os.path.join(output_dir, f"{case.case_id}_planning_result.json")
        csv_path = os.path.join(output_dir, f"{case.case_id}_scheme_compare.csv")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text_report)
        ReportExporter.export_json(json_path, case, plans, recommended)
        ReportExporter.export_csv(csv_path, case, plans, recommended)
    if write_plot:
        visualizer = PlanVisualizer(case)
        visualizer.save_plan(recommended, os.path.join(output_dir, f"{case.case_id}_{recommended.target_direction}_plan.png"))
        visualizer.save_comparison(
            plans,
            recommended,
            os.path.join(output_dir, f"{case.case_id}_scheme_comparison.png"),
        )
    return plans, recommended, text_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="露地蔬菜垄线规划模型")
    parser.add_argument("--cli", action="store_true", help="使用命令行模式；不加该参数时默认启动图形界面")
    parser.add_argument("--case", default="standard_rectangle", help="案例编号")
    parser.add_argument("--case-json", default="", help="外部案例库 JSON 文件")
    parser.add_argument("--list-cases", action="store_true", help="列出内置案例")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--report", action="store_true", help="导出 TXT/JSON/CSV 报告")
    parser.add_argument("--plot", action="store_true", help="导出推荐方案图片")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.cli and not args.list_cases:
        app = RidgePlanningDesktopApp()
        app.run()
        return
    library = CaseLibrary()
    if args.case_json:
        library.load_json(args.case_json)
    if args.list_cases:
        for case in library.list_cases():
            print(f"{case.case_id}\t{case.name}\t{case.crop_name}")
        return
    case = library.get(args.case)
    _, _, text_report = run_case(case, args.output_dir, args.report, args.plot)
    print(text_report)


if __name__ == "__main__":
    main()
