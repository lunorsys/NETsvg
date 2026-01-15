import random
import math
import svgwrite
import sys
from typing import Iterable, Iterator


# ============================================================
# OPTIONAL PROGRESS (tqdm if installed, fallback to stdout)
# ============================================================

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


def iterate_with_progress(
    iterable: Iterable[int],
    total: int,
    description: str,
    unit: str
) -> Iterator[int]:
    if tqdm is not None:
        yield from tqdm(iterable, total=total, desc=description, unit=unit)
        return

    if total <= 0:
        yield from iterable
        return

    update_every = max(1, total // 200)  # ~200 updates max
    current = 0

    for item in iterable:
        current += 1
        if current == 1 or current % update_every == 0 or current == total:
            percent = (current / total) * 100.0
            sys.stdout.write(f"\r{description}: {current}/{total} ({percent:5.1f}%)")
            sys.stdout.flush()
            if current == total:
                sys.stdout.write("\n")
        yield item


# ============================================================
# PARAMETERS
# ============================================================

# ----------------------------
# OUTPUT / RNG
# ----------------------------

output_filename = "net_cylinder.svg"

# None => random every run
# int  => reproducible run (also locks the boundary profile)
random_seed = None  # e.g. 12345


# ----------------------------
# CYLINDER (REAL WORLD)
# ----------------------------

# Base radius of the cylinder (in mm). The "edge" is derived from this.
cylinder_radius_mm = 250.0

# Thickness in Z direction (depth)
cylinder_height_mm = 220.0

# Margin/padding around the maximum projected radius (after projection)
canvas_margin_mm = 25.0


# ----------------------------
# ORGANIC CYLINDER EDGE (THIS IS THE ACTUAL "RAND")
# ----------------------------

# 0.0 => perfect circle boundary (exactly radius R everywhere)
# 1.0 => boundary can reach up to 2R (up to +R extension), depending on the generated profile
cylinder_boundary_irregularity_strength = 0.3

# How many "control points" around the circle define the profile.
# Lower => chunkier, more noticeable lobes ("zipfel").
# Higher => finer, more frequent waviness.
boundary_profile_control_point_count = 1

# How many circular smoothing passes are applied to the raw random profile.
# Higher => smoother edge (less spiky). Lower => more "growth" structures.
boundary_profile_smoothing_passes = 0


# ----------------------------
# SECTOR (CAKE SLICE) - CLOCK CONVENTION
# ----------------------------

# Clock angles:
# 0° = 12 o'clock, positive = clockwise
sector_start_angle_degrees = -120.0
sector_sweep_angle_degrees = 140.0  # 90 = quarter circle


# ----------------------------
# CAMERA / PROJECTION
# ----------------------------

# Must be > (cylinder_height_mm / 2). Larger => weaker perspective.
camera_distance_mm = 520.0

# Projection strength:
# 0.0 => effectively orthographic (no perspective scaling)
# 1.0 => physically-ish
# >1.0 => stronger perspective
perspective_strength = 0.4


# ----------------------------
# NODE DENSITY (CONSTANT OVER BASE AREA)
# ----------------------------

# Nodes per mm² over base disk area pi * r² (scaled by sector)
node_density_per_square_millimeter = 0.015


# ----------------------------
# NODES (DEPTH LOOK)
# ----------------------------

node_radius_min_mm = 0.20
node_radius_max_mm = 1.00

# 0.0 => white, 1.0 => original color
node_color_strength_background = 0.10
node_color_strength_foreground = 0.80

# 0.0 => no edge fade (recommended if you want "from above" without fade)
edge_fade_strength = 0.00

# Z distribution inside cylinder height:
# 1.0 = uniform; <1.0 => more extreme (front/back) => stronger depth
z_distribution_exponent = 0.90


# ----------------------------
# CONNECTIONS
# ----------------------------

required_neighbor_count = 3

connection_stroke_width_min_mm = 0.04
connection_stroke_width_max_mm = 0.30

# Long-edge lightening:
# 0.0 => long edges can become extremely light
# 0.15 => long edges never go below 15% strength
connection_distance_fade_minimum = 0.15

# Scale of "what counts as long" in projected space
connection_distance_fade_radius_multiplier = 1.6


# ----------------------------
# STYLE
# ----------------------------

stroke_color = "#312f2d"
fill_color = "#312f2d"


# ============================================================
# HELPERS
# ============================================================

def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalize_degrees(degrees_value: float) -> float:
    value = degrees_value % 360.0
    if value < 0.0:
        value += 360.0
    return value


def blend_hex_color_towards_white(hex_color: str, strength_01: float) -> str:
    """
    strength_01: 0.0 => white, 1.0 => original color
    """
    strength_01 = clamp(strength_01, 0.0, 1.0)

    color_value = hex_color.lstrip("#")
    red = int(color_value[0:2], 16)
    green = int(color_value[2:4], 16)
    blue = int(color_value[4:6], 16)

    red_blended = int(round(255.0 + (red - 255.0) * strength_01))
    green_blended = int(round(255.0 + (green - 255.0) * strength_01))
    blue_blended = int(round(255.0 + (blue - 255.0) * strength_01))

    return f"rgb({red_blended},{green_blended},{blue_blended})"


def create_svg_drawing_with_millimeter_viewbox(
    output_filename_value: str,
    canvas_width_millimeters: float,
    canvas_height_millimeters: float
) -> svgwrite.Drawing:
    drawing = svgwrite.Drawing(
        filename=output_filename_value,
        size=(f"{canvas_width_millimeters}mm", f"{canvas_height_millimeters}mm"),
        profile="tiny"
    )

    viewbox_width = int(round(canvas_width_millimeters))
    viewbox_height = int(round(canvas_height_millimeters))

    drawing.viewbox(0, 0, viewbox_width, viewbox_height)
    drawing.attribs["preserveAspectRatio"] = "xMidYMid meet"
    return drawing


# ============================================================
# ORGANIC CYLINDER EDGE (PROFILE)
# ============================================================

def build_boundary_radius_profile(
    boundary_irregularity_strength: float,
    control_point_count: int,
    smoothing_passes: int
) -> list[float]:
    """
    Returns a circular profile of multipliers.
    - strength = 0.0 => all multipliers = 1.0 (perfect circle)
    - strength = 1.0 => multipliers in [1.0 .. 2.0] (up to +1R)
    """
    if control_point_count <= 0:
        return [1.0]

    if boundary_irregularity_strength <= 0.0:
        return [1.0] * control_point_count

    multipliers = [
        1.0 + boundary_irregularity_strength * random.random()
        for _ in range(control_point_count)
    ]

    # Circular smoothing to create "organic" continuous edge
    for _ in range(max(0, smoothing_passes)):
        smoothed: list[float] = []
        for i in range(control_point_count):
            left = multipliers[(i - 1) % control_point_count]
            mid = multipliers[i]
            right = multipliers[(i + 1) % control_point_count]
            smoothed.append((left + 2.0 * mid + right) / 4.0)
        multipliers = smoothed

    return multipliers


def boundary_radius_for_clock_angle(
    base_radius_millimeters: float,
    clock_angle_degrees: float,
    boundary_profile: list[float]
) -> float:
    """
    Linear interpolation on the circular profile.
    clock_angle_degrees: 0° = 12 o'clock, positive clockwise
    """
    if not boundary_profile:
        return base_radius_millimeters

    profile_count = len(boundary_profile)
    angle_norm = normalize_degrees(clock_angle_degrees)
    t = (angle_norm / 360.0) * profile_count

    index0 = int(math.floor(t)) % profile_count
    index1 = (index0 + 1) % profile_count
    frac = t - math.floor(t)

    multiplier = boundary_profile[index0] * (1.0 - frac) + boundary_profile[index1] * frac
    return base_radius_millimeters * multiplier


def max_boundary_multiplier(boundary_profile: list[float]) -> float:
    if not boundary_profile:
        return 1.0
    return max(boundary_profile)


# ============================================================
# SECTOR / ANGLES
# ============================================================

def is_clock_angle_inside_sector(
    clock_angle_degrees: float,
    sector_start_clock_degrees: float,
    sector_sweep_clock_degrees: float
) -> bool:
    if sector_sweep_clock_degrees >= 360.0:
        return True

    start = normalize_degrees(sector_start_clock_degrees)
    sweep = max(0.0, sector_sweep_clock_degrees)
    end = start + sweep

    angle = normalize_degrees(clock_angle_degrees)

    if end <= 360.0:
        return start <= angle <= end

    end_wrapped = end - 360.0
    return angle >= start or angle <= end_wrapped


def compute_sector_bounds_in_svg_coordinates(
    max_projected_radius_millimeters: float,
    sector_start_clock_degrees: float,
    sector_sweep_clock_degrees: float
) -> tuple[float, float, float, float]:
    """
    Returns bounds (min_x, max_x, min_y, max_y) in local coordinates,
    where the wedge center is (0,0) and +Y points downward (SVG).
    """
    def point_on_circle(clock_degrees_value: float) -> tuple[float, float]:
        # clock -> math: math_deg = 90 - clock_deg
        math_radians = math.radians(90.0 - clock_degrees_value)
        x = math.cos(math_radians) * max_projected_radius_millimeters
        y = -math.sin(math_radians) * max_projected_radius_millimeters
        return x, y

    candidate_points: list[tuple[float, float]] = [(0.0, 0.0)]

    start_clock = sector_start_clock_degrees
    end_clock = sector_start_clock_degrees + sector_sweep_clock_degrees

    candidate_points.append(point_on_circle(start_clock))
    candidate_points.append(point_on_circle(end_clock))

    for cardinal_clock_degrees in (0.0, 90.0, 180.0, 270.0):
        if is_clock_angle_inside_sector(cardinal_clock_degrees, sector_start_clock_degrees, sector_sweep_clock_degrees):
            candidate_points.append(point_on_circle(cardinal_clock_degrees))

    x_values = [p[0] for p in candidate_points]
    y_values = [p[1] for p in candidate_points]
    return min(x_values), max(x_values), min(y_values), max(y_values)


# ============================================================
# CAMERA / PROJECTION
# ============================================================

def compute_max_perspective_scale_for_cylinder(
    camera_distance_millimeters: float,
    cylinder_half_height_millimeters: float,
    perspective_multiplier: float
) -> float:
    if perspective_multiplier <= 0.0:
        return 1.0

    denominator = camera_distance_millimeters - cylinder_half_height_millimeters
    if denominator <= 0.0001:
        denominator = 0.0001
    return (camera_distance_millimeters / denominator) ** perspective_multiplier


def project_3d_to_2d(
    x_millimeters: float,
    y_millimeters: float,
    z_millimeters: float,
    center_x_millimeters: float,
    center_y_millimeters: float,
    camera_distance_millimeters: float,
    perspective_multiplier: float
) -> tuple[float, float, float]:
    if perspective_multiplier <= 0.0:
        projected_x_millimeters = center_x_millimeters + x_millimeters
        projected_y_millimeters = center_y_millimeters + y_millimeters
        return projected_x_millimeters, projected_y_millimeters, 1.0

    denominator = camera_distance_millimeters - z_millimeters
    if abs(denominator) < 0.0001:
        denominator = 0.0001

    perspective_scale = (camera_distance_millimeters / denominator) ** perspective_multiplier
    projected_x_millimeters = center_x_millimeters + x_millimeters * perspective_scale
    projected_y_millimeters = center_y_millimeters + y_millimeters * perspective_scale
    return projected_x_millimeters, projected_y_millimeters, perspective_scale


def depth_from_z_for_cylinder(z_millimeters: float, cylinder_half_height_millimeters: float) -> float:
    return clamp(
        (z_millimeters + cylinder_half_height_millimeters) / (2.0 * cylinder_half_height_millimeters),
        0.0,
        1.0
    )


def edge_fade(
    projected_x_millimeters: float,
    projected_y_millimeters: float,
    center_x_millimeters: float,
    center_y_millimeters: float,
    radius_millimeters: float,
    fade_strength_value: float
) -> float:
    if fade_strength_value <= 0.0:
        return 1.0

    delta_x = projected_x_millimeters - center_x_millimeters
    delta_y = projected_y_millimeters - center_y_millimeters
    distance = math.hypot(delta_x, delta_y)

    normalized = clamp(distance / radius_millimeters, 0.0, 1.0)
    t = normalized * normalized
    return clamp(1.0 - t * fade_strength_value, 0.0, 1.0)


# ============================================================
# POINT GENERATION (uses boundary_profile => organic cylinder edge)
# ============================================================

def generate_point_in_cylinder_sector_with_constant_base_density(
    base_radius_millimeters: float,
    cylinder_half_height_millimeters: float,
    boundary_profile: list[float],
    z_exponent: float,
    sector_start_clock_degrees: float,
    sector_sweep_clock_degrees: float
) -> tuple[float, float, float]:
    """
    Uniform density inside an angle-dependent boundary radius.
    This makes the CYLINDER EDGE organic (continuous), not just random outliers.
    """
    random_clock_degrees = sector_start_clock_degrees + sector_sweep_clock_degrees * random.random()

    boundary_radius_millimeters = boundary_radius_for_clock_angle(
        base_radius_millimeters=base_radius_millimeters,
        clock_angle_degrees=random_clock_degrees,
        boundary_profile=boundary_profile
    )

    math_angle_radians = math.radians(90.0 - random_clock_degrees)

    # Uniform in the (angle-dependent) disk-sector: r = R(angle) * sqrt(u)
    radius_in_disk = boundary_radius_millimeters * math.sqrt(random.random())

    x_millimeters = math.cos(math_angle_radians) * radius_in_disk
    y_millimeters = -math.sin(math_angle_radians) * radius_in_disk

    sign = -1.0 if random.random() < 0.5 else 1.0
    z_millimeters = sign * cylinder_half_height_millimeters * (random.random() ** z_exponent)

    return x_millimeters, y_millimeters, z_millimeters


# ============================================================
# GRAPH BUILDING
# ============================================================

def build_edges_with_nearest_neighbors_and_stitch(
    nodes: list[tuple[float, float, float, float, float, float, float, float]],
    required_neighbor_count_value: int
) -> set[tuple[int, int]]:
    node_count_actual = len(nodes)
    if node_count_actual <= 1:
        return set()

    def compute_3d_distance(first_index: int, second_index: int) -> float:
        x1_3d, y1_3d, z1_3d, _, _, _, _, _ = nodes[first_index]
        x2_3d, y2_3d, z2_3d, _, _, _, _, _ = nodes[second_index]
        return math.sqrt(
            (x2_3d - x1_3d) ** 2 +
            (y2_3d - y1_3d) ** 2 +
            (z2_3d - z1_3d) ** 2
        )

    edges: set[tuple[int, int]] = set()

    # 1) KNN edges
    for node_index in iterate_with_progress(range(node_count_actual), node_count_actual, "Edges (KNN)", "node"):
        distance_candidates: list[tuple[float, int]] = []
        for other_index in range(node_count_actual):
            if other_index == node_index:
                continue
            distance_candidates.append((compute_3d_distance(node_index, other_index), other_index))

        distance_candidates.sort(key=lambda item: item[0])

        for _, neighbor_index in distance_candidates[:required_neighbor_count_value]:
            a = min(node_index, neighbor_index)
            b = max(node_index, neighbor_index)
            edges.add((a, b))

    # 2) Stitch components
    adjacency_by_index: list[list[int]] = [[] for _ in range(node_count_actual)]
    for a, b in edges:
        adjacency_by_index[a].append(b)
        adjacency_by_index[b].append(a)

    def collect_component(start_index: int, visited: set[int]) -> set[int]:
        stack = [start_index]
        component: set[int] = set()
        visited.add(start_index)

        while stack:
            current = stack.pop()
            component.add(current)
            for neighbor in adjacency_by_index[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)

        return component

    def find_components() -> list[set[int]]:
        visited: set[int] = set()
        components: list[set[int]] = []
        for index in range(node_count_actual):
            if index in visited:
                continue
            components.append(collect_component(index, visited))
        return components

    components = find_components()

    stitch_iteration = 0
    while len(components) > 1:
        stitch_iteration += 1
        components.sort(key=len)
        smallest_component = components[0]
        all_other_nodes = [n for component in components[1:] for n in component]

        best_distance = None
        best_a = None
        best_b = None

        for node_in_smallest in smallest_component:
            for node_outside in all_other_nodes:
                distance = compute_3d_distance(node_in_smallest, node_outside)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_a = node_in_smallest
                    best_b = node_outside

        if best_a is None or best_b is None:
            break

        a = min(best_a, best_b)
        b = max(best_a, best_b)

        if (a, b) not in edges:
            edges.add((a, b))
            adjacency_by_index[a].append(b)
            adjacency_by_index[b].append(a)

        components = find_components()

        if tqdm is None:
            sys.stdout.write(f"\rStitch: iteration {stitch_iteration}, components: {len(components)}")
            sys.stdout.flush()

    if tqdm is None and stitch_iteration > 0:
        sys.stdout.write("\n")

    return edges


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    # RNG
    if random_seed is not None:
        random.seed(random_seed)

    # Build a stable organic boundary ONCE per run (this is the cylinder edge)
    boundary_profile = build_boundary_radius_profile(
        boundary_irregularity_strength=cylinder_boundary_irregularity_strength,
        control_point_count=boundary_profile_control_point_count,
        smoothing_passes=boundary_profile_smoothing_passes
    )

    cylinder_half_height_mm = cylinder_height_mm / 2.0

    max_perspective_scale = compute_max_perspective_scale_for_cylinder(
        camera_distance_millimeters=camera_distance_mm,
        cylinder_half_height_millimeters=cylinder_half_height_mm,
        perspective_multiplier=perspective_strength
    )

    # Canvas must be large enough for the maximum possible boundary radius
    effective_max_radius_mm = cylinder_radius_mm * max_boundary_multiplier(boundary_profile)
    max_projected_radius_mm = effective_max_radius_mm * max_perspective_scale

    sector_min_x, sector_max_x, sector_min_y, sector_max_y = compute_sector_bounds_in_svg_coordinates(
        max_projected_radius_millimeters=max_projected_radius_mm,
        sector_start_clock_degrees=sector_start_angle_degrees,
        sector_sweep_clock_degrees=sector_sweep_angle_degrees
    )

    canvas_width_mm = (sector_max_x - sector_min_x) + 2.0 * canvas_margin_mm
    canvas_height_mm = (sector_max_y - sector_min_y) + 2.0 * canvas_margin_mm

    cylinder_center_x_mm = canvas_margin_mm - sector_min_x
    cylinder_center_y_mm = canvas_margin_mm - sector_min_y

    sector_fraction = clamp(sector_sweep_angle_degrees / 360.0, 0.0, 1.0)
    projected_sector_area_square_mm = math.pi * effective_max_radius_mm * effective_max_radius_mm * sector_fraction
    node_count = max(1, int(round(projected_sector_area_square_mm * node_density_per_square_millimeter)))

    if node_count > 6000:
        print(
            f"WARNING: node_count={node_count} is very high. "
            "Edge building is O(n^2) and will be extremely slow.\n"
            "Consider lowering node_density_per_square_millimeter or switching to a spatial index (k-d tree)."
        )

    drawing = create_svg_drawing_with_millimeter_viewbox(
        output_filename_value=output_filename,
        canvas_width_millimeters=canvas_width_mm,
        canvas_height_millimeters=canvas_height_mm
    )

    # (x3d, y3d, z3d, x2d, y2d, depth01, radius_mm, color_strength)
    nodes: list[tuple[float, float, float, float, float, float, float, float]] = []

    for _ in iterate_with_progress(range(node_count), node_count, "Nodes", "node"):
        x_3d_mm, y_3d_mm, z_3d_mm = generate_point_in_cylinder_sector_with_constant_base_density(
            base_radius_millimeters=cylinder_radius_mm,
            cylinder_half_height_millimeters=cylinder_half_height_mm,
            boundary_profile=boundary_profile,
            z_exponent=z_distribution_exponent,
            sector_start_clock_degrees=sector_start_angle_degrees,
            sector_sweep_clock_degrees=sector_sweep_angle_degrees
        )

        projected_x_mm, projected_y_mm, perspective_scale = project_3d_to_2d(
            x_millimeters=x_3d_mm,
            y_millimeters=y_3d_mm,
            z_millimeters=z_3d_mm,
            center_x_millimeters=cylinder_center_x_mm,
            center_y_millimeters=cylinder_center_y_mm,
            camera_distance_millimeters=camera_distance_mm,
            perspective_multiplier=perspective_strength
        )

        depth_01 = depth_from_z_for_cylinder(z_3d_mm, cylinder_half_height_mm)

        base_node_radius_mm = node_radius_min_mm + depth_01 * (node_radius_max_mm - node_radius_min_mm)
        radius_mm = base_node_radius_mm * (0.85 + 0.15 * perspective_scale)

        edge_multiplier = edge_fade(
            projected_x_millimeters=projected_x_mm,
            projected_y_millimeters=projected_y_mm,
            center_x_millimeters=cylinder_center_x_mm,
            center_y_millimeters=cylinder_center_y_mm,
            radius_millimeters=max_projected_radius_mm,
            fade_strength_value=edge_fade_strength
        )

        node_color_strength = clamp(
            (node_color_strength_background + depth_01 * (node_color_strength_foreground - node_color_strength_background))
            * edge_multiplier,
            0.0,
            1.0
        )

        nodes.append((
            x_3d_mm, y_3d_mm, z_3d_mm,
            projected_x_mm, projected_y_mm,
            depth_01,
            radius_mm,
            node_color_strength
        ))

    edges = build_edges_with_nearest_neighbors_and_stitch(
        nodes=nodes,
        required_neighbor_count_value=required_neighbor_count
    )

    def mean_depth_of_edge(edge: tuple[int, int]) -> float:
        a, b = edge
        return (nodes[a][5] + nodes[b][5]) / 2.0

    sorted_edges = sorted(edges, key=mean_depth_of_edge)

    # Draw lines (back to front)
    for edge_index in iterate_with_progress(range(len(sorted_edges)), len(sorted_edges), "Render lines", "line"):
        a, b = sorted_edges[edge_index]

        _, _, _, x1_2d, y1_2d, _, _, strength_a = nodes[a]
        _, _, _, x2_2d, y2_2d, _, _, strength_b = nodes[b]

        mean_depth = mean_depth_of_edge((a, b))
        stroke_width = connection_stroke_width_min_mm + mean_depth * (
            connection_stroke_width_max_mm - connection_stroke_width_min_mm
        )

        endpoint_color_strength = clamp((strength_a + strength_b) / 2.0, 0.0, 1.0)

        projected_distance = math.hypot(x2_2d - x1_2d, y2_2d - y1_2d)
        projected_distance_factor = clamp(
            1.0 - (projected_distance / (max_projected_radius_mm * connection_distance_fade_radius_multiplier)),
            connection_distance_fade_minimum,
            1.0
        )

        edge_color_strength = clamp(endpoint_color_strength * projected_distance_factor, 0.0, 1.0)
        edge_stroke_color = blend_hex_color_towards_white(stroke_color, edge_color_strength)

        drawing.add(drawing.line(
            start=(x1_2d, y1_2d),
            end=(x2_2d, y2_2d),
            stroke=edge_stroke_color,
            stroke_width=stroke_width
        ))

    # Draw nodes (back to front)
    nodes_sorted_by_depth = sorted(nodes, key=lambda n: n[5])

    for node_index in iterate_with_progress(range(len(nodes_sorted_by_depth)), len(nodes_sorted_by_depth), "Render nodes", "node"):
        _, _, _, x_2d, y_2d, _, radius_mm, node_color_strength = nodes_sorted_by_depth[node_index]
        node_fill_color = blend_hex_color_towards_white(fill_color, node_color_strength)

        drawing.add(drawing.circle(
            center=(x_2d, y_2d),
            r=radius_mm,
            fill=node_fill_color
        ))

    drawing.save()

    print("SVG erzeugt:", output_filename)
    print("node_count:", node_count)
    print("sector_start_angle_degrees:", sector_start_angle_degrees)
    print("sector_sweep_angle_degrees:", sector_sweep_angle_degrees)
    print("cylinder_boundary_irregularity_strength:", cylinder_boundary_irregularity_strength)
    print("boundary_profile_control_point_count:", boundary_profile_control_point_count)
    print("boundary_profile_smoothing_passes:", boundary_profile_smoothing_passes)
    print("perspective_strength:", perspective_strength)
    print("edge_fade_strength:", edge_fade_strength)


if __name__ == "__main__":
    main()
