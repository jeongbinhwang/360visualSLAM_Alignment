import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.placement
import ifcopenshell.util.unit

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool

from shapely.geometry import Polygon, MultiPolygon, Point, LineString, GeometryCollection
from shapely.ops import unary_union

# print("RUNNING UPDATED FILE - LEVEL 3 MULTI-SPACE HALL UNION DEBUG VERSION (ALL IN METERS)")

# =========================================================
# USER SETTINGS
# =========================================================
ifc_path = "/home/jb/workspace/IFCdata/FWH/FWH_Architectural.ifc"
path = "/home/jb/workspace/360video/graph_codes/"
level = "LEVEL 3"   # "LEVEL 1", "LEVEL 2", "LEVEL 3", "LEVEL 4"

TARGET_IFC_CLASS = "IfcSpace"

if level.lower() == "level 2":
    TARGET_SPACE_NAME = ['94'] ## 2nd floor of FWH has only one hall -> its ID is 94
if level.lower() == "level 3":
    TARGET_SPACE_NAME = ['3300', '3300A', '3100A']
else:
    TARGET_SPACE_NAME = None

TARGET_LONGNAME_KEYWORD = "CIRC"    # hall spaces' element name always include "CIRC" --> YOU MUST CHANGE IT BASED ON YOUR IFC FILE CHARACTERISTICS
# TARGET_SPACE_NAME = None            # OR IF YOU DON'T WANT TO USE HALL INFORMATION, USE THIS INSTEAD OF UPPER LINE.

USE_OBJECT_LEVEL_FILTER = False
MODEL_EDGE_Z_TOL = 0.5              # meters
exclude_guids = {"1kCsIarjLCq8yHmVMBUpKt"}

DRAW_DEBUG_INFO = True
PRINT_SPACE_DEBUG = True
PRINT_HOLE_DEBUG = False

SAVE_PER_SPACE_DEBUG_CSV = True
SAVE_UNION_DEBUG_CSV = True
SAVE_2D_DEBUG_PLOT = True
ENABLE_OPEN3D_VIS = True

# boundary 포함용 tolerance
GEOM_BUFFER_EPS = 1e-7

# door inclusion margin (meters)
DOOR_INCLUDE_MARGIN = 0.5

# outlier filtering
ENABLE_OUTLIER_FILTER = True
OUTLIER_MAX_BBOX_SIZE = 200.0         # meters
OUTLIER_MAX_DIST_FROM_MEDIAN = 300.0  # meters

# =========================================================
# FLOOR HEIGHT THRESHOLD (meters)
# =========================================================
if level == "LEVEL 1":
    level_threshold = 0.0
elif level == "LEVEL 2":
    level_threshold = 8.128
elif level == "LEVEL 3":
    level_threshold = 15.24
elif level == "LEVEL 4":
    level_threshold = 20.117
else:
    raise ValueError(f"Unknown level: {level}")

# =========================================================
# IFC LOAD
# =========================================================
ifc_file = ifcopenshell.open(ifc_path)
unit_scale = ifcopenshell.util.unit.calculate_unit_scale(ifc_file)
print(f"IFC unit scale to meters: {unit_scale}")

storeys = ifc_file.by_type("IfcBuildingStorey")
target_storey = None
for storey in storeys:
    if level in (storey.Name or "").upper():
        target_storey = storey
        break

if target_storey is None:
    raise ValueError(f"{level} storey cannot be found.")

print(f"Found Storey: {target_storey.Name}")

# =========================================================
# STOREY MEMBERSHIP CHECK
# =========================================================
def is_in_storey(element, storey, visited=None):
    if visited is None:
        visited = set()

    eid = element.id()
    if eid in visited:
        return False
    visited.add(eid)

    for rel in ifc_file.get_inverse(element):
        if rel.is_a("IfcRelContainedInSpatialStructure"):
            if rel.RelatingStructure == storey:
                return True
            if is_in_storey(rel.RelatingStructure, storey, visited):
                return True

    for rel in ifc_file.get_inverse(element):
        if rel.is_a("IfcRelAggregates") or rel.is_a("IfcRelNests"):
            parent = rel.RelatingObject
            if parent is not None and is_in_storey(parent, storey, visited):
                return True

    return False

# =========================================================
# BASIC POLYGON HELPERS
# =========================================================
def polygon_area(poly_xy):
    x = poly_xy[:, 0]
    y = poly_xy[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

def ensure_ccw(poly_xy):
    if len(poly_xy) < 3:
        return poly_xy
    if polygon_area(poly_xy) < 0:
        return poly_xy[::-1]
    return poly_xy

def ensure_closed(poly_xy, tol=1e-8):
    if len(poly_xy) == 0:
        return poly_xy
    if np.linalg.norm(poly_xy[0] - poly_xy[-1]) < tol:
        return poly_xy
    return np.vstack([poly_xy, poly_xy[0]])

def remove_duplicate_consecutive_points(poly_xy, tol=1e-8):
    if len(poly_xy) == 0:
        return poly_xy
    cleaned = [poly_xy[0]]
    for p in poly_xy[1:]:
        if np.linalg.norm(p - cleaned[-1]) > tol:
            cleaned.append(p)
    cleaned = np.array(cleaned, dtype=float)

    if len(cleaned) >= 2 and np.linalg.norm(cleaned[0] - cleaned[-1]) < tol:
        cleaned = cleaned[:-1]

    return cleaned

def close_poly_xy(poly_xy, tol=1e-8):
    poly_xy = np.array(poly_xy, dtype=float)
    if len(poly_xy) == 0:
        return poly_xy
    if np.linalg.norm(poly_xy[0] - poly_xy[-1]) < tol:
        return poly_xy
    return np.vstack([poly_xy, poly_xy[0]])

def polygon_xy_to_xyz(poly_xy, z_value):
    return np.column_stack([poly_xy, np.full(len(poly_xy), z_value, dtype=float)])

# =========================================================
# TRANSFORM HELPERS
# =========================================================
def normalize(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n

def cartesian_to_np(pt):
    return np.array(pt.Coordinates, dtype=float)

def direction_to_np(d):
    return np.array(d.DirectionRatios, dtype=float)

def scale_translation_4x4(T, s):
    T = np.array(T, dtype=float).copy()
    T[0:3, 3] *= s
    return T

def axis2placement2d_to_matrix(axis2):
    # translation -> meters
    origin = cartesian_to_np(axis2.Location) * unit_scale

    if getattr(axis2, "RefDirection", None):
        x_axis = normalize(direction_to_np(axis2.RefDirection))
    else:
        x_axis = np.array([1.0, 0.0], dtype=float)

    y_axis = np.array([-x_axis[1], x_axis[0]], dtype=float)

    T = np.eye(3, dtype=float)
    T[0:2, 0] = x_axis
    T[0:2, 1] = y_axis
    T[0:2, 2] = origin[:2]
    return T

def axis2placement3d_to_matrix(axis3):
    # translation -> meters
    origin = cartesian_to_np(axis3.Location) * unit_scale

    if getattr(axis3, "Axis", None):
        z_axis = normalize(direction_to_np(axis3.Axis))
    else:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)

    if getattr(axis3, "RefDirection", None):
        x_axis = normalize(direction_to_np(axis3.RefDirection))
    else:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = normalize(y_axis)

    x_axis = np.cross(y_axis, z_axis)
    x_axis = normalize(x_axis)

    T = np.eye(4, dtype=float)
    T[0:3, 0] = x_axis
    T[0:3, 1] = y_axis
    T[0:3, 2] = z_axis
    T[0:3, 3] = origin
    return T

def to_4x4_from_3x3(T2):
    T4 = np.eye(4, dtype=float)
    T4[0:2, 0:2] = T2[0:2, 0:2]
    T4[0:2, 3] = T2[0:2, 2]
    return T4

def apply_4x4_to_2d(points_xy, T4):
    pts_h = np.column_stack([points_xy, np.zeros(len(points_xy)), np.ones(len(points_xy))])
    out = (T4 @ pts_h.T).T
    return out[:, :3]

# =========================================================
# IFC PROFILE / REPRESENTATION HELPERS
# =========================================================
def polyline_to_xy(polyline):
    pts = []
    for p in polyline.Points:
        coords = list(p.Coordinates)
        pts.append(coords[:2])
    return np.array(pts, dtype=float)

def extract_profile_curves(profile):
    if profile.is_a("IfcArbitraryProfileDefWithVoids"):
        outer = profile.OuterCurve
        if not outer.is_a("IfcPolyline"):
            raise NotImplementedError(f"OuterCurve type not supported: {outer.is_a()}")
        outer_xy = polyline_to_xy(outer)

        inner_list = []
        for inner in profile.InnerCurves:
            if not inner.is_a("IfcPolyline"):
                raise NotImplementedError(f"InnerCurve type not supported: {inner.is_a()}")
            inner_list.append(polyline_to_xy(inner))

        return outer_xy, inner_list

    elif profile.is_a("IfcArbitraryClosedProfileDef"):
        outer = profile.OuterCurve
        if not outer.is_a("IfcPolyline"):
            raise NotImplementedError(f"OuterCurve type not supported: {outer.is_a()}")
        outer_xy = polyline_to_xy(outer)
        return outer_xy, []

    else:
        raise NotImplementedError(f"Profile type not supported: {profile.is_a()}")

def extract_polygons_from_extruded_area_solid(item, object_placement_4x4):
    profile = item.SweptArea
    outer_xy_local, inner_xy_local_list = extract_profile_curves(profile)

    if PRINT_HOLE_DEBUG:
        print("\n[DEBUG extract_polygons_from_extruded_area_solid]")
        print("profile type:", profile.is_a())
        print("outer local pts:", len(outer_xy_local))
        print("num inner local curves:", len(inner_xy_local_list))

    # local profile points -> meters
    outer_xy_local = outer_xy_local * unit_scale
    inner_xy_local_list = [arr * unit_scale for arr in inner_xy_local_list]

    if getattr(profile, "Position", None):
        T_profile_2d = axis2placement2d_to_matrix(profile.Position)
        T_profile_4 = to_4x4_from_3x3(T_profile_2d)
    else:
        T_profile_4 = np.eye(4, dtype=float)

    if getattr(item, "Position", None):
        T_item_4 = axis2placement3d_to_matrix(item.Position)
    else:
        T_item_4 = np.eye(4, dtype=float)

    # all transforms now in meters
    T_total = object_placement_4x4 @ T_item_4 @ T_profile_4

    outer_xyz_world = apply_4x4_to_2d(outer_xy_local, T_total)
    inner_xyz_world_list = [apply_4x4_to_2d(inner_xy, T_total) for inner_xy in inner_xy_local_list]

    outer_world_xy = close_poly_xy(outer_xyz_world[:, :2])

    inner_world_xy_list = []
    for arr in inner_xyz_world_list:
        hole_xy = close_poly_xy(arr[:, :2])
        if len(hole_xy) >= 4:
            inner_world_xy_list.append(hole_xy)

    return {
        "outer_world_xy": outer_world_xy,
        "outer_world_xyz": polygon_xy_to_xyz(outer_world_xy, level_threshold),
        "inner_world_xy_list": inner_world_xy_list,
        "inner_world_xyz_list": [polygon_xy_to_xyz(p, level_threshold) for p in inner_world_xy_list],
    }

def extract_space_polygons_from_representation_world(space):
    if not space.Representation:
        raise ValueError("Space has no Representation.")

    reps = space.Representation.Representations
    if len(reps) == 0:
        raise ValueError("No representations found.")

    body_rep = None
    for rep in reps:
        if (rep.RepresentationIdentifier or "").lower() == "body":
            body_rep = rep
            break
    if body_rep is None:
        body_rep = reps[0]

    if len(body_rep.Items) == 0:
        raise ValueError("No representation items found.")

    if getattr(space, "ObjectPlacement", None):
        T_object_4 = ifcopenshell.util.placement.get_local_placement(space.ObjectPlacement)
        T_object_4 = np.array(T_object_4, dtype=float)
        # placement translation -> meters
        T_object_4 = scale_translation_4x4(T_object_4, unit_scale)
    else:
        T_object_4 = np.eye(4, dtype=float)

    all_outer_xy = []
    all_outer_xyz = []
    all_inner_xy = []
    all_inner_xyz = []
    unsupported_items = []

    for item in body_rep.Items:
        try:
            if item.is_a("IfcExtrudedAreaSolid"):
                result = extract_polygons_from_extruded_area_solid(item, T_object_4)

                if len(result["outer_world_xy"]) >= 4:
                    all_outer_xy.append(result["outer_world_xy"])
                    all_outer_xyz.append(result["outer_world_xyz"])

                for pxy, pxyz in zip(result["inner_world_xy_list"], result["inner_world_xyz_list"]):
                    if len(pxy) >= 4:
                        all_inner_xy.append(pxy)
                        all_inner_xyz.append(pxyz)
            else:
                unsupported_items.append(item.is_a())

        except Exception as e:
            unsupported_items.append(f"{item.is_a()} ({e})")

    if PRINT_SPACE_DEBUG:
        print(f"\n[SPACE DEBUG] Space Name={space.Name}, LongName={space.LongName}, GlobalId={space.GlobalId}")
        print(f"  body items = {len(body_rep.Items)}")
        print(f"  supported outer polygons = {len(all_outer_xy)}")
        print(f"  supported inner hole polygons = {len(all_inner_xy)}")
        if len(unsupported_items) > 0:
            print(f"  unsupported/skipped items = {unsupported_items}")

    if len(all_outer_xy) == 0:
        raise ValueError("No supported polygon extracted from representation items.")

    return {
        "outer_world_xy_list": all_outer_xy,
        "outer_world_xyz_list": all_outer_xyz,
        "inner_world_xy_list": all_inner_xy,
        "inner_world_xyz_list": all_inner_xyz,
    }

# =========================================================
# SHAPELY HELPERS
# =========================================================
def xy_to_shapely_polygon(poly_xy):
    poly_xy = np.asarray(poly_xy, dtype=float)
    poly_xy = remove_duplicate_consecutive_points(poly_xy)

    if len(poly_xy) < 3:
        return None

    if np.linalg.norm(poly_xy[0] - poly_xy[-1]) < 1e-8:
        poly_xy = poly_xy[:-1]

    if len(poly_xy) < 3:
        return None

    try:
        geom = Polygon(poly_xy[:, :2])
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            return None
        return geom
    except Exception:
        return None

def build_valid_geom_from_space_result(result):
    outer_geoms = []
    inner_geoms = []

    for poly in result["outer_world_xy_list"]:
        g = xy_to_shapely_polygon(poly)
        if g is not None and not g.is_empty:
            outer_geoms.append(g)

    for poly in result["inner_world_xy_list"]:
        g = xy_to_shapely_polygon(poly)
        if g is not None and not g.is_empty:
            inner_geoms.append(g)

    if len(outer_geoms) == 0:
        return None

    outer_union = unary_union(outer_geoms)

    if len(inner_geoms) > 0:
        inner_union = unary_union(inner_geoms)
        valid_geom = outer_union.difference(inner_union)
    else:
        valid_geom = outer_union

    if not valid_geom.is_valid:
        valid_geom = valid_geom.buffer(0)

    if valid_geom.is_empty:
        return None

    return valid_geom

def extract_polygon_rings_from_geom(geom):
    outer_list = []
    hole_list = []

    if geom is None or geom.is_empty:
        return outer_list, hole_list

    if isinstance(geom, Polygon):
        outer_list.append(np.array(geom.exterior.coords, dtype=float))
        for interior in geom.interiors:
            hole_list.append(np.array(interior.coords, dtype=float))

    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            outer_list.append(np.array(g.exterior.coords, dtype=float))
            for interior in g.interiors:
                hole_list.append(np.array(interior.coords, dtype=float))

    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            o, h = extract_polygon_rings_from_geom(g)
            outer_list.extend(o)
            hole_list.extend(h)

    return outer_list, hole_list

def filter_points_by_geom(points_xyz, geom, margin=0.0, eps=GEOM_BUFFER_EPS):
    if len(points_xyz) == 0:
        return np.empty((0, 3), dtype=float)

    test_geom = geom.buffer(margin + eps)
    keep = []

    for p in points_xyz:
        pt = Point(float(p[0]), float(p[1]))
        if test_geom.covers(pt):
            keep.append(p)

    if len(keep) == 0:
        return np.empty((0, 3), dtype=float)

    return np.asarray(keep, dtype=float)

def object_intersects_geom(points_xyz, geom, margin=0.0, eps=GEOM_BUFFER_EPS):
    if len(points_xyz) == 0:
        return False

    test_geom = geom.buffer(margin + eps)
    for p in points_xyz:
        pt = Point(float(p[0]), float(p[1]))
        if test_geom.covers(pt):
            return True
    return False

def clip_edge_with_geom(p1, p2, geom, z_value, eps=GEOM_BUFFER_EPS):
    line = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
    inter = line.intersection(geom.buffer(eps))

    clipped_segments = []

    if inter.is_empty:
        return clipped_segments

    def add_linestring(ls):
        coords = list(ls.coords)
        if len(coords) < 2:
            return
        for i in range(len(coords) - 1):
            a = np.array([coords[i][0], coords[i][1], z_value], dtype=float)
            b = np.array([coords[i + 1][0], coords[i + 1][1], z_value], dtype=float)
            if np.linalg.norm(b - a) > 1e-10:
                clipped_segments.append((a, b))

    if inter.geom_type == "LineString":
        add_linestring(inter)

    elif inter.geom_type == "MultiLineString":
        for seg in inter.geoms:
            add_linestring(seg)

    elif inter.geom_type == "GeometryCollection":
        for g in inter.geoms:
            if g.geom_type == "LineString":
                add_linestring(g)
            elif g.geom_type == "MultiLineString":
                for seg in g.geoms:
                    add_linestring(seg)

    return clipped_segments

# =========================================================
# DEBUG / REPORT HELPERS
# =========================================================
def print_geom_bounds(label, geom):
    if geom is None or geom.is_empty:
        print(f"{label}: EMPTY")
        return
    minx, miny, maxx, maxy = geom.bounds
    print(
        f"{label}: bounds=({minx:.3f}, {miny:.3f}) ~ ({maxx:.3f}, {maxy:.3f}), "
        f"size=({maxx-minx:.3f}, {maxy-miny:.3f})"
    )

def print_space_bounds(space_list, geom_list):
    print("\n===== PER-SPACE BOUNDS =====")
    for i, (space, g) in enumerate(zip(space_list, geom_list)):
        if g is None or g.is_empty:
            print(f"[{i}] {space.GlobalId} EMPTY")
            continue
        minx, miny, maxx, maxy = g.bounds
        dx = maxx - minx
        dy = maxy - miny
        print(f"[{i}] Name={space.Name}, LongName={space.LongName}, GlobalId={space.GlobalId}")
        print(f"    bounds=({minx:.3f}, {miny:.3f}) ~ ({maxx:.3f}, {maxy:.3f}), size=({dx:.3f}, {dy:.3f})")

def filter_outlier_spaces(target_spaces, space_valid_geoms, max_size=200.0, max_dist_from_median=300.0):
    if len(space_valid_geoms) == 0:
        return target_spaces, space_valid_geoms

    centers = []
    for g in space_valid_geoms:
        minx, miny, maxx, maxy = g.bounds
        centers.append([(minx + maxx) * 0.5, (miny + maxy) * 0.5])
    centers = np.asarray(centers, dtype=float)

    median_center = np.median(centers, axis=0)

    kept_spaces = []
    kept_geoms = []

    print("\n===== OUTLIER CHECK =====")
    print("median_center =", median_center)

    for idx, (space, g) in enumerate(zip(target_spaces, space_valid_geoms)):
        minx, miny, maxx, maxy = g.bounds
        dx = maxx - minx
        dy = maxy - miny
        cx = 0.5 * (minx + maxx)
        cy = 0.5 * (miny + maxy)
        dist = np.linalg.norm(np.array([cx, cy]) - median_center)

        abnormal = False
        reasons = []

        if dx > max_size or dy > max_size:
            abnormal = True
            reasons.append(f"bbox too large ({dx:.2f}, {dy:.2f})")

        if dist > max_dist_from_median:
            abnormal = True
            reasons.append(f"center too far ({dist:.2f} m)")

        if abnormal:
            print(f"[OUTLIER][{idx}] Name={space.Name}, LongName={space.LongName}, GlobalId={space.GlobalId}")
            print("               reasons:", ", ".join(reasons))
            print(f"               bounds=({minx:.3f}, {miny:.3f}) ~ ({maxx:.3f}, {maxy:.3f})")
        else:
            kept_spaces.append(space)
            kept_geoms.append(g)

    return kept_spaces, kept_geoms

# =========================================================
# 2D DEBUG PLOT
# =========================================================
def plot_shapely_geom(ax, geom, color='r', linewidth=1.0, label=None):
    if geom is None or geom.is_empty:
        return

    if geom.geom_type == "Polygon":
        x, y = geom.exterior.xy
        ax.plot(x, y, color=color, linewidth=linewidth, label=label)
        for interior in geom.interiors:
            hx, hy = interior.xy
            ax.plot(hx, hy, color=color, linewidth=max(linewidth * 0.7, 0.5), linestyle='--')

    elif geom.geom_type == "MultiPolygon":
        first = True
        for g in geom.geoms:
            plot_shapely_geom(ax, g, color=color, linewidth=linewidth, label=(label if first else None))
            first = False

    elif geom.geom_type == "GeometryCollection":
        first = True
        for g in geom.geoms:
            plot_shapely_geom(ax, g, color=color, linewidth=linewidth, label=(label if first else None))
            first = False

def save_debug_2d_plot(filepath, hall_geom, target_spaces=None, space_valid_geoms=None, door_points=None, edge_segments=None):
    fig, ax = plt.subplots(figsize=(12, 12))

    if target_spaces is not None and space_valid_geoms is not None:
        for i, (space, g) in enumerate(zip(target_spaces, space_valid_geoms)):
            plot_shapely_geom(ax, g, color='magenta', linewidth=0.8)
            try:
                c = g.centroid
                ax.text(c.x, c.y, f"{i}", fontsize=7, color='magenta')
            except Exception:
                pass

    plot_shapely_geom(ax, hall_geom, color='red', linewidth=2.0, label='union hall')

    if door_points is not None and len(door_points) > 0:
        ax.scatter(door_points[:, 0], door_points[:, 1], s=4, c='blue', label='door vertices')

    if edge_segments is not None and len(edge_segments) > 0:
        for seg in edge_segments:
            p1, p2 = seg
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linewidth=0.4)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(loc='best')

    if hall_geom is not None and not hall_geom.is_empty:
        minx, miny, maxx, maxy = hall_geom.bounds
        cx = 0.5 * (minx + maxx)
        cy = 0.5 * (miny + maxy)
        dx = maxx - minx
        dy = maxy - miny
        span = max(dx, dy) * 0.65 + 5.0
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)

    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close(fig)
    print(f"Saved 2D debug plot: {filepath}")

# =========================================================
# OPEN3D HELPERS
# =========================================================
def make_lineset_from_xyz(poly_xyz, color):
    if len(poly_xyz) < 2:
        return None
    lines = [[i, i + 1] for i in range(len(poly_xyz) - 1)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(poly_xyz)
    ls.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=int))
    ls.colors = o3d.utility.Vector3dVector([color for _ in lines])
    return ls

def recenter_o3d_geometries(geoms):
    all_pts = []

    for g in geoms:
        if isinstance(g, o3d.geometry.PointCloud):
            pts = np.asarray(g.points)
            if len(pts) > 0:
                all_pts.append(pts)
        elif isinstance(g, o3d.geometry.LineSet):
            pts = np.asarray(g.points)
            if len(pts) > 0:
                all_pts.append(pts)

    if len(all_pts) == 0:
        return geoms

    all_pts = np.vstack(all_pts)
    center = np.mean(all_pts, axis=0)

    shifted = []
    for g in geoms:
        g2 = copy.deepcopy(g)
        g2.translate(-center)
        shifted.append(g2)

    print("Visualization recentered by:", center)
    return shifted

# =========================================================
# OCC / MESH SETTINGS
# =========================================================
settings_mesh = ifcopenshell.geom.settings()
settings_mesh.set(settings_mesh.USE_WORLD_COORDS, True)

settings_occ = ifcopenshell.geom.settings()
settings_occ.set(settings_occ.USE_WORLD_COORDS, True)
settings_occ.set(settings_occ.USE_PYTHON_OPENCASCADE, True)

# =========================================================
# FIND TARGET SPACES AND BUILD UNION HALL GEOMETRY
# =========================================================
spaces = ifc_file.by_type(TARGET_IFC_CLASS)

target_spaces = []
space_valid_geoms = []

per_space_outer_debug_xyz = []
per_space_hole_debug_xyz = []

print("\nSearching target IfcSpace objects from Representation...")
for idx, space in enumerate(spaces):
    if not is_in_storey(space, target_storey):
        continue

    longname_text = (space.LongName or "")
    if TARGET_LONGNAME_KEYWORD.lower() not in longname_text.lower():
        continue

    if TARGET_SPACE_NAME is not None and (space.Name or "") not in TARGET_SPACE_NAME:
        continue

    try:
        result = extract_space_polygons_from_representation_world(space)
        valid_geom = build_valid_geom_from_space_result(result)

        if valid_geom is None or valid_geom.is_empty:
            print(f"Skipped empty valid geom: {space.GlobalId}")
            continue

        target_spaces.append(space)
        space_valid_geoms.append(valid_geom)

        if SAVE_PER_SPACE_DEBUG_CSV:
            outer_list, hole_list = extract_polygon_rings_from_geom(valid_geom)

            for j, poly_xy in enumerate(outer_list):
                per_space_outer_debug_xyz.append(polygon_xy_to_xyz(poly_xy, level_threshold))
                save_path = os.path.join(
                    path,
                    f"debug_space_outer_{level.replace(' ', '_')}_{space.GlobalId}_{j}.csv"
                )
                np.savetxt(save_path, poly_xy, delimiter=",")
                print(f"Saved per-space outer debug: {save_path}")

            for j, hole_xy in enumerate(hole_list):
                per_space_hole_debug_xyz.append(polygon_xy_to_xyz(hole_xy, level_threshold))
                save_path = os.path.join(
                    path,
                    f"debug_space_hole_{level.replace(' ', '_')}_{space.GlobalId}_{j}.csv"
                )
                np.savetxt(save_path, hole_xy, delimiter=",")
                print(f"Saved per-space hole debug: {save_path}")

        print(
            f"Matched space | idx={len(target_spaces)-1} | Name: {space.Name} | "
            f"LongName: {space.LongName} | GlobalId: {space.GlobalId}"
        )

    except Exception as e:
        print(f"Failed to extract representation polygon from space {space.GlobalId}: {e}")

if len(space_valid_geoms) == 0:
    raise ValueError(
        f"No {TARGET_IFC_CLASS} found in {level} with LongName containing '{TARGET_LONGNAME_KEYWORD}'."
    )

print_space_bounds(target_spaces, space_valid_geoms)

if ENABLE_OUTLIER_FILTER:
    target_spaces, space_valid_geoms = filter_outlier_spaces(
        target_spaces,
        space_valid_geoms,
        max_size=OUTLIER_MAX_BBOX_SIZE,
        max_dist_from_median=OUTLIER_MAX_DIST_FROM_MEDIAN
    )

    if len(space_valid_geoms) == 0:
        raise ValueError("All matched spaces were filtered out as outliers.")

hall_geom = unary_union(space_valid_geoms)
if not hall_geom.is_valid:
    hall_geom = hall_geom.buffer(0)

if hall_geom.is_empty:
    raise ValueError("Hall geometry is empty after union.")

print_geom_bounds("hall_geom", hall_geom)

hall_outer_xy_list, hall_hole_xy_list = extract_polygon_rings_from_geom(hall_geom)
hall_outer_xyz_list = [polygon_xy_to_xyz(poly, level_threshold) for poly in hall_outer_xy_list]
hall_hole_xyz_list = [polygon_xy_to_xyz(poly, level_threshold) for poly in hall_hole_xy_list]

print(f"\nNumber of matched spaces after filtering: {len(target_spaces)}")
print(f"Union hall outer polygons: {len(hall_outer_xy_list)}")
print(f"Union hall hole polygons: {len(hall_hole_xy_list)}")

# =========================================================
# SAVE UNION HALL POLYGONS
# =========================================================
suffix = "TNAMEincluded" if TARGET_SPACE_NAME is not None else TARGET_LONGNAME_KEYWORD

if SAVE_UNION_DEBUG_CSV:
    for i, poly in enumerate(hall_outer_xy_list):
        poly_csv = os.path.join(
            path,
            f"target_space_polygon_{level.replace(' ', '_')}_{suffix}_{i}.csv"
        )
        np.savetxt(poly_csv, poly, delimiter=",")
        print(f"Saved union target polygon: {poly_csv}")

    for i, poly in enumerate(hall_hole_xy_list):
        poly_csv = os.path.join(
            path,
            f"target_space_hole_polygon_{level.replace(' ', '_')}_{suffix}_{i}.csv"
        )
        np.savetxt(poly_csv, poly, delimiter=",")
        print(f"Saved union target hole polygon: {poly_csv}")

# =========================================================
# PART 1: DOOR CORNERS
# =========================================================
doors = list(ifc_file.by_type("IfcDoor"))
sliding_plates = [
    e for e in ifc_file.by_type("IfcPlate")
    if "sliding doors" in (e.Name or "").lower()
]
doors.extend(sliding_plates)

vertices_list = []
centers_list = []

print("\nExtracting door vertices in union hall geometry...")
for door in doors:
    if not is_in_storey(door, target_storey):
        continue

    try:
        shape = ifcopenshell.geom.create_shape(settings_mesh, door)
        verts = np.array(shape.geometry.verts).reshape(-1, 3)

        # USE_WORLD_COORDS=True 이므로 meter 기준으로 그대로 사용
        if USE_OBJECT_LEVEL_FILTER:
            if object_intersects_geom(verts, hall_geom, margin=DOOR_INCLUDE_MARGIN):
                vertices_list.append(verts)
                centers_list.append(verts.mean(axis=0))
                print(f"Selected whole object: {door.is_a()} | {door.GlobalId}")
        else:
            verts_filtered = filter_points_by_geom(
                verts,
                hall_geom,
                margin=DOOR_INCLUDE_MARGIN
            )
            if verts_filtered.shape[0] > 0:
                vertices_list.append(verts_filtered)
                centers_list.append(verts_filtered.mean(axis=0))
                print(
                    f"Selected filtered vertices: {door.is_a()} | {door.GlobalId} | "
                    f"kept {len(verts_filtered)}/{len(verts)}"
                )

    except Exception as e:
        print(f"Geometry failed: {door.GlobalId}, Reason: {e}")

if vertices_list:
    all_verts = np.vstack(vertices_list)
    print(f"\nAll selected door vertices: {all_verts.shape}")
else:
    print("\nCouldn't extract selected door corners.")
    all_verts = np.empty((0, 3))

door_corner_csv = os.path.join(
    path,
    f"BIMcorners_door_{level.replace(' ','_')}_{suffix}.csv"
)
door_center_csv = os.path.join(
    path,
    f"BIMcorners_door_{level.replace(' ','_')}_center_{suffix}.csv"
)

np.savetxt(door_corner_csv, all_verts, delimiter=",")
if len(centers_list) > 0:
    np.savetxt(door_center_csv, np.array(centers_list), delimiter=",")
else:
    np.savetxt(door_center_csv, np.empty((0, 3)), delimiter=",")

print(f"Saved: {door_corner_csv}")
print(f"Saved: {door_center_csv}")

# =========================================================
# PART 2: BIM MODEL EDGES NEAR FLOOR HEIGHT
# =========================================================
model = ifcopenshell.open(ifc_path)

level_products = []
for rel in model.by_type("IfcRelContainedInSpatialStructure"):
    if rel.RelatingStructure == target_storey:
        level_products.extend(rel.RelatedElements)

z0_edges = []

print("\nExtracting BIM edges near floor elevation in union hall geometry...")
for product in level_products:
    if getattr(product, "GlobalId", None) in exclude_guids:
        continue

    try:
        shape = ifcopenshell.geom.create_shape(settings_occ, product)
        occ_shape = shape.geometry

        exp_edge = TopExp_Explorer(occ_shape, TopAbs_EDGE)
        while exp_edge.More():
            edge = exp_edge.Current()

            exp_vertex = TopExp_Explorer(edge, TopAbs_VERTEX)
            points = []

            while exp_vertex.More():
                vertex = exp_vertex.Current()
                pnt = BRep_Tool.Pnt(vertex)
                coord = (pnt.X(), pnt.Y(), pnt.Z())
                points.append(coord)
                exp_vertex.Next()

            if len(points) == 2:
                z_coords = [pt[2] for pt in points]

                if all(abs(z - level_threshold) < MODEL_EDGE_Z_TOL for z in z_coords):
                    p1, p2 = points
                    clipped_segments = clip_edge_with_geom(p1, p2, hall_geom, level_threshold)
                    z0_edges.extend(clipped_segments)

            exp_edge.Next()

    except Exception:
        continue

print(f"Number of clipped floor-edge segments: {len(z0_edges)}")

points = []
lines = []

for seg in z0_edges:
    p1, p2 = seg
    points.append(p1)
    points.append(p2)
    idx = len(points)
    lines.append([idx - 2, idx - 1])

points_np = np.array(points) if len(points) > 0 else np.empty((0, 3))
lines_np = np.array(lines) if len(lines) > 0 else np.empty((0, 2), dtype=int)

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points_np)
line_set.lines = o3d.utility.Vector2iVector(lines_np)

colors = [[0, 0, 0] for _ in range(len(lines))]
if len(colors) > 0:
    line_set.colors = o3d.utility.Vector3dVector(colors)

# =========================================================
# SAVE 2D DEBUG PLOT
# =========================================================
if SAVE_2D_DEBUG_PLOT:
    debug_plot_path = os.path.join(
        path,
        f"debug_plot_{level.replace(' ','_')}_{suffix}.png"
    )

    save_debug_2d_plot(
        filepath=debug_plot_path,
        hall_geom=hall_geom,
        target_spaces=target_spaces,
        space_valid_geoms=space_valid_geoms,
        door_points=all_verts if len(all_verts) > 0 else None,
        edge_segments=z0_edges
    )

# =========================================================
# OPEN3D VISUALIZATION
# =========================================================
if ENABLE_OPEN3D_VIS:
    vis_list = []

    # Door point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_verts)
    pcd.paint_uniform_color([0, 0.5, 1])
    vis_list.append(pcd)

    # BIM clipped edges
    vis_list.append(line_set)

    # Union hall outer polygons - red
    for poly_xyz in hall_outer_xyz_list:
        ls = make_lineset_from_xyz(poly_xyz, [1, 0, 0])
        if ls is not None:
            vis_list.append(ls)

    # Union hall holes - green
    for hole_xyz in hall_hole_xyz_list:
        ls = make_lineset_from_xyz(hole_xyz, [0, 1, 0])
        if ls is not None:
            vis_list.append(ls)

    # per-space debug outer - magenta
    for poly_xyz in per_space_outer_debug_xyz:
        ls = make_lineset_from_xyz(poly_xyz, [1, 0, 1])
        if ls is not None:
            vis_list.append(ls)

    # per-space debug holes - yellow
    for hole_xyz in per_space_hole_debug_xyz:
        ls = make_lineset_from_xyz(hole_xyz, [1, 1, 0])
        if ls is not None:
            vis_list.append(ls)

    if DRAW_DEBUG_INFO:
        print("\nVisualization colors:")
        print("  blue    = selected door vertices")
        print("  black   = BIM edges near floor elevation (intersection-clipped)")
        print("  red     = union hall outer polygons")
        print("  green   = union hall hole polygons")
        print("  magenta = per-space valid outer debug")
        print("  yellow  = per-space valid hole debug")

    vis_list = recenter_o3d_geometries(vis_list)
    o3d.visualization.draw_geometries(vis_list)