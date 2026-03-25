from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import logging

logger = logging.getLogger("roster-neural")
from langchain.tools import tool


@tool
def parse_and_extract_cad_features(
    file_path: str,
    tessellation_tolerance_mm: float = 0.1,
    wall_sample_count: int = 300,
    thin_wall_threshold_mm: float = 1.5,
) -> dict:
    """
    Parse a STEP or STL CAD file and extract all geometric features
    needed for material selection.

    Args:
        file_path:
            Absolute path to a .step, .stp, or .stl file.
        tessellation_tolerance_mm:
            Chord-height tolerance for STEP tessellation (mm).
            Lower = finer mesh = slower.
            Guidelines: micro parts < 5 mm → 0.01,
                        typical parts 10–500 mm → 0.1 (default),
                        large structures > 1 m → 1.0.
            Ignored for STL (already tessellated).
        wall_sample_count:
            Surface sample points for wall thickness estimation.
            100 = fast preview, 300 = default, 1000 = high accuracy.
        thin_wall_threshold_mm:
            Walls below this mark has_thin_features=True.
            Default 1.5 mm suits typical FDM / injection moulding limits.

    Returns dict with keys:

        source_format            "step" | "stl_binary" | "stl_ascii"
        unit_assumption          always "mm"

        volume_mm3               float
        surface_area_mm2         float

        bbox_x_mm                float
        bbox_y_mm                float
        bbox_z_mm                float
        centroid_mm              [x, y, z]

        aspect_ratio             float  (max_bbox / min_bbox)
        dominant_axis            "elongated" | "flat" | "compact" | "degenerate"

        min_wall_thickness_mm    float | null
        median_wall_thickness_mm float | null
        has_thin_features        bool
        thin_wall_threshold_mm   float

        triangle_count           int
        vertex_count             int
        is_watertight            bool
        multi_body               bool

        warnings                 list[str]

    Raises:
        FileNotFoundError  file_path does not exist
        ValueError         unsupported extension or unrecoverable parse failure
        ImportError        trimesh or cadquery not installed
    """
    logger.info(f"[Tool] Executing parse_and_extract_cad_features on file: {file_path}")
    path = _validate_path(file_path)
    ext  = path.suffix.lower()

    if ext in {".step", ".stp"}:
        mesh, source_format, multi_body, load_warnings = _load_step(
            path, tessellation_tolerance_mm
        )
    elif ext == ".stl":
        mesh, source_format, multi_body, load_warnings = _load_stl(path)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            "Accepted: .step  .stp  .stl"
        )

    features, extract_warnings = _extract_features(
        mesh,
        wall_sample_count=wall_sample_count,
        thin_wall_threshold_mm=thin_wall_threshold_mm,
    )

    return {
        "source_format":          source_format,
        "unit_assumption":        "mm",
        "multi_body":             multi_body,
        **features,
        "thin_wall_threshold_mm": thin_wall_threshold_mm,
        "warnings":               load_warnings + extract_warnings,
    }




def _validate_path(file_path: str) -> Path:
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"CAD file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a regular file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"File is empty: {path}")
    return path


# ===========================================================================
# STEP loader
# ===========================================================================

def _load_step(
    path: Path,
    tolerance_mm: float,
) -> tuple:
    """
    Load a STEP file via cadquery and return a trimesh.Trimesh.

    cadquery wraps pythonOCC, which handles the full AP203/AP214/AP242
    B-rep grammar — splines, revolutions, boolean operations, assemblies.
    We tessellate to a polygon mesh here so the rest of the pipeline
    works with a single consistent representation.

    Returns (Trimesh, source_format, multi_body_bool, warnings_list).
    """
    try:
        import cadquery as cq                           # type: ignore
        from trimesh import Trimesh                     # type: ignore
    except ImportError as e:
        raise ImportError(
            "cadquery and trimesh are required for STEP files. "
            "Install: pip install cadquery trimesh"
        ) from e

    warnings: list[str] = []

    # --- Load ---
    try:
        result = cq.importers.importStep(str(path))
    except Exception as e:
        raise ValueError(
            f"cadquery failed to load '{path.name}': {e}. "
            "Verify the file is a valid AP203/AP214/AP242 STEP export."
        ) from e

    shape = result.val()

    # --- Multi-body check ---
    multi_body = False
    try:
        solids = shape.Solids()
        if len(solids) > 1:
            multi_body = True
            warnings.append(
                f"STEP file contains {len(solids)} separate solids. "
                "They are merged into one mesh for analysis. "
                "If solids serve different structural roles, analyse "
                "each body separately."
            )
    except Exception:
        pass  # Solids() may not exist on all Shape types

    # --- Tessellate ---
    try:
        vertices_cq, faces_cq = shape.tessellate(
            tolerance_mm,
            angularTolerance=0.1,   # degrees — controls curve fidelity
        )
    except Exception as e:
        raise ValueError(
            f"Tessellation of '{path.name}' failed: {e}. "
            f"Try a larger tessellation_tolerance_mm "
            f"(current: {tolerance_mm} mm)."
        ) from e

    if len(vertices_cq) == 0 or len(faces_cq) == 0:
        raise ValueError(
            f"'{path.name}' tessellated to an empty mesh. "
            "Possible causes: file contains only 2D sketch geometry "
            "(no solids), or tolerance is too coarse for the part scale. "
            f"Current tolerance: {tolerance_mm} mm."
        )

    # Convert cadquery Vectors to numpy
    verts = np.array([[v.x, v.y, v.z] for v in vertices_cq], dtype=np.float64)
    faces = np.array(faces_cq, dtype=np.int32)

    # process=True: trimesh removes duplicate vertices and degenerate faces,
    # and fixes winding. Essential for STEP tessellations which can have
    # floating-point duplicates at seam edges between adjacent faces.
    mesh = Trimesh(vertices=verts, faces=faces, process=True)

    if len(mesh.faces) == 0:
        raise ValueError(
            f"'{path.name}' produced an empty mesh after repair. "
            "The geometry may be degenerate or surface-only."
        )

    return mesh, "step", multi_body, warnings


# ===========================================================================
# STL loader
# ===========================================================================

def _load_stl(path: Path) -> tuple:
    """
    Load an STL file via trimesh.

    trimesh's STL loader resolves the binary/ASCII ambiguity correctly:
    it checks the binary triangle-count formula before attempting ASCII
    parse, so files that begin with 'solid' (valid ASCII prefix, but also
    present in many binary exports) are correctly identified.

    Multi-body STLs produce a trimesh.Scene; we merge into one Trimesh.

    Returns (Trimesh, source_format, multi_body_bool, warnings_list).
    """
    try:
        import trimesh                                  # type: ignore
    except ImportError as e:
        raise ImportError(
            "trimesh is required for STL files. "
            "Install: pip install trimesh"
        ) from e

    import struct

    warnings: list[str] = []

    try:
        # force='mesh' asks trimesh to merge Scene into a single Trimesh
        loaded = trimesh.load(str(path), force="mesh")
    except Exception as e:
        raise ValueError(
            f"trimesh failed to load '{path.name}': {e}. "
            "Ensure the file is a valid binary or ASCII STL."
        ) from e

    # Guard: if force='mesh' didn't collapse a Scene (edge case in some
    # trimesh versions), do it manually.
    multi_body = False
    if hasattr(loaded, "geometry"):
        bodies = list(loaded.geometry.values())
        if len(bodies) > 1:
            multi_body = True
            warnings.append(
                f"STL contains {len(bodies)} disconnected bodies. "
                "Merged into one mesh for analysis."
            )
        import trimesh as tm                            # type: ignore
        loaded = tm.util.concatenate(bodies) if len(bodies) > 1 else bodies[0]

    if len(loaded.faces) == 0:
        raise ValueError(
            f"STL file '{path.name}' contains no triangle faces. "
            "The file may be empty or malformed."
        )

    # Binary vs ASCII detection (informational only)
    source_format = "stl_ascii"
    try:
        raw = path.read_bytes()
        if len(raw) >= 84:
            n = struct.unpack_from("<I", raw, 80)[0]
            if n > 0 and len(raw) == 84 + 50 * n:
                source_format = "stl_binary"
    except Exception:
        pass

    warnings.append(
        "STL files carry no unit information. "
        "Geometry assumed to be in millimetres. "
        "If exported in metres or inches, scale before uploading."
    )

    return loaded, source_format, multi_body, warnings


# ===========================================================================
# Feature extraction
# ===========================================================================

def _extract_features(
    mesh,
    wall_sample_count: int,
    thin_wall_threshold_mm: float,
) -> tuple[dict, list[str]]:
    """
    Compute all scalar features from a trimesh.Trimesh.
    Returns (features_dict, warnings_list).
    """
    warnings: list[str] = []

    # ── Watertight ────────────────────────────────────────────────────────
    is_watertight = bool(mesh.is_watertight)
    if not is_watertight:
        warnings.append(
            "Mesh is not watertight (open edges present). "
            "Volume and wall-thickness estimates may be inaccurate. "
            "Causes: incomplete STEP export, missing STL faces, or "
            "surface-only geometry."
        )

    # ── Volume ────────────────────────────────────────────────────────────
    volume = abs(float(mesh.volume))
    if mesh.volume < 0:
        warnings.append(
            "Mesh has inverted winding (negative signed volume). "
            "Volume reported as absolute value."
        )

    # ── Surface area ──────────────────────────────────────────────────────
    surface_area = float(mesh.area)

    # ── Bounding box ──────────────────────────────────────────────────────
    # mesh.bounds: (2, 3) — [[xmin,ymin,zmin], [xmax,ymax,zmax]]
    dims     = mesh.bounds[1] - mesh.bounds[0]
    centroid = mesh.center_mass

    if np.any(dims < 1e-6):
        warnings.append(
            f"Near-zero bounding-box dimension detected: {dims.tolist()} mm. "
            "Geometry may be degenerate."
        )

    # ── Shape ─────────────────────────────────────────────────────────────
    aspect_ratio  = _aspect_ratio(dims)
    dominant_axis = _classify_shape(dims)

    # ── Wall thickness ────────────────────────────────────────────────────
    min_wall, median_wall = _wall_thickness(
        mesh, wall_sample_count, warnings
    )

    has_thin = (
        min_wall is not None and min_wall < thin_wall_threshold_mm
    )

    return {
        "volume_mm3":               round(volume, 4),
        "surface_area_mm2":         round(surface_area, 4),
        "bbox_x_mm":                round(float(dims[0]), 4),
        "bbox_y_mm":                round(float(dims[1]), 4),
        "bbox_z_mm":                round(float(dims[2]), 4),
        "centroid_mm":              [round(float(c), 4) for c in centroid],
        "aspect_ratio":             round(aspect_ratio, 4),
        "dominant_axis":            dominant_axis,
        "min_wall_thickness_mm":    round(min_wall,    4) if min_wall    is not None else None,
        "median_wall_thickness_mm": round(median_wall, 4) if median_wall is not None else None,
        "has_thin_features":        has_thin,
        "triangle_count":           int(len(mesh.faces)),
        "vertex_count":             int(len(mesh.vertices)),
        "is_watertight":            is_watertight,
    }, warnings


# ===========================================================================
# Wall thickness estimation
# ===========================================================================

def _wall_thickness(
    mesh,
    n_samples: int,
    warnings: list[str],
) -> tuple[Optional[float], Optional[float]]:
    """
    Estimate wall thickness by inward ray casting from surface samples.

    1. trimesh.sample.sample_surface — area-weighted random surface points.
       Area-weighting avoids bias toward dense triangulation regions.
    2. Inward normal = negated outward face normal of the sampled triangle.
    3. Origin is offset EPS_OFFSET inward to avoid self-intersecting the
       source triangle.
    4. trimesh RayMeshIntersector uses a BVH — O(n log n), much faster
       than naive O(n * triangles) for meshes > ~10k triangles.
    5. multiple_hits=False: we only need the nearest opposite-wall hit.
    6. Filter: remove hits closer than 10×EPS_OFFSET (still self-hits
       despite offset), then remove top-5% outliers (long rays through
       internal voids or open mesh boundaries).
    """
    try:
        import trimesh.sample as tms                    # type: ignore
        import trimesh.ray.ray_triangle as trt          # type: ignore
    except ImportError as e:
        warnings.append(f"Wall thickness skipped — trimesh ray module unavailable: {e}")
        return None, None

    if len(mesh.faces) < 4:
        warnings.append("Too few triangles (< 4) for wall thickness estimation.")
        return None, None

    # Sample surface points
    try:
        points, face_indices = tms.sample_surface(mesh, n_samples)
    except Exception as e:
        warnings.append(f"Surface sampling failed: {e}. Wall thickness not computed.")
        return None, None

    outward_normals = mesh.face_normals[face_indices]   # (n_samples, 3)
    inward_dirs     = -outward_normals

    EPS_OFFSET = 1e-4  # mm
    origins = points + inward_dirs * EPS_OFFSET

    # Ray cast
    intersector = trt.RayMeshIntersector(mesh)
    try:
        locations, ray_idx, _ = intersector.intersects_location(
            ray_origins=origins,
            ray_directions=inward_dirs,
            multiple_hits=False,
        )
    except Exception as e:
        warnings.append(f"Ray casting failed: {e}. Wall thickness not computed.")
        return None, None

    if len(locations) == 0:
        warnings.append(
            "No opposite-wall intersections found during ray casting. "
            "Mesh may be an open shell or have uniform thin geometry. "
            "Wall thickness not computed."
        )
        return None, None

    # Distance from ray origin to hit
    thicknesses = np.linalg.norm(locations - origins[ray_idx], axis=1)

    # Remove self-hits (should be caught by offset, but guard anyway)
    thicknesses = thicknesses[thicknesses > EPS_OFFSET * 10]

    if len(thicknesses) < 3:
        warnings.append(
            f"Only {len(thicknesses)} valid thickness measurements from "
            f"{n_samples} rays. Result unreliable; not reported."
        )
        return None, None

    # Remove top-5% outliers (long rays through voids or open boundaries)
    p95       = np.percentile(thicknesses, 95)
    filtered  = thicknesses[thicknesses <= p95]

    if len(filtered) == 0:
        warnings.append("All thickness samples were outliers. Not reported.")
        return None, None

    return float(filtered.min()), float(np.median(filtered))


# ===========================================================================
# Shape helpers
# ===========================================================================

def _classify_shape(dims: np.ndarray) -> str:
    """
    elongated  — rod / beam / shaft  (one dominant long axis)
    flat       — plate / panel / shell (one thin axis, two large)
    compact    — block / equidimensional
    degenerate — near-zero in ≥ 1 dimension
    """
    s = sorted(float(d) for d in dims)     # [min, mid, max]
    if s[0] < 1e-6:
        return "degenerate"
    # elongated: max >> mid (long axis stands out)
    if (s[2] / s[1]) >= 2.0 and (s[2] / s[0]) >= 4.0:
        return "elongated"
    # flat: mid >> min (thin axis stands out)
    if (s[1] / s[0]) >= 2.0 and (s[2] / s[0]) >= 4.0:
        return "flat"
    return "compact"


def _aspect_ratio(dims: np.ndarray) -> float:
    min_d = float(dims.min())
    if min_d < 1e-9:
        return float("inf")
    return float(dims.max()) / min_d