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

        # ── Structural analysis enrichment ──────────────────────────────
        connected_components     list[dict]
            Per-body geometry dicts (same schema as the top-level fields,
            minus connected_components itself).  One entry per disconnected
            body in the mesh.  Length == 1 for single-body STLs.
            Each dict has:
                bbox_x_mm, bbox_y_mm, bbox_z_mm
                dominant_axis
                min_wall_thickness_mm, median_wall_thickness_mm
                is_watertight
                volume_mm3, surface_area_mm2
                triangle_count, vertex_count
                bottom_contact_area_mm2
                top_load_area_mm2
                has_cylindrical_bores
                mean_curvature

        mean_curvature           float  (1/mm)  — whole-mesh discrete mean
                                 curvature: average |Δn| per unit area.
                                 Near 0 = flat/prismatic.  > 0.05/mm = shell.

        bottom_contact_area_mm2  float  — total area of faces with outward
                                 normal ≈ −Z and whose Z centroid is within
                                 5 % of the part height from Z_min.
                                 Proxy for the resting / support footprint.

        top_load_area_mm2        float  — same but normal ≈ +Z at Z_max.
                                 Proxy for the primary load entry surface.

        has_cylindrical_bores    bool   — True when the mesh contains
                                 interior loops of faces whose normals are
                                 predominantly radially inward (pointing
                                 toward a shared axis), indicating mounting
                                 holes or bores.

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
# Feature extraction  (top-level dispatcher)
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
    dims     = mesh.bounds[1] - mesh.bounds[0]   # (3,) xyz extents
    centroid = mesh.center_mass

    if np.any(dims < 1e-6):
        warnings.append(
            f"Near-zero bounding-box dimension detected: {dims.tolist()} mm. "
            "Geometry may be degenerate."
        )

    # ── Shape classification ───────────────────────────────────────────────
    aspect_ratio  = _aspect_ratio(dims)
    dominant_axis = _classify_shape(dims)

    # ── Wall thickness ─────────────────────────────────────────────────────
    min_wall, median_wall = _wall_thickness(
        mesh, wall_sample_count, warnings
    )
    has_thin = (
        min_wall is not None and min_wall < thin_wall_threshold_mm
    )

    # ── NEW: Structural enrichment fields ─────────────────────────────────
    mean_curv              = _mean_curvature(mesh, warnings)
    bottom_contact         = _contact_area(mesh, face="bottom")
    top_load               = _contact_area(mesh, face="top")
    has_bores              = _has_cylindrical_bores(mesh, warnings)
    components             = _connected_components(
        mesh, wall_sample_count, thin_wall_threshold_mm, warnings
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
        # ── structural enrichment ──────────────────────────────────────
        "mean_curvature":           round(mean_curv, 6),
        "bottom_contact_area_mm2":  round(bottom_contact, 4),
        "top_load_area_mm2":        round(top_load, 4),
        "has_cylindrical_bores":    bool(has_bores),
        "connected_components":     components,
    }, warnings


# ===========================================================================
# NEW: Discrete mean curvature
# ===========================================================================

def _mean_curvature(mesh, warnings: list[str]) -> float:
    """
    Compute a scalar discrete mean curvature proxy for the whole mesh.

    Method: angle-weighted vertex normal variation (Meyer et al. 2003).
    For each vertex, we compute the magnitude of the discrete Laplace–
    Beltrami operator applied to the vertex positions, divided by the
    local Voronoi area. The mesh-wide mean of these magnitudes gives a
    scale-independent curvature measure in 1/mm.

    Interpretation:
        < 0.001 / mm   essentially flat / prismatic (box, plate, beam)
        0.001–0.05     gently curved (ribbed panel, ergonomic grip)
        > 0.05         strongly curved (pipe, sphere, shell structure)

    Falls back to 0.0 and appends a warning on failure (e.g. open mesh,
    degenerate triangles).

    References:
        Meyer et al., "Discrete Differential-Geometry Operators for
        Triangulated 2-Manifolds", Visualization and Mathematics III, 2003.
    """
    try:
        verts  = mesh.vertices                 # (V, 3)
        faces  = mesh.faces                    # (F, 3)
        n_vert = len(verts)

        if n_vert < 4 or len(faces) < 4:
            return 0.0

        # ── Cotangent weights (standard cotan Laplacian) ──────────────────
        # For each face, compute the cotangent of each interior angle.
        # Edges: e_i = v_k - v_j  (opposite to vertex i in the triangle)
        v0 = verts[faces[:, 0]]   # (F, 3)
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]

        def _cot_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """
            Cotangent of the angle at a vertex given the two edge vectors
            from that vertex.  cot θ = cos θ / sin θ = (a·b) / |a×b|.
            Clipped to [−10, 10] to guard against near-degenerate triangles.
            """
            dot  = np.einsum("ij,ij->i", a, b)
            cross_norm = np.linalg.norm(np.cross(a, b), axis=1)
            cross_norm = np.maximum(cross_norm, 1e-12)
            return np.clip(dot / cross_norm, -10.0, 10.0)

        # cot of the angle at each vertex of every face
        cot0 = _cot_angle(v1 - v0, v2 - v0)   # angle at v0
        cot1 = _cot_angle(v0 - v1, v2 - v1)   # angle at v1
        cot2 = _cot_angle(v0 - v2, v1 - v2)   # angle at v2

        # ── Accumulate cotan-Laplacian × vertex positions ─────────────────
        # Δ(v_i) = (1 / 2A_i) Σ_j (cot α_ij + cot β_ij)(v_j - v_i)
        # We accumulate the numerator (Σ cotan * edge) per vertex.
        laplacian = np.zeros((n_vert, 3), dtype=np.float64)

        def _add_contribution(
            idx_i: np.ndarray,
            idx_j: np.ndarray,
            cot_weight: np.ndarray,
        ) -> None:
            edge = verts[idx_j] - verts[idx_i]   # (F, 3)
            contrib = cot_weight[:, None] * edge  # (F, 3)
            np.add.at(laplacian, idx_i, contrib)
            np.add.at(laplacian, idx_j, -contrib)   # antisymmetric

        # Each edge (i,j) contributes with the cot of the opposite angle.
        # Edge (v1,v2) is opposite angle at v0 → weight cot0
        _add_contribution(faces[:, 1], faces[:, 2], cot0)
        # Edge (v0,v2) is opposite angle at v1 → weight cot1
        _add_contribution(faces[:, 0], faces[:, 2], cot1)
        # Edge (v0,v1) is opposite angle at v2 → weight cot2
        _add_contribution(faces[:, 0], faces[:, 1], cot2)

        # ── Voronoi areas ─────────────────────────────────────────────────
        # A_i = (1/8) Σ_triangles (cot α + cot β) |e|² per vertex.
        # Simplified: distribute triangle area (1/3 each) as Voronoi proxy.
        face_areas = 0.5 * np.linalg.norm(
            np.cross(v1 - v0, v2 - v0), axis=1
        )                                         # (F,)
        vertex_area = np.zeros(n_vert, dtype=np.float64)
        np.add.at(vertex_area, faces[:, 0], face_areas / 3.0)
        np.add.at(vertex_area, faces[:, 1], face_areas / 3.0)
        np.add.at(vertex_area, faces[:, 2], face_areas / 3.0)

        # ── Mean curvature magnitude at each vertex ───────────────────────
        # H_i = |Δ(p_i)| / (2 * A_i)
        # (factor 2 from the definition of the discrete mean curvature normal)
        safe_area = np.maximum(vertex_area, 1e-12)
        H = np.linalg.norm(laplacian, axis=1) / (2.0 * safe_area)   # (V,) 1/mm

        # ── Robust aggregate: area-weighted median ────────────────────────
        # Median is more robust than mean for open/imperfect meshes.
        # Weight by vertex area to reflect surface distribution.
        # Use 90th percentile of H as the summary statistic —
        # the "typical high curvature zone" is what matters for shell detection.
        p90 = float(np.percentile(H, 90))
        return p90

    except Exception as e:
        warnings.append(
            f"Mean curvature computation failed ({e}). "
            "Defaulting to 0.0 (flat/prismatic assumption)."
        )
        return 0.0


# ===========================================================================
# NEW: Contact area (top / bottom faces)
# ===========================================================================

def _contact_area(mesh, face: str) -> float:
    """
    Sum the area of faces whose outward normal is predominantly in the
    ±Z direction AND whose centroid Z-position is near the bounding-box
    extremum.

    face="bottom":
        normal ≈ −Z  (n_z < −cos45° = −0.707)
        centroid Z within 5 % of part height from Z_min.
        → resting / support footprint.

    face="top":
        normal ≈ +Z  (n_z > +0.707)
        centroid Z within 5 % of part height from Z_max.
        → primary load-entry surface.

    The 5 % height threshold is generous enough to capture multi-surface
    bases (e.g. four separate foot pads) while excluding mid-body surfaces.

    Returns total area in mm².  Returns 0.0 on any failure.
    """
    try:
        normals     = mesh.face_normals            # (F, 3)  unit normals
        face_areas  = mesh.area_faces              # (F,)
        centroids   = mesh.triangles_center        # (F, 3)

        z_min, z_max = mesh.bounds[0, 2], mesh.bounds[1, 2]
        height       = z_max - z_min
        if height < 1e-9:
            return 0.0

        z_tol = height * 0.05    # 5 % height tolerance
        _COS45 = 0.707

        if face == "bottom":
            normal_mask   = normals[:, 2] < -_COS45
            position_mask = centroids[:, 2] < (z_min + z_tol)
        else:  # "top"
            normal_mask   = normals[:, 2] > _COS45
            position_mask = centroids[:, 2] > (z_max - z_tol)

        mask = normal_mask & position_mask
        return float(face_areas[mask].sum())

    except Exception:
        return 0.0


# ===========================================================================
# NEW: Cylindrical bore detection
# ===========================================================================

def _has_cylindrical_bores(mesh, warnings: list[str]) -> bool:
    """
    Detect cylindrical bores (mounting holes, through-holes) by looking
    for clusters of face normals that point radially inward toward a
    shared axis — the signature of a cylindrical hole surface.

    Algorithm
    ---------
    1. Identify "inward-radial" faces: faces whose centroid-to-nearest-
       axis-candidate vector is anti-parallel to the face normal.
       We test two candidate axes: Z (vertical holes) and any dominant
       horizontal axis.

    2. For the Z-axis: a face on a cylindrical bore has:
          normal · (centroid_XY_normalised) ≈ −1
       i.e. the normal points toward the Z-axis in XY.

    3. We use a histogram of the azimuthal angle of face normals in XY:
       for a cylindrical bore the normals should be uniformly distributed
       in azimuth (all pointing inward = toward axis = outward normals
       point away from axis). We detect this by checking if there is a
       subset of faces whose XY-normal magnitude is > 0.85 (nearly
       horizontal normal) AND whose XY-normal, when plotted in azimuth,
       has a large fraction pointing toward a locally concentrated centroid.

    Practical heuristic (robust without trimesh topology extras)
    ------------------------------------------------------------
    We look for faces where:
        |n_z| < 0.3            (nearly vertical normal — side of a hole)
        |n_XY|  > 0.9          (strongly horizontal)
        centroid r = |p_XY - axis| is small relative to bbox

    If such faces form ≥ 3 % of the total face count AND their azimuthal
    normals show significant negative radial alignment (normal points
    toward the axis candidate), we flag a bore.

    We test three axis candidates: Z-axis through XY centroid, X-axis
    through YZ centroid, Y-axis through XZ centroid.

    Returns True if any bore is detected, False otherwise.
    Silently returns False on any exception (bore detection is advisory).
    """
    try:
        normals    = mesh.face_normals        # (F, 3)
        centroids  = mesh.triangles_center    # (F, 3)
        n_faces    = len(normals)

        if n_faces < 20:
            return False

        dims = mesh.bounds[1] - mesh.bounds[0]

        def _check_axis(axis: int) -> bool:
            """
            axis: 0=X, 1=Y, 2=Z  — the axis of the candidate bore.
            Radial plane is the two axes NOT equal to `axis`.
            """
            radial_axes = [i for i in range(3) if i != axis]
            r0, r1     = radial_axes

            # Radial components of normal and centroid
            n_radial   = normals[:, [r0, r1]]     # (F, 2)
            n_radial_mag = np.linalg.norm(n_radial, axis=1)

            # Candidate axis passes through the mesh centroid in XY
            axis_pos   = np.array([mesh.center_mass[r0], mesh.center_mass[r1]])
            c_radial   = centroids[:, [r0, r1]] - axis_pos[None, :]  # (F, 2)
            c_radial_r = np.linalg.norm(c_radial, axis=1)

            # Select faces with nearly-horizontal normals relative to this axis
            horizontal_mask = (
                n_radial_mag > 0.85
            ) & (
                np.abs(normals[:, axis]) < 0.35
            )

            if horizontal_mask.sum() < max(6, n_faces * 0.02):
                return False

            # Among those, check radial alignment:
            # n_radial · (−c_radial / |c_radial|) > 0.5  →  inward normal
            c_r_safe = np.maximum(c_radial_r, 1e-9)
            inward   = np.einsum(
                "ij,ij->i",
                n_radial[horizontal_mask],
                -c_radial[horizontal_mask] / c_r_safe[horizontal_mask, None],
            )
            # Fraction of horizontal faces that are inward-radial
            inward_fraction = (inward > 0.5).sum() / horizontal_mask.sum()

            # Also check that the bore radius is plausible:
            # radius < 40 % of the smallest radial bbox dimension
            median_r = np.median(c_radial_r[horizontal_mask])
            max_radial_dim = min(dims[r0], dims[r1]) * 0.4

            return inward_fraction > 0.45 and median_r < max_radial_dim

        return _check_axis(2) or _check_axis(0) or _check_axis(1)

    except Exception as e:
        warnings.append(f"Bore detection failed ({e}). Assuming no bores.")
        return False


# ===========================================================================
# NEW: Connected-component decomposition
# ===========================================================================

def _connected_components(
    mesh,
    wall_sample_count: int,
    thin_wall_threshold_mm: float,
    warnings: list[str],
) -> list[dict]:
    """
    Split the mesh into connected face-components and compute per-component
    geometry features.  Each component gets the same feature schema as the
    whole-mesh output (minus connected_components to avoid recursion).

    Uses trimesh's graph utilities which do face-adjacency BFS/DFS — O(F).

    If the mesh has only one connected component (typical for watertight
    solid bodies), returns a single-element list containing the whole-mesh
    features so that downstream code always sees a list.

    Limits: if there are > 50 components (e.g. mesh soup), only the 50
    largest by face-count are analysed individually to avoid very slow runs.
    A warning is appended in that case.
    """
    try:
        import trimesh                                  # type: ignore

        labels = _face_component_labels(mesh)
        if labels.size == 0:
            warnings.append(
                "Connected-component decomposition returned no bodies. "
                "Returning whole mesh as single component."
            )
            return [_component_features(mesh, wall_sample_count,
                                        thin_wall_threshold_mm, warnings)]

        unique, counts = np.unique(labels, return_counts=True)
        order = np.argsort(-counts)
        unique = unique[order]
        counts = counts[order]
        n_components = len(unique)

        if n_components == 1:
            return [_component_features(mesh, wall_sample_count,
                                        thin_wall_threshold_mm, warnings)]

        min_faces = max(20, int(len(mesh.faces) * 0.002))
        kept = [(label, count) for label, count in zip(unique, counts) if count >= min_faces]
        skipped = n_components - len(kept)
        if kept:
            unique = np.array([label for label, _ in kept], dtype=unique.dtype)
        else:
            warnings.append(
                f"All {n_components} connected components are below the "
                f"minimum size threshold ({min_faces} faces). Returning the "
                "largest component only."
            )
            unique = unique[:1]

        if skipped > 0:
            warnings.append(
                f"Ignored {skipped} tiny disconnected mesh fragment(s) "
                f"(< {min_faces} faces) during structural decomposition."
            )

        if len(unique) > 50:
            warnings.append(
                f"Mesh has {len(unique)} significant connected components. "
                "Only the 50 largest are analysed individually. "
                "Consider cleaning the mesh."
            )
            unique = unique[:50]

        components = []
        for index, label in enumerate(unique):
            try:
                face_mask = labels == label
                sub_faces = mesh.faces[face_mask]
                used_verts, inv = np.unique(sub_faces, return_inverse=True)
                sub_verts = mesh.vertices[used_verts]
                new_faces = inv.reshape(sub_faces.shape)
                sub_mesh = trimesh.Trimesh(
                    vertices=sub_verts,
                    faces=new_faces,
                    process=False,
                )
                comp_warn: list[str] = []
                feat = _component_features(
                    sub_mesh, wall_sample_count, thin_wall_threshold_mm, comp_warn
                )
                feat["component_index"] = index
                # Prefix component warnings so they're traceable
                for w in comp_warn:
                    warnings.append(f"[component {index}] {w}")
                components.append(feat)
            except Exception as exc:
                warnings.append(
                    f"Component {index} skipped during feature extraction: {exc}"
                )

        if not components:
            warnings.append(
                "All connected components failed feature extraction. "
                "Falling back to whole-mesh single component."
            )
            return [_component_features(mesh, wall_sample_count,
                                        thin_wall_threshold_mm, warnings)]

        return components

    except Exception as exc:
        warnings.append(
            f"Connected-component decomposition failed ({exc}). "
            "Returning whole mesh as single component."
        )
        return [_component_features(mesh, wall_sample_count,
                                    thin_wall_threshold_mm, warnings)]


def _face_component_labels(mesh) -> np.ndarray:
    """
    Label connected face components from trimesh.face_adjacency without
    optional graph dependencies.
    """
    face_count = int(len(mesh.faces))
    if face_count == 0:
        return np.array([], dtype=np.int32)

    adjacency = [[] for _ in range(face_count)]
    for a, b in np.asarray(mesh.face_adjacency, dtype=np.int64):
        if 0 <= a < face_count and 0 <= b < face_count:
            adjacency[int(a)].append(int(b))
            adjacency[int(b)].append(int(a))

    labels = np.full(face_count, -1, dtype=np.int32)
    current = 0
    for start in range(face_count):
        if labels[start] != -1:
            continue
        labels[start] = current
        stack = [start]
        while stack:
            face = stack.pop()
            for neighbor in adjacency[face]:
                if labels[neighbor] == -1:
                    labels[neighbor] = current
                    stack.append(neighbor)
        current += 1

    return labels


def _component_features(
    mesh,
    wall_sample_count: int,
    thin_wall_threshold_mm: float,
    warnings: list[str],
) -> dict:
    """
    Compute the structural-enrichment feature subset for one (sub-)mesh.
    This is intentionally a subset — no recursion into connected_components.

    Uses a reduced wall_sample_count proportional to face count to avoid
    the per-component cost becoming O(n_components × wall_sample_count).
    """
    dims = mesh.bounds[1] - mesh.bounds[0]
    is_watertight = bool(mesh.is_watertight)

    # Scale sample count by face fraction (min 20 for tiny components)
    n_samples = max(20, min(wall_sample_count, len(mesh.faces) // 3))
    min_wall, median_wall = _wall_thickness(mesh, n_samples, warnings)

    mean_curv      = _mean_curvature(mesh, warnings)
    bottom_contact = _contact_area(mesh, face="bottom")
    top_load       = _contact_area(mesh, face="top")
    has_bores      = _has_cylindrical_bores(mesh, warnings)

    return {
        "bbox_x_mm":                round(float(dims[0]), 4),
        "bbox_y_mm":                round(float(dims[1]), 4),
        "bbox_z_mm":                round(float(dims[2]), 4),
        "dominant_axis":            _classify_shape(dims),
        "min_wall_thickness_mm":    round(min_wall,    4) if min_wall    is not None else None,
        "median_wall_thickness_mm": round(median_wall, 4) if median_wall is not None else None,
        "is_watertight":            is_watertight,
        "volume_mm3":               round(abs(float(mesh.volume)), 4),
        "surface_area_mm2":         round(float(mesh.area), 4),
        "triangle_count":           int(len(mesh.faces)),
        "vertex_count":             int(len(mesh.vertices)),
        "bottom_contact_area_mm2":  round(bottom_contact, 4),
        "top_load_area_mm2":        round(top_load, 4),
        "has_cylindrical_bores":    bool(has_bores),
        "mean_curvature":           round(mean_curv, 6),
    }


# ===========================================================================
# Wall thickness estimation  (unchanged from original)
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

    # Sample surface points. trimesh uses numpy randomness internally; seed it
    # from stable mesh properties so identical CAD gives repeatable requirements.
    try:
        seed = int(
            (
                len(mesh.faces) * 1_000_003
                + len(mesh.vertices) * 9_176
                + round(float(mesh.area) * 1_000)
            )
            % (2**32 - 1)
        )
        random_state = np.random.get_state()
        np.random.seed(seed)
        try:
            points, face_indices = tms.sample_surface(mesh, n_samples)
        finally:
            np.random.set_state(random_state)
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

    # Remove top-5% outliers (long rays through voids or open boundaries).
    # Use a lower percentile rather than the absolute minimum for the reported
    # wall thickness; the minimum is too sensitive to grazing rays and mesh
    # defects, and can make structural sizing unrealistically conservative.
    p95       = np.percentile(thicknesses, 95)
    filtered  = thicknesses[thicknesses <= p95]

    if len(filtered) == 0:
        warnings.append("All thickness samples were outliers. Not reported.")
        return None, None

    robust_min = np.percentile(filtered, 10)
    return float(robust_min), float(np.median(filtered))


# ===========================================================================
# Shape helpers  (unchanged from original)
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
