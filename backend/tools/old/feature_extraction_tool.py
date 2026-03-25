import trimesh
import numpy as np
import json
from langchain.tools import tool

    
@tool
def extract_features(file_path: str) -> str:
    """
    Advanced CAD feature extraction tool.

    Capabilities:
    - Handles assemblies and multiple parts
    - Computes robust geometric + structural descriptors
    - Detects degeneracies and mesh issues
    - Estimates thickness distribution and curvature
    - Provides FEA-relevant metadata

    Output:
    JSON with part-level structured engineering descriptors
    """
    try:
        loaded = trimesh.load(file_path, force='scene')
        parts = loaded.dump()

        assembly = []

        for i, mesh in enumerate(parts):
            if not isinstance(mesh, trimesh.Trimesh):
                continue

            mesh.remove_degenerate_faces()
            mesh.remove_unreferenced_vertices()

                
            volume = mesh.volume if mesh.is_volume else 0
            area = mesh.area

            bbox = mesh.bounding_box.extents
            scale = np.linalg.norm(bbox)

                
            inertia = mesh.moment_inertia
            eigvals, _ = np.linalg.eigh(inertia)
            eigvals = np.sort(eigvals)[::-1]
            ratios = eigvals / (eigvals[0] + 1e-8)

                
            try:
                curvature = trimesh.curvature.discrete_gaussian_curvature_measure(
                    mesh, mesh.vertices, radius=scale * 0.05
                )
                curvature_stats = {
                    "mean": float(np.mean(curvature)),
                    "std": float(np.std(curvature))
                }
            except:
                curvature_stats = {"mean": 0, "std": 0}

                
            try:
                thickness_samples = []
                for _ in range(100):
                    origin = mesh.sample(1)[0]
                    direction = np.random.randn(3)
                    direction /= np.linalg.norm(direction)

                    locations, _, _ = mesh.ray.intersects_location(
                        [origin], [direction]
                    )

                    if len(locations) >= 2:
                        d = np.linalg.norm(locations[0] - locations[1])
                        thickness_samples.append(d)

                thickness = float(np.median(thickness_samples)) if thickness_samples else 0
            except:
                thickness = 0

                
            genus = 1 - (mesh.euler_number / 2)

            assembly.append({
                "part_id": f"part_{i}",
                "validity": {
                    "watertight": mesh.is_watertight,
                    "manifold": mesh.is_winding_consistent,
                    "degenerate_faces_removed": True
                },
                "scale": {
                    "bbox": bbox.tolist(),
                    "global_size": float(scale)
                },
                "mass_properties": {
                    "volume": float(volume),
                    "surface_area": float(area),
                    "center_of_mass": mesh.center_mass.tolist()
                },
                "shape_descriptors": {
                    "principal_ratios": ratios.tolist(),
                    "curvature": curvature_stats,
                    "genus": int(genus)
                },
                "structural_hints": {
                    "estimated_thickness": thickness,
                    "slenderness_ratio": float(max(bbox) / (min(bbox) + 1e-6))
                }
            })

        return json.dumps(assembly, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})