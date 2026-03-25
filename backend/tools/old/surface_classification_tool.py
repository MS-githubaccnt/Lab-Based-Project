import trimesh
import numpy as np
import json
from langchain.tools import tool

@tool
def classify_surfaces(file_path: str) -> str:
    """
    Robust surface segmentation using adjacency + curvature.
    Detects functional regions for FEA:
    - flat contact faces
    - curved load-bearing regions
    - sharp transitions
    """
    try:
        mesh = trimesh.load_mesh(file_path)

        adjacency = mesh.face_adjacency
        normals = mesh.face_normals

        visited = set()
        regions = []

        def grow_region(seed):
            stack = [seed]
            region = []

            while stack:
                f = stack.pop()
                if f in visited:
                    continue

                visited.add(f)
                region.append(f)

                neighbors = adjacency[adjacency[:, 0] == f][:, 1]
                neighbors = np.append(neighbors,
                                      adjacency[adjacency[:, 1] == f][:, 0])

                for n in neighbors:
                    if n not in visited:
                        angle = np.dot(normals[f], normals[n])
                        if angle > 0.95:  # smooth continuation
                            stack.append(n)

            return region

        for i in range(len(mesh.faces)):
            if i not in visited:
                reg = grow_region(i)
                regions.append(reg)

        output = []

        for r_id, faces in enumerate(regions):
            area = mesh.area_faces[faces].sum()
            avg_normal = normals[faces].mean(axis=0)
            avg_normal /= np.linalg.norm(avg_normal)

            curvature = np.std(normals[faces], axis=0).mean()

            output.append({
                "region_id": r_id,
                "area": float(area),
                "normal": avg_normal.tolist(),
                "flat": curvature < 0.05,
                "type": "planar" if curvature < 0.05 else "curved"
            })

        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})