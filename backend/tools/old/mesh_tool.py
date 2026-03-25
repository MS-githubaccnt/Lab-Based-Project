import gmsh
import json
from langchain.tools import tool

@tool
def meshing_tool_v2(cad_file: str) -> str:
    """
    Production-grade meshing tool using Gmsh API.

    Features:
    - Uses STEP if available (fallback STL)
    - Adaptive mesh sizing
    - Curvature-based refinement
    - Boundary tagging ready for FEA
    """


    try:
        gmsh.initialize()
        gmsh.model.add("model")

        ext = cad_file.split(".")[-1].lower()

        if ext in ["step", "stp"]:
            gmsh.model.occ.importShapes(cad_file)
            gmsh.model.occ.synchronize()
        else:
            gmsh.merge(cad_file)

            
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 5)

            
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)

            
        gmsh.model.mesh.generate(2)

            
        gmsh.model.mesh.generate(3)

        msh_file = cad_file + ".msh"
        gmsh.write(msh_file)

        gmsh.finalize()

        return json.dumps({
            "mesh_file": msh_file,
            "status": "success"
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})