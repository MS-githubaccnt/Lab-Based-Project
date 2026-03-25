from langchain.tools import tool
import json

def validate_input(data):
    required = ["mesh_file", "material", "loads", "constraints"]

    for key in required:
        if key not in data:
            raise ValueError(f"Missing key: {key}")

    mat = data["material"]
    if "E" not in mat or "nu" not in mat:
        raise ValueError("Material must contain E and nu")

    if not data["loads"]:
        raise ValueError("No loads provided")

    if not data["constraints"]:
        raise ValueError("No constraints provided")

def parse_msh(msh_file):
    nodes = []
    elements = []

    with open(msh_file, "r") as f:
        lines = f.readlines()

    reading_nodes = False
    reading_elements = False

    for line in lines:
        if "$Nodes" in line:
            reading_nodes = True
            continue
        if "$EndNodes" in line:
            reading_nodes = False
            continue
        if "$Elements" in line:
            reading_elements = True
            continue
        if "$EndElements" in line:
            reading_elements = False
            continue

        if reading_nodes:
            parts = line.split()
            if len(parts) == 4:
                nodes.append(parts)

        if reading_elements:
            parts = line.split()
            if len(parts) > 4:
                elements.append(parts)

    return nodes, elements

def build_inp(data, nodes, elements):
    inp_file = data["mesh_file"].replace(".msh", ".inp")

    with open(inp_file, "w") as f:

        # Nodes
        f.write("*NODE\n")
        for n in nodes:
            f.write(",".join(n) + "\n")

        # Elements (C3D4 for now)
        f.write("*ELEMENT, TYPE=C3D4\n")
        for e in elements:
            f.write(",".join(e[:5]) + "\n")

        # Material
        E = data["material"]["E"]
        nu = data["material"]["nu"]

        f.write("*MATERIAL, NAME=MAT1\n")
        f.write("*ELASTIC\n")
        f.write(f"{E}, {nu}\n")

        f.write("*SOLID SECTION, ELSET=ALL, MATERIAL=MAT1\n")

        # Boundary conditions
        f.write("*BOUNDARY\n")
        for c in data["constraints"]:
            f.write(f"{c['region']}, 1, 6\n")

        # Loads
        f.write("*CLOAD\n")
        for l in data["loads"]:
            f.write(f"{l['region']}, 3, {-l['value']}\n")

        # Step
        f.write("*STEP\n")
        f.write("*STATIC\n")
        f.write("*END STEP\n")

    return inp_file

import subprocess

def run_solver(inp_file):
    cmd = ["ccx", inp_file.replace(".inp", "")]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    
def parse_results(dat_file):
    max_stress = 0
    max_disp = 0

    with open(dat_file, "r") as f:
        for line in f:
            if "MISES" in line:
                val = float(line.split()[-1])
                max_stress = max(max_stress, val)

            if "U" in line:
                val = float(line.split()[-1])
                max_disp = max(max_disp, val)

    return {
        "max_stress": max_stress,
        "max_displacement": max_disp
    }

@tool
def fea_solver_tool(input_json: str) -> str:
    """
    Runs Finite Element Analysis using CalculiX.

    Input:
    - mesh_file (.msh)
    - material (E, nu)
    - loads
    - constraints

    Output:
    - max_stress
    - max_displacement
    - status
    """

    try:
        data = json.loads(input_json)

        validate_input(data)

        nodes, elements = parse_msh(data["mesh_file"])

        inp_file = build_inp(data, nodes, elements)

        run_solver(inp_file)

        results = parse_results(inp_file.replace(".inp", ".dat"))

        return json.dumps({
            "status": "success",
            "results": results
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })