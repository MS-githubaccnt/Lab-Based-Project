# Create CSV file as requested
import pandas as pd

data = [
    # ---------------- PLA DATA ----------------
    ["PLA", "Polylactic acid (PLA)", 3.2, 49, 70, 2.5],
    ["PLA", "PLA-5% Lignin", None, 48.39, 37, 2.4],
    ["PLA", "PLA-63% starch-24% cellulose-2.9% carnauba wax", None, 3.27, 0.29687, 0.77],
    ["PLA", "70% PLA-20% PBAT-10% office waste paper", None, 49, 73, 3.6],
    ["PLA", "PLA-30% rice straw", None, 22.27, 26, 1.63],
    ["PLA", "PLA-30% kraft lignin", None, 25.3, 68, 1.4],
    ["PLA", "PLA-50% pine wood flour", None, 66.2, 98, 1.6],
    ["PLA", "PLA-30% banana–sisal fiber", None, 79, 125, 1.1],
    ["PLA", "PLA-8% oil seed fillers", None, 62.6, None, 7.8],
    ["PLA", "PLA-10% sugar beet pulp", None, 38, None, 3.1],
    ["PLA", "PLA-30% okra fiber", None, 58.4, None, 1.9],
    ["PLA", "PLA-60% EFB", None, 12.4, 9, 3.3],
    ["PLA", "PLA-60% kenaf", None, 5.2, 28, 4.4],
    ["PLA", "PLA-30% kenaf bast fiber", None, 32, 40.5, 7],
    ["PLA", "PLA-10% coir fiber", None, 57.9, 107.1, 3.7],
    ["PLA", "PLA-40% banana fiber", None, 78.6, 65.4, 0.24],
    ["PLA", "PLA-10% kenaf fiber", None, 37, 40.5, 1.26],
    ["PLA", "PLA-hemp fiber", None, 72.1, 96.5, 5.6],
    ["PLA", "PLA-50% jute fiber", None, 32.3, 41.8, 2.2],
    ["PLA", "PLA-50% flax fiber", None, 151, 215, 8.3],
    ["PLA", "PLA-30% ramie fiber", None, 53, 104, 3.2],
    ["PLA","PLA 96:4 L:D injection mold grade",3.5,59,106,11.3],
    ["PLA","l-PLA (Mw 66000)", None,59,106,7],
    ["PLA","Annealed l-PLA (Mw 66000)",None,66,119,4],
    ["PLA","d,l-PLA (Mw 114000)",None,44,88,5.4],

    # ---------------- PHA DATA ----------------

    # Basic PHB properties
    ["PHA", "PHB", 3.5, 40, None, 5],
    ["PHA", "PHBV (20% HV)", 1.9, 26, None, None],

    # Additional PHB mechanical (from table)
    ["PHA", "PHB (general)", 1.7, 35, None, 10],

    # Blends (Young’s modulus converted MPa → GPa)
    ["PHA", "PLA/PHBV (80/20) + DBPH", None, None, None, 15.95],
    ["PHA", "PE/PHBV (80/20)", 0.348, 25.93, None, None],
    ["PHA", "PE/PHBV (70/30)", 0.300, 17.5, None, None],
    ["PHA", "PHBV/PBS (50/50)", 1.9, 36, None, None],
    ["PHA", "PHB/PEG (9:1)", 0.431, 12.57, None, 3.34],
    ["PHA", "PHB/PEG (8/1)", 0.254, 3.4, None, 24],
    ["PHA", "PHB-5% TABC", None, 14.8, None, 6.3],
    ["PHA", "PHBV-5% MFC/epoxidized soybean oil", 2.67, 27.3, None, 1.27],
    ["PHA", "starch/12% PHA", None, 3.75, None, 72.4],
    ["PHA", "PHA-g-MA/palm fiber", 0.338, 12.9, None, None],
    ["PHA", "PHA-g-MA/TPF", 0.424, 23.7, None, None],
    ["PHA", "PHA/50% PCL", 0.28, 6.4, None, 51.9],
    ["PHA", "crosslinked starch/PHA", None, 8.55, None, 38.6],
    ["PHA", "PHA + 2% SSS", 0.342, 15, None, 518],
    ["PHA", "PHA-AA + 2% SSS", 0.372, 22, None, 565],
    ["PHA", "CNC/PHA (1%)", 0.72, 22.5, None, 10.43],
    ["PHA", "PHA + 20% MF", 0.94, 24.9, None, 3.78],

    # scl-PHAs
    ["PHA", "P3HB", 1.1, 40, None, 5],
    ["PHA", "P4HB", 0.15, 13.8, None, None],
    ["PHA", "P3HV", None, 31.2, None, 14],
    ["PHA", "P3HBco4HB", None, 3, None, None],
    ["PHA", "P3HBco3HV", 0.2, 20, None, 50],

    # mcl-PHAs
    ["PHA", "PHO", 0.02, 12, None, 312.9],
    ["PHA", "P3HBcoHHx", 0.024, 9, None, 380],

        # ---------------- BIO-PET DATA (from image) ----------------
    ["Bio-PET", "Bio-PET100", 0.777, 50.7, None, 378.4],
    ["Bio-PET", "Bio-PET99/RCF01", 0.843, 48.1, None, 8.1],
    ["Bio-PET", "Bio-PET98/RCF02", 0.898, 42.9, None, 6.5],
    ["Bio-PET", "Bio-PET97/RCF03", 0.907, 39.7, None, 6.2],
    ["Bio-PET", "Bio-PET96/RCF04", 0.908, 36.7, None, 5.7],
    ["Bio-PET", "Bio-PET95/RCF05", 0.950, 29.9, None, 4.2],
    ["Bio-PET", "Bio-PET90/RCF10", 1.124, 24.4, None, 2.8],
]

df = pd.DataFrame(data, columns=[
    "material_name",
    "material_specification",
    "tensile_modulus_GPa",
    "tensile_strength_MPa",
    "flexural_strength_MPa",
    "elongation_percent"
])

file_path = "/mnt/data/materials_combined.csv"
df.to_csv(file_path, index=False)