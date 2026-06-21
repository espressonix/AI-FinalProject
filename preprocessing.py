"""
Preprocessing pipeline - Pyrolysis Plastic Waste ML Project
Privan Peter & Charlie Dumingan
"""
import csv
from collections import Counter

IN_PATH = "pyrolysis_dataset_main.csv"
OUT_PATH = "pyrolysis_dataset_preprocessed.csv"

def parse_primary_fraction(blend_ratio, feedstock_secondary):
    if not blend_ratio or not blend_ratio.strip():
        return 1.0
    s = blend_ratio.strip()
    if ":" in s:
        try:
            a, b = s.split(":")
            a, b = float(a), float(b)
            if a + b == 0:
                return None
            return a / (a + b)
        except ValueError:
            return None
    try:
        val = float(s)
        return 1.0 - val
    except ValueError:
        return None

def to_float_or_none(s):
    if s is None or s.strip() == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None

def catalyst_family(name):
    n = name.lower()
    if n in ("none", ""):
        return "none"
    if "zsm" in n or "zeolite" in n or "si/al" in n:
        return "zeolite"
    if "koh" in n or "na2co3" in n or "alkali" in n:
        return "alkali"
    if "fe" in n or "ni" in n or "cr" in n or "mgo" in n:
        return "metal_doped"
    if "fcc" in n or "ecat" in n or "mcf" in n or "hzsm" in n:
        return "fcc_zeolite_industrial"
    return "other_catalyst"

def feedstock_family(name):
    n = (name or "").lower()
    mapping = {
        "pe": "PE", "hdpe": "PE", "ldpe": "PE",
        "pp": "PP", "pt": "PP",
        "ps": "PS", "pet": "PET",
        "vfr": "mixed_waste", "cfw": "mixed_waste", "efb": "biomass",
        "plastic_container": "mixed_waste", "unspecified_plastic": "mixed_waste",
        "algae_ulva": "biomass", "ulva_lactuca": "biomass",
        "wax": "wax", "pe_wax": "wax", "br": "other", "csgfp_blend": "mixed_waste",
    }
    return mapping.get(n, "other")

rows_out = []
catalyst_types = set()
feedstock_types = set()

with open(IN_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        oil = to_float_or_none(r["oil_yield_pct"])
        if oil is None:
            continue

        primary_fraction = parse_primary_fraction(r["blend_ratio"], r["feedstock_secondary"])
        catalyst = r["catalyst"].strip() if r["catalyst"] else "none"
        has_catalyst = 0 if catalyst.lower() in ("none", "") else 1
        catalyst_loading = to_float_or_none(r["catalyst_loading_pct"]) or 0.0
        temperature = to_float_or_none(r["temperature_C"])

        catalyst_types.add(catalyst)
        feedstock_types.add(r["feedstock_primary"])

        rows_out.append({
            "paper_id": r["paper_id"],
            "group_id": r["paper_id"],
            "feedstock_primary": r["feedstock_primary"],
            "feedstock_family": feedstock_family(r["feedstock_primary"]),
            "feedstock_secondary": r["feedstock_secondary"] or "none",
            "primary_fraction": primary_fraction,
            "catalyst_type": catalyst,
            "catalyst_family": catalyst_family(catalyst),
            "has_catalyst": has_catalyst,
            "catalyst_loading_pct": catalyst_loading,
            "temperature_C": temperature if temperature is not None else "",
            "temperature_missing": 1 if temperature is None else 0,
            "oil_yield_pct": oil,
            "gas_yield_pct_REFERENCE_ONLY": r["gas_yield_pct"],
            "char_yield_pct_REFERENCE_ONLY": r["char_yield_pct"],
            "data_confidence": r["data_confidence"],
            "notes": r["notes"],
        })

fieldnames = list(rows_out[0].keys())
with open(OUT_PATH, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows_out)

print(f"Baris siap training (punya oil_yield_pct): {len(rows_out)}")
print(f"Jumlah paper (group) tersisa: {len(set(r['group_id'] for r in rows_out))}")
print(f"Jumlah temperature_C kosong: {sum(1 for r in rows_out if r['temperature_missing']==1)}")
print(f"catalyst_type unik: {len(catalyst_types)} -> catalyst_family: {len(set(r['catalyst_family'] for r in rows_out))} kategori")
print(f"feedstock_primary unik: {len(feedstock_types)} -> feedstock_family: {len(set(r['feedstock_family'] for r in rows_out))} kategori")
print("Distribusi feedstock_family:", Counter(r['feedstock_family'] for r in rows_out))
print("Distribusi catalyst_family:", Counter(r['catalyst_family'] for r in rows_out))
print("Baris per paper:")
for k, v in sorted(Counter(r['group_id'] for r in rows_out).items()):
    print(f"  {k}: {v}")
