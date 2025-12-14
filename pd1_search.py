import requests
import pandas as pd
from typing import List, Dict, Any

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
UNIPROT = "Q15116"  # PD-1 (PDCD1) human
ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{}"
NONPOLYMER_ENTITY_BASE = "https://data.rcsb.org/rest/v1/core/nonpolymer_entity"
POLYMER_ENTITY_BASE = "https://data.rcsb.org/rest/v1/core/polymer_entity"

# non-trivial ligand filter: exclude common solvent/ions/water
NON_LIGAND_COMP_IDS = {
    "HOH", "WAT", "DOD",
    "NA", "CL", "K", "CA", "MG", "ZN", "MN", "FE", "CO", "NI", "CU", "IOD",
    "SO4", "PO4", "PEG",
}

DEBUG = False


def build_entry_query_by_uniprot_text(uniprot: str) -> Dict[str, Any]:
    # Use text service on reference_sequence_identifiers.database_accession to fetch entry IDs
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "operator": "exact_match",
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                        "value": uniprot,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }


def fetch_pdb_ids_for_uniprot(uniprot: str, timeout_s: int = 60) -> List[str]:
    query = build_entry_query_by_uniprot_text(uniprot)
    resp = requests.post(SEARCH_URL, json=query, timeout=timeout_s)
    if not resp.ok:
        # Print server response to help diagnose issues
        try:
            print(f"[ERROR] RCSB search failed. HTTP {resp.status_code}. Response:\n{resp.text}")
        finally:
            resp.raise_for_status()
    data = resp.json()
    pdb_ids = sorted({it["identifier"].upper() for it in data.get("result_set", []) if "identifier" in it})
    return pdb_ids


def safe_get(d: Dict[str, Any], path: List[Any], default=None):
    cur = d
    for p in path:
        if cur is None:
            return default
        if isinstance(p, int):
            if not isinstance(cur, list) or len(cur) <= p:
                return default
            cur = cur[p]
        else:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
    return cur if cur is not None else default


def fetch_entry(pdb_id: str) -> Dict[str, Any]:
    resp = requests.get(ENTRY_URL.format(pdb_id), timeout=60)
    resp.raise_for_status()
    return resp.json()


def parse_entry_fields(pdb_id: str) -> Dict[str, Any]:
    e = fetch_entry(pdb_id)
    # Helpers for PD-1 residue count
    def fetch_polymer_entity(entry_id: str, entity_id: str):
        ent_ident = f"{entry_id}_{entity_id}"
        # Correct REST path uses slash-separated entry/entity: /polymer_entity/<entry>/<entity_id>
        entry = str(entry_id).upper()
        entno = str(entity_id)
        url = f"{POLYMER_ENTITY_BASE}/{entry}/{entno}"
        r = requests.get(url, timeout=60)
        if not r.ok:
            return None
        try:
            return r.json()
        except Exception as ex:
            return None

    def is_entity_pd1(ent: Dict[str, Any]) -> bool:
        accs = safe_get(ent, ["rcsb_polymer_entity_container_identifiers", "uniprot_accession"], default=None)
        if isinstance(accs, list) and any(str(a).upper() == UNIPROT for a in accs):
            return True
        if isinstance(accs, str) and accs.upper() == UNIPROT:
            return True
        refs = safe_get(ent, ["rcsb_polymer_entity_container_identifiers", "reference_sequence_identifiers"], default=[])
        if isinstance(refs, list):
            for rsi in refs:
                db = str(safe_get(rsi, ["database_name"], default="")).lower()
                acc = str(safe_get(rsi, ["database_accession"], default="")).upper()
                if db in {"uniprot", "unp", "uniprotkb"} and acc == UNIPROT:
                    return True
        return False

    def entity_length(ent: Dict[str, Any]) -> int | None:
        length = safe_get(ent, ["entity_poly", "rcsb_sample_sequence_length"], default=None)
        if isinstance(length, int):
            return length
        seq = safe_get(ent, ["entity_poly", "pdbx_seq_one_letter_code_can"], default=None)
        if isinstance(seq, str) and seq:
            letters = [ch for ch in seq.upper() if "A" <= ch <= "Z"]
            if letters:
                return len(letters)
        pl = safe_get(ent, ["rcsb_polymer_entity", "polymer_length"], default=None)
        if isinstance(pl, int):
            return pl
        return None
    # method
    method = safe_get(e, ["exptl", 0, "method"], default=None)
    # resolution (first combined)
    resolution = None
    res_list = safe_get(e, ["rcsb_entry_info", "resolution_combined"], default=None)
    if isinstance(res_list, list) and res_list:
        resolution = res_list[0]
    # mutation flag from title keywords
    title = safe_get(e, ["struct", "title"], default="") or ""
    has_mutation = False
    for kw in ("mutant", "mutation", "variant"):
        if kw.lower() in title.lower():
            has_mutation = True
            break
    # ligand IDs
    comp_ids: List[str] = []
    # Prefer embedded nonpolymer_entities list when present
    nonpoly_list = safe_get(e, ["nonpolymer_entities"], default=None)
    if isinstance(nonpoly_list, list):
        for ne in nonpoly_list:
            # Try chem_comp.id first
            cid = safe_get(ne, ["chem_comp", "id"], default=None)
            if not cid:
                cid = safe_get(ne, ["pdbx_entity_nonpoly", "comp_id"], default=None)
            if cid:
                comp_ids.append(str(cid).upper())
    # Fallback: call nonpolymer_entity endpoints using IDs
    if not comp_ids:
        nonpoly_ids = safe_get(e, ["rcsb_entry_container_identifiers", "nonpolymer_entity_ids"], default=None)
        if isinstance(nonpoly_ids, list) and nonpoly_ids:
            for np_id in nonpoly_ids:
                try:
                    entry = str(pdb_id).upper()
                    npno = str(np_id)
                    url_np = f"{NONPOLYMER_ENTITY_BASE}/{entry}/{npno}"
                    r = requests.get(url_np, timeout=60)
                    if not r.ok:
                        continue
                    np_json = r.json()
                    cid = safe_get(np_json, ["chem_comp", "id"], default=None)
                    if not cid:
                        cid = safe_get(np_json, ["nonpolymer_entity", "pdbx_entity_nonpoly", "comp_id"], default=None)
                    if cid:
                        comp_ids.append(str(cid).upper())
                except Exception:
                    continue
    comp_ids = sorted(set(comp_ids))
    drug_like = [c for c in comp_ids if c not in NON_LIGAND_COMP_IDS]
    has_ligand_nontrivial = bool(drug_like)

    # PD-1 residue count: check polymer entities mapped to Q15116; take max length if multiple
    pd1_residue_count = None
    protein_polymer_count = 0
    entity_ids = safe_get(e, ["rcsb_entry_container_identifiers", "polymer_entity_ids"], default=None)
    ent_identifiers: List[str] = []
    if isinstance(entity_ids, list) and entity_ids:
        for ent_id in entity_ids:
            ent_identifiers.append(f"{pdb_id}_{ent_id}")
    else:
        n_entities = safe_get(e, ["rcsb_entry_info", "polymer_entity_count"], default=0) or 0
        for i in range(1, n_entities + 1):
            ent_identifiers.append(f"{pdb_id}_{i}")
    # Also try entry.polymer_entities[*].rcsb_id if present
    poly_entities = safe_get(e, ["polymer_entities"], default=None)
    if isinstance(poly_entities, list):
        for pe in poly_entities:
            rid = safe_get(pe, ["rcsb_id"], default=None)
            if isinstance(rid, str) and rid:
                ent_identifiers.append(rid)
    # De-duplicate preserving order
    seen = set()
    ent_identifiers = [x for x in ent_identifiers if not (x in seen or seen.add(x))]
    for ent_ident in ent_identifiers:
        parts = str(ent_ident).split("_", 1)
        if len(parts) != 2:
            continue
        ent = fetch_polymer_entity(parts[0], parts[1])
        if not ent:
            continue
        # count protein-like polymer entities (helps distinguish antibody-containing complexes)
        poly_type = (safe_get(ent, ["entity_poly", "type"], default="") or safe_get(ent, ["rcsb_polymer_entity", "type"], default="") or "").lower()
        if "polypeptide" in poly_type:
            protein_polymer_count += 1
        if is_entity_pd1(ent):
            length_val = entity_length(ent)
            if isinstance(length_val, int):
                if pd1_residue_count is None or length_val > pd1_residue_count:
                    pd1_residue_count = length_val
    is_single_protein = (protein_polymer_count == 1)
    return {
        "pdb_id": pdb_id,
        "method": method,
        "resolution_A": resolution,
        "pd1_residue_count": pd1_residue_count,
        "has_mutation_flag": has_mutation,
        "has_ligand_nontrivial": has_ligand_nontrivial,
        "is_single_protein": is_single_protein,
    }


def main():
    pdb_ids = fetch_pdb_ids_for_uniprot(UNIPROT)
    print(f"UniProt {UNIPROT} → {len(pdb_ids)} PDB entries")
    rows: List[Dict[str, Any]] = []
    for pid in pdb_ids:
        try:
            row = parse_entry_fields(pid)
            rows.append(row)
        except Exception as ex:
            print(f"[WARN] Failed to parse {pid}: {ex}")
    df = pd.DataFrame(rows)
    # Flags for sorting
    df["is_xray"] = df["method"].fillna("").str.contains("X-RAY", case=False)
    # Coerce dtypes to ensure intended ordering
    df["has_mutation_flag"] = df["has_mutation_flag"].fillna(False).astype(bool)
    df["has_ligand_nontrivial"] = df["has_ligand_nontrivial"].fillna(False).astype(bool)
    df["is_single_protein"] = df["is_single_protein"].fillna(False).astype(bool)
    df["resolution_A"] = pd.to_numeric(df["resolution_A"], errors="coerce")
    df["pd1_residue_count"] = pd.to_numeric(df["pd1_residue_count"], errors="coerce")
    # Sorting (priority):
    # 1) is_xray: True first (desc)
    # 2) is_single_protein: True first (desc)
    # 3) has_mutation_flag: False first (asc)
    # 4) resolution_A: smaller first (asc), NaN last
    # 5) pd1_residue_count: larger first (desc), NaN last
    # 6) has_ligand_nontrivial: False first (asc)
    df_sorted = df.sort_values(
        by=["is_xray", "is_single_protein", "has_mutation_flag", "resolution_A", "pd1_residue_count", "has_ligand_nontrivial"],
        ascending=[False, False, True, True, False, True],
        na_position="last"
    )
    out_path = "pd1_Q15116_structures.csv"
    df_sorted.to_csv(out_path, index=False)
    print(f"Saved sorted results to {out_path}")


if __name__ == "__main__":
    main()

