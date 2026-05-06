# -*- coding: utf-8 -*-


import os
import json
import numpy as np
import pandas as pd
from Bio import PDB
import freesasa
from sklearn.naive_bayes import GaussianNB
import argparse
from tqdm import tqdm

# 列出 PDB 中的所有链
def list_chains(pdb_path):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    chains = {}
    for chain in structure.get_chains():
        residues = list(chain.get_residues())
        res_count = len(residues)
        first_res = residues[0].get_resname() if residues else "?"
        chains[chain.id] = f"{res_count} 个残基，首残基={first_res}"
    return chains

#  提取抗体链（重链+轻链）
def extract_antibody_chains(pdb_path, heavy_chain, light_chain, output="clean_fv.pdb"):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("ab", pdb_path)
    io = PDB.PDBIO()

    class SelectAbChains(PDB.Select):
        def accept_chain(self, chain):
            return chain.id in [heavy_chain, light_chain]

    io.set_structure(structure)
    io.save(output, SelectAbChains())
    print(f" 已提取抗体链 {heavy_chain}+{light_chain} → {output}")
    return output

#  特征计算函数
PKA_TABLE = {'N_term': 8.0, 'C_term': 3.67, 'ASP': 3.90, 'GLU': 4.30, 'HIS': 6.00,
             'CYS': 8.50, 'TYR': 10.0, 'LYS': 10.5, 'ARG': 12.5}

def get_residue_charge(res_name: str, pH: float = 6.0) -> float:
    if res_name in ['ASP', 'GLU']:
        pKa = PKA_TABLE[res_name]
        return -1 / (1 + 10**(pH - pKa))
    elif res_name == 'HIS':
        pKa = PKA_TABLE['HIS']
        return 1 / (1 + 10**(pH - pKa))
    elif res_name in ['LYS', 'ARG']:
        pKa = PKA_TABLE[res_name]
        return 1 / (1 + 10**(pH - pKa))
    elif res_name in ['CYS', 'TYR']:
        pKa = PKA_TABLE[res_name]
        return -1 / (1 + 10**(pH - pKa))
    return 0.0

def compute_net_charge(pdb_path: str, pH: float = 6.0) -> float:
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("ab", pdb_path)
    total = 0.0
    for chain in structure.get_chains():
        residues = list(chain.get_residues())
        total += 1 / (1 + 10**(pH - PKA_TABLE['N_term']))
        total -= 1 / (1 + 10**(pH - PKA_TABLE['C_term']))
        for res in residues:
            total += get_residue_charge(res.get_resname(), pH)
    return round(total, 4)

HYDROPHOBIC_RES = {'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'TYR', 'CYS'}

def compute_patch_hyd_percent(pdb_path: str) -> float:
    """修复版：兼容 freesasa 新旧版本"""
    structure = freesasa.Structure(pdb_path)
    result = freesasa.calc(structure)
    total_sasa = result.totalArea()
    hyd_sasa = 0.0

    # 兼容不同版本的 freesasa API
    for chain_label in result.residueAreas():
        for res_num in result.residueAreas()[chain_label]:
            area = result.residueAreas()[chain_label][res_num]
            # 新版可能用 residue_name，旧版用 residueName
            res_name = getattr(area, 'residueName', None) or getattr(area, 'residue_name', None) or ""
            if res_name.upper() in HYDROPHOBIC_RES:
                hyd_sasa += area.areaTotal

    if total_sasa > 0:
        return round((hyd_sasa / total_sasa * 100), 4)
    return 0.0

KYTE_DOOLITTLE = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
                  'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
                  'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}

def compute_moments(pdb_path: str, pH: float = 6.0):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("ab", pdb_path)
    ca_atoms = [atom for atom in structure.get_atoms() if atom.get_name() == "CA"]
    if not ca_atoms:
        return 0.0, 0.0
    com = np.mean([atom.get_coord() for atom in ca_atoms], axis=0)

    dipole_vec = np.zeros(3)
    hyd_vec = np.zeros(3)
    for chain in structure.get_chains():
        for res in chain.get_residues():
            if res.get_resname() == "HOH" or res.id[0] != " ":
                continue
            try:
                ca = res["CA"]
                r = ca.get_coord() - com
                charge = get_residue_charge(res.get_resname(), pH)
                dipole_vec += charge * r
                hyd = KYTE_DOOLITTLE.get(res.get_resname()[0], 0.0)
                hyd_vec += hyd * r
            except:
                continue
    return round(np.linalg.norm(dipole_vec), 4), round(np.linalg.norm(hyd_vec), 4)

# 单个 PDB 处理 
def process_single_pdb(pdb_path: str, heavy_chain: str = "H", light_chain: str = "G"):
    print(f"\n正在处理: {pdb_path}")

    print("PDB 中的链信息：")
    for cid, info in list_chains(pdb_path).items():
        print(f"  链 {cid}: {info}")

    clean_pdb = extract_antibody_chains(pdb_path, heavy_chain, light_chain)

    app_charge = compute_net_charge(clean_pdb, pH=6.0)
    patch_hyd = compute_patch_hyd_percent(clean_pdb)
    dipole, hyd_m = compute_moments(clean_pdb, pH=6.0)

    print(f"\n近似特征值：")
    print(f"  app_charge     = {app_charge}")
    print(f"  dipole_moment  = {dipole}")
    print(f"  hyd_moment     = {hyd_m}")
    print(f"  patch_hyd_%    = {patch_hyd}")

    # 加载模型
    data = pd.read_csv("tradeoff_model_moe_features_measured_reduced.csv", index_col=0)
    target = pd.read_csv("7.12.21_data_targets.csv", index_col=0)
    target.loc[target['CS-SINS Score'] > 0.35, 'CS-SINS Label'] = 1
    target.loc[target['SMP Score'] > 0.19, 'SMP Label'] = 1
    target['CS-SINS Label'] = target['CS-SINS Label'].fillna(0)
    target['SMP Label'] = target['SMP Label'].fillna(0)

    X_sins = data[['app_charge', 'dipole_moment', 'patch_hyd_%']].values
    clf_sins = GaussianNB().fit(X_sins, target['CS-SINS Label'].values)

    X_smp = data[['hyd_moment', 'app_charge', 'patch_hyd_%']].values
    clf_smp = GaussianNB().fit(X_smp, target['SMP Label'].values)

    feat_sins = np.array([[app_charge, dipole, patch_hyd]])
    prob_low_sins = clf_sins.predict_proba(feat_sins)[0, 0]

    feat_smp = np.array([[hyd_m, app_charge, patch_hyd]])
    prob_low_smp = clf_smp.predict_proba(feat_smp)[0, 0]

    print("\n=== 预测结果 ===")
    print(f"自聚集 (CS-SINS) 低风险概率：{prob_low_sins:.3f}  (>0.5 为优)")
    print(f"非特异性结合 (SMP) 低风险概率：{prob_low_smp:.3f}  (>0.5 为优)")

    if prob_low_sins > 0.5 and prob_low_smp > 0.5:
        print(" 综合表现优秀！")
    else:
        print("建议优化 CDR 区")

    return {
        "pdb": os.path.basename(pdb_path),
        "app_charge": app_charge,
        "dipole": dipole,
        "hyd_moment": hyd_m,
        "patch_hyd": patch_hyd,
        "prob_CS_SINS": float(prob_low_sins),
        "prob_SMP": float(prob_low_smp)
    }

# 批量扫描（忽略 reference）
def parse_pdb_files(data_dir: str):
    pdb_files = []
    reference_dir = os.path.abspath(os.path.join(data_dir, 'reference'))

    for root, dirs, files in os.walk(data_dir):
        if os.path.abspath(root).startswith(reference_dir):
            continue
        for fname in files:
            if fname.lower().endswith('.pdb'):
                full_path = os.path.join(root, fname)
                if os.path.getsize(full_path) > 0:
                    pdb_files.append(full_path)
    return sorted(pdb_files)

#主函数 
def main():
    parser = argparse.ArgumentParser(description="批量预测抗体自聚集和非特异性结合")
    parser.add_argument("--data_dir", required=True, help="包含 PDB 文件的目录")
    parser.add_argument("--heavy_chain", default="H", help="重链标识符")
    parser.add_argument("--light_chain", default="G", help="轻链标识符")
    parser.add_argument("--output", default="developability_results.json", help="输出 JSON 文件")
    args = parser.parse_args()

    pdb_files = parse_pdb_files(args.data_dir)
    if not pdb_files:
        print("未找到任何 .pdb 文件！")
        return

    print(f"找到 {len(pdb_files)} 个 PDB 文件，开始批量预测...\n")

    all_results = []
    for pdb_path in tqdm(pdb_files, desc="预测进度"):
        result = process_single_pdb(pdb_path, args.heavy_chain, args.light_chain)
        all_results.append(result)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n全部处理完成！结果保存至: {args.output}")

if __name__ == "__main__":
    main()