#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
抗体 CDR 几何真实性评估：JSD、MAE、MCE（支持多 CDR 区域 + 自动链检测，优先从文件名解析链标识符）

本脚本根据 GeoAB 论文定义，自动分析以下 CDR 区域（IMGT 编号）：
- CDR-H1 (H链, 残基27-38)
- CDR-H2 (H链, 残基56-65)
- CDR-H3 (H链, 残基95-102)
- CDR-L1 (L链, 残基27-38)
- CDR-L2 (L链, 残基56-65)
- CDR-L3 (L链, 残基89-97)

链标识符解析顺序：
1. 从文件名中解析（格式：*_H_L_A.pdb 或 *_H_L.pdb）
2. 若解析失败，自动检测每个 PDB 文件的链长度，最长链为重链，次长链为轻链
3. 用户可通过 --heavy_chain / --light_chain 手动指定，此时将忽略前两者

使用方法 (legacy 模式)：
    python eval_geo.py --data_dir /path/to/data --mode legacy --output results.json

使用方法 (diffab 模式)：
    python eval_geo.py --data_dir /path/to/data --mode diffab --output results.json

可选参数：
    --cdr H3 L3             只分析指定的 CDR（默认全部）
    --heavy_chain A         手动指定重链链标识符（关闭自动检测）
    --light_chain B         手动指定轻链链标识符（关闭自动检测）
    --auto_detect           自动检测链标识符（默认 True）

依赖库：biopython, numpy, scipy, tqdm
"""

import os
import json
import re
import argparse
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import jensenshannon
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from tqdm import tqdm


# ========================== 几何特征提取函数（不变） ==========================
def get_atom_coord(res, atom_name):
    """获取残基中指定原子的坐标，若不存在则返回 None"""
    if atom_name in res:
        return res[atom_name].get_coord()
    return None


def calc_bond_length(coord1, coord2):
    """计算两点之间的欧氏距离（键长）"""
    return np.linalg.norm(coord1 - coord2)


def calc_bond_angle(coord1, coord2, coord3):
    """
    计算三点构成的键角（弧度），以 coord2 为顶点。
    向量：coord1 - coord2 和 coord3 - coord2
    """
    v1 = coord1 - coord2
    v2 = coord3 - coord2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def calc_torsion_angle(coord1, coord2, coord3, coord4):
    """
    计算四个原子构成的二面角（弧度），范围 (-π, π]。
    使用标准公式：通过两个平面的法向量点积计算角度，再通过三重积确定符号。
    """
    v1 = coord2 - coord1
    v2 = coord3 - coord2
    v3 = coord4 - coord3
    n1 = np.cross(v1, v2)          # 平面1的法向量
    n2 = np.cross(v2, v3)          # 平面2的法向量
    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)
    if norm_n1 < 1e-6 or norm_n2 < 1e-6:
        return 0.0
    cos_angle = np.dot(n1, n2) / (norm_n1 * norm_n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    # 确定符号： (v1 × v2) · v3 的正负
    sign = np.dot(np.cross(v1, v2), v3)
    if sign < 0:
        angle = -angle
    return angle


def extract_geometries_from_pdb(pdb_file, chain_id, start_pos, end_pos):
    """
    从 PDB 文件中提取指定链和残基范围内的主链几何特征。
    始终使用 PDB 残基编号进行筛选，不依赖于序列位置。
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_file)
    except Exception as e:
        print(f"读取 PDB 文件失败 {pdb_file}: {e}")
        return None

    residues = []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                # 检查是否为氨基酸
                try:
                    if is_aa(res):
                        residues.append(res)
                except:
                    # 备用方法：通过残基名称判断
                    aa_names = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                                'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                                'THR', 'TRP', 'TYR', 'VAL']
                    if res.get_resname() in aa_names:
                        residues.append(res)

    # 按残基编号筛选
    filtered = [r for r in residues if start_pos <= r.id[1] <= end_pos]
    if len(filtered) == 0:
        return None
    residues = filtered

    bond_lengths = defaultdict(list)
    bond_angles = defaultdict(list)
    torsion_angles = defaultdict(list)

    n_res = len(residues)
    for i, res in enumerate(residues):
        N = get_atom_coord(res, 'N')
        CA = get_atom_coord(res, 'CA')
        C = get_atom_coord(res, 'C')
        O = get_atom_coord(res, 'O')

        if N is None or CA is None or C is None or O is None:
            continue

        # 残基内键长
        bond_lengths['N-CA'].append(float(calc_bond_length(N, CA)))
        bond_lengths['CA-C'].append(float(calc_bond_length(CA, C)))
        # 残基内键角
        bond_angles['N-CA-C'].append(float(calc_bond_angle(N, CA, C)))

        # 残基间几何（需要下一个残基）
        if i < n_res - 1:
            next_res = residues[i + 1]
            next_N = get_atom_coord(next_res, 'N')
            next_CA = get_atom_coord(next_res, 'CA')
            if next_N is None:
                continue

            # 残基间键长 C-N
            bond_lengths['C-N'].append(float(calc_bond_length(C, next_N)))

            # 残基间键角 CA-C-N
            if next_CA is not None:
                bond_angles['CA-C-N'].append(
                    float(calc_bond_angle(CA, C, next_N))
                )

            # 柔性扭转角 C-N-CA-C
            next_C = get_atom_coord(next_res, 'C')
            if next_C is not None and next_CA is not None:
                torsion_angles['C-N-CA-C'].append(
                    float(calc_torsion_angle(C, next_N, next_CA, next_C))
                )

            # 刚性扭转角 O=C-N-CA（肽键平面性）
            if next_CA is not None:
                torsion_angles['O=C-N-CA'].append(
                    float(calc_torsion_angle(O, C, next_N, next_CA))
                )

    if not bond_lengths and not bond_angles and not torsion_angles:
        return None

    return {
        'bond_lengths': dict(bond_lengths),
        'bond_angles': dict(bond_angles),
        'torsion_angles': dict(torsion_angles)
    }


# ========================== 指标计算函数（不变） ==========================
def angular_difference(angle1, angle2):
    """
    计算两个角度的最小有符号差值，结果在 [-π, π] 之间。
    参数可以是标量或numpy数组，返回相同形状。
    """
    diff = (angle2 - angle1) % (2 * np.pi)
    # 对于大于π的差值，减去2π；使用np.where向量化操作
    mask = diff > np.pi
    diff = np.where(mask, diff - 2 * np.pi, diff)
    return diff


def compute_jsd(pred_values, true_values, n_bins=50, range_limits=None, is_angular=False):
    """计算两个数值分布之间的 Jensen-Shannon 散度。
    若 is_angular=True，则使用固定范围 [-π, π] 进行分箱（用于角度/二面角）。
    """
    if len(pred_values) == 0 or len(true_values) == 0:
        return np.nan
    if is_angular:
        range_limits = (-np.pi, np.pi)
    if range_limits is None:
        min_val = min(np.min(pred_values), np.min(true_values))
        max_val = max(np.max(pred_values), np.max(true_values))
    else:
        min_val, max_val = range_limits
    if min_val == max_val:
        return 0.0
    bins = np.linspace(min_val, max_val, n_bins + 1)
    hist_pred, _ = np.histogram(pred_values, bins=bins, density=True)
    hist_true, _ = np.histogram(true_values, bins=bins, density=True)
    eps = 1e-10
    hist_pred = hist_pred + eps
    hist_true = hist_true + eps
    hist_pred = hist_pred / np.sum(hist_pred)
    hist_true = hist_true / np.sum(hist_true)
    return float(jensenshannon(hist_pred, hist_true))


def compute_mae(pred_values, true_values):
    """计算平均绝对误差（用于键长）"""
    if len(pred_values) == 0 or len(true_values) == 0:
        return np.nan
    pred = np.array(pred_values)
    true = np.array(true_values)
    return float(np.mean(np.abs(pred - true)))


def compute_mce(pred_angles, true_angles):
    """计算平均余弦误差（用于角度），公式：MCE = 1 - cos(θ_true - θ_pred)，
    使用角差规范化确保差值在 [-π, π] 内。
    """
    if len(pred_angles) == 0 or len(true_angles) == 0:
        return np.nan
    pred = np.array(pred_angles)
    true = np.array(true_angles)
    diff = angular_difference(pred, true)   # 规范化角差
    return float(np.mean(1.0 - np.cos(diff)))


# ========================== CDR 定义（抽象链标识符） ==========================
CDR_DEFINITIONS = {
    'H1': {'chain': 'H', 'start': 27, 'end': 38, 'desc': 'CDR-H1 (重链, 第27-38位残基)'},
    'H2': {'chain': 'H', 'start': 56, 'end': 65, 'desc': 'CDR-H2 (重链, 第56-65位残基)'},
    'H3': {'chain': 'H', 'start': 95, 'end': 102, 'desc': 'CDR-H3 (重链, 第95-102位残基)'},
    'L1': {'chain': 'L', 'start': 27, 'end': 38, 'desc': 'CDR-L1 (轻链, 第27-38位残基)'},
    'L2': {'chain': 'L', 'start': 56, 'end': 65, 'desc': 'CDR-L2 (轻链, 第56-65位残基)'},
    'L3': {'chain': 'L', 'start': 89, 'end': 97, 'desc': 'CDR-L3 (轻链, 第89-97位残基)'},
}


# ========================== 链标识符解析（新增：从文件名解析） ==========================
def parse_chains_from_filename(filepath):
    """
    从文件名中解析重链、轻链、抗原链标识符。
    支持格式：
        *prefix*_H_L_A.pdb   -> heavy='H', light='L', antigen='A'
        *prefix*_H_L.pdb     -> heavy='H', light='L', antigen=None
    返回 (heavy, light, antigen) 元组，若解析失败则返回 (None, None, None)
    """
    basename = os.path.basename(filepath)
    # 匹配模式：任意内容_单字符_单字符(_单字符)?\.pdb$
    # 例如：2adf_H_L_A.pdb 或 2adf_H_L.pdb
    pattern = r'_([A-Z])_([A-Z])(?:_([A-Z]))?\.pdb$'
    match = re.search(pattern, basename)
    if not match:
        return (None, None, None)
    heavy = match.group(1)
    light = match.group(2)
    antigen = match.group(3) if match.group(3) else None
    return (heavy, light, antigen)


def get_heavy_light_chains(pdb_file):
    """
    原有的自动检测函数：根据链长度自动检测重链和轻链。
    返回 (heavy_chain_id, light_chain_id)，若链数量不足2则返回 (None, None)。
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_file)
    except Exception as e:
        print(f"无法解析 {pdb_file}: {e}")
        return None, None

    chain_lengths = []
    for chain in structure.get_chains():
        residues = list(chain.get_residues())
        # 过滤掉水分子等非氨基酸残基
        aa_residues = [r for r in residues if is_aa(r)]
        chain_lengths.append((chain.id, len(aa_residues)))

    if len(chain_lengths) < 2:
        print(f"警告：{pdb_file} 中氨基酸链数量不足2条，无法自动分配重链/轻链")
        return None, None

    # 按残基数降序排序，最长链视为重链，次长链视为轻链
    chain_lengths.sort(key=lambda x: x[1], reverse=True)
    heavy = chain_lengths[0][0]
    light = chain_lengths[1][0]
    return heavy, light


def get_chains_for_pair(pred_file, ref_file, use_filename_parsing=True):
    """
    为预测-参考对确定链标识符。
    优先从 pred_file 的文件名解析（如果 use_filename_parsing=True 且解析成功），
    否则回退到自动检测（按链长度排序）。
    返回 (heavy_chain_id, light_chain_id)。
    """
    if use_filename_parsing:
        heavy, light, _ = parse_chains_from_filename(pred_file)
        if heavy is not None and light is not None:
            return heavy, light
    # 回退：使用原有的链长度自动检测（从预测文件或参考文件读取，通常两者一致）
    heavy, light = get_heavy_light_chains(pred_file)
    if heavy is None or light is None:
        heavy, light = get_heavy_light_chains(ref_file)
    return heavy, light


# ========================== 任务构建函数（保持不变） ==========================
def parse_list(data_dir: str, include_relaxed: bool) -> list:
    pdb_files = []
    pdb_pattern = re.compile(r'\.pdb$')
    relax_pattern = re.compile(r'_relaxed\.pdb$')
    reference_dir = os.path.abspath(os.path.join(data_dir, 'reference'))
    for root, _, files in os.walk(data_dir):
        if os.path.abspath(root).startswith(reference_dir):
            continue
        for fname in files:
            if not pdb_pattern.search(fname):
                continue
            if os.path.getsize(os.path.join(root, fname)) == 0:
                continue
            is_relaxed = bool(relax_pattern.search(fname))
            if include_relaxed and not is_relaxed:
                continue
            if not include_relaxed and is_relaxed:
                continue
            pdb_files.append(os.path.join(root, fname))
    return pdb_files


def build_tasks_legacy(data_dir: str, include_relaxed: bool):
    pred_files = parse_list(data_dir, include_relaxed=include_relaxed)
    reference_dir = os.path.join(data_dir, 'reference')
    tasks = []
    for pred_file in pred_files:
        stem = os.path.splitext(os.path.basename(pred_file))[0].split('@')[0]
        if stem.endswith('_relaxed'):
            stem = stem[:-len('_relaxed')]
        ref_file = os.path.join(reference_dir, f"{stem}.pdb")
        if os.path.exists(ref_file):
            tasks.append((pred_file, ref_file, {}))
    return tasks


def _pick_best_regions(region_names):
    best = {}
    for rn in region_names:
        if "-O" in rn:
            base, step_str = rn.split("-O", 1)
            try:
                step = int(step_str)
            except ValueError:
                step = 0
        else:
            base, step = rn, 0
        cur = best.get(base)
        if cur is None or step > cur[0]:
            best[base] = (step, rn)
    return [v[1] for v in best.values()]


def build_tasks_diffab(data_dir: str, include_relaxed: bool):
    tasks = []
    data_dir = os.path.abspath(data_dir)
    job_dir_re = re.compile(
        r'^(?P<idx>\d{4})_(?P<pdbid>[^_]+)_(?P<chains>.+?)_'
        r'(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})__'
        r'(?P<h>\d{2})_(?P<m>\d{2})_(?P<s>\d{2})$'
    )
    region_dir_re = re.compile(r'^[HL]_CDR[123](?:-O\d+)?$')
    sample_re = re.compile(r'^\d{4}_relaxed\.pdb$' if include_relaxed else r'^\d{4}\.pdb$')
    ref_re = re.compile(r'^REF\d+\.pdb$', re.IGNORECASE)

    def parse_job(job_name: str):
        m = job_dir_re.match(job_name)
        if not m:
            return None
        target = m.group("pdbid")
        chain_tokens = [t for t in m.group("chains").split("_") if t]
        if len(chain_tokens) < 2:
            return None
        heavy, light = chain_tokens[0], chain_tokens[1]
        antigen = "".join(chain_tokens[2:])
        return {"target": target, "heavy": heavy, "light": light, "antigen": antigen}

    for job_name in sorted(os.listdir(data_dir)):
        job_path = os.path.join(data_dir, job_name)
        if not os.path.isdir(job_path) or not job_dir_re.match(job_name):
            continue
        meta_job = parse_job(job_name)
        if not meta_job:
            continue

        region_names = []
        for rn in sorted(os.listdir(job_path)):
            rp = os.path.join(job_path, rn)
            if os.path.isdir(rp) and region_dir_re.match(rn):
                region_names.append(rn)
        region_names = _pick_best_regions(region_names)

        for region_name in region_names:
            region_path = os.path.join(job_path, region_name)
            ref_candidates = [f for f in sorted(os.listdir(region_path)) if ref_re.match(f)]
            if not ref_candidates:
                continue
            ref_file = os.path.join(region_path, "REF1.pdb" if "REF1.pdb" in ref_candidates else ref_candidates[0])

            for fn in sorted(os.listdir(region_path)):
                if not sample_re.match(fn):
                    continue
                pred_file = os.path.join(region_path, fn)
                sample_id = fn.split('_')[0].split('.')[0]
                meta = {
                    "mode": "diffab",
                    "job": job_name,
                    "region": region_name,
                    "sample_id": sample_id,
                    "target": meta_job["target"],
                    "heavy": meta_job["heavy"],
                    "light": meta_job["light"],
                    "antigen": meta_job["antigen"],
                }
                tasks.append((pred_file, ref_file, meta))
    return tasks


# ========================== 终端打印函数（不变） ==========================
def print_results(results):
    """在终端中美观地打印结果"""
    print("\n" + "=" * 80)
    print("抗体 CDR 几何真实性评估结果")
    print("=" * 80)

    print("\n【Jensen-Shannon 散度 (JSD)】")
    print("说明：值越小表示预测结构的内几何分布越接近真实结构\n")
    jsd_results = results.get("JSD", {})
    for cdr_name, metrics in jsd_results.items():
        cdr_desc = CDR_DEFINITIONS.get(cdr_name, {}).get('desc', cdr_name)
        print(f"  {cdr_desc}:")
        if metrics:
            for metric_name, value in metrics.items():
                geo_type = metric_name.replace('JSD_', '')
                print(f"    {geo_type}: {value:.6f}")
        else:
            print(f"    (无有效数据)")
        print()

    print("\n【平均绝对误差 (MAE) 和平均余弦误差 (MCE)】")
    print("说明：MAE 用于键长（单位：埃），MCE 用于角度（范围 0-2，0 表示完全一致）\n")
    mae_mce = results.get("MAE_MCE_average", {})
    for cdr_name, metrics in mae_mce.items():
        cdr_desc = CDR_DEFINITIONS.get(cdr_name, {}).get('desc', cdr_name)
        print(f"  {cdr_desc}:")
        if metrics:
            for metric_name, value in metrics.items():
                print(f"    {metric_name}: {value:.6f}")
        else:
            print(f"    (无有效数据)")
        print()

    print("\n【样本统计】")
    samples_per_cdr = results.get("num_samples_per_CDR", {})
    for cdr_name, count in samples_per_cdr.items():
        cdr_desc = CDR_DEFINITIONS.get(cdr_name, {}).get('desc', cdr_name)
        print(f"  {cdr_desc}: {count} 个有效样本")

    print("\n" + "-" * 80)
    print("【整体平均结果（跨所有 CDR）】")
    print("-" * 80)

    all_jsd = []
    for cdr_name, metrics in jsd_results.items():
        for value in metrics.values():
            all_jsd.append(value)
    if all_jsd:
        print(f"  平均 JSD: {np.mean(all_jsd):.6f} ± {np.std(all_jsd):.6f}")
    else:
        print("  平均 JSD: 无有效数据")

    all_mae = []
    all_mce = []
    for cdr_name, metrics in mae_mce.items():
        for metric_name, value in metrics.items():
            if metric_name.startswith('MAE'):
                all_mae.append(value)
            else:
                all_mce.append(value)
    if all_mae:
        print(f"  平均 MAE (键长): {np.mean(all_mae):.6f} ± {np.std(all_mae):.6f}")
    else:
        print("  平均 MAE (键长): 无有效数据")
    if all_mce:
        print(f"  平均 MCE (角度): {np.mean(all_mce):.6f} ± {np.std(all_mce):.6f}")
    else:
        print("  平均 MCE (角度): 无有效数据")

    print("\n" + "=" * 80)


# ========================== 可导入的几何指标计算函数（修改：移除 use_sequence_pos） ==========================
def compute_geometry_metrics_for_pair(pred_file, ref_file, heavy_chain=None, light_chain=None,
                                       cdr_list=None, auto_detect=True, n_bins=50):
    """
    计算单个预测-参考对的几何指标（JSD、MAE、MCE），返回扁平化的字典。

    参数:
        pred_file (str): 预测 PDB 文件路径
        ref_file (str):  参考 PDB 文件路径
        heavy_chain (str): 手动指定重链标识符（如 'A'），若为 None 且 auto_detect=True 则自动检测
        light_chain (str): 手动指定轻链标识符（如 'B'）
        cdr_list (list):  要分析的 CDR 列表，默认为全部六个
        auto_detect (bool): 是否自动检测链标识符（优先从文件名解析）
        n_bins (int): JSD 直方图区间数

    返回:
        dict: 扁平化的指标字典，例如 {'JSD_H1_N-CA': 0.123, 'MAE_H1_N-CA': 0.045, ...}
              若无法提取任何有效指标，返回空字典。
    """
    if cdr_list is None:
        cdr_list = list(CDR_DEFINITIONS.keys())

    # 确定链标识符
    if heavy_chain is not None and light_chain is not None:
        pred_chain_map = {'H': heavy_chain, 'L': light_chain}
        ref_chain_map = {'H': heavy_chain, 'L': light_chain}
    else:
        # 优先从文件名解析（如果 auto_detect 为 True）
        if auto_detect:
            heavy, light = get_chains_for_pair(pred_file, ref_file, use_filename_parsing=True)
        else:
            # 回退到默认 H/L
            heavy, light = ('H', 'L')
        if heavy is None or light is None:
            return {}
        pred_chain_map = {'H': heavy, 'L': light}
        ref_chain_map = {'H': heavy, 'L': light}

    # 初始化结果收集器
    all_geoms = {cdr: {'bond_lengths': defaultdict(list),
                       'bond_angles': defaultdict(list),
                       'torsion_angles': defaultdict(list)}
                 for cdr in cdr_list}
    sample_errors = {cdr: {} for cdr in cdr_list}

    # 对每个 CDR 提取几何特征
    for cdr_name in cdr_list:
        defn = CDR_DEFINITIONS[cdr_name]
        abstract_chain = defn['chain']
        start, end = defn['start'], defn['end']

        pred_chain = pred_chain_map.get(abstract_chain)
        ref_chain = ref_chain_map.get(abstract_chain)
        if pred_chain is None or ref_chain is None:
            continue

        pred_geo = extract_geometries_from_pdb(pred_file, pred_chain, start, end)
        ref_geo = extract_geometries_from_pdb(ref_file, ref_chain, start, end)
        if pred_geo is None or ref_geo is None:
            continue

        # 收集分布数据（用于 JSD）
        for gtype in ['bond_lengths', 'bond_angles', 'torsion_angles']:
            for key in ref_geo[gtype].keys():
                all_geoms[cdr_name][gtype][key].extend(pred_geo[gtype].get(key, []))
                all_geoms[cdr_name][gtype][key + "_true"].extend(ref_geo[gtype].get(key, []))

        # 计算该 CDR 的 MAE/MCE
        errors = {}
        for key in ['N-CA', 'CA-C', 'C-N']:
            pred_vals = pred_geo['bond_lengths'].get(key, [])
            true_vals = ref_geo['bond_lengths'].get(key, [])
            if len(pred_vals) == len(true_vals) and len(pred_vals) > 0:
                errors[f'MAE_{key}'] = compute_mae(pred_vals, true_vals)
        for key in ['N-CA-C', 'CA-C-N']:
            pred_vals = pred_geo['bond_angles'].get(key, [])
            true_vals = ref_geo['bond_angles'].get(key, [])
            if len(pred_vals) == len(true_vals) and len(pred_vals) > 0:
                errors[f'MCE_{key}'] = compute_mce(pred_vals, true_vals)
        for key in ['C-N-CA-C', 'O=C-N-CA']:
            pred_vals = pred_geo['torsion_angles'].get(key, [])
            true_vals = ref_geo['torsion_angles'].get(key, [])
            if len(pred_vals) == len(true_vals) and len(pred_vals) > 0:
                errors[f'MCE_{key}'] = compute_mce(pred_vals, true_vals)

        sample_errors[cdr_name] = errors

    # 计算每个 CDR 的 JSD
    jsd_values = {}
    for cdr_name in cdr_list:
        for gtype in ['bond_lengths', 'bond_angles', 'torsion_angles']:
            for key in list(all_geoms[cdr_name][gtype].keys()):
                if key.endswith("_true"):
                    continue
                true_key = key + "_true"
                if true_key in all_geoms[cdr_name][gtype]:
                    pred_vals = all_geoms[cdr_name][gtype][key]
                    true_vals = all_geoms[cdr_name][gtype][true_key]
                    if len(pred_vals) > 0 and len(true_vals) > 0:
                        is_angular = (gtype == 'torsion_angles')
                        jsd = compute_jsd(pred_vals, true_vals, n_bins=n_bins, is_angular=is_angular)
                        jsd_values[f'JSD_{cdr_name}_{key}'] = jsd

    # 将 MAE/MCE 扁平化并合并
    flat_errors = {}
    for cdr_name, err_dict in sample_errors.items():
        for k, v in err_dict.items():
            flat_errors[f'{k}_{cdr_name}'] = v

    # 合并所有指标
    all_metrics = {**jsd_values, **flat_errors}
    return all_metrics


# ========================== 主流程（修改：移除 use_sequence_pos） ==========================
def main():
    parser = argparse.ArgumentParser(description="计算抗体多 CDR 几何评估指标 JSD, MAE, MCE")
    parser.add_argument("--data_dir", required=True, help="数据根目录（与 eval_metric.py 一致）")
    parser.add_argument("--mode", choices=['legacy', 'diffab'], default='legacy',
                        help="legacy: 参考文件在 data_dir/reference/ 下；diffab: 参考文件为 REF*.pdb")
    parser.add_argument("--output", default="geo_results.json", help="输出 JSON 文件路径")
    parser.add_argument("--heavy_chain", default=None, help="手动指定重链链标识符（例如 A），设置后将关闭自动检测")
    parser.add_argument("--light_chain", default=None, help="手动指定轻链链标识符（例如 B），设置后将关闭自动检测")
    parser.add_argument("--auto_detect", action='store_true', default=True,
                        help="自动检测重链和轻链（先按文件名解析，失败后按链长度），当手动指定链时自动忽略")
    parser.add_argument("--cdr", nargs='+', choices=list(CDR_DEFINITIONS.keys()),
                        default=list(CDR_DEFINITIONS.keys()), help="要分析的 CDR 区域（默认全部）")
    parser.add_argument("--n_bins", type=int, default=50, help="JSD 直方图区间数")
    args = parser.parse_args()

    # 构建任务列表
    include_relaxed = args.data_dir.rstrip('/').endswith('_relaxed')
    if args.mode == "legacy":
        tasks = build_tasks_legacy(args.data_dir, include_relaxed)
    else:
        tasks = build_tasks_diffab(args.data_dir, include_relaxed)

    if not tasks:
        print("错误：未找到任何有效的预测-参考配对。")
        return

    # 处理链标识符规则
    if args.heavy_chain is not None and args.light_chain is not None:
        fixed_heavy = args.heavy_chain
        fixed_light = args.light_chain
        auto_detect = False
        print(f"使用手动指定的链: 重链={fixed_heavy}, 轻链={fixed_light}")
    else:
        # 尝试从第一个任务的文件名解析全局链
        sample_pred, _, _ = tasks[0]
        heavy, light, _ = parse_chains_from_filename(sample_pred)
        if heavy is not None and light is not None:
            fixed_heavy, fixed_light = heavy, light
            auto_detect = False
            print(f"从文件名（{os.path.basename(sample_pred)}）解析到链: 重链={fixed_heavy}, 轻链={fixed_light}")
        else:
            auto_detect = args.auto_detect
            fixed_heavy = fixed_light = None
            if auto_detect:
                print("将自动检测每个 PDB 文件的重链和轻链（优先从文件名解析，失败后按链长度）")
            else:
                print("警告: 未指定重链/轻链且 auto_detect 为 False，将尝试使用默认 'H' 和 'L'")
                fixed_heavy, fixed_light = 'H', 'L'

    print(f"找到 {len(tasks)} 个任务，开始分析...")
    print(f"将分析以下 CDR: {', '.join(args.cdr)}")

    # 存储每个 CDR 的全局几何值（用于 JSD）
    all_geoms = {cdr: {'bond_lengths': defaultdict(list),
                       'bond_angles': defaultdict(list),
                       'torsion_angles': defaultdict(list)}
                 for cdr in args.cdr}

    # 存储每个样本每个 CDR 的 MAE/MCE
    sample_errors = {cdr: [] for cdr in args.cdr}

    for pred_file, ref_file, _ in tqdm(tasks, desc="处理 PDB 对"):
        # 确定当前任务的链标识符
        if fixed_heavy is not None and fixed_light is not None:
            pred_chain_map = {'H': fixed_heavy, 'L': fixed_light}
            ref_chain_map = {'H': fixed_heavy, 'L': fixed_light}
        else:
            # 自动检测（文件名优先）
            if auto_detect:
                heavy, light = get_chains_for_pair(pred_file, ref_file, use_filename_parsing=True)
            else:
                heavy, light = ('H', 'L')
            if heavy is None or light is None:
                continue
            pred_chain_map = {'H': heavy, 'L': light}
            ref_chain_map = {'H': heavy, 'L': light}

        for cdr_name in args.cdr:
            defn = CDR_DEFINITIONS[cdr_name]
            abstract_chain = defn['chain']   # 'H' 或 'L'
            start = defn['start']
            end = defn['end']

            pred_chain = pred_chain_map.get(abstract_chain)
            ref_chain = ref_chain_map.get(abstract_chain)
            if pred_chain is None or ref_chain is None:
                continue

            pred_geo = extract_geometries_from_pdb(pred_file, pred_chain, start, end)
            ref_geo = extract_geometries_from_pdb(ref_file, ref_chain, start, end)
            if pred_geo is None or ref_geo is None:
                continue

            # 收集全局分布数据
            for gtype in ['bond_lengths', 'bond_angles', 'torsion_angles']:
                for key in ref_geo[gtype].keys():
                    all_geoms[cdr_name][gtype][key].extend(pred_geo[gtype].get(key, []))
                    all_geoms[cdr_name][gtype][key + "_true"].extend(ref_geo[gtype].get(key, []))

            # 计算该样本该 CDR 的 MAE/MCE
            error_sample = {}
            for key in ['N-CA', 'CA-C', 'C-N']:
                pred_vals = pred_geo['bond_lengths'].get(key, [])
                true_vals = ref_geo['bond_lengths'].get(key, [])
                if len(pred_vals) == len(true_vals) and len(pred_vals) > 0:
                    error_sample[f'MAE_{key}'] = compute_mae(pred_vals, true_vals)
            for key in ['N-CA-C', 'CA-C-N']:
                pred_vals = pred_geo['bond_angles'].get(key, [])
                true_vals = ref_geo['bond_angles'].get(key, [])
                if len(pred_vals) == len(true_vals) and len(pred_vals) > 0:
                    error_sample[f'MCE_{key}'] = compute_mce(pred_vals, true_vals)
            for key in ['C-N-CA-C', 'O=C-N-CA']:
                pred_vals = pred_geo['torsion_angles'].get(key, [])
                true_vals = ref_geo['torsion_angles'].get(key, [])
                if len(pred_vals) == len(true_vals) and len(pred_vals) > 0:
                    error_sample[f'MCE_{key}'] = compute_mce(pred_vals, true_vals)

            if error_sample:
                sample_errors[cdr_name].append(error_sample)

    # 计算每个 CDR 的全局 JSD
    jsd_results = {}
    for cdr_name in args.cdr:
        jsd_results[cdr_name] = {}
        for gtype in ['bond_lengths', 'bond_angles', 'torsion_angles']:
            for key in list(all_geoms[cdr_name][gtype].keys()):
                if key.endswith("_true"):
                    continue
                true_key = key + "_true"
                if true_key in all_geoms[cdr_name][gtype]:
                    pred_vals = all_geoms[cdr_name][gtype][key]
                    true_vals = all_geoms[cdr_name][gtype][true_key]
                    if len(pred_vals) > 0 and len(true_vals) > 0:
                        is_angular = (gtype == 'torsion_angles')
                        jsd = compute_jsd(pred_vals, true_vals, n_bins=args.n_bins, is_angular=is_angular)
                        jsd_results[cdr_name][f'JSD_{key}'] = jsd

    # 计算每个 CDR 的平均 MAE/MCE
    mae_mce_avg = {}
    for cdr_name in args.cdr:
        summary = defaultdict(list)
        for err in sample_errors[cdr_name]:
            for k, v in err.items():
                summary[k].append(v)
        mae_mce_avg[cdr_name] = {k: np.mean(v) for k, v in summary.items() if v}

    # 最终结果
    results = {
        "JSD": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in jsd_results.items()},
        "MAE_MCE_average": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in mae_mce_avg.items()},
        "num_samples_per_CDR": {cdr: len(sample_errors[cdr]) for cdr in args.cdr}
    }

    # 终端输出
    print_results(results)

    # 保存到 JSON
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n详细结果已保存至 {args.output}")


if __name__ == "__main__":
    main()