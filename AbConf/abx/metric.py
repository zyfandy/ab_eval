import os
import argparse
import functools
import multiprocessing as mp
import logging
import re
import pandas as pd
import traceback
from tqdm import tqdm
from collections import OrderedDict
import numpy as np

# BioPython 相关模块：序列处理、结构解析、比对
from Bio.SeqUtils import seq1
from Bio.PDB.PDBParser import PDBParser
from Bio import pairwise2
from Bio.Align import substitution_matrices

# 抗体评估工具包中的自定义模块
from abx.common.ab_utils import calc_ab_metrics
from abx.preprocess.numbering import renumber_ab_seq, get_ab_regions
from abx.common.pdb_utils import AgAbComplex
from abx.evaluation.tm_score import tm_score
from abx.evaluation.lddt import lddt
from abx.evaluation.dockq import dockq
from abx.evaluation.configs import CONTACT_DIST, NUM_INTERFACE_RESIDUES

# ---------- 能量计算相关（PyRosetta 懒加载）----------
_PYROSETTA_READY = False   # 全局标志，避免重复初始化

def _chain_seq_from_model(model, chain_id: str) -> str:
    """
    从 Bio.PDB 模型中提取指定链的氨基酸序列（一字母码）。
    如果链不存在或没有残基，返回空字符串。
    """
    try:
        residues = list(model[chain_id].get_residues())
    except KeyError:
        return ""
    aa = []
    for r in residues:
        try:
            aa.append(seq1(r.get_resname()))
        except Exception:
            continue
    return "".join(aa)

def _autodetect_heavy_light(model):
    """
    自动识别模型中的重链和轻链。
    使用 ANARCI 方案（通过 renumber_ab_seq 检测）来确定链类型。
    返回 (重链ID, 轻链ID, 所有链ID列表)
    """
    chain_ids = [c.id for c in model.get_chains()]
    heavy_cands, light_cands = [], []
    for cid in chain_ids:
        seq = _chain_seq_from_model(model, cid)
        if not seq:
            continue
        # 尝试将序列识别为重链
        try:
            h = renumber_ab_seq(seq, allow=['H'], scheme='imgt')
            if h.get('domain_numbering') is not None:
                heavy_cands.append((cid, len(seq)))
        except Exception:
            pass
        # 尝试识别为轻链（Kappa/Lambda）
        try:
            l = renumber_ab_seq(seq, allow=['K', 'L'], scheme='imgt')
            if l.get('domain_numbering') is not None:
                light_cands.append((cid, len(seq)))
        except Exception:
            pass

    # 如果识别出多条链，选择序列最长的作为最可能的重链/轻链
    heavy = max(heavy_cands, key=lambda x: x[1])[0] if heavy_cands else None
    light = max(light_cands, key=lambda x: x[1])[0] if light_cands else None
    return heavy, light, chain_ids

def _infer_hl_from_name_or_path(pdb_file: str, model=None):
    """
    从文件名或目录路径中推断重链和轻链的标识符。
    尝试多种规则：
      1. 从预测文件名（如 1a3r_A_B.pdb）解析第2、3部分
      2. 从上级 job 目录名（diffab 格式）解析
      3. 如果提供了 model，调用自动检测函数
    返回 (重链ID, 轻链ID) 可能为 None
    """
    base = os.path.splitext(os.path.basename(pdb_file))[0].split('@')[0]
    if base.endswith('_relaxed'):
        base = base[:-len('_relaxed')]
    parts = base.split('_')
    if len(parts) >= 3:
        return parts[1], parts[2]

    # 尝试从 job 目录名解析（diffab 风格）
    jobdir = os.path.basename(os.path.dirname(os.path.dirname(pdb_file)))
    jparts = jobdir.split('_')
    year_idx = None
    for i, p in enumerate(jparts):
        if re.fullmatch(r'\d{4}', p):
            year_idx = i
            break
    if year_idx is not None and year_idx >= 4:
        chain_tokens = jparts[2:year_idx]
        if len(chain_tokens) >= 2:
            return chain_tokens[0], chain_tokens[1]

    # 最后尝试用自动检测
    if model is not None:
        h, l, _ = _autodetect_heavy_light(model)
        return h, l

    return None, None

def _ensure_pyrosetta_ready(mute_banner: bool = True):
    """
    确保 PyRosetta 已经初始化（只执行一次）。
    mute_banner=True 时屏蔽 PyRosetta 启动时的大量输出。
    """
    global _PYROSETTA_READY
    if _PYROSETTA_READY:
        return
    if mute_banner:
        import contextlib, io
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            import pyrosetta
            pyrosetta.init(
                "-use_input_sc -input_ab_scheme AHo_Scheme -ignore_unrecognized_res "
                "-ignore_zero_occupancy false -load_PDB_components true "
                "-relax:default_repeats 2 -no_fconfig -mute all"
            )
    else:
        import pyrosetta
        pyrosetta.init(
            "-use_input_sc -input_ab_scheme AHo_Scheme -ignore_unrecognized_res "
            "-ignore_zero_occupancy false -load_PDB_components true "
            "-relax:default_repeats 2 -no_fconfig"
        )
    _PYROSETTA_READY = True

def pyrosetta_interface_energy(pdb_path, interface):
    """
    使用 PyRosetta 的 InterfaceAnalyzerMover 计算给定接口的结合自由能 (dG_separated)。
    """
    _ensure_pyrosetta_ready(mute_banner=True)
    import pyrosetta
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta import create_score_function
    pose = pyrosetta.pose_from_pdb(pdb_path)
    mover = InterfaceAnalyzerMover()
    mover.set_interface(interface)
    mover.set_scorefunction(create_score_function('ref2015'))
    mover.apply(pose)
    return pose.scores['dG_separated']

def InterfaceEnergy(pdb_file):
    """
    计算抗体-抗原界面结合能 (dG)。
    自动识别重链、轻链和抗原链，构造接口字符串（如 "HL_A"），调用 PyRosetta 计算。
    若失败返回 NaN。
    """
    try:
        parser = PDBParser(QUIET=True)
        model = parser.get_structure("s", pdb_file)[0]
        heavy_chain_id, light_chain_id = _infer_hl_from_name_or_path(pdb_file, model=model)
        if not heavy_chain_id or not light_chain_id:
            raise ValueError(f"Cannot infer heavy/light for {os.path.basename(pdb_file)}")

        all_chain_ids = [c.id for c in model.get_chains()]
        antigen_chain_ids = [c for c in all_chain_ids if c not in {heavy_chain_id, light_chain_id}]
        if not antigen_chain_ids:
            return np.nan

        antibody_chains_str = f"{heavy_chain_id}{light_chain_id}"
        antigen_str = "".join(antigen_chain_ids)
        interface = f"{antibody_chains_str}_{antigen_str}"

        return pyrosetta_interface_energy(pdb_file, interface)
    except Exception as e:
        logging.error(f"Failed to calculate InterfaceEnergy for {pdb_file}: {e}")
        return np.nan

def get_aligned_ab_data(pdb_file: str, scheme: str = 'imgt') -> dict:
    """
    从 PDB 文件中提取抗体 Fv 区（重链+轻链）的对齐数据。
    使用 ANARCI 进行 Chothia/IMGT 编号对齐，返回：
        - coords: Cα 原子坐标 (N×3)
        - str_seq: 对齐后的氨基酸序列（一字母）
        - cdr_def: 每个残基的 CDR 类型标记（5=H3, 3=H1 等）
        - residues: Bio.PDB 残基对象列表（用于后续接触计算）
    如果链识别或 ANARCI 失败，抛出异常。
    """
    parser = PDBParser(QUIET=1)
    model = parser.get_structure('s', pdb_file)[0]
    heavy_id, light_id = _infer_hl_from_name_or_path(pdb_file, model=model)
    if not heavy_id or not light_id:
        raise ValueError(f"Cannot infer heavy/light chains for {os.path.basename(pdb_file)}")

    # 原始链的所有残基
    heavy_residues_full = list(model[heavy_id].get_residues())
    light_residues_full = list(model[light_id].get_residues())

    heavy_str_seq_full = "".join([seq1(r.get_resname()) for r in heavy_residues_full])
    light_str_seq_full = "".join([seq1(r.get_resname()) for r in light_residues_full])

    # ANARCI 编号与对齐
    heavy_anarci_res = renumber_ab_seq(heavy_str_seq_full, allow=['H'], scheme='imgt')
    light_anarci_res = renumber_ab_seq(light_str_seq_full, allow=['K', 'L'], scheme='imgt')

    h_domain_num_raw = heavy_anarci_res.get('domain_numbering')
    l_domain_num_raw = light_anarci_res.get('domain_numbering')

    if h_domain_num_raw is None or l_domain_num_raw is None:
        raise ValueError(f"ANARCI failed to process chains in {os.path.basename(pdb_file)}")

    # 标记有效残基（ANARCI 能分配编号的残基）
    h_valid_mask = [num is not None for num in h_domain_num_raw]
    l_valid_mask = [num is not None for num in l_domain_num_raw]

    # ANARCI 输出的起止索引（相对于原始序列）
    h_start = heavy_anarci_res.get('start', 0)
    h_end = heavy_anarci_res.get('end', len(heavy_residues_full))
    l_start = light_anarci_res.get('start', 0)
    l_end = light_anarci_res.get('end', len(light_residues_full))

    # 截取 Fv 区（ANARCI 建议的范围）
    heavy_residues_fv = heavy_residues_full[h_start:h_end]
    light_residues_fv = light_residues_full[l_start:l_end]

    # 仅保留有效残基（能对齐的部分）
    heavy_residues_aligned = [res for res, is_valid in zip(heavy_residues_fv, h_valid_mask) if is_valid]
    light_residues_aligned = [res for res, is_valid in zip(light_residues_fv, l_valid_mask) if is_valid]

    all_residues_aligned = heavy_residues_aligned + light_residues_aligned
    h_domain_num_aligned = [num for num in h_domain_num_raw if num is not None]
    l_domain_num_aligned = [num for num in l_domain_num_raw if num is not None]

    aligned_str_seq = "".join([seq1(r.get_resname()) for r in all_residues_aligned])
    aligned_coords = np.zeros((len(all_residues_aligned), 3))
    for i, r in enumerate(all_residues_aligned):
        aligned_coords[i] = r['CA'].get_coord()

    # 根据对齐后的编号确定 CDR 定义
    heavy_cdr_def_aligned = get_ab_regions(h_domain_num_aligned, chain_id='H')
    light_cdr_def_aligned = get_ab_regions(l_domain_num_aligned, chain_id='L')
    aligned_cdr_def = np.concatenate([heavy_cdr_def_aligned, light_cdr_def_aligned], axis=0)

    # 数据一致性检查
    if not (aligned_coords.shape[0] == len(aligned_str_seq) == aligned_cdr_def.shape[0]):
        raise AssertionError("Internal logic error in get_aligned_ab_data")

    return {
        'coords': aligned_coords,
        'str_seq': aligned_str_seq,
        'cdr_def': aligned_cdr_def,
        'residues': all_residues_aligned
    }

def eval_metric(pred_file: str, ref_file: str, args: argparse.Namespace, meta: dict = None):
    """
    主评估函数：计算预测结构与参考结构之间的多种指标。
    包括：
        - 基于抗体 Fv 区的 RMSD、序列一致率、CDR RMSD（通过 calc_ab_metrics）
        - TM-score、LDDT（全抗体结构比对）
        - DockQ（抗体-抗原对接质量）
        - 可选：界面结合能 dG (--energy)
    参数:
        pred_file: 预测 PDB 路径
        ref_file:  参考 PDB 路径
        args:      命令行参数（包含 H3、energy 等标志）
        meta:      可选的元数据，如 heavy/light 链标识符（用于 diffab 模式）
    返回:
        包含所有计算指标的字典；若出现严重错误返回 None
    """
    pdb_name = os.path.splitext(os.path.basename(pred_file))[0].split('@')[0]
    base_result = {'code': pdb_name, 'file_path': pred_file}

    try:
        # 获取预测和参考的抗体 Fv 对齐数据
        pred_data = get_aligned_ab_data(pred_file)
        ref_data = get_aligned_ab_data(ref_file)

        gt_ab_ca = ref_data['coords']
        gt_ab_str_seq = ref_data['str_seq']
        cdr_def_ref = ref_data['cdr_def']
        ref_residues = ref_data['residues']

        pred_ab_ca = pred_data['coords']
        pred_ab_str_seq = pred_data['str_seq']

        # ---------- 获取参考结构中的抗体链和抗原链 ----------
        parser = PDBParser(QUIET=1)
        ref_model = parser.get_structure('ref', ref_file)[0]

        if meta and "heavy" in meta and "light" in meta:
            # diffab 模式通过 meta 直接提供链 ID
            ref_H_chain = meta["heavy"]
            ref_L_chain = meta["light"]
            ag_chain_ids = list(meta.get("antigen", ""))
        else:
            # legacy 模式：从文件名或自动检测推断
            ref_H_chain, ref_L_chain = _infer_hl_from_name_or_path(ref_file, model=ref_model)
            all_ref_chains = [c.id for c in ref_model.get_chains()]
            ag_chain_ids = [c for c in all_ref_chains if c not in {ref_H_chain, ref_L_chain}]

        # ---------- 构建接触掩码（用于 Interface RMSD 等）----------
        # 收集抗原残基的所有重原子坐标
        ag_residues = []
        for ag_id in ag_chain_ids:
            if ag_id in ref_model:
                for res in ref_model[ag_id].get_residues():
                    if res.id[0] == ' ':  # 跳过异质原子（如水）
                        heavy_atoms = [atom.get_coord() for atom in res.get_atoms() if atom.element != 'H']
                        if heavy_atoms:
                            ag_residues.append(np.array(heavy_atoms))

        contact_mask = np.zeros(len(ref_residues), dtype=bool)

        if len(ag_residues) > 0:
            # 确定 CDR-H3 残基的 Cα 坐标
            h3_indices = np.where(cdr_def_ref == 5)[0]
            h3_heavy_atoms = []
            for idx in h3_indices:
                h3_heavy_atoms.extend([atom.get_coord() for atom in ref_residues[idx].get_atoms() if atom.element != 'H'])

            if len(h3_heavy_atoms) > 0:
                h3_heavy_atoms = np.array(h3_heavy_atoms)
                # 计算每个抗原残基到 H3 的最小距离
                ag_to_h3_dists = []
                for ag_res_atoms in ag_residues:
                    dist_mat = np.linalg.norm(ag_res_atoms[:, None, :] - h3_heavy_atoms[None, :, :], axis=-1)
                    ag_to_h3_dists.append(np.min(dist_mat))
                ag_to_h3_dists = np.array(ag_to_h3_dists)

                # 选取距离 H3 最近的若干个抗原残基作为表位（epitope）
                topk = min(len(ag_to_h3_dists), NUM_INTERFACE_RESIDUES)
                epitope_indices = np.argpartition(ag_to_h3_dists, topk - 1)[:topk]
                epitope_residues = [ag_residues[i] for i in epitope_indices]

                # 判定每个抗体残基是否与表位残基有原子接触（距离 < CONTACT_DIST）
                for i, ab_res in enumerate(ref_residues):
                    ab_res_atoms = np.array([atom.get_coord() for atom in ab_res.get_atoms() if atom.element != 'H'])
                    if len(ab_res_atoms) == 0:
                        continue

                    is_contact = False
                    for ep_res_atoms in epitope_residues:
                        dist_mat = np.linalg.norm(ab_res_atoms[:, None, :] - ep_res_atoms[None, :, :], axis=-1)
                        if np.min(dist_mat) < CONTACT_DIST:
                            is_contact = True
                            break
                    contact_mask[i] = is_contact

        # ---------- 处理序列长度不一致（通过全局序列比对）----------
        if len(gt_ab_str_seq) != len(pred_ab_str_seq):
            alignments = pairwise2.align.globalds(
                gt_ab_str_seq, pred_ab_str_seq,
                substitution_matrices.load("BLOSUM62"), -10, -0.5
            )
            if not alignments:
                raise ValueError("Pairwise alignment failed.")

            best_aln = alignments[0]
            aligned_gt_seq, aligned_pred_seq, _, _, _ = best_aln

            # 记录对齐后保留的残基索引
            aligned_gt_indices = []
            aligned_pred_indices = []
            gt_idx, pred_idx = 0, 0
            for gt_char, pred_char in zip(aligned_gt_seq, aligned_pred_seq):
                if gt_char != '-' and pred_char != '-':
                    aligned_gt_indices.append(gt_idx)
                    aligned_pred_indices.append(pred_idx)
                if gt_char != '-':
                    gt_idx += 1
                if pred_char != '-':
                    pred_idx += 1

            if not aligned_gt_indices:
                raise ValueError("Alignment resulted in no common residues.")

            # 根据对齐索引选取对应坐标、序列、CDR 定义和接触掩码
            gt_ab_ca_aligned = gt_ab_ca[aligned_gt_indices]
            pred_ab_ca_aligned = pred_ab_ca[aligned_pred_indices]
            cdr_def_aligned = cdr_def_ref[aligned_gt_indices]
            gt_ab_str_seq_aligned = "".join(np.array(list(gt_ab_str_seq))[aligned_gt_indices])
            pred_ab_str_seq_aligned = "".join(np.array(list(pred_ab_str_seq))[aligned_pred_indices])
            contact_mask_aligned = contact_mask[aligned_gt_indices]
        else:
            # 长度相同，直接使用
            gt_ab_ca_aligned, pred_ab_ca_aligned = gt_ab_ca, pred_ab_ca
            cdr_def_aligned = cdr_def_ref
            gt_ab_str_seq_aligned, pred_ab_str_seq_aligned = gt_ab_str_seq, pred_ab_str_seq
            contact_mask_aligned = contact_mask

        # ---------- 计算抗体结构指标（RMSD、序列一致性、CDR 指标等）----------
        ab_metrics = calc_ab_metrics(
            gt_ab_ca_aligned,
            pred_ab_ca_aligned,
            cdr_def_aligned,
            gt_ab_str_seq_aligned,
            pred_ab_str_seq_aligned,
            contact_mask_aligned
        )
        base_result.update(ab_metrics)

        # ---------- 计算全局指标：TM-score、LDDT、DockQ ----------
        try:
            parser_mod = PDBParser(QUIET=1)
            mod_model_temp = parser_mod.get_structure('mod', pred_file)[0]
            mod_chains = [c.id for c in mod_model_temp.get_chains()]

            if not mod_chains:
                raise ValueError("Prediction PDB is empty")

            # 推断预测结构中的重链、轻链和抗原链 ID（优先使用参考链 ID，若不存在则合理猜测）
            mod_H_chain = ref_H_chain if ref_H_chain in mod_chains else ('H' if 'H' in mod_chains else mod_chains[0])
            mod_L_chain = ref_L_chain if ref_L_chain in mod_chains else ('L' if 'L' in mod_chains else (mod_chains[1] if len(mod_chains) > 1 else mod_chains[-1]))
            mod_A_chains = [c for c in ag_chain_ids if c in mod_chains]

            # 构建 AgAbComplex 对象（抗体-抗原复合物）
            mod_cplx = AgAbComplex.from_pdb(pred_file, mod_H_chain, mod_L_chain, mod_A_chains, skip_epitope_cal=True)
            ref_cplx = AgAbComplex.from_pdb(ref_file, ref_H_chain, ref_L_chain, ag_chain_ids, skip_epitope_cal=False)

            base_result['TMscore'] = tm_score(mod_cplx.antibody, ref_cplx.antibody)
            lddt_val, _ = lddt(mod_cplx.antibody, ref_cplx.antibody)
            base_result['LDDT'] = lddt_val
            base_result['DockQ'] = dockq(mod_cplx, ref_cplx, cdrh3_only=getattr(args, 'H3', False))

        except Exception as e:
            logging.warning(f"Failed to calculate macroscopic metrics for {pdb_name}: {e}")

        # ---------- 可选：能量计算 ----------
        if args.energy:
            pred_dG = InterfaceEnergy(pred_file)
            ref_dG = InterfaceEnergy(ref_file)
            base_result.update({
                'dG_gen': pred_dG,
                'dG_ref': ref_dG,
                'ddG': pred_dG - ref_dG if not (np.isnan(pred_dG) or np.isnan(ref_dG)) else np.nan
            })

        return base_result

    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {os.path.basename(pred_file)}. "
                      f"Skipping. Error: {traceback.format_exc()}")
        return None


# =================================================================
# 任务列表扫描与构建（支持 legacy / diffab 两种目录结构）
# =================================================================
def parse_list(data_dir: str, include_relaxed: bool) -> list:
    """
    递归扫描 data_dir 下的所有 PDB 文件，排除 reference 子目录。
    根据 include_relaxed 筛选是否包含 '_relaxed.pdb' 文件。
    返回 PDB 文件路径列表。
    """
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

def _pick_best_regions(region_names):
    """
    从 region 名称列表中选择最优版本（例如 H_CDR3-O2 优于 H_CDR3）。
    规则：同一基础名称（去掉 -O数字）下，取数字最大的版本。
    """
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

def build_tasks_legacy(data_dir: str, include_relaxed: bool):
    """
    legacy 模式：参考结构放在 reference/ 子目录下，预测文件散落在其他子目录。
    预测文件名基础名（去除 _relaxed 和 @ 后缀）与参考文件同名时配对。
    返回 (预测路径, 参考路径, 元数据) 列表，元数据为空字典。
    """
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
        else:
            logging.warning(f"No reference file found for prediction: {os.path.basename(pred_file)}")
    return tasks

def build_tasks_diffab(data_dir: str, include_relaxed: bool):
    """
    diffab 模式：目录结构为 job/region/sample.pdb，每个 region 下有 REF*.pdb 作为参考。
    自动挑选每个 region 的最优版本（如 H_CDR3-O2 优于 H_CDR3）。
    返回 (预测路径, 参考路径, 元数据) 列表，元数据包含 heavy/light/antigen/target 等信息。
    """
    tasks = []
    data_dir = os.path.abspath(data_dir)

    # job 目录名正则：例如 0001_1a3r_A_B_2000_01_01__00_00_00
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

        # 收集该 job 下所有 region 目录
        region_names = [rn for rn in os.listdir(job_path) if os.path.isdir(os.path.join(job_path, rn)) and region_dir_re.match(rn)]
        region_names = _pick_best_regions(region_names)

        for region_name in region_names:
            region_path = os.path.join(job_path, region_name)
            ref_candidates = [f for f in sorted(os.listdir(region_path)) if ref_re.match(f)]
            if not ref_candidates:
                continue
            # 优先使用 REF1.pdb，否则使用第一个匹配的参考文件
            ref_file = os.path.join(region_path, "REF1.pdb" if "REF1.pdb" in ref_candidates else ref_candidates[0])

            for fn in sorted(os.listdir(region_path)):
                if not sample_re.match(fn):
                    continue
                pred_file = os.path.join(region_path, fn)
                sample_id = fn.split('_')[0].split('.')[0]
                meta = {
                    "mode": "diffab", "job": job_name, "region": region_name, "sample_id": sample_id,
                    "target": meta_job["target"], "heavy": meta_job["heavy"], "light": meta_job["light"], "antigen": meta_job["antigen"],
                }
                tasks.append((pred_file, ref_file, meta))
    return tasks

def eval_metric_with_meta(pred_file: str, ref_file: str, meta: dict, args: argparse.Namespace):
    """
    封装 eval_metric，将元数据合并到结果字典中。
    用于多进程映射。
    """
    r = eval_metric(pred_file, ref_file, args, meta=meta)
    if r is None:
        return None
    if meta:
        r.update(meta)
    return r

def main(args):
    """
    主函数：扫描任务、多进程评估、汇总结果并输出 CSV。
    """
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s [%(processName)s] %(levelname)s - %(message)s')
    try:
        logging.info(f"[mode={args.mode}] Scanning for evaluation tasks in '{args.data_dir}'...")
        # 根据路径是否以 _relaxed 结尾自动决定 include_relaxed
        include_relaxed = args.data_dir.rstrip('/').endswith('_relaxed')

        if args.mode == "legacy":
            tasks = build_tasks_legacy(args.data_dir, include_relaxed=include_relaxed)
        else:
            tasks = build_tasks_diffab(args.data_dir, include_relaxed=include_relaxed)

        if not tasks:
            logging.info("No valid prediction-reference pairs found to process.")
            return

        logging.info(f"Found {len(tasks)} tasks. Starting evaluation with {args.cpus} CPUs...")
        func = functools.partial(eval_metric_with_meta, args=args)

        # 多进程执行，tqdm 显示进度条
        with mp.Pool(processes=args.cpus) as pool:
            all_results = list(tqdm(pool.starmap(func, tasks), total=len(tasks)))

        results = [r for r in all_results if r is not None]
        if not results:
            logging.warning("No files were processed successfully.")
            return

        logging.info("Evaluation complete. Aggregating results...")
        df = pd.DataFrame(results)

        # 计算并打印所有数值型指标的平均值
        avg_metrics = [col for col in df.columns if col not in ['code', 'file_path']]
        if avg_metrics:
            print("\n" + "-" * 21)
            print("Average Results for each Metric")
            print("-" * 21)
            print(df[avg_metrics].mean(numeric_only=True).to_string(float_format=lambda x: f"{x:.6f}"))

        # 保存完整结果到 CSV
        output_path = os.path.join(args.data_dir, 'evaluation_results.csv')
        df.to_csv(output_path, index=False, float_format='%.4f')
        logging.info(f"Full results saved to {output_path}")

    except Exception as e:
        logging.error(f"A critical error occurred in the main pipeline: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    # 设置多进程启动方式为 spawn（避免 fork 导致的锁问题）
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_dir', type=str, required=True, help="数据根目录")
    parser.add_argument('-c', '--cpus', type=int, default=12, help="并行进程数")
    parser.add_argument('-e', '--energy', action='store_true', help="启用界面结合能计算（需要 PyRosetta）")
    parser.add_argument('-v', '--verbose', action='store_true', help="输出详细日志")
    parser.add_argument('--H3', action='store_true', help="DockQ 仅计算 CDR-H3 环")
    parser.add_argument('--mode', type=str, choices=['legacy', 'diffab'], default='legacy',
                        help="目录结构模式：legacy (旧格式) 或 diffab (新格式)")
    args = parser.parse_args()

    # 确保 root logger 至少有一个 StreamHandler
    if not any(isinstance(h, logging.StreamHandler) for h in logging.getLogger().handlers):
        logging.getLogger().addHandler(logging.StreamHandler())

    main(args)