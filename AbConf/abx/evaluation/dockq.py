#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re

from abx.common.pdb_utils import AgAbComplex, Protein
from abx.evaluation.configs import DOCKQ_DIR, CACHE_DIR

import logging
import tempfile
from Bio.PDB import PDBParser, PDBIO

from time import time

def get_time_sign(prefix='', suffix='') -> str:
    time_note = time()
    unique_id = round(round(time_note - round(time_note), 3) * 1000)
    unique_id = unique_id if unique_id > 0 else -unique_id
    return prefix + str(unique_id) + suffix

def save_merged_pdb_final_text_based(
    complex_obj,
    output_pdb_path: str,
    antibody_chain_ids=None,
    antigen_chain_ids=None,
):
    """
    辅助函数，通过纯文本处理，将任意数量的抗体链合并为'R'，
    任意数量的抗原链合并为'G'。它能正确处理 AgAbComplex 和 Protein 对象。
    """
    # 1) 如果外部显式给了链集合，直接用（最稳）
    if antibody_chain_ids is not None or antigen_chain_ids is not None:
        antibody_chain_ids = set([c for c in (antibody_chain_ids or []) if c])
        antigen_chain_ids  = set([c for c in (antigen_chain_ids  or []) if c])
    else:
        # 1. 动态识别抗体链和抗原链
        antibody_chain_ids = set()
        antigen_chain_ids = set()
        
        
        # 根据对象类型使用不同的方式获取链ID
        if isinstance(complex_obj, AgAbComplex):
            antibody_chain_ids.add(complex_obj.heavy_chain)
            antibody_chain_ids.add(complex_obj.light_chain)
            try:
                for cid in complex_obj.antigen.peptides.keys():
                    antigen_chain_ids.add(cid)
            except AttributeError:
                logging.warning(f"Could not find .antigen.peptides for {complex_obj.get_id()}")
        elif isinstance(complex_obj, Protein):
            # 对于在cdrh3_only模式下创建的Protein对象，我们需要推断
            # 假设文件名中的最后一个 '_' 分隔的部分是抗原链
            try:
                pdb_id = complex_obj.get_id()
                parts = pdb_id.split('_')
                ag_chains_from_id = list(parts[-1].replace('(filename)', ''))
                
                # 遍历Protein对象中实际存在的链
                for chain_id in complex_obj.peptides.keys():
                    if chain_id in ag_chains_from_id:
                        antigen_chain_ids.add(chain_id)
                    else:
                        antibody_chain_ids.add(chain_id)
            except IndexError:
                raise ValueError(f"Could not determine antigen/antibody chains from Protein object with ID: {complex_obj.get_id()}")

    # 2. 创建链映射
    chain_map = {}
    for cid in antibody_chain_ids:
        if cid: chain_map[cid] = 'R'
    for cid in antigen_chain_ids:
        if cid: chain_map[cid] = 'G'
    
    if 'R' not in chain_map.values() or 'G' not in chain_map.values():
         logging.warning(f"Complex {complex_obj.get_id()} does not appear to contain both antibody ('R') and antigen ('G') components. Map: {chain_map}")

    # 3. 将复合物写入临时文件，然后逐行读写以重命名
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False, dir=CACHE_DIR) as tmp_f:
        complex_obj.to_pdb(tmp_f.name)
        tmp_pdb_path = tmp_f.name

    try:
        with open(tmp_pdb_path, 'r') as infile, open(output_pdb_path, 'w') as outfile:
            for line in infile:
                if line.startswith(('ATOM', 'HETATM')):
                    original_chain_id = line[21]
                    if original_chain_id in chain_map:
                        new_line = line[:21] + chain_map[original_chain_id] + line[22:]
                        outfile.write(new_line)
                    # 如果链不在map中，我们选择丢弃它，以确保PDB只包含R和G
                else:
                    outfile.write(line)
    finally:
        if os.path.exists(tmp_pdb_path):
            os.remove(tmp_pdb_path)


def dockq(mod_cplx: AgAbComplex, ref_cplx: AgAbComplex, cdrh3_only: bool = False):
    """
    计算DockQ分数。采用统一的、健壮的“合并链”策略，并能正确处理两种模式。
    """
    processed_mod = mod_cplx
    processed_ref = ref_cplx
    L = getattr(ref_cplx, "light_chain", None)
    H = ref_cplx.heavy_chain
    # 抗原链来自 antigen.peptides.keys()
    try:
        ag_chain_ids = set(ref_cplx.antigen.peptides.keys())
    except Exception:
        ag_chain_ids = set()
    
    if cdrh3_only:
        ab_chain_ids = {H}  # 只有 H3 截断出来的那条链
    else:
        ab_chain_ids = set([c for c in [H, L] if c])
        
    if cdrh3_only:
        logging.info("Preparing truncated complexes for CDR-H3 only mode.")
        
        mod_cdr3 = mod_cplx.get_cdr('H3')
        ref_cdr3 = ref_cplx.get_cdr('H3')
        
        if mod_cdr3 is None or ref_cdr3 is None:
            missing = "model" if mod_cdr3 is None else "reference"
            raise ValueError(f"Cannot run in cdrh3_only mode: CDR-H3 not found in {missing} complex for {ref_cplx.get_id()}.")

        # 创建新的peptides字典，只包含抗原和H3
        mod_peptides = {**mod_cplx.antigen.peptides, H: mod_cdr3}
        ref_peptides = {**ref_cplx.antigen.peptides, H: ref_cdr3}
        
        # 创建新的、被截断的Protein对象
        processed_mod = Protein(mod_cplx.get_id(), mod_peptides)
        processed_ref = Protein(ref_cplx.get_id(), ref_peptides)

    # 生成临时文件名时建议做个净化，避免 (filename) 这种字符污染路径
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(processed_ref.get_id()))
    prefix = get_time_sign(suffix=safe_id)
    mod_pdb_merged = os.path.join(CACHE_DIR, f"{prefix}_mod_merged.pdb")
    ref_pdb_merged = os.path.join(CACHE_DIR, f"{prefix}_ref_merged.pdb")

    try:
        # 统一调用最终的、基于文本的合并保存函数
        # ✅ 关键：显式传入链集合，别再从 get_id() 猜
        save_merged_pdb_final_text_based(processed_mod, mod_pdb_merged,
                                         antibody_chain_ids=ab_chain_ids,
                                         antigen_chain_ids=ag_chain_ids)
        save_merged_pdb_final_text_based(processed_ref, ref_pdb_merged,
                                         antibody_chain_ids=ab_chain_ids,
                                         antigen_chain_ids=ag_chain_ids)

        dockq_script = os.path.join(DOCKQ_DIR, "DockQ.py")
        mapping_str = "RG:RG"
        command = f'python3 "{dockq_script}" "{mod_pdb_merged}" "{ref_pdb_merged}" --mapping {mapping_str} --no_align'
        
        logging.info(f"Executing DockQ command: {command}")
        p = os.popen(command)
        text = p.read()
        p.close()

        if "could not find interfaces" in text.lower():
            logging.warning(f"DockQ found no interfaces for {processed_ref.get_id()}. Returning score 0.")
            return 0.0

        if "error" in text.lower() and "unrecognized arguments" not in text.lower():
             # 忽略旧的unrecognized arguments错误，因为它可能只是警告
            raise RuntimeError(f"DockQ execution failed. Output:\n{text}")

        pattern = r"(GlobalDockQ|DockQ)\s*:?\s+([0-1]\.[0-9]+)"
        res = re.search(pattern, text)

        if not res:
            logging.error(f"Could not parse DockQ score for {processed_ref.get_id()}. Full output:\n---\n{text}\n---")
            return 0.0
            
        score = float(res.group(2))
        return score
    except Exception as e:
        # 重新抛出异常，让上层调用者知道发生了错误并可以进行处理
        raise e

    finally:
        if os.path.exists(mod_pdb_merged):
            os.remove(mod_pdb_merged)
        if os.path.exists(ref_pdb_merged):
            os.remove(ref_pdb_merged)
            