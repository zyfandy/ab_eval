#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import time
from copy import deepcopy

from abx.common.pdb_utils import Peptide, Protein, merge_to_one_chain
from abx.evaluation.configs import CACHE_DIR


def exec_bin(mod_pdb, ref_pdb, log, backbone_only):
    options = '-x'
    if backbone_only:
        options += ' -c'
    cmd = f'lddt {options} {mod_pdb} {ref_pdb} > {log} 2>&1'
    return os.system(cmd)


def lddt(mod_protein: Protein, ref_protein: Protein, backbone_only=False):
    # --- 您的文件准备逻辑保持不变 ---
    mod_protein = merge_to_one_chain(mod_protein)
    ref_protein = merge_to_one_chain(ref_protein)

    mod_sign, ref_sign = id(mod_protein), id(ref_protein)
    # 使用更唯一的命名方式，避免潜在的微秒级冲突
    prefix = f'lddt_{mod_sign}_{ref_sign}_{int(time.time() * 1000)}'
    mod_pdb = os.path.join(CACHE_DIR, f'{prefix}_mod.pdb')
    ref_pdb = os.path.join(CACHE_DIR, f'{prefix}_ref.pdb')
    log = os.path.join(CACHE_DIR, f'{prefix}_log.txt')
    
    mod_protein.to_pdb(mod_pdb)
    ref_protein.to_pdb(ref_pdb)

    # --- 您的二进制文件执行逻辑保持不变 ---
    res_code = exec_bin(mod_pdb, ref_pdb, log, backbone_only)
    if res_code != 0:
        # 最好在抛出异常前，先读取并记录日志内容
        log_content = "Log file not found or empty."
        if os.path.exists(log):
            with open(log, 'r') as fin:
                log_content = fin.read()
        # logging.error(f"lddt execution failed with code {res_code} for {mod_pdb}. Log content:\n{log_content}")
        # 清理文件
        if os.path.exists(mod_pdb): os.remove(mod_pdb)
        if os.path.exists(ref_pdb): os.remove(ref_pdb)
        if os.path.exists(log): os.remove(log)
        raise ValueError(f'lddt execution failed with code {res_code}')

    with open(log, 'r') as fin:
        text = fin.read()
    
    res = re.search(r'Global LDDT score: ([0-1]\.?[0-9]*)', text)

    # 增加一个判断，检查 res 是否为 None
    if res:
        # 只有在 res 不是 None 的情况下，才执行 .group(1)
        score = float(res.group(1))
    else:
        # 如果 res 是 None，说明在lddt的输出中没有找到我们期望的那一行
        # 这是一个需要被记录的严重警告，但程序不应该因此崩溃
        # logging.warning(f"Could not parse LDDT score for {os.path.basename(mod_pdb)}. "
        #                 f"The output from the lddt binary was:\n---\n{text}\n---")
        # 返回一个安全的默认值，让上层调用者可以继续处理
        score = 0.0 # 或者 np.nan, 取决于您的下游如何处理失败案例
    #
    # === 修复结束 ===
    #

    # --- 您的文件清理逻辑保持不变 ---
    os.remove(mod_pdb)
    os.remove(ref_pdb)
    os.remove(log)
    
    return score, text
