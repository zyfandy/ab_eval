# cacl_energy_ddg.py (支持 relaxed / raw；output 默认写回 pdb_dir)

import os
import re
import argparse
import multiprocessing as mp
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import pyrosetta
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover


import hashlib
_WORKER_INIT = False
_SCORE_FXN = None

def _stable_seed_from_str(s: str, base: int = 20250101) -> int:
    """Stable 32-bit seed from string (reproducible across runs/machines)."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) ^ base) & 0x7fffffff

def _set_rosetta_seed(seed: int):
    """
    Re-seed Rosetta RNG per task to make packer deterministic even in multiprocessing.
    """
    try:
        pyrosetta.rosetta.basic.random.init_random_generators(seed, "mt19937")
    except Exception as e:
        logging.warning(f"[Seed] init_random_generators failed: {type(e).__name__}: {e}")

def _init_pyrosetta_worker(init_opts: str, base_seed: int):
    """
    Called once per spawned worker process.
    """
    global _WORKER_INIT, _SCORE_FXN
    if _WORKER_INIT:
        return
    pyrosetta.init(init_opts)
    _SCORE_FXN = pyrosetta.create_score_function("ref2015")
    _WORKER_INIT = True


# --- 1. 初始化 PyRosetta ---
def _pyrosetta_interface_energy(pdb_path: str, interface_override: Optional[str] = None) -> float:
    try:
        interface = interface_override or _parse_interface_from_filename(pdb_path)
        if interface is None:
            logging.warning(f"[Skip] Cannot determine interface for: {os.path.basename(pdb_path)}")
            return np.nan

        # ===== [ADD] per-task deterministic seed =====
        # Use pdb_path as identity; if you want gen/ref share same seed, use basename only.
        seed = _stable_seed_from_str(os.path.abspath(pdb_path))
        _set_rosetta_seed(seed)
        # ===== [ADD END] =====

        pose = pyrosetta.pose_from_pdb(pdb_path)

        mover = InterfaceAnalyzerMover()
        mover.set_interface(interface)

        # ===== [CHANGE] reuse per-worker scorefxn if available =====
        scorefxn = _SCORE_FXN if _SCORE_FXN is not None else pyrosetta.create_score_function("ref2015")
        mover.set_scorefunction(scorefxn)
        # ===== [CHANGE END] =====

        mover.set_pack_separated(True)
        mover.apply(pose)

        return float(pose.scores.get("dG_separated", np.nan))
    except Exception as e:
        logging.warning(f"Energy calculation failed for {os.path.basename(pdb_path)}: {e}")
        return np.nan



# --- 2. 任务封装 ---
@dataclass
class EnergyTask:
    pred_path: str
    ref_path: str
    name: str
    run_folder: str
    variant: str  # "relaxed" or "raw"
    interface: Optional[str] = None  # ✅ diffab 用，从 jobdir 解析得到 "HL_AG"
    scores: Dict[str, Any] = field(default_factory=dict)


def _strip_relaxed_suffix(stem: str) -> str:
    # 统一去掉 relaxed 后缀（兼容 _relaxed / -relaxed 等）
    if stem.endswith("_relaxed"):
        return stem[: -len("_relaxed")]
    return stem


def _parse_interface_from_filename(pdb_path: str) -> Optional[str]:
    """
    从文件名解析 interface: heavy+light _ antigen_ids
    约定：{code}_{H}_{L}_{AG}.pdb 或 {code}_{H}_{L}_{AG}_relaxed.pdb
    """
    stem = os.path.splitext(os.path.basename(pdb_path))[0]
    stem = _strip_relaxed_suffix(stem)
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    # 只取最后三段更稳（防止 code 内含 '_' 的极端情况）
    heavy_id, light_id, antigen_ids = parts[-3], parts[-2], parts[-1]
    antibody = f"{heavy_id}{light_id}"
    return f"{antibody}_{antigen_ids}"


def calculate_ddG(task: EnergyTask) -> EnergyTask:
    dG_gen = _pyrosetta_interface_energy(task.pred_path, interface_override=task.interface)
    dG_ref = _pyrosetta_interface_energy(task.ref_path, interface_override=task.interface)
    ddG = dG_gen - dG_ref if not (np.isnan(dG_gen) or np.isnan(dG_ref)) else np.nan

    task.scores.update({
        "dG_gen": dG_gen,
        "dG_ref": dG_ref,
        "ddG": ddG,
    })
    return task


def _load_name_set(name_idx_path: str) -> set:
    df = pd.read_csv(name_idx_path, names=["name"], header=None)
    return set(df["name"].astype(str).tolist())


def _choose_pred_file(run_folder_path: str, name: str, mode: str) -> List[Tuple[str, str]]:
    """
    返回 [(pred_path, variant), ...]
    mode:
      - "auto": 优先 relaxed，找不到用 raw
      - "relaxed": 只用 relaxed
      - "raw": 只用 raw
      - "both": 两者都算（如果存在）
    """
    relaxed_path = os.path.join(run_folder_path, f"{name}_relaxed.pdb")
    raw_path = os.path.join(run_folder_path, f"{name}.pdb")

    out = []
    if mode == "relaxed":
        if os.path.exists(relaxed_path):
            out.append((relaxed_path, "relaxed"))
        return out

    if mode == "raw":
        if os.path.exists(raw_path):
            out.append((raw_path, "raw"))
        return out

    if mode == "both":
        if os.path.exists(relaxed_path):
            out.append((relaxed_path, "relaxed"))
        if os.path.exists(raw_path):
            out.append((raw_path, "raw"))
        return out

    # auto
    if os.path.exists(relaxed_path):
        out.append((relaxed_path, "relaxed"))
    elif os.path.exists(raw_path):
        out.append((raw_path, "raw"))
    return out

def _load_name_set_lines(name_idx_path: str) -> set:
    """
    更稳的 idx 读取：空行/空文件 -> 返回空 set（diffab 可视为不过滤）
    """
    s = set()
    if not name_idx_path:
        return s
    if not os.path.exists(name_idx_path):
        raise FileNotFoundError(f"name_idx not found: {name_idx_path}")
    with open(name_idx_path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            s.add(t)
    return s

def _parse_region_dir_list(s: Optional[str]) -> Optional[set]:
    """
    "H_CDR3" -> {"H_CDR3"}
    "H_CDR3,H_CDR2" -> {"H_CDR3","H_CDR2"}
    None/"" -> None
    """
    if s is None:
        return None
    items = [x.strip() for x in str(s).split(",") if x.strip()]
    return set(items) if items else None


def _pick_best_regions(region_names):
    """
    同一 base（如 H_CDR3）只保留最大 O-step（H_CDR3-O2 优先于 H_CDR3）
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


def _parse_diffab_jobdir(job_name: str) -> Optional[dict]:
    """
    diffab jobdir 例子：
      0000_3rkd_D_C_B_2026_01_13__18_38_07
    解析出：
      pdbid=3rkd, H=D, L=C, antigen_ids="B" (多 token/多字符时会展开)
    """
    parts = [p for p in job_name.split("_") if p]
    year_idx = None
    for i in range(1, len(parts) - 2):
        if (re.fullmatch(r"20\d{2}", parts[i]) and
            re.fullmatch(r"\d{2}", parts[i + 1]) and
            re.fullmatch(r"\d{2}", parts[i + 2])):
            year_idx = i
            break
    if year_idx is None or year_idx < 4:
        return None

    pdbid = parts[1]
    chain_tokens = parts[2:year_idx]
    if len(chain_tokens) < 2:
        return None

    H = chain_tokens[0]
    L = chain_tokens[1]
    antigen_tokens = chain_tokens[2:]

    # token 逐字符展开：例如 "AF" -> ['A','F']
    A_list = []
    for t in antigen_tokens:
        A_list.extend(list(t))
    antigen_ids = "".join(A_list)

    return {"pdbid": pdbid, "H": H, "L": L, "antigen_ids": antigen_ids}


def _choose_pred_file_diffab(region_path: str, sample_stem: str, mode: str) -> List[Tuple[str, str]]:
    """
    diffab region 下 sample 命名：0000.pdb / 0000_relaxed.pdb
    返回 [(pred_path, variant), ...]
    """
    relaxed_path = os.path.join(region_path, f"{sample_stem}_relaxed.pdb")
    raw_path = os.path.join(region_path, f"{sample_stem}.pdb")

    out = []
    if mode == "relaxed":
        if os.path.exists(relaxed_path):
            out.append((relaxed_path, "relaxed"))
        return out
    if mode == "raw":
        if os.path.exists(raw_path):
            out.append((raw_path, "raw"))
        return out
    if mode == "both":
        if os.path.exists(relaxed_path):
            out.append((relaxed_path, "relaxed"))
        if os.path.exists(raw_path):
            out.append((raw_path, "raw"))
        return out

    # auto
    if os.path.exists(relaxed_path):
        out.append((relaxed_path, "relaxed"))
    elif os.path.exists(raw_path):
        out.append((raw_path, "raw"))
    return out


def prepare_tasks_diffab(pred_root_dir: str, name_idx_path: str, mode: str, diffab_region_dir: Optional[str] = None) -> List[EnergyTask]:
    """
    diffab 目录结构：
      root/
        0000_3rkd_D_C_B_2026_.../
          H_CDR3/ or H_CDR3-O2/
            0000.pdb ... 0099.pdb (或 *_relaxed.pdb)
            REF1.pdb (或 REF2.pdb...)

    - name_idx_path: 可用于过滤；空文件 -> 不过滤（全算）
      支持匹配：job_name / job_index(0000) / pdbid / pdbid_H_L_AG
    """
    tasks: List[EnergyTask] = []
    pred_root_dir = os.path.abspath(pred_root_dir)

    targets = _load_name_set_lines(name_idx_path)
    # 空 idx -> 不过滤
    do_filter = len(targets) > 0

    job_dir_re = re.compile(r"^\d{4}_.+_\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}$")
    region_dir_re = re.compile(r"^[HL]_CDR[123](?:-O\d+)?$")
    ref_re = re.compile(r"^REF\d+\.pdb$", re.IGNORECASE)
    sample_any_re = re.compile(r"^(\d{4})(?:_relaxed)?\.pdb$", re.IGNORECASE)

    all_subdirs = [d for d in os.listdir(pred_root_dir) if os.path.isdir(os.path.join(pred_root_dir, d))]
    job_dirs = sorted([d for d in all_subdirs if job_dir_re.match(d)])
    logging.info(f"[diffab] Found {len(job_dirs)} job folders in '{pred_root_dir}'.")

    for job_name in job_dirs:
        meta = _parse_diffab_jobdir(job_name)
        if meta is None:
            continue

        # 过滤逻辑（尽量宽松）
        job_index = job_name.split("_", 1)[0]
        key_full = f"{meta['pdbid']}_{meta['H']}_{meta['L']}_{meta['antigen_ids']}"
        if do_filter:
            hit = (
                (job_name in targets) or
                (job_index in targets) or
                (meta["pdbid"] in targets) or
                (key_full in targets) or
                any(t.split("/", 1)[0] == job_name for t in targets)  # 允许写 job/region/sample
            )
            if not hit:
                continue

        interface = f"{meta['H']}{meta['L']}_{meta['antigen_ids']}"

        job_path = os.path.join(pred_root_dir, job_name)

        # region dirs + O-step 去重
        region_names = []
        for rn in sorted(os.listdir(job_path)):
            rp = os.path.join(job_path, rn)
            if os.path.isdir(rp) and region_dir_re.match(rn):
                region_names.append(rn)
        # ✅ diffab only: optionally restrict to EXACT region folder names
        # 注意：这里是“目录名精确匹配”，不会做 base-match，也不会做 -O* picking
        region_dir_allow = _parse_region_dir_list(diffab_region_dir)
        if region_dir_allow is not None:
            region_names = [rn for rn in region_names if rn in region_dir_allow]
        else:
            # 不指定时维持你现有行为：对每个 base 取最大 O-step
            region_names = _pick_best_regions(region_names)
        if not region_names:
            continue

        for region_name in region_names:
            region_path = os.path.join(job_path, region_name)

            # reference
            ref_candidates = [f for f in sorted(os.listdir(region_path)) if ref_re.match(f)]
            if not ref_candidates:
                continue
            ref_fn = "REF1.pdb" if "REF1.pdb" in ref_candidates else ref_candidates[0]
            ref_path = os.path.join(region_path, ref_fn)
            if not os.path.exists(ref_path):
                continue

            # sample stems
            stems = set()
            for fn in os.listdir(region_path):
                m = sample_any_re.match(fn)
                if not m:
                    continue
                stems.add(m.group(1))

            for stem in sorted(stems):
                pred_candidates = _choose_pred_file_diffab(region_path, stem, mode)
                for pred_path, variant in pred_candidates:
                    if not os.path.exists(pred_path):
                        continue
                    tasks.append(EnergyTask(
                        pred_path=pred_path,
                        ref_path=ref_path,
                        name=f"{region_name}/{stem}",
                        run_folder=job_name,
                        variant=variant,
                        interface=interface,  # ✅ diffab 的关键：显式传 interface
                    ))

    return tasks


def prepare_tasks(pred_root_dir: str, name_idx_path: str, mode: str) -> List[EnergyTask]:
    """
    pred_root_dir:
      design/
        reference/
          {name}.pdb
        0000/
          {name}.pdb or {name}_relaxed.pdb
        0001/
          ...
    """
    tasks: List[EnergyTask] = []
    pred_root_dir = os.path.abspath(pred_root_dir)

    reference_dir = os.path.join(pred_root_dir, "reference")
    if not os.path.isdir(reference_dir):
        raise FileNotFoundError(f"Reference directory not found: {reference_dir}")

    target_names = _load_name_set(name_idx_path)

    all_subdirs = [d for d in os.listdir(pred_root_dir) if os.path.isdir(os.path.join(pred_root_dir, d))]
    run_folders = sorted([d for d in all_subdirs if d != "reference"])
    logging.info(f"Found {len(run_folders)} run folders in '{pred_root_dir}'.")

    # 加速：先把 reference 里存在的 name 过滤掉，避免无意义匹配
    ref_exist = set()
    for n in target_names:
        if os.path.exists(os.path.join(reference_dir, f"{n}.pdb")):
            ref_exist.add(n)
    if not ref_exist:
        logging.warning("No reference pdb matched any name in idx.")
        return []

    for run_folder in run_folders:
        run_folder_path = os.path.join(pred_root_dir, run_folder)

        # 加速：扫描该目录所有 pdb，推回 name 候选
        # 支持 name.pdb 与 name_relaxed.pdb
        present_names = set()
        for fn in os.listdir(run_folder_path):
            if not fn.endswith(".pdb"):
                continue
            stem = os.path.splitext(fn)[0]
            stem = _strip_relaxed_suffix(stem)
            present_names.add(stem)

        common = (present_names & ref_exist)
        if not common:
            continue

        for name in sorted(common):
            ref_file_path = os.path.join(reference_dir, f"{name}.pdb")
            pred_candidates = _choose_pred_file(run_folder_path, name, mode)

            for pred_path, variant in pred_candidates:
                if os.path.exists(pred_path) and os.path.exists(ref_file_path):
                    tasks.append(EnergyTask(
                        pred_path=pred_path,
                        ref_path=ref_file_path,
                        name=name,
                        run_folder=run_folder,
                        variant=variant,
                    ))

    return tasks


def _summarize_imp(df: pd.DataFrame):
    if "ddG" not in df.columns:
        return
    df_clean = df.dropna(subset=["ddG"])
    if df_clean.empty:
        return

    # 同时按 run_folder 和 variant 分组，避免混在一起误读
    grp = df_clean.groupby(["run_folder", "variant"])
    imp = grp.apply(lambda x: (x["ddG"] < 0).mean() * 100.0).reset_index(name="IMP(%)")

    print("\n--- IMP (%) per Run Folder & Variant ---")
    # 更可读
    for _, row in imp.iterrows():
        print(f"{row['run_folder']}\t{row['variant']}\t{row['IMP(%)']:.2f}")

    overall_imp = (df_clean["ddG"] < 0).mean() * 100.0
    logging.info(f"FINAL_IMP_SCORE (overall, all variants): {overall_imp:.2f}%")


def main(args):
    pdb_dir = os.path.abspath(args.pdb_dir)
    output_dir = os.path.abspath(args.output_dir or args.pdb_dir)  # ✅ 默认 output = input

    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Scanning in '{pdb_dir}' with idx '{args.name_idx}' (mode={args.mode})")
    if args.layout == "legacy":
        tasks = prepare_tasks(pdb_dir, args.name_idx, args.mode)
    else:
        tasks = prepare_tasks_diffab(
            pdb_dir,
            args.name_idx,
            args.mode,
            diffab_region_dir=getattr(args, "diffab_region_dir", None)
        )

    if not tasks:
        logging.info("No valid (prediction, reference) file pairs found.")
        return

    logging.info(f"Found {len(tasks)} tasks. Using {args.cpus} CPUs.")

    if args.cpus <= 1:
        # Single process: init here (safe)
        init_opts = (
            f"-use_input_sc -input_ab_scheme AHo_Scheme -ignore_unrecognized_res "
            f"-ignore_zero_occupancy false -load_PDB_components false -no_fconfig -mute all "
            f"-multithreading:total_threads 1 "
            f"-constant_seed -jran {args.seed} "
        )
        try:
            pyrosetta.init(init_opts)
        except RuntimeError:
            pass

        final_results = [calculate_ddG(t) for t in tqdm(tasks, desc="Calculating ddG")]
    else:
        # Multi-process: use spawn + per-worker init (avoid fork issues)
        init_opts = (
            "-use_input_sc -input_ab_scheme AHo_Scheme -ignore_unrecognized_res "
            "-ignore_zero_occupancy false -load_PDB_components false -no_fconfig -mute all "
            "-multithreading:total_threads 1 "
        )

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.cpus,
            initializer=_init_pyrosetta_worker,
            initargs=(init_opts, args.seed),
        ) as pool:
            final_results = list(
                tqdm(pool.imap(calculate_ddG, tasks, chunksize=1),
                     total=len(tasks), desc="Calculating ddG")
            )

    report_data = []
    for t in final_results:
        report_data.append({
            "name": t.name,
            "run_folder": t.run_folder,
            "variant": t.variant,
            "pred_path": t.pred_path,
            "ref_path": t.ref_path,
            **t.scores
        })

    df = pd.DataFrame(report_data)
    _summarize_imp(df)

    out_csv = os.path.join(output_dir, args.csv_name)
    df.to_csv(out_csv, index=False, float_format="%.6f")
    logging.info(f"Saved: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate interface ddG/IMP for relaxed and/or raw PDBs.")
    parser.add_argument("--pdb_dir", type=str, required=True,
                        help="Root directory (e.g., .../design) containing 'reference' and run subfolders.")
    parser.add_argument("--name_idx", type=str, required=True,
                        help="Path to .idx file with target names (one per line).")

    # output 默认等于 pdb_dir
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write csv (default: same as pdb_dir).")
    parser.add_argument("--csv_name", type=str, default="ddG_imp_results.csv",
                        help="CSV filename (default: ddG_imp_results.csv).")

    # relaxed/raw 选择
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "relaxed", "raw", "both"],
                        help="auto: prefer *_relaxed.pdb else raw; relaxed/raw: only that; both: compute both if present.")

    parser.add_argument("-c", "--cpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for deterministic Rosetta packing.")
    parser.add_argument("--layout", type=str, default="legacy",
                    choices=["legacy", "diffab"],
                    help="legacy: reference/ + run_folders; diffab: job/region/sample + REF*.pdb")
    parser.add_argument(
        "--diffab_region_dir",
        type=str,
        default=None,
        help="(layout=diffab only) Only evaluate EXACT region folder(s), comma-separated. "
             "E.g. 'H_CDR3' or 'H_CDR3-O8' or 'H_CDR3,H_CDR2'. "
             "If not set, will keep current behavior: pick max -O* per base."
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main(args)
