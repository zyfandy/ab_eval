import os
import argparse
import functools
import multiprocessing as mp
import logging
import re
import pandas as pd
import traceback
from tqdm import tqdm
import sys

from abx.metric import eval_metric

# 导入几何指标计算模块（外部路径）
geo_module_path = '/home/data3/cjm/project-zyf/ab_eval/AbConf/ab-geo'
if geo_module_path not in sys.path:
    sys.path.insert(0, geo_module_path)

import eval_geo


# ------------------------------------------------------------
# 辅助函数：扫描目录下所有 PDB 文件，跳过 reference 子目录
# ------------------------------------------------------------
def parse_list(data_dir: str, include_relaxed: bool) -> list:
    """
    递归扫描 data_dir 下的所有 .pdb 文件，排除 reference 目录。
    根据 include_relaxed 过滤是否包含 _relaxed.pdb 文件。
    返回 PDB 文件路径列表。
    """
    pdb_files = []
    pdb_pattern = re.compile(r'\.pdb$')
    relax_pattern = re.compile(r'_relaxed\.pdb$')

    reference_dir = os.path.abspath(os.path.join(data_dir, 'reference'))

    for root, _, files in os.walk(data_dir):
        # 跳过 reference 目录（参考结构存放处）
        if os.path.abspath(root).startswith(reference_dir):
            continue

        for fname in files:
            if not pdb_pattern.search(fname):
                continue
            # 跳过空文件
            if os.path.getsize(os.path.join(root, fname)) == 0:
                continue

            is_relaxed = bool(relax_pattern.search(fname))
            if include_relaxed:
                if not is_relaxed:
                    continue
            else:
                if is_relaxed:
                    continue

            pdb_files.append(os.path.join(root, fname))
    return pdb_files


# ------------------------------------------------------------
# diffab 模式专用：选取最优的 CDR 区域（如 H3-O2 优于 H3-O1）
# ------------------------------------------------------------
def _pick_best_regions(region_names):
    """
    从 region 名称列表（如 ["H3", "H3-O1", "L2"]）中选取最优版本。
    规则：同一基础名（去除 -O数字 后缀）下，取数字最大的版本返回。
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


# ------------------------------------------------------------
# legacy 模式：构建 (预测文件, 参考文件, 元数据) 任务列表
# ------------------------------------------------------------
def build_tasks_legacy(data_dir: str, include_relaxed: bool):
    """
    legacy 模式目录结构：
        data_dir/
            reference/          # 存放参考 PDB
            (其它子目录)         # 存放预测 PDB
    根据 include_relaxed 过滤预测文件后，与 reference 中同名的 PDB 配对。
    """
    pred_files = parse_list(data_dir, include_relaxed=include_relaxed)
    reference_dir = os.path.join(data_dir, 'reference')

    tasks = []
    for pred_file in pred_files:
        # 基础名去掉扩展名，并去掉 @ 后缀及 _relaxed 标记
        stem = os.path.splitext(os.path.basename(pred_file))[0].split('@')[0]
        if stem.endswith('_relaxed'):
            stem = stem[:-len('_relaxed')]

        ref_file = os.path.join(reference_dir, f"{stem}.pdb")

        if os.path.exists(ref_file):
            tasks.append((pred_file, ref_file, {}))
        else:
            logging.warning(f"No reference file found for prediction: {os.path.basename(pred_file)} (expected {ref_file})")
    return tasks


# ------------------------------------------------------------
# diffab 模式：构建任务列表（支持多 job、多 region、多 sample）
# ------------------------------------------------------------
def build_tasks_diffab(data_dir: str, include_relaxed: bool):
    """
    diffab 模式目录结构示例：
        data_dir/
            0001_1a3r_A_B_2000_01_01__00_00_00/
                H3/
                    REF1.pdb
                    0001.pdb
                    0002_relaxed.pdb (若 include_relaxed=True 则包括)
                 L3/
                    ...
            0002_...
    解析 job 名称获取靶点、重链、轻链等信息，并自动选择最优 region 版本。
    """
    tasks = []
    data_dir = os.path.abspath(data_dir)

    # 匹配 job 目录名格式
    job_dir_re = re.compile(
        r'^(?P<idx>\d{4})_'
        r'(?P<pdbid>[^_]+)_'
        r'(?P<chains>.+?)_'
        r'(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})__'
        r'(?P<h>\d{2})_(?P<m>\d{2})_(?P<s>\d{2})$'
    )
    region_dir_re = re.compile(r'^[HL]_CDR[123](?:-O\d+)?$')   # 例如 H_CDR1, H_CDR3-O2

    if include_relaxed:
        sample_re = re.compile(r'^\d{4}_relaxed\.pdb$')
    else:
        sample_re = re.compile(r'^\d{4}\.pdb$')

    ref_re = re.compile(r'^REF\d+\.pdb$', re.IGNORECASE)

    def parse_job(job_name: str):
        """从 job 目录名中提取靶点、轻重链和抗原信息"""
        m = job_dir_re.match(job_name)
        if not m:
            return None

        target = m.group("pdbid")
        chain_tokens = [t for t in m.group("chains").split("_") if t]
        if len(chain_tokens) < 2:
            return None

        heavy = chain_tokens[0]
        light = chain_tokens[1]
        antigen_tokens = chain_tokens[2:]
        antigen = "".join(antigen_tokens) if antigen_tokens else ""

        return {
            "target": target,
            "heavy": heavy,
            "light": light,
            "antigen": antigen,
            "pdbname": f"{target}_{heavy}_{light}_{antigen}" if antigen else f"{target}_{heavy}_{light}_"
        }

    for job_name in sorted(os.listdir(data_dir)):
        job_path = os.path.join(data_dir, job_name)
        if not os.path.isdir(job_path) or not job_dir_re.match(job_name):
            continue

        meta_job = parse_job(job_name)
        if meta_job is None:
            logging.warning(f"Skip job (cannot parse): {job_name}")
            continue

        # 收集该 job 下所有 region 目录
        region_names = []
        for rn in sorted(os.listdir(job_path)):
            rp = os.path.join(job_path, rn)
            if os.path.isdir(rp) and region_dir_re.match(rn):
                region_names.append(rn)

        # 保留每个 region 的最优版本（如 H_CDR3-O2 优先于 H_CDR3）
        region_names = _pick_best_regions(region_names)

        for region_name in region_names:
            region_path = os.path.join(job_path, region_name)

            # 查找参考文件（优先 REF1.pdb）
            ref_candidates = [f for f in sorted(os.listdir(region_path)) if ref_re.match(f)]
            if not ref_candidates:
                logging.warning(f"No REF*.pdb under {region_path}, skip.")
                continue
            if "REF1.pdb" in ref_candidates:
                ref_file = os.path.join(region_path, "REF1.pdb")
            else:
                ref_file = os.path.join(region_path, ref_candidates[0])

            # 遍历所有预测样本文件
            for fn in sorted(os.listdir(region_path)):
                if not sample_re.match(fn):
                    continue
                pred_file = os.path.join(region_path, fn)

                sample_id = fn.split('_')[0].split('.')[0]   # 如 "0001"
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


# ------------------------------------------------------------
# rabd 模式：构建任务列表（results/*_generated.pdb + templates/*_template.pdb）
# ------------------------------------------------------------
def build_tasks_rabd(data_dir: str, include_relaxed: bool):
    """
    rabd 模式目录结构：
        data_dir/
            results/          # 存放预测文件，命名如 1a3r_generated.pdb
            templates/        # 存放参考文件，命名如 1a3r_template.pdb
    include_relaxed 参数在此模式下被忽略（保留兼容）。
    """
    tasks = []
    results_dir = os.path.join(data_dir, 'results')
    templates_dir = os.path.join(data_dir, 'templates')

    if not os.path.isdir(results_dir):
        logging.error(f"Results directory not found: {results_dir}")
        return tasks
    if not os.path.isdir(templates_dir):
        logging.error(f"Templates directory not found: {templates_dir}")
        return tasks

    pattern = re.compile(r'^(.+)_generated\.pdb$')

    for fname in os.listdir(results_dir):
        if not fname.endswith('.pdb'):
            continue
        m = pattern.match(fname)
        if not m:
            continue
        base_id = m.group(1)   # 如 "1a3r"
        pred_file = os.path.join(results_dir, fname)
        ref_file = os.path.join(templates_dir, f"{base_id}_template.pdb")
        if not os.path.exists(ref_file):
            logging.warning(f"Template not found for {pred_file}: expected {ref_file}")
            continue
        tasks.append((pred_file, ref_file, {}))
    return tasks


# ------------------------------------------------------------
# 单个任务的评测函数（封装 eval_metric 并可选添加几何指标）
# ------------------------------------------------------------
def eval_metric_with_meta(pred_file: str, ref_file: str, meta: dict, args: argparse.Namespace):
    """
    调用 abx.metric.eval_metric 计算 DockQ 等指标，
    若 args.geo 为 True，则进一步调用 eval_geo.compute_geometry_metrics_for_pair 计算几何指标（JSD, MAE, MCE）。
    返回包含所有指标的字典，失败返回 None。
    """
    r = eval_metric(pred_file, ref_file, args, meta=meta)
    if r is None:
        return None
    if meta:
        r.update(meta)

    # 计算几何指标（仅当命令行指定 --geo 且模块可用时）
    if args.geo:
        try:
            heavy = meta.get('heavy') if meta else None
            light = meta.get('light') if meta else None
            geo_metrics = eval_geo.compute_geometry_metrics_for_pair(
                pred_file, ref_file,
                heavy_chain=heavy,
                light_chain=light,
                auto_detect=(heavy is None or light is None),
                cdr_list=None,
                n_bins=50
            )
            r.update(geo_metrics)
        except Exception as e:
            logging.warning(f"Geometry metrics failed for {pred_file}: {e}")
    return r


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def main(args):
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(processName)s] %(levelname)s - %(message)s'
    )

    try:
        logging.info(f"[mode={args.mode}] Scanning for evaluation tasks in '{args.data_dir}'...")

        # 根据 data_dir 路径是否以 _relaxed 结尾自动决定 include_relaxed
        include_relaxed = args.data_dir.rstrip('/').endswith('_relaxed')

        # 根据模式构建任务列表
        if args.mode == "legacy":
            tasks = build_tasks_legacy(args.data_dir, include_relaxed=include_relaxed)
        elif args.mode == "rabd":
            tasks = build_tasks_rabd(args.data_dir, include_relaxed=include_relaxed)
        else:   # diffab 模式
            tasks = build_tasks_diffab(args.data_dir, include_relaxed=include_relaxed)

        if not tasks:
            logging.info("No valid prediction-reference pairs found to process.")
            return

        logging.info(f"Found {len(tasks)} tasks. Starting evaluation with {args.cpus} CPUs...")

        func = functools.partial(eval_metric_with_meta, args=args)

        # 多进程执行评测
        with mp.Pool(processes=args.cpus) as pool:
            all_results = list(tqdm(pool.starmap(func, tasks), total=len(tasks)))
        results = [r for r in all_results if r is not None]

        # 可选：单进程调试（注释掉的代码）
        # results = []
        # for pred_file, ref_file, meta in tqdm(tasks, desc="Processing tasks"):
        #     try:
        #         result = func(pred_file, ref_file, meta)
        #         if result is not None:
        #             results.append(result)
        #     except Exception as e:
        #         logging.error(f"Error processing {pred_file}: {e}")
        #         traceback.print_exc()

        if not results:
            logging.warning("No files were processed successfully. Please check logs for errors.")
            return

        # --- 结果汇总与输出 ---
        logging.info("Evaluation complete. Aggregating results...")
        df = pd.DataFrame(results)

        avg_metrics = [col for col in df.columns if col not in ['code', 'file_path']]

        if avg_metrics:
            print("\n" + "-" * 21)
            print("Average Results for each Metric")
            print("-" * 21)
            # 输出各指标的平均值（忽略 NaN）
            print(df[avg_metrics].mean(numeric_only=True).to_string(float_format=lambda x: f"{x:.6f}"))

        # 保存详细结果到 CSV
        output_path = os.path.join(args.data_dir, 'evaluation_results.csv')
        df.to_csv(output_path, index=False, float_format='%.4f')
        logging.info(f"Full results saved to {output_path}")

    except Exception as e:
        logging.error(f"A critical error occurred in the main pipeline: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    # 设置多进程启动方式为 spawn，避免 Linux 下 fork 导致的问题
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_dir', type=str, required=True, help="数据目录路径")
    parser.add_argument('-c', '--cpus', type=int, default=1, help="并行使用的 CPU 核心数")
    parser.add_argument('-e', '--energy', action='store_true', help="启用能量计算（传递给底层 eval_metric）")
    parser.add_argument('-v', '--verbose', action='store_true', help="输出详细日志")
    parser.add_argument('--H3', action='store_true', help="仅计算 H3 CDR 环的 DockQ")
    parser.add_argument('--mode', type=str, choices=['legacy', 'diffab', 'rabd'], default='legacy',
                        help="选择目录结构模式: legacy | diffab | rabd")
    parser.add_argument('--geo', action='store_true', help="计算几何指标（JSD, MAE, MCE，依赖 eval_geo 模块）")

    args = parser.parse_args()

    # 确保至少有一个日志处理器
    if not any(isinstance(h, logging.StreamHandler) for h in logging.getLogger().handlers):
        logging.getLogger().addHandler(logging.StreamHandler())

    main(args)

# 使用示例:
#   计算所有 Fv DockQ:  python metric.py -i /path/to/data --cpus 32
#   只计算 H3 DockQ:    python metric.py -i /path/to/data --cpus 32 --H3