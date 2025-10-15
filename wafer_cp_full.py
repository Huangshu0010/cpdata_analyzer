#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wafer_cp_full.py
合并同一晶圆 → 逐晶圆分析 → Lot 汇总
"""
import re, sys, pathlib, itertools
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict

# ---------- 0. 结果根目录 ----------
RESULT_ROOT = pathlib.Path('./CP_Results')
RESULT_ROOT.mkdir(exist_ok=True)

# ---------- 1. 正则提取 lot & wafer ----------
LOT_WAFER_RE = re.compile(r'^(.*)_(FA\d+-\d+)_((FA\d+-\d+)-(\d+))_CP\d*_', re.I)

def parse_filename(name):
    m = LOT_WAFER_RE.match(name)
    if not m:
        return None, None
    return m.group(2), m.group(3)          # lot, wafer

# ---------- 2. 读单文件 ----------
def read_single_cp_csv(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.rstrip() for l in f]
    start_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^\s*SITE_NUM,', line):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f'{path} 未找到 SITE_NUM 起始行')
    header = lines[start_idx].split(',')
    rows   = [l.split(',') for l in lines[start_idx+1:]]

    unit_row, low_row, high_row = rows[0], rows[1], rows[2]
    data_rows = rows[3:]
    df = pd.DataFrame(data_rows, columns=header)

    para_lim = {}
    for col in header:
        idx  = header.index(col)
        u    = unit_row[idx]
        L    = low_row[idx]  if low_row[idx]  != '' else np.nan
        U    = high_row[idx] if high_row[idx] != '' else np.nan
        try: L = float(L)
        except: L = np.nan
        try: U = float(U)
        except: U = np.nan
        para_lim[col] = {'L': L, 'U': U, 'Unit': u}

    df_lim = pd.DataFrame([unit_row, low_row, high_row], columns=header)
    df = pd.concat([df_lim, df], ignore_index=True)
    return df, para_lim

# ---------- 3. 类型转换 ----------
def convert_df_types(df):
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            # 任何异常都保持原字符串
            pass
    return df

def drop_duplicate_die(df):
    """
    按 X_COORD+Y_COORD 去重，保留最后一条（文件已按测试顺序排列）
    返回去重后的 DataFrame（含 Limit 行）
    """
    # 保留前 3 行 Limit 数据
    limit_rows = df.iloc[:3].copy()
    data       = df.iloc[3:].copy()

    # 生成唯一键
    data['die_key'] = data['X_COORD'].astype(str) + ',' + data['Y_COORD'].astype(str)
    data = data.drop_duplicates(subset=['die_key'], keep='last')
    data = data.drop(columns=['die_key'])

    # 拼回 Limit
    return pd.concat([limit_rows, data], ignore_index=True)
    
# ---------- 4. 单晶圆分析 ----------
def analyze_one_wafer(df, para_lim, out_root, wafer):
    save_dir = out_root / wafer
    save_dir.mkdir(parents=True, exist_ok=True)
    # ===== 去重：同坐标保留最后一次 =====
    df = drop_duplicate_die(df)
    # ===================================
    # 4.1 良率
    df['PASSFG'] = df['PASSFG'].astype(str).str.lower().isin({'true', '1', 'pass', 'yes'})
    yield_df = df[['PART_ID', 'PASSFG']].copy()
    yield_df.columns = ['PART_ID', 'Pass']
    yield_file = save_dir / f'{wafer}_yield.csv'
    yield_df.to_csv(yield_file, index=False)
    total = len(yield_df)
    pass_cnt = yield_df['Pass'].sum()
    print(f'  晶圆 {wafer}  总计 {total} 颗  通过 {pass_cnt}  良率 {pass_cnt/total*100:.2f}%')

    # 4.2 参数统计
    stats = []
    for col in df.columns:
        if col in {'SITE_NUM','PART_ID','PASSFG','SOFT_BIN','T_TIME',
                   'X_COORD','Y_COORD','TEST_NUM'}:
            continue
        vals = pd.to_numeric(df[col].iloc[3:], errors='coerce').dropna()
        if len(vals) == 0:
            continue
        mean, std, mn, mx = vals.mean(), vals.std(ddof=1), vals.min(), vals.max()
        L = para_lim[col]['L']
        U = para_lim[col]['U']
        fail = 0
        if not pd.isna(L): fail += (vals < L).sum()
        if not pd.isna(U): fail += (vals > U).sum()
        cpk_l = (mean - L)/(3*std) if not pd.isna(L) and std != 0 else np.nan
        cpk_u = (U - mean)/(3*std) if not pd.isna(U) and std != 0 else np.nan
        cpk   = np.nanmin([cpk_l, cpk_u])
        stats.append({
            'Parameter': col,
            'Unit': para_lim[col]['Unit'],
            'Mean': mean, 'Std': std, 'Min': mn, 'Max': mx,
            'FailCount': fail, 'Cpk': cpk
        })
    stats_df = pd.DataFrame(stats)
    stats_file = save_dir / f'{wafer}_stats.csv'
    stats_df.to_csv(stats_file, index=False)

    # 4.3 3×3 直方图  (X轴 = Limit ±50%)
    hist_file = save_dir / f'{wafer}_histograms.pdf'
    param_list = [p for p in df.columns
                  if p not in {'SITE_NUM','PART_ID','PASSFG','SOFT_BIN',
                               'T_TIME','X_COORD','Y_COORD','TEST_NUM'}]
    with PdfPages(hist_file) as pdf:
        for i in range(0, len(param_list), 9):
            fig, axes = plt.subplots(3, 3, figsize=(15, 10))
            axes = axes.flatten()
            for ax, param in zip(axes, param_list[i:i+9]):
                vals = pd.to_numeric(df[param].iloc[3:], errors='coerce').dropna()
                if len(vals) == 0:
                    ax.axis('off')
                    continue
                # 计算显示区间 & 柱宽
                L = para_lim[param]['L']
                U = para_lim[param]['U']
                if pd.isna(L) or pd.isna(U):
                    bins = 50          # 无规格线用默认
                else:
                    span  = (U - L) * 1.2            # L-50% → U+50%
                    width = span / 40                # 每柱约 20 单位
                    bins  = np.arange(L - span*0.5, U + span*0.5 + width, width)
                # 绘制
                ax.hist(vals, bins=bins, edgecolor='k', align='mid')
                # ------ X 轴范围 = Limit ±50% ------
                if not (pd.isna(L) and pd.isna(U)):
                    span = (U - L) if not (pd.isna(L) or pd.isna(U)) else (
                        U - vals.min() if pd.isna(L) else vals.max() - L)
                    span = max(span, (vals.max() - vals.min()) * 0.1)
                    x_left  = L - span * 0.5 if not pd.isna(L) else vals.min() - span * 0.2
                    x_right = U + span * 0.5 if not pd.isna(U) else vals.max() + span * 0.2
                    ax.set_xlim(x_left, x_right)
                # -----------------------------------
                ax.set_title(f'{param}\nn={len(vals)}')
                ax.set_xlabel(f"{param} ({para_lim[param]['Unit']})")
                if not pd.isna(L): ax.axvline(L, color='r', ls='--', lw=1)
                if not pd.isna(U): ax.axvline(U, color='r', ls='--', lw=1)
            for j in range(len(param_list[i:i+9]), 9):
                axes[j].axis('off')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

# ---------- 5. Lot 汇总 ----------
def lot_summary(out_root, lot, wafers, para_lim):
    print(f'\n>>> 汇总 Lot {lot}')
    # 5.1 合并各晶圆 stats
    stats_list = []
    for wafer in wafers:
        stat_file = out_root / wafer / f'{wafer}_stats.csv'
        if not stat_file.exists():
            print(f'  警告：找不到 {stat_file}')
            continue
        df = pd.read_csv(stat_file)
        df['Wafer'] = wafer
        stats_list.append(df)
    if not stats_list:
        print(f'  Lot {lot} 无有效统计文件，跳过汇总')
        return
    lot_df = pd.concat(stats_list, ignore_index=True)

    lot_stat = (lot_df.groupby('Parameter', as_index=False)
                      .agg({'Mean': ['mean', 'std', 'min', 'max'],
                            'Cpk': ['mean', 'min']}))
    lot_stat.columns = ['Parameter',
                        'Mean_avg', 'Mean_std', 'Mean_min', 'Mean_max',
                        'Cpk_avg', 'Cpk_min']
    lot_stat_file = out_root / f'{lot}_lot_stats.csv'
    lot_stat.to_csv(lot_stat_file, index=False)
    print(f'  Lot 统计 -> {lot_stat_file}')

    # 5.2 各晶圆 Cpk 对比
    cpk_pivot = lot_df.pivot(index='Parameter', columns='Wafer', values='Cpk')
    cpk_file = out_root / f'{lot}_wafer_cpk.csv'
    cpk_pivot.to_csv(cpk_file)
    print(f'  Cpk 对比 -> {cpk_file}')

    # 5.3 Lot 级直方图（合并全部原始数据）
    lot_raw_list = []
    for wafer in wafers:
        csv_files = list(out_root.parent.glob(f'*{wafer}*CP*.csv'))
        for csv in csv_files:
            df, _ = read_single_cp_csv(csv)
            df = convert_df_types(df)
            lot_raw_list.append(df)
    if lot_raw_list:
        lot_raw = pd.concat(lot_raw_list, ignore_index=True)
        hist_file = out_root / f'{lot}_lot_histograms.pdf'
        param_list = [p for p in lot_raw.columns
                      if p not in {'SITE_NUM','PART_ID','PASSFG','SOFT_BIN',
                                   'T_TIME','X_COORD','Y_COORD','TEST_NUM'}]
        with PdfPages(hist_file) as pdf:
            for i in range(0, len(param_list), 9):
                fig, axes = plt.subplots(3, 3, figsize=(15, 10))
                axes = axes.flatten()
                for ax, param in zip(axes, param_list[i:i+9]):
                    vals = pd.to_numeric(df[param].iloc[3:], errors='coerce').dropna()
                    if len(vals) == 0:
                        ax.axis('off')
                        continue
                    # 计算显示区间 & 柱宽
                    L = para_lim[param]['L']
                    U = para_lim[param]['U']
                    if pd.isna(L) or pd.isna(U):
                        bins = 50          # 无规格线用默认
                    else:
                        span  = (U - L) * 1.2            # L-50% → U+50%
                        width = span / 40                # 每柱约 20 单位
                        bins  = np.arange(L - span*0.5, U + span*0.5 + width, width)
                    # 绘制
                    ax.hist(vals, bins=bins, edgecolor='k', align='mid')
                    # ------ 同样 Limit ±50% ------
                    if not (pd.isna(L) and pd.isna(U)):
                        span = (U - L) if not (pd.isna(L) or pd.isna(U)) else (
                            U - vals.min() if pd.isna(L) else vals.max() - L)
                        span = max(span, (vals.max() - vals.min()) * 0.1)
                        x_left  = L - span * 0.5 if not pd.isna(L) else vals.min() - span * 0.2
                        x_right = U + span * 0.5 if not pd.isna(U) else vals.max() + span * 0.2
                        ax.set_xlim(x_left, x_right)
                    # ----------------------------
                    ax.set_title(f'{param}  Lot {lot}\nn={len(vals)}')
                    ax.set_xlabel(f"{param} ({para_lim[param]['Unit']})")
                    if not pd.isna(L): ax.axvline(L, color='r', ls='--', lw=1)
                    if not pd.isna(U): ax.axvline(U, color='r', ls='--', lw=1)
                for j in range(len(param_list[i:i+9]), 9):
                    axes[j].axis('off')
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
        print(f'  Lot 直方图 -> {hist_file}')

# ---------- 6. 主流程 ----------
def main():
    root  = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path.cwd()
    files = [f for f in root.glob('*CP*.csv') if f.is_file()]
    if not files:
        print('未找到 *CP*.csv 文件'); return

    # 6.1 按晶圆分组
    wafer_files = {}
    for csv in files:
        lot, wafer = parse_filename(csv.name)
        if wafer is None:
            print(f'跳过无法解析的文件: {csv.name}'); continue
        wafer_files.setdefault(wafer, []).append(csv)

    # 6.2 逐晶圆合并并分析
    lot_wafers = defaultdict(list)
    for wafer, csvs in wafer_files.items():
        print(f'\n>>> 处理晶圆 {wafer}  共 {len(csvs)} 个文件')
        df_list, lim0 = [], None
        for csv in csvs:
            df, lim = read_single_cp_csv(csv)
            df_list.append(df); lim0 = lim
        merged = pd.concat(df_list, ignore_index=True)
        merged = convert_df_types(merged)
        analyze_one_wafer(merged, lim0, RESULT_ROOT, wafer)
        lot = '-'.join(wafer.split('-')[:2])   # FA56-7390
        lot_wafers[lot].append(wafer)

    # 6.3 Lot 汇总
    for lot, wafers in lot_wafers.items():
        _, para_lim = read_single_cp_csv(wafer_files[wafers[0]][0])
        lot_summary(RESULT_ROOT, lot, wafers, para_lim)

    print('\n全部完成！结果统一保存在:', RESULT_ROOT.absolute())

if __name__ == '__main__':
    main()
