#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ========== USER CONFIGURATION ==========
main_csv = "Data/input.csv"
test_csv = "Data/test.csv"
output_csv = "Data/combined_train_test.csv"
summary_dir = "summaries_joint"
outdir = "plots_iso_deam_splits"
window_min = 1
window_max = 6
n_folds = 5
rate_cutoffs = [2, 5]
# =========================================

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
os.makedirs(summary_dir, exist_ok=True)
os.makedirs(outdir, exist_ok=True)





# ========================= HELPERS ========================

def total_deviation(percent, global_pct):
    return np.sum(np.abs(percent - global_pct))


def calc_fold_pos_percent(df, assignments, label_col, n_folds):
    percent = np.zeros(n_folds)
    N = np.zeros(n_folds, dtype=int)
    Pos = np.zeros(n_folds, dtype=int)
    for f in range(n_folds):
        m = (assignments == f)
        N[f] = m.sum()
        if N[f] > 0:
            Pos[f] = df.loc[m, label_col].sum()
            percent[f] = Pos[f] / N[f]
        else:
            percent[f] = 0.
    return percent, N, Pos


def load_and_filter(main_csv: str, skip_pH=False) -> pd.DataFrame:
    df = pd.read_csv(main_csv)
    if "windows_5" in df.columns:
        df = df.drop(columns=["windows_5"])
    df = df.rename(columns={"N.1":"N-1","N.1.1":"N+1"})
    if not skip_pH and "pH" in df.columns:
        df = df[(df["N+1"]!="PRO") & (df["pH"].isin([5.5,6.0]))].reset_index(drop=True)
    else:
        df = df[df["N+1"]!="PRO"].reset_index(drop=True)
    return df

def extract_sequence(df: pd.DataFrame) -> pd.DataFrame:
    def get_seq(row):
        key = f"Chain{row['Chain']}_Seq"
        if key in row and pd.notna(row[key]):
            return row[key]
        for c in ['ChainA_Seq', 'ChainB_Seq', 'ChainC_Seq', 'ChainD_Seq', 'ChainE_Seq', 'ChainF_Seq']:
            if c in row and pd.notna(row[c]):
                return row[c]
        return np.nan
    df = df.copy()
    df["Sequence"] = df.apply(get_seq, axis=1)
    return df

def window_indices(pos, length, win):
    indices = [pos]
    if win >= 1:
        if pos + 1 < length:
            indices.append(pos + 1)
    if win >= 2:
        if pos - 1 >= 0:
            indices.insert(0, pos - 1)
    if win >= 3:
        if pos + 2 < length:
            indices.append(pos + 2)
    if win >= 4:
        if pos - 2 >= 0:
            indices.insert(0, pos - 2)
    if win >= 5:
        if pos + 3 < length:
            indices.append(pos + 3)
    if win >= 6:
        if pos - 3 >= 0:
            indices.insert(0, pos - 3)
    return sorted(set(indices))

def make_window_seq(seq, pos, win):
    idxs = window_indices(pos, len(seq), win)
    return ''.join([seq[i] for i in idxs])

def assign_clusters_equal_size(df, group_col, n_folds):
    group_df = df.groupby(group_col).size().reset_index(name="count")
    group_df = group_df.sort_values("count",ascending=False).reset_index(drop=True)
    fold_sizes = [0]*n_folds
    mapping = {}
    for _,row in group_df.iterrows():
        g = row[group_col]; c = row["count"]
        f = fold_sizes.index(min(fold_sizes))
        mapping[g] = f
        fold_sizes[f] += c
    return df[group_col].map(mapping)

def refine_twofold_swap(df, group_col, label_col, cur, target_ratio):
    df2 = df.copy(); df2["ClusterID"] = cur
    stats = df2.groupby("ClusterID")[label_col].agg(["sum","count"])
    stats["ratio"] = stats["sum"]/stats["count"]
    low_f  = stats["ratio"].idxmin()
    high_f = stats["ratio"].idxmax()
    low_r  = stats.loc[low_f,"ratio"]
    high_r = stats.loc[high_f,"ratio"]
    old_dev_low  = target_ratio - low_r
    old_dev_high = high_r - target_ratio

    best_imp = 0
    best_swap = None
    # Try classic: swap negatives with positives only
    for g in df2[group_col].unique():
        this_fold = cur[df[group_col]==g].iloc[0]
        g_targets = set(df2.loc[df2[group_col]==g, label_col])
        # Only attempt "classic" swap if applicable
        if (this_fold == high_f and g_targets == {1}) or (this_fold == low_f and g_targets == {0}):
            for h in df2[group_col].unique():
                that_fold = cur[df[group_col]==h].iloc[0]
                h_targets = set(df2.loc[df2[group_col]==h, label_col])
                if (this_fold == high_f and that_fold == low_f and h_targets == {0}) or \
                   (this_fold == low_f and that_fold == high_f and g_targets == {0}):
                    # test out the swap
                    df_sim = df2.copy()
                    df_sim.loc[df2[group_col]==g,"ClusterID"] = that_fold
                    df_sim.loc[df2[group_col]==h,"ClusterID"] = this_fold
                    s2 = df_sim.groupby("ClusterID")[label_col].agg(["sum","count"])
                    s2["ratio"] = s2["sum"]/s2["count"]
                    new_low_r  = s2.loc[low_f,"ratio"]
                    new_high_r = s2.loc[high_f,"ratio"]
                    dev_low2   = target_ratio - new_low_r
                    dev_high2  = new_high_r - target_ratio
                    imp = (old_dev_low - dev_low2) + (old_dev_high - dev_high2)
                    if imp > best_imp:
                        best_imp, best_swap = imp, (g,h)
    if best_swap:
        g,h = best_swap
        f_g = cur[df[group_col]==g].iloc[0]
        f_h = cur[df[group_col]==h].iloc[0]
        cur.loc[df[group_col]==g] = f_h
        cur.loc[df[group_col]==h] = f_g
        return cur

    # No classic swap, try a mixed cluster
    # In "high" fold, get all mixed clusters, pick the one with highest pos%
    mixed_clusters = []
    for g in df2[group_col].unique():
        this_fold = cur[df[group_col]==g].iloc[0]
        g_targets = set(df2.loc[df2[group_col]==g, label_col])
        if this_fold == high_f and len(g_targets) > 1:
            pc = df2.loc[df2[group_col]==g, label_col].mean()
            mixed_clusters.append((g, pc))
    if mixed_clusters:
        # Sort by highest positive rate first
        mixed_clusters.sort(key=lambda x: -x[1])
        for g, pc in mixed_clusters:
            # Try every negative cluster in low_f
            for h in df2[group_col].unique():
                that_fold = cur[df[group_col]==h].iloc[0]
                h_targets = set(df2.loc[df2[group_col]==h, label_col])
                if that_fold == low_f and h_targets == {0}:
                    df_sim = df2.copy()
                    df_sim.loc[df2[group_col]==g,"ClusterID"] = low_f
                    df_sim.loc[df2[group_col]==h,"ClusterID"] = high_f
                    s2 = df_sim.groupby("ClusterID")[label_col].agg(["sum","count"])
                    s2["ratio"] = s2["sum"]/s2["count"]
                    new_low_r  = s2.loc[low_f,"ratio"]
                    new_high_r = s2.loc[high_f,"ratio"]
                    dev_low2   = target_ratio - new_low_r
                    dev_high2  = new_high_r - target_ratio
                    imp = (old_dev_low - dev_low2) + (old_dev_high - dev_high2)
                    if imp > best_imp:
                        best_imp, best_swap = imp, (g,h)
        if best_swap:
            g,h = best_swap
            f_g = cur[df[group_col]==g].iloc[0]
            f_h = cur[df[group_col]==h].iloc[0]
            cur.loc[df[group_col]==g] = f_h
            cur.loc[df[group_col]==h] = f_g
    return cur

# ===================== MAIN WORK =====================

df = load_and_filter(main_csv, skip_pH=False)
df = extract_sequence(df)
test_df = load_and_filter(test_csv, skip_pH=True)
test_df = extract_sequence(test_df)
df['is_test'] = 0
test_df['is_test'] = 1

base_cols = sorted(set(df.columns) | set(test_df.columns))
for col in base_cols:
    if col not in df.columns:
        df[col] = np.nan
    if col not in test_df.columns:
        test_df[col] = np.nan

full_df = pd.concat([df, test_df], sort=False, ignore_index=True)
full_df.insert(0, "SerialID", np.arange(1, len(full_df) + 1))

window_sizes = list(range(window_min, window_max + 1))
all_summaries = []

for cutoff in rate_cutoffs:
    cutoffstr = f"cut_{cutoff}"
    full_df[f"Rate_{cutoffstr}"] = (full_df["Rate"] > cutoff).astype(int)
    for win in window_sizes:
        winstr = f"win_{win}"
        windowseq = []
        for _, row in full_df.iterrows():
            seq = row["Sequence"]
            try:
                pos = int(row["ResNum"]) - 1
            except:
                windowseq.append("")
                continue
            if isinstance(seq, str) and seq and 0 <= pos < len(seq):
                winseq = make_window_seq(seq, pos, win)
            else:
                winseq = ""
            windowseq.append(winseq)
        full_df[f"WindowSeq_{winstr}"] = windowseq
        # Only assign clusters for training (is_test == 0)
        in_train = (full_df['is_test'] == 0)
        train_windowseq = full_df.loc[in_train, f"WindowSeq_{winstr}"].tolist()
        train_target = full_df.loc[in_train, f"Rate_{cutoffstr}"].astype(int).tolist()
        group_col = f"WindowSeq_{winstr}"
        if len(set(train_windowseq)) <= n_folds:
            continue

        df_train = full_df.loc[in_train].copy()
        df_train['Target'] = train_target

        max_swaps = 10
        tolerance = 0.10  # allow ±10%

        assignments_fold = assign_clusters_equal_size(df_train, group_col, n_folds)
        global_pos_pct = df_train['Target'].mean()

        swaps = 0
        per_fold_pct, fold_Ns, fold_Ps = calc_fold_pos_percent(df_train, assignments_fold, "Target", n_folds)
        prev_deviation = total_deviation(per_fold_pct, global_pos_pct)

        while swaps < max_swaps:
            if np.all(np.abs(per_fold_pct - global_pos_pct) <= tolerance):
                print(f"Window={win} Cutoff={cutoff}: Class balance criterion met after {swaps} swaps.")
                break
            new_assignments = refine_twofold_swap(df_train, group_col, "Target", assignments_fold.copy(), global_pos_pct)
            new_per_fold_pct, _, _ = calc_fold_pos_percent(df_train, new_assignments, "Target", n_folds)
            new_deviation = total_deviation(new_per_fold_pct, global_pos_pct)

            # Accept swap only if it improves total deviation
            if new_deviation < prev_deviation:
                assignments_fold = new_assignments
                per_fold_pct = new_per_fold_pct
                prev_deviation = new_deviation
                swaps += 1
                print(f"Window={win} Cutoff={cutoff}: Swap {swaps}, deviation now {prev_deviation:.4f}")
            else:
                print(f"Window={win} Cutoff={cutoff}: No improving swap at swap {swaps+1}. Stopping.")
                break

        if swaps == max_swaps:
            print(f"Window={win} Cutoff={cutoff}: Reached max {max_swaps} swaps. Class balance may not be fully optimal.")

        col_name = f"{winstr}_cut_{cutoff}_ClusterID"
        full_df[col_name] = np.nan
        full_df.loc[in_train, col_name] = assignments_fold.values
        n_clusters = len(set(train_windowseq))
        clusters = df_train.groupby(group_col)
        cluster_info = []
        for cname, grp in clusters:
            vals = set(grp['Target'])
            n_pos = grp['Target'].sum()
            n_neg = len(grp) - n_pos
            cluster_info.append((cname, vals, len(grp), n_pos, n_neg))
        n_both = sum(1 for x in cluster_info if (0 in x[1] and 1 in x[1]))
        n_only_pos = sum(1 for x in cluster_info if x[1] == {1})
        n_only_neg = sum(1 for x in cluster_info if x[1] == {0})
        summary_rows = []
        for f in range(n_folds):
            m = (assignments_fold == f)
            N = m.sum()
            Pos = df_train.loc[m, "Target"].sum()
            Neg = N - Pos
            clusters_in_fold = df_train.loc[m, group_col].nunique()
            wvals = df_train.loc[m].groupby(group_col)["Target"].agg(lambda x: set(x))
            fold_n_both = (wvals.apply(lambda s: (0 in s) and (1 in s))).sum()
            pct_pos = (Pos / N * 100.0) if N > 0 else 0.0
            info = {
                "window_size": win,
                "cutoff": cutoff,
                "fold": f,
                "n_clusters": n_clusters,
                "n_only_pos": n_only_pos,
                "n_only_neg": n_only_neg,
                "n_both": n_both,
                "fold_N": N,
                "fold_Pos": Pos,
                "fold_Neg": Neg,
                "fold_clusters": clusters_in_fold,
                "fold_mixclusters": fold_n_both,
                "fold_PositivePercent": pct_pos,  # Added statistic
            }
            summary_rows.append(info)
            all_summaries.append(info)
        summary_fn = os.path.join(summary_dir, f"summary_win{win}_cut{cutoff}.csv")
        pd.DataFrame(summary_rows).to_csv(summary_fn, index=False)

full_df.to_csv(output_csv, index=False)
print(f"\nWrote full multi-split dataset to {output_csv}")
# ========== SUMMARY PRINTOUT & EXAMPLE USAGE ==========
master_summary_df = pd.DataFrame(all_summaries)
if not master_summary_df.empty:
    smry = master_summary_df.groupby(['window_size','cutoff']).agg(
        total_clusters=("n_clusters","first"),
        only_pos_clusters=("n_only_pos","first"),
        only_neg_clusters=("n_only_neg","first"),
        both_clusters=("n_both","first"),
        total_pos=("fold_Pos","sum"),
        total_neg=("fold_Neg","sum"),
        total_samples=("fold_N","sum")
    ).reset_index()
    print("\n==== MASTER SPLIT SUMMARY ====")
    print(smry.to_string(index=False))
    print("\nCheck summary CSVs for per-fold details.")
else:
    print("\nNo split summary was produced (likely not enough clusters/samples for some window/cutoff).")

# Example of train/val/test split for CV for window w and cutoff c:
# train_df = full_df[(full_df['is_test']==0) & (full_df[f'win_{w}_cut_{c}_ClusterID']!=f)]
# val_df   = full_df[(full_df['is_test']==0) & (full_df[f'win_{w}_cut_{c}_ClusterID']==f)]
# test_df  = full_df[full_df['is_test']==1]

# ========== PLOTTING ACROSS SPLITS ==========

sns.set_style("whitegrid")

def autolabel(ax, bars, is_frac=False):
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            txt = f"{height:.2f}" if is_frac else f"{height:.0f}"
            ax.annotate(txt, xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 2),
                textcoords="offset points", ha='center', va='bottom', fontsize=8)

group_sizes_plot = list(range(window_min, window_max + 1))
bar_colors = {'neg': '#1f77b4', 'pos': '#ff7f0e', 'mix': '#7f7f7f'}

for cutoff in rate_cutoffs:
    d = master_summary_df[master_summary_df["cutoff"] == cutoff].copy()
    # --- Cluster composition by group size ---
    summ = d.groupby("window_size").agg({
        "n_clusters": "first",
        "n_only_pos": "first",
        "n_only_neg": "first",
        "n_both": "first"
    }).reset_index().rename(columns={"window_size": "group_size"})
    plt.figure(figsize=(10,6))
    bars1 = plt.bar(summ.group_size-0.2, summ.n_only_pos, width=0.2, label="Pos only clusters")
    bars2 = plt.bar(summ.group_size    , summ.n_only_neg, width=0.2, label="Neg only clusters")
    bars3 = plt.bar(summ.group_size+0.2, summ.n_both, width=0.2, label="Mixed clusters")
    plt.plot(summ.group_size, summ.n_clusters, 'ko--', label="Total clusters")
    [autolabel(plt.gca(), b) for b in [bars1, bars2, bars3]]
    plt.xlabel("Group Size")
    plt.ylabel("Count")
    plt.title(f"Cluster Types vs Group Size (cutoff={cutoff})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"clusters_types_groupsize_cutoff{cutoff}.png"))
    plt.close()

    stats = ['fold_Pos', 'fold_Neg', 'fold_clusters', 'fold_mixclusters']
    stats_labels = ['Pos', 'Neg', 'Clusters', 'Mixed clusters']
    x = np.arange(len(group_sizes_plot))
    group_width = 0.8
    bar_width = group_width / len(stats)

    for fold in range(n_folds):
        plt.figure(figsize=(14,7))
        fold_data = d[d['fold']==fold].set_index('window_size').sort_index()
        fold_data.index.name = "group_size"
        fold_data = fold_data.rename_axis("group_size").reset_index()
        for i, (stat, stats_label) in enumerate(zip(stats, stats_labels)):
            offset = (i - (len(stats)-1)/2) * bar_width
            # Ensure we use group_size as x axis
            bars = plt.bar(x + offset, fold_data[stat].reindex(group_sizes_plot, fill_value=0),
                width=bar_width*0.96, label=stats_label)
            autolabel(plt.gca(), bars)
        plt.xticks(x, group_sizes_plot)
        plt.xlabel("Group Size")
        plt.ylabel("Count")
        plt.title(f"Fold {fold+1}: Pos, Neg, Cluster, & Mixed Cluster Count vs Group Size (cutoff={cutoff})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"fold_stats_hist_fold{fold+1}_cutoff{cutoff}.png"))
        plt.close()

    # Positive fraction per fold with debug print
    plt.figure(figsize=(13,7))
    palette = sns.color_palette("tab10", n_folds)
    bar_width_fold = group_width / n_folds
    for fold in range(n_folds):
        fold_d = d[d['fold']==fold].set_index('window_size').sort_index()
        fold_d.index.name = "group_size"
        fold_d = fold_d.rename_axis("group_size").reset_index()
        # Fill in any missing group sizes with zeros
        pos = fold_d.set_index('group_size')['fold_Pos'].reindex(group_sizes_plot, fill_value=0)
        neg = fold_d.set_index('group_size')['fold_Neg'].reindex(group_sizes_plot, fill_value=0)
        frac = (pos / (pos + neg)).replace([np.inf, -np.inf], np.nan).fillna(0)
        # --- DEBUG PRINT ---
        print(f"For cutoff={cutoff} fold={fold+1}:")
        print("  group_sizes:", group_sizes_plot)
        print("  pos:", list(pos.values))
        print("  neg:", list(neg.values))
        print("  frac:", list(frac.values))
        # -------------------
        offset = (fold - (n_folds-1)/2) * bar_width_fold
        bars = plt.bar(x + offset, frac, width=bar_width_fold*0.96, label=f"Fold {fold+1}", color=palette[fold])
        autolabel(plt.gca(), bars, is_frac=True)
    plt.xticks(x, group_sizes_plot)
    plt.xlabel("Group Size")
    plt.ylabel("Positive Fraction")
    plt.ylim(0, 0.5)
    plt.title(f"Positive Fraction per Fold vs Group Size (cutoff={cutoff})")
    plt.legend()
    plt.xlim(group_sizes_plot[0] - 0.5, group_sizes_plot[-1] + 0.2)  # <-- Add this line!
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"fold_posfrac_hist_per_fold_cutoff{cutoff}.png"))
    plt.close()

    # Stacked bar by group_size/cut/fold
    width = 0.12
    group_pad = 0.02
    for fold in range(n_folds):
        all_neg, all_pos, all_mix = [], [], []
        for gs in group_sizes_plot:
            folddf = d[(d["window_size"] == gs) & (d["fold"] == fold)]
            if len(folddf) == 0:
                all_neg.append(0)
                all_pos.append(0)
                all_mix.append(0)
            else:
                all_pos.append(int(folddf['fold_Pos'].iloc[0]))
                all_neg.append(int(folddf['fold_Neg'].iloc[0]))
                all_mix.append(int(folddf['fold_mixclusters'].iloc[0]))
        offset = (fold - (n_folds-1)/2) * (width + group_pad)
        plt.figure(figsize=(12,8))
        ax = plt.gca()
        bars_neg = ax.bar(x + offset, all_neg, width, label="Negatives" if fold == 0 else "", color=bar_colors['neg'], alpha=0.5, zorder=5)
        bars_pos = ax.bar(x + offset, all_pos, width, bottom=all_neg, label="Positives" if fold == 0 else "", color=bar_colors['pos'], alpha=0.8, zorder=5)
        bars_mix = ax.bar(x + offset, all_mix, width, bottom=(np.array(all_neg)+np.array(all_pos)),
                label="Mixed Clusters" if fold == 0 else "", color=bar_colors['mix'], alpha=0.7, zorder=5)
        for xi, (n, p, m) in enumerate(zip(all_neg, all_pos, all_mix)):
            if n > 0:
                ax.text(xi + offset, n/2, str(n), ha='center', va='center', color='white', fontsize=8)
            if p > 0:
                ax.text(xi + offset, n + p/2, str(p), ha='center', va='center', color='white', fontsize=8)
            if m > 0:
                ax.text(xi + offset, n + p + m/2, str(m), ha='center', va='center', color='white', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(group_sizes_plot)
        ax.set_xlabel("Group Size")
        ax.set_ylabel("Count")
        ax.set_title(f"Stacked Bar by Group Size and Fold {fold+1} (cutoff={cutoff})")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"merged_stacked_bar_fold{fold+1}_cutoff{cutoff}.png"))
        plt.close()

print(f"\nAll summary plots are saved in '{outdir}/'")