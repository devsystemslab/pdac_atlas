import os
import sys
import torch
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
import argparse

from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

from scarches.dataset.trvae.data_handling import remove_sparsity
from scarches.models.scpoli import scPoli
import scarches as sca
import scvi
from scib_metrics.benchmark import Benchmarker

import matplotlib as mpl

from matplotlib.pyplot import rc_context
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['pdf.fonttype'] = 42
sc.settings.set_figure_params(facecolor=(0,0,0,0), dpi=300)
sns.set_style({'axes.grid' : False})

sc.set_figure_params(frameon=False)
sc.set_figure_params(dpi=300)
sc.set_figure_params(figsize=(5, 5))


def laod_seurat(adata_pca, in_dir, method):
    adata_seurat = adata_pca.copy()
    pcs = pd.read_csv(f'{in_dir}/gut_{method}_integration_pcs.txt', sep='\t')
    adata_seurat.obsm['X_pca']=pcs.to_numpy()
    sc.pp.neighbors(adata_seurat)
    sc.tl.umap(adata_seurat)

    return adata_seurat

def load_data(in_dir):

    adata_pca= sc.read_h5ad(f"{in_dir}/gut_mergepy_integration.h5ad")
    adata_harmony = sc.read_h5ad(f"{in_dir}/gut_harmonypy_integration.h5ad")
    adata_bbknn = sc.read_h5ad(f"{in_dir}/gut_bbknn_integration.h5ad")
    adata_combat = sc.read_h5ad(f"{in_dir}/gut_combat_integration.h5ad")
    adata_scvi = sc.read_h5ad(f"{in_dir}/gut_scvi_integration2.h5ad")
    adata_scanvi = sc.read_h5ad(f"{in_dir}/gut_scanvi_integration2.h5ad")
    adata_scpoli = sc.read_h5ad(f"{in_dir}/gut_scpoli_integration2.h5ad")
    adata_rsspearson = sc.read_h5ad(f"{in_dir}/gut_csspearson_integration.h5ad")
    adata_rssspearman= sc.read_h5ad(f"{in_dir}/gut_cssspearman_integration.h5ad")

    adata_seurat3 = laod_seurat(adata_pca, in_dir, 'seurat3')
    adata_seurat4 = laod_seurat(adata_pca, in_dir, 'seurat4')
    adata_seurat5 = laod_seurat(adata_pca, in_dir, 'seurat5')

    return adata_pca, adata_harmony, adata_bbknn, adata_combat, \
            adata_scvi, adata_scanvi, adata_scpoli, adata_rsspearson, \
            adata_rssspearman, adata_seurat3, adata_seurat4, adata_seurat5

def plot_umap(adata_list, in_dir):

    adata_pca, adata_harmony, adata_bbknn, adata_combat, \
    adata_scvi, adata_scanvi, adata_scpoli, adata_rsspearson, \
    adata_rssspearman, adata_seurat3, adata_seurat4, adata_seurat5=adata_list

    fig, axs = plt.subplots(4, 3, figsize=(10, 13))

    sc.pl.umap(adata_harmony, color=['level_2'],frameon=False, size=5, show=False,  
            legend_loc=None, ax=axs[0,0], title='harmony')
    sc.pl.umap(adata_bbknn, color=['level_2'],frameon=False, size=5, show=False, 
            legend_loc=None, ax=axs[0,1], title='bbkn')
    sc.pl.umap(adata_combat, color=['level_2'],frameon=False, size=5, show=False, 
            legend_loc=None, ax=axs[0,2], title='combat')

    sc.pl.umap(adata_rsspearson, color=['level_2'],frameon=False, size=5, show=False, 
            legend_loc=None, ax=axs[1,0], title='rsspearson')
    sc.pl.umap(adata_rssspearman, color=['level_2'],frameon=False, size=5, show=False, 
            legend_loc=None, ax=axs[1,1], title='rssspearman')
    sc.pl.umap(adata_pca, color=['level_2'],frameon=False, size=5, show=False, 
            legend_loc=None, ax=axs[1,2], title='pca')

    sc.pl.umap(adata_scvi, color=['level_2'],frameon=False, size=5, show=False, 
            legend_loc=None, ax=axs[2,0], title='scvi')
    sc.pl.umap(adata_scanvi, color=['level_2'],frameon=False, size=5, show=False, 
            legend_loc=None, ax=axs[2,1], title='scanvi')
    sc.pl.umap(adata_scpoli, color=['level_2'],frameon=False, size=5, show=False, 
            legend_loc=None, ax=axs[2,2], title='scpoli')

    sc.pl.umap(adata_seurat3, color=['level_2'],frameon=False, size=5, show=False, 
            legend_loc=None, ax=axs[3,0], title='seurat3')
    sc.pl.umap(adata_seurat4, color=['level_2'],frameon=False, size=5, show=False, 
            legend_loc=None, ax=axs[3,1], title='seurat4')
    sc.pl.umap(adata_seurat5, color=['level_2'],frameon=False, size=5, show=False, 
            legend_loc=None, ax=axs[3,2], title='seurat5')

    plt.savefig(f"{in_dir}/bench_umap.pdf", bbox_inches="tight")

def run_scbi(adata_list, in_dir):

    adata_pca, adata_harmony, adata_bbknn, adata_combat, \
    adata_scvi, adata_scanvi, adata_scpoli, adata_rsspearson, \
    adata_rssspearman, adata_seurat3, adata_seurat4, adata_seurat5=adata_list

    adata_scanvi.obsm['X_scPoli'] = adata_scpoli.obsm['X_scPoli']
    adata_scanvi.obsm['X_harmony'] = adata_harmony.obsm['X_pca_harmony']
    adata_scanvi.obsm['X_bbknn'] = adata_bbknn.obsm['X_pca']
    adata_scanvi.obsm['X_combat'] = adata_combat.obsm['X_pca']
    adata_scanvi.obsm['X_csspearson'] = adata_rsspearson.obsm['X_corr']
    adata_scanvi.obsm['X_cssspearman'] = adata_rssspearman.obsm['X_corr']
    adata_scanvi.obsm['X_seurat3'] = adata_seurat3.obsm['X_pca']
    adata_scanvi.obsm['X_seurat4'] = adata_seurat4.obsm['X_pca']
    adata_scanvi.obsm['X_seurat5'] = adata_seurat5.obsm['X_pca']

    bm = Benchmarker(
        adata_scanvi,
        batch_key="sample_id",
        label_key="level_2",
        embedding_obsm_keys=["X_pca", "X_scVI", "X_scANVI", "X_scPoli", 
                            "X_harmony", "X_bbknn", "X_combat", 
                            "X_csspearson", "X_cssspearman", 
                            'X_seurat3','X_seurat4','X_seurat5'],
        n_jobs=-1,
    )
    bm.benchmark()

    df1 = bm.get_results(min_max_scale=False)
    df1.to_csv(f"{in_dir}/scib_results.txt", sep="\t")

    df2 = bm.get_results(min_max_scale=True)
    df2.to_csv(f"{in_dir}/scib_results_scale.txt", sep="\t")

    bench_results1 = bm.plot_results_table(min_max_scale=False, save_dir=f"{in_dir}/")
    bench_results2 = bm.plot_results_table(min_max_scale=True, save_dir=f"{in_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rep_num')

    args             = parser.parse_args()
    rep_num        = args.rep_num


    in_dir = f"/home/xuq44/projects/hgioa/data/results/inte_benchmarker/whole/rep_{rep_num}"

    adata_list = load_data(in_dir)
    plot_umap(adata_list, in_dir)
    run_scbi(adata_list, in_dir)
