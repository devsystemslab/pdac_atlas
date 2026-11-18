import os
import sys
import argparse

import umap
import pickle

import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import scvi
import anndata

import scanpy.external as sce
from scvi.model.utils import mde

from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

from scarches.dataset.trvae.data_handling import remove_sparsity
from scarches.models.scpoli import scPoli

sys.path.append('/home/xuq44/git/')
import util_single_cell.scripts.py_util as qsc


def read_samples(project_dir, sample_sheet, annot_dir):
    adata_list=[]
    for sample in sample_sheet.sample_id:
        sample_file=f"{project_dir}/h5ad_pyraw/{sample}_pyraw.h5ad"
        adata0 = sc.read_h5ad(sample_file)
        adata0.obs['sample_id'] = sample
        adata0.var = adata0.var.drop(adata0.var.columns, axis=1)
        # adata0 = cc_reg(adata0, ccgenes_file)
        meta_file = f"{annot_dir}/{sample}_annotation.txt"
        if os.path.exists(meta_file) and os.path.getsize(meta_file)>1:
            meta_data = pd.read_csv(meta_file, sep="\t", index_col=0)
            adata0.obs = pd.merge(adata0.obs, meta_data, left_index=True, right_index=True)
        else:
            adata0.obs['level_1'] = 'epithelial'
            adata0.obs['level_2'] = 'na'

        adata_list.append(adata0)
    adata = anndata.AnnData.concatenate(*adata_list, join='outer', fill_value=0)
    del adata_list
    adata.obs['publication'] = ['_'.join(i.split('_')[:3]) for i in adata.obs.sample_id.tolist()]

    adata.obs.index.name = "cells"
    
    return adata
    
def clear_genes(adata):
    clear_genes = pd.read_csv("/home/xuq44/refgenomes/hg38/hg_genes_clear.txt", header=None)[0].tolist()
    sub_clear_genes = [i for i in clear_genes if i in adata.var.index.tolist()]
    adata = adata[:, sub_clear_genes]
    
    return adata

def cc_reg(adata, ccgenes_file):
    cell_cycle_genes = [x.strip() for x in open(ccgenes_file)]
    s_genes = cell_cycle_genes[:43]
    g2m_genes = cell_cycle_genes[43:]
    cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    sc.pp.regress_out(adata, ['S_score', 'G2M_score'])
    return adata

def pre_inti(adata):

    adata.raw = adata
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, batch_key='sample_id')
    adata = adata[:,adata.var.highly_variable]

    return adata

def run_merge(adata):
    sc.pp.neighbors(adata, n_pcs=30)
    sc.tl.umap(adata)
    return adata

def run_scvi(adata, method='scANVI'):
    adata = adata.copy()
    scvi.model.SCVI.setup_anndata(adata, layer='counts', batch_key="sample_id")
    vae = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
    vae.train()
    adata.obsm["X_scVI"] = vae.get_latent_representation()

    if method == 'scVI':
        sc.pp.neighbors(adata, use_rep="X_scVI")
        sc.tl.leiden(adata)
        sc.tl.umap(adata)

        return adata

    else:
        lvae = scvi.model.SCANVI.from_scvi_model(
            vae,
            adata=adata,
            labels_key='level_2',
            unlabeled_category="na",
        )
        lvae.train(max_epochs=20, n_samples_per_label=100)
        
        adata.obsm["X_scANVI"] = lvae.get_latent_representation(adata)

        sc.pp.neighbors(adata, use_rep="X_scANVI")
        sc.tl.leiden(adata)
        sc.tl.umap(adata)

        return adata

def run_scvi_one(adata):
    adata = adata.copy()
    scvi.model.SCVI.setup_anndata(adata, layer='counts', batch_key="sample_id")
    vae = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
    vae.train()
    adata.obsm["X_scVI"] = vae.get_latent_representation()

    adata2 = adata.copy()

    sc.pp.neighbors(adata, use_rep="X_scVI")
    sc.tl.leiden(adata)
    sc.tl.umap(adata)

    lvae = scvi.model.SCANVI.from_scvi_model(
        vae,
        adata=adata2,
        labels_key='level_2',
        unlabeled_category="na",
    )
    lvae.train(max_epochs=20, n_samples_per_label=100)
    
    adata2.obsm["X_scANVI"] = lvae.get_latent_representation(adata2)

    sc.pp.neighbors(adata2, use_rep="X_scANVI")
    sc.tl.leiden(adata2)
    sc.tl.umap(adata2)

    return adata, adata2

def run_scpoli(adata):
    early_stopping_kwargs = {
        "early_stopping_metric": "val_prototype_loss",
        "mode": "min",
        "threshold": 0,
        "patience": 20,
        "reduce_lr": True,
        "lr_patience": 13,
        "lr_factor": 0.1,
    }

    scpoli_model = scPoli(
        adata=adata,
        unknown_ct_names=['na'],
        condition_key='sample_id',
        cell_type_keys=['level_1', 'level_2'],
        embedding_dim=3,
        hidden_layer_sizes=[int(np.sqrt(adata.shape[0]))]
    )

    scpoli_model.train(
        n_epochs=5,
        pretraining_epochs=4,
        early_stopping_kwargs=early_stopping_kwargs,
        eta=10,
        alpha_epoch_anneal=100
    )

    adata.obsm["X_scPoli"] = scpoli_model.get_latent(
        adata.X, 
        adata.obs["sample_id"].values,
        mean=True
    )
    sc.pp.neighbors(adata, use_rep="X_scPoli")
    sc.tl.leiden(adata)
    sc.tl.umap(adata)

    return scpoli_model, adata

def mk_new_annot(adata, group_name, level):
    aa = adata.obs.groupby([group_name, level]).size().reset_index()
    new_clust_map={}

    for i in aa[group_name].unique():
        bb = aa[aa[group_name]==i]
        new_clust_map[i]=bb.loc[bb[0].idxmax()][level]

    adata.obs[f'{level}_late'] = adata.obs[group_name].map(new_clust_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--inti_method', required=True, help='intigration method')
    parser.add_argument('-p', '--project_dir', default="/home/xuq44/projects/scPDAC/results/run_20241130_scp/all")
    o = parser.parse_args()
    inti_method = o.inti_method
    project_dir = o.project_dir

    adata = sc.read_h5ad(f"{project_dir}/pdac_raw.h5ad")
    adata.uns['log1p']["base"] = None

    if o.inti_method == 'scanvi':
        adata = pre_inti(adata)
        adata1, adata2 = run_scvi_one(adata)
        adata1.obs = adata1.obs[['sample_id', 'level_1', 'level_2', 'publication']]
        adata2.obs = adata2.obs[['sample_id', 'level_1', 'level_2', 'publication']]
        adata1.write_h5ad(f"{project_dir}/pdac_scvi_integration.h5ad", compression="gzip")
        adata2.write_h5ad(f"{project_dir}/pdac_scanvi_integration.h5ad", compression="gzip")

