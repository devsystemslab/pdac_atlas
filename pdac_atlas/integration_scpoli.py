import os
import argparse

import numpy as np
import pandas as pd
import umap
import pickle
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import scanpy as sc
import anndata
import scanpy.external as sce

from scarches.models.scpoli import scPoli


def umap_quan(adata, use_rep=None):
    if use_rep==None:
        use_rep = 'X'
        model = umap.UMAP(n_neighbors=5, random_state=1, min_dist=0.1).fit(adata.X)
    
    adata.obsm['X_umap'] = model.embedding_
    # adata.uns['umap_model'] = model

    return adata, model

def load_data(sam2publication, sub_celltype='all'):
    adata_list=[]
    for sam in sam2publication:
        adata0 = sc.read_h5ad(f'/home/xuq44/projects/scPDAC/results/annot_res/{sam2publication[sam]}/{sam}.h5ad')
        if sub_celltype!='all':
            adata0=adata0[adata0.obs.level_1==sub_celltype]
        if adata0.n_obs>1:
            adata0.obs['publication']=sam2publication[sam]
            adata0 = adata0.raw.to_adata()
            if sam2publication[sam] == 'Hwang_NatGenet_2022':
                adata0.obs = adata0.obs[['publication', 'level_1', 'level_2', 
                                         'n_genes_by_counts', 'total_counts', 'pct_counts_mt', 
                                         'Level 1 Annotation', 'Level 2 Annotation', 'Level 3 Annotation']]
            else:
                adata0.obs = adata0.obs[['publication','level_1', 'level_2', 
                                        'n_genes_by_counts', 'total_counts', 'pct_counts_mt']]
            adata0.obs['sample_id'] = sam
            adata0.var = adata0.var[[]]
            adata_list.append(adata0)

    adata_all = anndata.AnnData.concatenate(*adata_list, join='outer', fill_value=0)

    return adata_all

def clear_genes(adata):
    clear_genes = pd.read_csv("/home/xuq44/refgenomes/hg38/hg_genes_clear.txt", header=None)[0].tolist()
    malat1 = [i for i in clear_genes if i.startswith('MALAT1')]
    mito_genes = [i for i in clear_genes if i.startswith('MT-')]
    xist = [i for i in clear_genes if i.startswith('XIST')]
    clear_genes = [i for i in clear_genes if i not in malat1+mito_genes+xist]

    sub_clear_genes = [i for i in clear_genes if i in adata.var.index.tolist()]
    adata = adata[:, sub_clear_genes]
    
    return adata

def data_quity_ctrl(adata_all):
    sc.pp.normalize_total(adata_all, target_sum=1e4)
    sc.pp.log1p(adata_all)
    adata_all.raw = adata_all
    sc.pp.highly_variable_genes(adata_all, n_top_genes=3000, batch_key='batch')
    adata_all = adata_all[:,adata_all.var.highly_variable]

    adata_all.obs['level_1'] = adata_all.obs['level_1'].astype('str')
    adata_all.obs['level_2'] = adata_all.obs['level_2'].astype('str')

    return adata_all

def run_scpoli(adata_all, unknow_cts, celltype_key):
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
        adata=adata_all,
        unknown_ct_names=['nan']+unknow_cts,
        condition_key='batch',
        cell_type_keys=[celltype_key],
        embedding_dim=3,
        hidden_layer_sizes=[int(np.sqrt(adata_all.shape[0]))]
    )

    scpoli_model.train(
        n_epochs=12,
        pretraining_epochs=10,
        early_stopping_kwargs=early_stopping_kwargs,
        eta=10,
        alpha_epoch_anneal=100
    )
    adata_all.obsm["X_scPoli"] = scpoli_model.get_latent(
        adata_all.X, 
        adata_all.obs["batch"].values,
        mean=True
    )

    ref_path = f'{project_dir}/scpoli_ref/scpoli_model/'
    scpoli_model.save(ref_path, overwrite=True)

    sc.pp.neighbors(adata_all, use_rep="X_scPoli")
    # sc.tl.leiden(adata_all, resolution=10.0)
    sc.tl.umap(adata_all)

    return adata_all

def run_harmony(adata):
    sce.pp.harmony_integrate(adata, 'sample_id')
    sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_pcs=30)
    sc.tl.umap(adata)
    return adata

def run_bbknn(adata):
    adata2 = adata.raw.to_adata() 
    adata2.uns['log1p']['base']=None

    sc.pp.highly_variable_genes(adata2, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key = 'sample_id')
    var_genes_batch = adata2.var.highly_variable_nbatches > 0
    
    sce.pp.bbknn(adata2, batch_key='sample_id', n_pcs=30)  # running bbknn 1.3.6
    sc.tl.umap(adata2)

    return adata2

def run_combat(adata):
    adata_combat = sc.AnnData(X=adata.raw.X, var=adata.raw.var, obs = adata.obs)
    adata_combat.raw = adata_combat

    sc.pp.combat(adata_combat, key='sample_id')

    sc.pp.highly_variable_genes(adata_combat)
    sc.pp.pca(adata_combat, n_comps=30, use_highly_variable=True, svd_solver='arpack')
    sc.pp.neighbors(adata_combat)
    sc.tl.umap(adata_combat)

    return adata_combat

def mk_new_annot(adata, group_name, level):
    aa = adata.obs.groupby([group_name, level]).size().reset_index()
    new_clust_map={}

    for i in aa[group_name].unique():
        bb = aa[aa[group_name]==i]
        new_clust_map[i]=bb.loc[bb[0].idxmax()][level]

    adata.obs[f'{level}_late'] = adata.obs[group_name].map(new_clust_map)

def grouped_obs_mean(adata, group_key, layer=None, gene_symbols=None):
    if layer == 'raw':
        adata = adata.raw.to_adata()
        
    if layer is not None and layer != 'raw':
        getX = lambda x: x.layers[layer]
    else:
        getX = lambda x: x.X
    if gene_symbols is not None:
        new_idx = adata.var[idx]
    else:
        new_idx = adata.var_names

    grouped = adata.obs.groupby(group_key)
    out = pd.DataFrame(
        np.zeros((adata.shape[1], len(grouped)), dtype=np.float64),
        columns=list(grouped.groups.keys()),
        index=adata.var_names
    )

    for group, idx in grouped.indices.items():
        X = getX(adata[idx])
        out[group] = np.ravel(X.mean(axis=0, dtype=np.float64))
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir')
    parser.add_argument('--sub_celltype')
    parser.add_argument('--inti_method')
    parser.add_argument('--celltype_key')
    parser.add_argument('--sample_file')
    parser.add_argument('--tumor_only', action="store_true", default=False)
    parser.add_argument('--pancreas_only', action="store_true", default=False)
    parser.add_argument('--use_late', action="store_true", default=False)
    parser.add_argument('--peng_only', action="store_true", default=False)

    args               = parser.parse_args()
    out_dir            = args.out_dir
    sub_celltype       = args.sub_celltype
    inti_method        = args.inti_method
    celltype_key       = args.celltype_key
    sample_file        = args.sample_file
    tumor_only         = args.tumor_only
    pancreas_only      = args.pancreas_only
    use_late           = args.use_late
    peng_only          = args.peng_only

    project_dir0 = f"/home/xuq44/projects/scPDAC/results/{out_dir}"
    project_dir = f"/home/xuq44/projects/scPDAC/results/{out_dir}/{sub_celltype}"
    os.system(f'mkdir -pv {project_dir}')

    sample_table = pd.read_csv(f"/home/xuq44/projects/scPDAC/data/db/{sample_file}", 
                                sep='\t', comment='#', dtype=str).set_index(["sample_id"], 
                                drop=False)
    
    if tumor_only:
        sample_table = sample_table[sample_table.Type=='Tumor']
    if pancreas_only:
        sample_table = sample_table[sample_table.Tissue=='Pancreas']
    if peng_only:
        sample_table = sample_table[sample_table.publication=='Peng_CellRes_2019']

    sam2dir = sample_table.to_dict()['proj_dir']
    sam2publication = sample_table.to_dict()['publication']


    if sub_celltype=='all':

        adata = sc.read_h5ad(f"{project_dir}/pdac_raw.h5ad")
        adata.uns['log1p']["base"] = None
        adata.raw = adata
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, batch_key='batch')
        adata = adata[:,adata.var.highly_variable]

        adata = run_scpoli(adata, ['epithelial','mesenchyme'], celltype_key)
        adata.write_h5ad(f'{project_dir}/pdac_scpoli.h5ad', compression="gzip")
        sc.tl.leiden(adata, resolution=10)
        mk_new_annot(adata, 'leiden', 'level_1')
        mk_new_annot(adata, 'leiden', 'level_2')
        adata.write_h5ad(f'{project_dir}/pdac_scpoli_reannot.h5ad', compression="gzip")
        
        with plt.rc_context({"figure.figsize": (5, 5), "figure.dpi": (600)}):
            sc.pl.umap(adata, color=[
                'publication', 'level_1', 'level_2', 'level_1_late', 'level_2_late'], 
               wspace=0.6, show=False, size=2, frameon=False, ncols=2)
            plt.savefig(f"{project_dir}/pdac_scpoli_reannot_umap.pdf", bbox_inches="tight")

    else:
        if use_late:
            adata = sc.read_h5ad(f'{project_dir0}/all/pdac_scpoli_reannot_clear.h5ad')
            if sub_celltype == 'epi_mes':
                adata=adata[adata.obs.level_1_late.isin([sub_celltype, 'epithelial','mesenchyme'])]
            else:
                adata=adata[adata.obs.level_1_late==sub_celltype]
        else:
            adata = load_data(sam2publication, sub_celltype=sub_celltype)
            adata = clear_genes(adata)
            adata = data_quity_ctrl(adata)

        if inti_method == 'scpoli':
            adata = run_scpoli(adata, [sub_celltype], celltype_key)

        if inti_method == 'harmonypy':
            adata = run_harmony(adata)

        if inti_method == 'bbknn':
            adata = run_bbknn(adata)

        if inti_method == 'combat':
            adata = run_combat(adata)

        if use_late:
            adata.write_h5ad(f"{project_dir}/pdac_{inti_method}_late.h5ad", compression="gzip")
        else:
            adata.write_h5ad(f"{project_dir}/pdac_{inti_method}_raw.h5ad", compression="gzip")

