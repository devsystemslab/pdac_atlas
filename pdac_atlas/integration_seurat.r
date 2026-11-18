library(Seurat)


project_dir="/home/xuq44/projects/scPDAC/results/run_20241130_scp/all"


load_data <- function(){
    mtx = Read10X(paste0(project_dir, '/sub_counts'))
    seurat = CreateSeuratObject(mtx)

    meta.data = read.table(paste0(project_dir, '/sub_counts/barcodes.tsv.gz'), sep='\t', row.names=1)

    seurat@meta.data$publication <- meta.data$V2
    seurat@meta.data$sample_id <- meta.data$V8
    seurat@meta.data$level_1 <- meta.data$V3
    seurat@meta.data$level_2 <- meta.data$V4

    seurat
}

inte_seurat_v34 <- function(seurat, method){
    seurat <- NormalizeData(seurat)
    obj.list <- SplitObject(seurat, split.by = "sample_id")
    anchors <- FindIntegrationAnchors(object.list = obj.list, dims = 1:30)
    saveRDS(object = seurat, file = paste0(project_dir, '/pdac_', method, '_integration_pcs.rds'))
    seurat <- IntegrateData(anchors, dims = 1:30, k.weight = 50)
    # seurat <- IntegrateData(anchors, dims = 1:30, k.weight = 50) seurat3 use dims=1:10
    seurat <- ScaleData(seurat)

    seurat <- RunPCA(seurat, npcs = 50)

    write.table(seurat@reductions$pca@cell.embeddings, 
                paste0(project_dir, '/pdac_', method, '_integration_pcs2.txt'), 
                quote = FALSE, sep='\t')
}

inte_seurat_v5 <- function(seurat, method){
    seurat[["RNA"]] <- split(seurat[["RNA"]], f = seurat$sample_id)

    seurat <- NormalizeData(seurat)
    seurat <- FindVariableFeatures(seurat)
    seurat <- ScaleData(seurat)
    seurat <- RunPCA(seurat)

    seurat <- IntegrateLayers(object = seurat, method = CCAIntegration, 
                            orig.reduction = "pca", new.reduction = "integrated.cca",
                            verbose = FALSE)

    write.table(seurat@reductions$integrated.cca@cell.embeddings, 
                paste0(project_dir, '/pdac_', method, '_integration_pcs2.txt'), 
                quote = FALSE, sep='\t')
}

args = commandArgs(trailingOnly=TRUE)

method <- args[1]

seurat <- load_data()
if (method=='seurat3'){
    inte_seurat_v34(seurat, method)
} else if (method=='seurat4'){
    inte_seurat_v34(seurat, method)
} else if (method=='seurat5'){
    inte_seurat_v34(seurat, method)
}
