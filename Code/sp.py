import pandas as pd

## Spearman 
tcga_spearman = {}
gdsc_spearman = {}

for fold in range(5):
    print(f"======================================={fold}_th fold========================================")


    pre = pd.read_csv(f"/data2/zyt/ResGitDR-main/shiyan/NEWRESMamba2Linearbatch901/cross_validation/ResGit/TCGA/fold_{fold}/test/resultspredicts.csv", index_col=0)
    tar = pd.read_csv(f"/data2/zyt/ResGitDR-main/shiyan/NEWRESMamba2Linearbatch901/cross_validation/ResGit/TCGA/fold_{fold}/test/resultstargets.csv", index_col=0)

    ## TCGA 
    tcga_sga_df = pd.read_csv("/data2/zyt/ResGitDR-main/ResGitDR_data/data/data/TCGA_sga_data.csv", index_col=0)
    tcga_RNAseq_df = pd.read_csv("/data2/zyt/ResGitDR-main/ResGitDR_data/data/data/TCGA_RNAseq_data.csv", index_col=0)
    tcga_RNAseq_df[tcga_RNAseq_df < 0] = 0
    tcga_can_df = pd.read_csv("/data2/zyt/ResGitDR-main/ResGitDR_data/data/data/TCGA_xena_map_cancertype.csv", index_col=0)

    tcga_RNAseq_df = tcga_RNAseq_df.loc[tcga_sga_df.index, :]
    tcga_can_df = tcga_can_df.loc[tcga_sga_df.index, ["cancer_type"]]

    ##  GDSC 
    gdsc_sga_df = pd.read_csv("/data2/zyt/ResGitDR-main/ResGitDR_data/data/data/GDSC_sga_data.csv", index_col=0)
    gdsc_RNAseq_df = pd.read_csv("/data2/zyt/ResGitDR-main/ResGitDR_data/data/data/GDSC_RNAseq_data.csv", index_col=0)
    gdsc_RNAseq_df[gdsc_RNAseq_df < 0] = 0
    gdsc_can_df = pd.read_csv("/data2/zyt/ResGitDR-main/ResGitDR_data/data/data/GDSC_cancer_type.csv", index_col=0)

    gdsc_RNAseq_df = gdsc_RNAseq_df.loc[gdsc_sga_df.index, :]
    gdsc_can_df = gdsc_can_df.loc[gdsc_sga_df.index,["cancer_type"]]

    tcga_can_df.index = tcga_can_df.index.astype(str)
    gdsc_can_df.index = gdsc_can_df.index.astype(str)

    tcga_cell = list(tcga_can_df.index)
    gdsc_cell = list(gdsc_can_df.index)
    pre.index = pre.index.astype(str)
    tar.index = tar.index.astype(str)

    print("")
    for j in range(len(pre.index)):
        p = pre.iloc[j]
        t = tar.iloc[j]
        data = {'p': p, 't': t}
        df = pd.DataFrame(data)
        spearman_corr = df.corr(method='spearman')
        corr_value = spearman_corr.loc['t', 'p']

        #  TCGA  Spearman 
        if p.name in tcga_cell:
            cancer_type = tcga_can_df.loc[p.name, 'cancer_type']
            if cancer_type in tcga_spearman:
                tcga_spearman[cancer_type].append(corr_value)
            else:
                tcga_spearman[cancer_type] = [corr_value]

        #  GDSC Spearman 
        if p.name in gdsc_cell:
            cancer_type = gdsc_can_df.loc[p.name, 'cancer_type']
            if cancer_type in gdsc_spearman:
                gdsc_spearman[cancer_type].append(corr_value)
            else:
                gdsc_spearman[cancer_type] = [corr_value]

##  TCGA 
TCGA_Spearman = pd.DataFrame.from_dict(tcga_spearman, orient='index').T
TCGA_Spearman = TCGA_Spearman.sort_index(axis=1)
TCGA_Spearman.to_csv("/data2/zyt/ResGitDR-main/shiyan/NEWRESMamba2KANNOCANbatch901/cross_validation/TCGA_spearman.csv", index=None)

##  GDSC 
#GDSC_Spearman = pd.DataFrame.from_dict(gdsc_spearman, orient='index').T
#GDSC_Spearman = GDSC_Spearman.sort_index(axis=1)
#GDSC_Spearman.to_csv("/data2/zyt/ResGitDR-main/shiyan/NEWRESMamba2KANNOCANbatch901/cross_validation/GDSC_spearman.csv", index=None)