import pandas as pd
import numpy as np

tcga_pearson = {}
tcga_rmse = {}
gdsc_pearson = {}
gdsc_rmse = {}

for fold in range(5):
    print(f"======================================={fold}_th fold========================================")
    
    # load data
    pre = pd.read_csv(f"/data2/zyt/ResGitDR-main/shiyan/NEWRESMamba2Linearbatch901/cross_validation/ResGit/TCGA/fold_{fold}/test/resultspredicts.csv", index_col=0)
    tar = pd.read_csv(f"/data2/zyt/ResGitDR-main/shiyan/NEWRESMamba2Linearbatch901/cross_validation/ResGit/TCGA/fold_{fold}/test/resultstargets.csv", index_col=0)
    
    #  TCGA 
    tcga_sga_df = pd.read_csv("/data2/zyt/ResGitDR-main/ResGitDR_data/data/data/TCGA_sga_data.csv", index_col=0)
    tcga_RNAseq_df = pd.read_csv("/data2/zyt/ResGitDR-main/ResGitDR_data/data/data/TCGA_RNAseq_data.csv", index_col=0)
    tcga_RNAseq_df[tcga_RNAseq_df < 0] = 0
    tcga_can_df = pd.read_csv("/data2/zyt/ResGitDR-main/ResGitDR_data/data/data/TCGA_xena_map_cancertype.csv", index_col=0)
    tcga_RNAseq_df = tcga_RNAseq_df.loc[tcga_sga_df.index, :]
    tcga_can_df = tcga_can_df.loc[tcga_sga_df.index, ["cancer_type"]]
    
    #  GDSC 
    gdsc_sga_df = pd.read_csv("/data2/zyt/ResGitDR-main/ResGitDR_data/data/data/GDSC_sga_data.csv", index_col=0)
    gdsc_RNAseq_df = pd.read_csv("/data2/zyt/ResGitDR-main/ResGitDR_data/data/data/GDSC_RNAseq_data.csv", index_col=0)
    gdsc_RNAseq_df[gdsc_RNAseq_df < 0] = 0
    gdsc_can_df = pd.read_csv("/data2/zyt/ResGitDR-main/ResGitDR_data/data/data/GDSC_cancer_type.csv", index_col=0)
    gdsc_RNAseq_df = gdsc_RNAseq_df.loc[gdsc_sga_df.index, :]
    gdsc_can_df = gdsc_can_df.loc[gdsc_sga_df.index, :]

    tcga_can_df.index = tcga_can_df.index.astype(str)
    gdsc_can_df.index = gdsc_can_df.index.astype(str)
    
    # combine TCGA  GDSC 
    sga_df = pd.concat([tcga_sga_df, gdsc_sga_df])
    deg_df = pd.concat([tcga_RNAseq_df, gdsc_RNAseq_df])
    can_df = pd.concat([tcga_can_df, gdsc_can_df])
    
    tcga_cell = list(tcga_can_df.index)
    gdsc_cell = list(gdsc_can_df.index)
    
    pre.index = pre.index.astype(str)
    tar.index = tar.index.astype(str)

    for idx in range(len(pre.index)):
        p = pre.iloc[idx]
        t = tar.iloc[idx]
        
        data = {'p': p, 't': t}
        df = pd.DataFrame(data)
        pearson_corr = df.corr(method='pearson')
        corr_value = pearson_corr.loc['t', 'p']
        
        # RMSE
        rmse_value = np.sqrt(((p - t) ** 2).mean())
        
        if p.name in tcga_cell:
            cancer_type = tcga_can_df.loc[p.name, 'cancer_type']
            if cancer_type in tcga_pearson:
                tcga_pearson[cancer_type].append(corr_value)
                tcga_rmse[cancer_type].append(rmse_value)
            else:
                tcga_pearson[cancer_type] = [corr_value]
                tcga_rmse[cancer_type] = [rmse_value]
        if p.name in gdsc_cell:
            cancer_type = gdsc_can_df.loc[p.name, 'cancer_type']
            if cancer_type in gdsc_pearson:
                gdsc_pearson[cancer_type].append(corr_value)
                gdsc_rmse[cancer_type].append(rmse_value)
            else:
                gdsc_pearson[cancer_type] = [corr_value]
                gdsc_rmse[cancer_type] = [rmse_value]


#TCGA_Pearson = pd.DataFrame.from_dict(gdsc_pearson, orient='index').T.sort_index(axis=1)
#TCGA_Pearson.to_csv('/data2/zyt/ResGitDR-main/shiyan/NEWRESMamba2KANNOCANbatch901/cross_validation/GDSCPearson.csv', index=None)

#TCGA_RMSE = pd.DataFrame.from_dict(gdsc_rmse, orient='index').T.sort_index(axis=1)
#TCGA_RMSE.to_csv('/data2/zyt/ResGitDR-main/shiyan/NEWRESMamba2KANNOCANbatch901/cross_validation/GDSCRMSE.csv', index=None)


TCGA_Pearson = pd.DataFrame.from_dict(tcga_pearson, orient='index').T.sort_index(axis=1)
TCGA_Pearson.to_csv('/data2/zyt/ResGitDR-main/shiyan/NEWRESMamba2KANNOCANbatch901/cross_validation/TCGAPearson.csv', index=None)

TCGA_RMSE = pd.DataFrame.from_dict(tcga_rmse, orient='index').T.sort_index(axis=1)
TCGA_RMSE.to_csv('/data2/zyt/ResGitDR-main/shiyan/NEWRESMamba2KANNOCANbatch901/cross_validation/TCGARMSE.csv', index=None)

