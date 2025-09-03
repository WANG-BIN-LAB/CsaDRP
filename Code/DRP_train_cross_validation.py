import copy
import csv
import shutil
import os
import random

import numpy as np
import pandas as pd
import torch
import math
import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.optim import lr_scheduler
from tqdm import tqdm
from sklearn.preprocessing import normalize, scale
from models.ResGit import ResGit
import torch.nn.utils.prune as prune

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV



import warnings
warnings.filterwarnings("ignore")

'''
    This script is to predict drug sensitivity using the hidden representation of ResGit, and it used
    cross-validation which training-testing dataset is same with GIT-RINN.
'''












class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
































'''
    print cpu information and set random seed
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Use device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(random_seed)

'''
    set hyperparameters
'''

permute_labels = False


'''
    load data
'''
Drug_sensi_df = pd.read_csv("/data2/zyt/ResGitDR-main/ResGitDR_data/data/data/GDSC1_drug_response_binaryAUC.csv",index_col=0)
Drug_sensi_df.index = Drug_sensi_df.index.astype(str)
Drug_sensi_df_copy = Drug_sensi_df.copy()
print(Drug_sensi_df.shape)


if permute_labels == True:
    Drug_sensi_df = Drug_sensi_df.sample(frac=1).reset_index(drop=True)
    Drug_sensi_df = Drug_sensi_df.set_axis(Drug_sensi_df_copy.index)
    print(Drug_sensi_df)


gdsc_mut_df = pd.read_csv("/data2/zyt/ResGitDR-main/ResGitDR_data/data/data/GDSC_sga_data.csv",index_col=0)
gdsc_mut_df.index = gdsc_mut_df.index.astype(str)
#common sample between drug sensitivity file and sga file
common_samples_list = [sample for sample in Drug_sensi_df.index if sample in gdsc_mut_df.index]
#select sga gene list
sga_gene_list = pd.read_csv("/data2/zyt/ResGitDR-main/ResGitDR_data/data/data/TCI_fondation1_driver_dataset_combine_gene_list.txt",header=None,names=["gene_name"])
gdsc_mut_df = gdsc_mut_df.loc[common_samples_list,sga_gene_list["gene_name"].values]
gdsc_mut_df.columns = ["sga_"+ gene for gene in gdsc_mut_df.columns]

'''
    load hyperparameter tuning results for Elastic Net
'''

hyperparameter_file_dir = "/data2/zyt/ResGitDR-main/ResGitDR_data/data/data/hyperparameters_tuning_result.csv"
hyperparameter_file = pd.read_csv(hyperparameter_file_dir,index_col=0)

'''
    load hidden representation data for each cell line
'''
###load hidden Para
data_dir= "/data2/zyt/ResGitDR-main/shiyan/NEWRESMamba2KANNOCANbatch901/cross_validation/ResGit/GDSC/"

'''
    store result
'''
res_dir = "/data2/zyt/ResGitDR-main/shiyan/NEWRESMamba2KANNOCANbatch901/cross_validation/NEWDR/"

res_headers = ["drug", "sga_hiddens_f1"," sga_hiddens_auroc"]



#This function is to get features for cell lines
def create_features(train_hidden_dir,phase):
    train_features = {}
    ResGit_preds = pd.read_csv(train_hidden_dir + "/" + phase+ "/resultspredicts.csv", index_col=0)
    ResGit_preds.index = ResGit_preds.index.astype(str)
    GDSC_sample_index = []
    for sample_name in ResGit_preds.index.to_list():
        if "TCGA" not in str(sample_name):
            GDSC_sample_index.append(sample_name)
    curr_sample_name = list(set(GDSC_sample_index) &set(common_samples_list))

    train_features ["sga"] = gdsc_mut_df.loc[curr_sample_name]
    train_combined_features_df_list = []
    for idx in range(len(os.listdir(train_hidden_dir+ "/" + phase + "/hidden/"))):
        f = "hidden_outs_" + str(idx) + ".csv"
        prefix = "hidden_" + str(idx) + "_"
        curr_df = pd.read_csv(train_hidden_dir + "/" + phase + "/hidden/" + f, index_col=0)
        curr_df.index = curr_df.index.astype(str)
        curr_df = curr_df.loc[curr_sample_name]
        cols = [prefix + str(i) for i in range(curr_df.shape[1])]
        curr_df.columns = cols
        train_features [f] = curr_df
        train_combined_features_df_list.append(curr_df)
    train_features ["all_hiddens"] = pd.concat(train_combined_features_df_list,axis=1)
    train_features ["sga_hiddens"] = pd.concat([train_features["all_hiddens"],train_features["sga"]],axis=1)
    return train_features

feature = "sga_hiddens"




# cross validation
for fold_dir in os.listdir(data_dir):
    print("fold_dir",fold_dir)
    #make fold to store the result of drug sensitivity prediction
    train_res_file = res_dir + "/" + fold_dir + "/train.csv"
    test_res_file = res_dir + "/" + fold_dir + "/test.csv"
    # parameters_file = res_dir + "/" + fold_dir+ "/parameters"

    if os.path.exists(res_dir + "/" + fold_dir):
        shutil.rmtree(res_dir + "/" + fold_dir)

    os.makedirs(res_dir + "/" + fold_dir)
    with open(train_res_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(res_headers)
    with open(test_res_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(res_headers)

    #load hidden represention from ResGit
    train_hidden_dir = data_dir + fold_dir
    test_hidden_dir = data_dir + fold_dir

    train_features = create_features(train_hidden_dir,"train")
    test_features = create_features(test_hidden_dir,"test")
    print(test_features)


    '''
        Only select the data of the interested feature
    '''
    train_features = {key: train_features[key] for key in train_features.keys()}
    test_features = {key: test_features[key] for key in test_features.keys()}
    # print("curr_feature",test_features)


    train_preds = np.empty((train_features[feature].shape[0], Drug_sensi_df.shape[1]))
    train_preds[:] = np.nan

    test_preds = np.empty((test_features[feature].shape[0], Drug_sensi_df.shape[1]))
    test_preds[:] = np.nan

    train_preds = pd.DataFrame(train_preds, index=train_features[feature].index, columns=Drug_sensi_df.columns)

    train_targets = Drug_sensi_df.loc[train_preds.index, :]

    test_preds = pd.DataFrame(test_preds, index=test_features[feature].index, columns=Drug_sensi_df.columns)
    # print("test_preds",test_preds.index)
    test_targets = Drug_sensi_df.loc[test_preds.index, :]

    train_targets.to_csv(train_res_file.replace('train.csv', 'train_targets.csv'))
    test_targets.to_csv(test_res_file.replace('test.csv', 'test_targets.csv'))

    for i in range(0, Drug_sensi_df.shape[1]):
    # for i in range(0, 2): 
        col = Drug_sensi_df.iloc[:, i]  # current col
        drug = col.name
        print("Currnt drug ID: ", drug)

        nan_indicies = col.index[col.apply(np.isnan)]
        labeled_indicies = col.index[~col.apply(np.isnan)]  # remove rows with 'nan' values
        drug_labeled = col[labeled_indicies]

        res_dict = {"train" : {}, "test" : {}}
        for layer_file_name, hidden_rep_df in train_features.items():
            curr_train = pd.concat((hidden_rep_df, drug_labeled), axis=1, join='inner')
            if curr_train.shape[0] == 0:
                continue
            curr_test = pd.concat((test_features[layer_file_name], drug_labeled), axis=1, join='inner')


             # Prepare CNN input data

            

            #using "ElasticNet" tunned hyperparameters:
            C = hyperparameter_file.iloc[i,1]
            l1_ratio = hyperparameter_file.iloc[i,2]
            #print(C,l1_ratio)
            model = LogisticRegression(penalty = 'elasticnet', solver = 'saga',C=C, l1_ratio = l1_ratio)
            model.fit(curr_train.iloc[:,:-1], curr_train.iloc[:, -1])
            #model.fit(train_combined,drug_labeled)
            train_pred_prob = model.predict_proba(curr_train.iloc[:,:-1])[:,1]
            #train_pred_prob = model.predict_proba(train_combined.iloc[:,:-1])[:,1]
            #test_pred_prob = model.predict_proba(test_combined.iloc[:,:-1])[:,1]
            test_pred_prob = model.predict_proba(curr_test.iloc[:,:-1])[:,1]
            # print("test_pred_prob",test_pred_prob)






            train_preds.loc[curr_train.index, drug] = train_pred_prob
            test_preds.loc[curr_test.index, drug] = test_pred_prob
            #train_preds.loc[train_features[feature].index, drug] = train_pred_prob
            #test_preds.loc[test_features[feature].index, drug] = test_pred_prob


            # if i == 0:
            #     with open(os.path.join(parameters_file,layer_file_name),'a') as f:
            #         writer = csv.writer(f)
            #         writer.writerow(["drug_id"] + hidden_rep_df.columns.to_list())
            # with open(os.path.join(parameters_file,layer_file_name),'a') as f:
            #     writer = csv.writer(f)
            #     writer.writerow([drug] + list(model.coef_[0]))


            try:
                train_auroc = round(roc_auc_score(curr_train.iloc[:, -1],train_pred_prob),3)
                # train_auroc = round(roc_auc_score(drug_labeled, train_pred_prob), 3)
            except:
                train_auroc = 0.5

            try:
                test_auroc = round(roc_auc_score(curr_test.iloc[:, -1], test_pred_prob),3)
                #test_auroc = round(roc_auc_score(test_targets[drug], test_pred_prob), 3)
            except:
                test_auroc = 0.5

            print(f"train_auroc: {train_auroc}, test_auroc: {test_auroc}")
            res_dict['train'][layer_file_name] = {'auroc':train_auroc}
            res_dict['test'][layer_file_name] = {'auroc':test_auroc}

        train_preds.to_csv(train_res_file.replace('train.csv', 'train_preds.csv'))
        test_preds.to_csv(test_res_file.replace('test.csv', 'test_preds.csv'))

        # write train results
        with open(train_res_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([drug,res_dict['train']['sga']['auroc'],
                             res_dict['train']['hidden_outs_0.csv']['auroc'],
                             res_dict['train']['hidden_outs_1.csv']['auroc'],
                             res_dict['train']['hidden_outs_2.csv']['auroc'],
                             res_dict['train']['hidden_outs_3.csv']['auroc'],
                             res_dict['train']['all_hiddens']['auroc'],
                             res_dict['train']['sga_hiddens']['auroc']])

        # write test results
        with open(test_res_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([drug,res_dict['test']['sga']['auroc'],
                             res_dict['test']['hidden_outs_0.csv']['auroc'],
                             res_dict['test']['hidden_outs_1.csv']['auroc'],
                             res_dict['test']['hidden_outs_2.csv']['auroc'],
                             res_dict['test']['hidden_outs_3.csv']['auroc'],
                             res_dict['test']['all_hiddens']['auroc'],
                             res_dict['test']['sga_hiddens']['auroc']])
