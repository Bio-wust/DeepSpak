import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score

from mymodel.network import GCFAggMVC
from mymodel.loss_test import Loss
import numpy as np
import argparse
import random
import scanpy as sc
from mymodel.preprocess_3M import adjacent_matrix_preprocessing
import os
from mymodel.utils import clustering, read_list_from_file
from torch.cuda.amp import GradScaler

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class SpatialOmicsTrainer:
    def __init__(self, args, data):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.walk_steps = args.walk_steps
        self.alpha = args.alpha
        
        self.adata_omics1 = data['adata_omics1']
        self.adata_omics2 = data['adata_omics2']
        self.adata_omics3 = data.get('adata_omics3', None)
        
        self.is_three_modal = self.adata_omics3 is not None
        
        print(f"Training mode: {'Three-modal' if self.is_three_modal else 'Dual-modal'}")
        
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2, self.adata_omics3)

        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)
        
        if self.is_three_modal:
            self.adj_spatial_omics3 = self.adj['adj_spatial_omics3'].to(self.device)
            self.adj_feature_omics3 = self.adj['adj_feature_omics3'].to(self.device)
        else:
            self.adj_spatial_omics3 = self.adj_spatial_omics2
            self.adj_feature_omics3 = self.adj_feature_omics2

        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        
        if self.is_three_modal:
            self.features_omics3 = torch.FloatTensor(self.adata_omics3.obsm['feat'].copy()).to(self.device)
        else:
            self.features_omics3 = self.features_omics2

        input_dims = [
            self.features_omics1.shape[1], 
            self.features_omics2.shape[1], 
            self.features_omics3.shape[1]
        ]
        self.model = GCFAggMVC(
            view=3,  
            input_size=input_dims,
            low_feature_dim=args.low_feature_dim,
            high_feature_dim=args.high_feature_dim,
            device=self.device
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + 
            list(self.model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        self.criterion = Loss(args.batch_size, args.temperature_f, self.device, args.walk_steps, args.alpha).to(self.device)
        self.mse = torch.nn.MSELoss()


    def pre_train(self, epoch):
        """Pre-training stage: mainly optimizes reconstruction loss"""
        self.model.train()
        tot_loss = 0.

        xs = [self.features_omics1, self.features_omics2, self.features_omics3]
        feature_adj = [self.adj_feature_omics1, self.adj_feature_omics2, self.adj_feature_omics3]
        spatial_adj = [self.adj_spatial_omics1, self.adj_spatial_omics2, self.adj_spatial_omics3]
        
        self.optimizer.zero_grad()
        xrs, _, _ = self.model(xs, feature_adj, spatial_adj)

        loss_list = []
        for v in range(3):
            loss_list.append(self.mse(xs[v], xrs[v]))
        loss = sum(loss_list)

        loss.backward()
        self.optimizer.step()
        tot_loss += loss.item()
        print('Pre-train Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss))

    def fine_tune(self, epoch):
        """Fine-tuning stage: combines reconstruction loss and structure-guided contrastive loss"""
        tot_loss = 0.

        self.model.train()

        xs = [self.features_omics1, self.features_omics2, self.features_omics3]
        feature_adj = [self.adj_feature_omics1, self.adj_feature_omics2, self.adj_feature_omics3]
        spatial_adj = [self.adj_spatial_omics1, self.adj_spatial_omics2, self.adj_spatial_omics3]
        
        self.optimizer.zero_grad()
        xrs, fused_features, hs = self.model(xs, feature_adj, spatial_adj)
        commonz, S = self.model.GCFAgg(xs, fused_features)


        loss_list = []
        
        
        for v in range(3):
            loss_list.append(self.criterion.Structure_guided_Contrastive_Loss(
                hs[v], 
                commonz, 
                S
            ))
            loss_list.append(self.mse(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        self.optimizer.step()
        tot_loss += loss.item()
        print('Fine-tune Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss))

    def evaluate_clustering(self, cluster_result):
        adata1 = self.adata_omics1.copy()
        adata1.obsm['emb_combined'] = cluster_result
        y4 = clustering(adata1, key='emb_combined', add_key='emb_combined', n_clusters=5, method='mclust', use_pca=True)
        y4 = torch.tensor(y4.values, device=self.device) if y4 is not None else None
        
        label = adata1.obs['emb_combined']
        
        ids = label.index.astype(str).str[:4]
        int_list = [int(num_str) for num_str in ids]
        list_pred = [-1 for i in range(len(int_list))]
        for i in range(len(int_list)):
            list_pred[int_list[i]] = label[i]
                
        try:
            label_file = 'E:/code/mymodel/data/Simulation/GT.txt'
            labels_true = read_list_from_file(label_file)
            
            ari = adjusted_rand_score(labels_true, list_pred)
            print(f"Current clustering ARI: {ari:.6f}")
        except Exception as e:
            print(f"Error evaluating clustering results: {str(e)}")

    def get_embeddings(self):
        self.model.eval()  # Set to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            xs = [self.features_omics1, self.features_omics2, self.features_omics3]
            feature_adj = [self.adj_feature_omics1, self.adj_feature_omics2, self.adj_feature_omics3]
            spatial_adj = [self.adj_spatial_omics1, self.adj_spatial_omics2, self.adj_spatial_omics3]
            
            xrs, fused_features, hs = self.model(xs, feature_adj, spatial_adj)
            commonz, S= self.model.GCFAgg(xs, fused_features)

            emb_omics1 = F.normalize(hs[0], p=2, eps=1e-12, dim=1)
            emb_omics2 = F.normalize(hs[1], p=2, eps=1e-12, dim=1)
            emb_combined = F.normalize(commonz, p=2, eps=1e-12, dim=1)
            
            embeddings = {
                'emb_omics1': emb_omics1.cpu().numpy(),
                'emb_omics2': emb_omics2.cpu().numpy(),
                'emb_combined': emb_combined.cpu().numpy(),
                'attention_weights': S.cpu().numpy()
            }
            
            if self.is_three_modal:
                emb_omics3 = F.normalize(hs[2], p=2, eps=1e-12, dim=1)
                embeddings['emb_omics3'] = emb_omics3.cpu().numpy()

            return embeddings

    def train(self):
        for epoch in range(1, self.args.rec_epochs + 1):
            self.pre_train(epoch)

        for epoch in range(self.args.rec_epochs + 1, self.args.rec_epochs + self.args.fine_tune_epochs + 1):
            self.fine_tune(epoch)

        embeddings = self.get_embeddings()
        print('Embedding extraction completed')

        return embeddings

