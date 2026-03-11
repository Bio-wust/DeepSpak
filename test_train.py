import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score

from mymodel.network import GCFAggMVC
import numpy as np
import argparse
import random
from mymodel.preprocess import adjacent_matrix_preprocessing
import os
from mymodel.utils import  read_list_from_file
from mymodel.loss_test import Loss
from torch.cuda.amp import GradScaler
from mymodel.utils import clustering
import torch.nn as nn

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class SpatialOmicsTrainer:
    def __init__(self, args, data, fusion_weights=None):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.walk_steps = args.walk_steps if hasattr(args, 'walk_steps') else 3
        self.alpha = args.alpha if hasattr(args, 'alpha') else 0.5
        self.fusion_weights = fusion_weights
        self.adata_omics1 = data['adata_omics1']
        self.adata_omics2 = data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)

        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)

        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)

        input_dims = [self.features_omics1.shape[1], self.features_omics2.shape[1]]
        self.model = GCFAggMVC(
            view=2,
            input_size=input_dims,
            low_feature_dim=args.low_feature_dim,
            high_feature_dim=args.high_feature_dim,
            device=self.device,
            fusion_weights=fusion_weights  
        ).to(self.device)


        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        self.criterion = Loss(args.batch_size, args.temperature_f, self.device, 
                               self.walk_steps, self.alpha).to(self.device)
        self.mse = torch.nn.MSELoss()

    def pre_train(self, epoch):
        self.model.train()
        xs = [self.features_omics1, self.features_omics2]
        feature_adj = [self.adj_feature_omics1, self.adj_feature_omics2]
        spatial_adj = [self.adj_spatial_omics1, self.adj_spatial_omics2]
        
        
        self.optimizer.zero_grad()
        xrs, _, _ = self.model(xs, feature_adj, spatial_adj)

        loss_list = []
        for v in range(2):
            loss_list.append(self.mse(xs[v], xrs[v]))
        loss = sum(loss_list)
        
        loss.backward()
        self.optimizer.step()

        print('Pre-train Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(loss.item()))

    def fine_tune(self, epoch):
        self.model.train()
        xs = [self.features_omics1, self.features_omics2]
        feature_adj = [self.adj_feature_omics1, self.adj_feature_omics2]
        spatial_adj = [self.adj_spatial_omics1, self.adj_spatial_omics2]
        
        self.optimizer.zero_grad()
        xrs, fused_features, hs = self.model(xs, feature_adj, spatial_adj)
        commonz, S= self.model.GCFAgg(xs, fused_features)


        loss_list = []
        
        for v in range(2):
            loss_list.append(self.criterion.Structure_guided_Contrastive_Loss(
                hs[v], 
                commonz, 
                S
            ))
            loss_list.append(self.mse(xs[v], xrs[v]))
        
        loss = sum(loss_list)
        loss.backward()
        self.optimizer.step()

        print('Fine-tune Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(loss.item()))
        
        if epoch % 20 == 0 and hasattr(self.model, 'feature_fusion') and hasattr(self.model.feature_fusion[0], 'get_weights'):
            weights = []
            for v in range(2):
                w = self.model.feature_fusion[v].get_weights()
                avg_feature_weight = w[:, 0].mean().item()
                avg_spatial_weight = w[:, 1].mean().item()
                weights.append((avg_feature_weight, avg_spatial_weight))
            print(f"Epoch {epoch} Fusion weights - View1: Feature={weights[0][0]:.3f}, Spatial={weights[0][1]:.3f} | "
                  f"View2: Feature={weights[1][0]:.3f}, Spatial={weights[1][1]:.3f}")


    def get_embeddings(self):
        self.model.eval()  
        with torch.no_grad():  
            xs = [self.features_omics1, self.features_omics2]
            feature_adj = [self.adj_feature_omics1, self.adj_feature_omics2]
            spatial_adj = [self.adj_spatial_omics1, self.adj_spatial_omics2]
            
            xrs, fused_features, hs = self.model(xs, feature_adj, spatial_adj)
            commonz, S= self.model.GCFAgg(xs, fused_features)

            emb_omics1 = F.normalize(hs[0], p=2, dim=1)
            emb_omics2 = F.normalize(hs[1], p=2, dim=1)
            emb_combined = F.normalize(commonz, p=2, dim=1)

            embeddings = {
                'emb_omics1': emb_omics1.cpu().numpy(),
                'emb_omics2': emb_omics2.cpu().numpy(),
                'emb_combined': emb_combined.cpu().numpy(),
                'attention_weights': S.cpu().numpy()  
            }
            
            if hasattr(self.model, 'feature_fusion') and hasattr(self.model.feature_fusion[0], 'get_weights'):
                fusion_weights = []
                for v in range(2):
                    fusion_weights.append(self.model.feature_fusion[v].get_weights().cpu().numpy())
                embeddings['fusion_weights'] = fusion_weights

            return embeddings

    def train(self):
        for epoch in range(1, self.args.rec_epochs + 1):
            self.pre_train(epoch)

        for epoch in range(self.args.rec_epochs + 1, self.args.rec_epochs + self.args.fine_tune_epochs + 1):
            self.fine_tune(epoch)

        embeddings = self.get_embeddings()
        print('Embedding extraction completed')

        return embeddings

