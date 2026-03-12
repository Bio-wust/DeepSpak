import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.functional import normalize

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

class CrossViewAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = torch.sqrt(torch.FloatTensor([feature_dim]))
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x, other_views):
        # x: [batch_size, seq_len, feature_dim]
        # other_views: list of tensors each with shape [batch_size, seq_len, feature_dim]
        
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            other_views = [v.unsqueeze(0) if len(v.shape) == 2 else v for v in other_views]
            
        batch_size = x.shape[0]
        
        Q = self.query(x)  # [batch_size, seq_len, feature_dim]
        K = self.key(torch.cat(other_views, dim=1))  # [batch_size, total_other_len, feature_dim]
        V = self.value(torch.cat(other_views, dim=1))  # [batch_size, total_other_len, feature_dim]
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(x.device)
        attention = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attention, V)
        
        output = self.layer_norm(context + x)
        
        if len(x.shape) == 3:
            return output.squeeze(0), attention
        return output, attention

class GCMA(nn.Module):
    def __init__(self, view, input_size, low_feature_dim, high_feature_dim, device, 
                 fusion_weights=None, act=F.relu, dropout=0.):
        super().__init__()
        self.view = view
        self.encoders = []
        self.spatial_encoders = [] 
        self.decoders = []
        self.cross_attentions = []
        self.dropout = dropout
        self.act = act
        self.fusion_weights = fusion_weights
        
        for v in range(view):
            self.encoders.append(GraphConvolution(input_size[v], low_feature_dim, self.dropout, self.act).to(device))
            self.spatial_encoders.append(GraphConvolution(input_size[v], low_feature_dim, self.dropout, self.act).to(device))
            self.decoders.append(GraphConvolution(low_feature_dim, input_size[v], self.dropout, self.act).to(device))
            self.cross_attentions.append(CrossViewAttention(low_feature_dim).to(device))
            
        self.encoders = nn.ModuleList(self.encoders)
        self.spatial_encoders = nn.ModuleList(self.spatial_encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.cross_attentions = nn.ModuleList(self.cross_attentions)
        
        view_weights = None
        if fusion_weights is not None:
            if isinstance(fusion_weights[0], (int, float)):
                total = fusion_weights[0] + fusion_weights[1]
                normalized_weights = [fusion_weights[0]/total, fusion_weights[1]/total]
                view_weights = [normalized_weights for _ in range(view)]
            elif len(fusion_weights) == view:
                view_weights = fusion_weights
                
        self.feature_fusion = nn.ModuleList([
            AdaptiveFeatureFusion(
                low_feature_dim, 
                fixed_weights=None if view_weights is None else view_weights[v]
            ).to(device) for v in range(view)
        ])
        
        self.Specific_view = nn.Sequential(
            nn.Linear(low_feature_dim, high_feature_dim),
        )
        self.Common_view = nn.Sequential(
            nn.Linear(low_feature_dim * view, high_feature_dim),
        )
        
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=low_feature_dim*view, 
            nhead=1, 
            dim_feedforward=256
        )
        
    def forward(self, xs, feature_adjs, spatial_adjs):
        feature_encoded = []
        for v in range(self.view):
            z = self.encoders[v](xs[v], feature_adjs[v])
            feature_encoded.append(z)
            
        spatial_encoded = []
        for v in range(self.view):
            z = self.spatial_encoders[v](xs[v], spatial_adjs[v])
            spatial_encoded.append(z)
            
        fused_features = []
        fusion_weights = []  
        for v in range(self.view):
            fused, weights = self.feature_fusion[v](feature_encoded[v], spatial_encoded[v])
            fused_features.append(fused)
            fusion_weights.append(weights)
            
        xrs = []
        hs = []
        for v in range(self.view):
            xr = self.decoders[v](feature_encoded[v], feature_adjs[v])
            xrs.append(xr)
            
            h = normalize(self.Specific_view(fused_features[v]), dim=1)
            hs.append(h)
            
        return xrs, fused_features, hs

    def GCFAgg(self, xs,fused_features):
        
        zs = []
        for v in range(self.view):
            zs.append(fused_features[v])
        commonz = torch.cat(zs, 1) 
        commonz, S = self.TransformerEncoderLayer(commonz)
        commonz = normalize(self.Common_view(commonz), dim=1)

        return commonz, S  

class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, feature_dim, fixed_weights=None):
        super(AdaptiveFeatureFusion, self).__init__()
        self.feature_dim = feature_dim
        self.fixed_weights = fixed_weights
        
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2),
            #nn.Sigmoid()  
            nn.Softmax(dim=1)
        )
        
        self.transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        self.last_weights = None
        
        self.min_feature_weight = 0.1  
        self.min_spatial_weight = 0.1  
    
    def forward(self, feature_encoded, spatial_encoded):
        if self.fixed_weights is not None:
            weights = torch.tensor([self.fixed_weights[0], self.fixed_weights[1]], 
                                  device=feature_encoded.device).expand(feature_encoded.shape[0], 2)
        else:
            combined = torch.cat([feature_encoded, spatial_encoded], dim=1)
            raw_weights = self.attention_net(combined)
            biased_weights = raw_weights
            feature_weight = torch.clamp(biased_weights[:, 0], min=self.min_feature_weight)
            spatial_weight = torch.clamp(biased_weights[:, 1], min=self.min_spatial_weight)
            total_weight = feature_weight + spatial_weight
            weights = torch.stack([
                feature_weight / total_weight,
                spatial_weight / total_weight
            ], dim=1)
        
        self.last_weights = weights
        feature_weight = weights[:, 0].unsqueeze(1)
        spatial_weight = weights[:, 1].unsqueeze(1)
        fused_features = feature_weight * feature_encoded + spatial_weight * spatial_encoded
        transformed = self.transform(fused_features)
        
        return transformed, weights
    
    def get_weights(self):
        if self.last_weights is None:
            raise RuntimeError("Weights not computed yet, please call forward method first")
        return self.last_weights