import torch

def cluster_features(features, features_config):
    from sklearn.cluster import KMeans

    clustered_feats = []
    for i in range(features.shape[0]):
        # (4400, 768) → (K, 768)
        kmeans = KMeans(n_clusters=features_config['k']).fit(features[i].cpu().numpy())
        centers = torch.tensor(kmeans.cluster_centers_)
        clustered_feats.append(centers)  # shape: (5, 768)

    return torch.stack(clustered_feats, dim=0).sum(dim=1) 

def l2_norm_features(features):
    ### object-aware weighted pooling
    ### 1. L2-norm
    norms = features.norm(dim=-1)  # (B, N)
    weights = torch.nn.functional.softmax(norms, dim=1) # (B, N)

    return (features * weights.unsqueeze(-1)).sum(dim=1).cpu().detach()  # (B, D) 

def topk_features(features):        
    ### 2. saliency-like: top-k patch pooling 
    topk = 100
    norms = features.norm(dim=-1)  # (B, N)
    values, indices = torch.topk(norms, topk, dim=1)  # (B, topk)
    pooled = []
    for i in range(features.shape[0]):
        top_feats = features[i][indices[i]]  # (topk, 768)
        pooled.append(top_feats.mean(dim=0))    # (768,)

    return torch.stack(pooled, dim=0).cpu().detach()    # (B, 768)

def attention_feature(features, attn_list):
    ### 3. self-attention 
    # 현재 이미지의 CLS→patch attention
    attn = attn_list[-1]                 # attn: (B, heads, tokens, tokens)
    cls_attn = attn[:, :, 0, 1:]         # cls_attn: (B, heads, tokens)
    avg_attn = cls_attn.mean(dim=1)  
    
    #     # ks = list(map(int, ks))
    #     # ks = list(set([min(k, min(dims)) for k in ks]))
    #     # ks.sort()
    #     # pool_dims = [[dims[0] - k+1, dims[1]-k+1] for k in ks]
    #     # feat_pool = [torch.nn.AdaptiveAvgPool2d(k) for k in pool_dims]
    #     # thresh = [1+(np.log(d[0])/np.log(2)) for d in pool_dims]
    #     # # compute entropy at each resolution
    #     # ents = compute_block_entropy(map_, feat_pool)
    #     # # Check if map contain any object
    #     # pass_ = [l < t for l, t in zip(ents, thresh)]
    #     # # If atleast 50% of the maps agree there is an object, we pick it
    #     # if sum(pass_) >= 0.5 * len(pass_):
    #     #     seeds.append(i)
                

    # ### topk 
    # # topk = 5
    # # _, important_idx = avg_attn.topk(topk, dim=1)  # (B, topk)

    # # # (B, topk, D)
    # # important_features = features.gather(1, important_idx.unsqueeze(-1).expand(-1, -1, features.size(-1)))
       
    return (features * avg_attn.unsqueeze(-1)).sum(dim=1).cpu().detach()  # (1, D)
         