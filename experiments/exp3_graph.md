```
(torch) root@nnkmpjoblweenhmd-snow-7c84fdbd88-wfkk4:/data/coding/dm-final-explainable-rec# python experiments/exp3_graph.py
============================================================
  Experiment 3: Graph Model Comparison (NCF vs LightGCN)
  RQ3: Can graph structure better model user-item relationships?
============================================================
  Dataset: 4963 users, 3000 items
  Train: 106328 | Val: 4963 | Test: 4963
  Device: cuda

============================================================
  1/2: Training NCF
============================================================
    Early stopping at epoch 33                                                                                                                                                     
NCF Training:  64%|████████████████████████████████████████████████████████▉                                | 32/50 [08:47<04:56, 16.48s/epoch, train_loss=0.3109, val_loss=0.6757]
  Training chart saved to outputs/plots/ncf_training.png

  ── NCF Evaluation ──
  Metric      @5        @10       @20       
  ──────────────────────────────────────────
  Precision   0.0015    0.0016    0.0014    
  Recall      0.0075    0.0155    0.0274    
  HitRate     0.0075    0.0155    0.0274    
  MAP         0.0030    0.0040    0.0048    
  NDCG        0.0041    0.0067    0.0096    
  NCF model saved to outputs/models/ncf_best.pt
  Time: 539.7s

============================================================
  2/2: Training LightGCN
============================================================
  Building normalized adjacency matrix ...
  Adj matrix shape: (7963, 7963)
LightGCN Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [06:00<00:00,  7.20s/epoch, bpr_loss=0.4652]
  Training chart saved to outputs/plots/lightgcn_training.png                                                                                                                      

  ── LightGCN Evaluation ──
  Metric      @5        @10       @20       
  ──────────────────────────────────────────
  Precision   0.0015    0.0016    0.0014    
  Recall      0.0077    0.0163    0.0282    
  HitRate     0.0077    0.0163    0.0282    
  MAP         0.0033    0.0045    0.0053    
  NDCG        0.0044    0.0072    0.0102    
  LightGCN model saved to outputs/models/lightgcn_best.pt
  Time: 445.0s


============================================================
  EXPERIMENT 3: GRAPH MODEL COMPARISON SUMMARY
============================================================

  Precision:
  Model          @5          @10         @20         
  ───────────────────────────────────────────────────
  NCF            0.0015      0.0016      0.0014      
  LightGCN       0.0015      0.0016      0.0014      

  Recall:
  Model          @5          @10         @20         
  ───────────────────────────────────────────────────
  NCF            0.0075      0.0155      0.0274      
  LightGCN       0.0077      0.0163      0.0282      

  HitRate:
  Model          @5          @10         @20         
  ───────────────────────────────────────────────────
  NCF            0.0075      0.0155      0.0274      
  LightGCN       0.0077      0.0163      0.0282      

  MAP:
  Model          @5          @10         @20         
  ───────────────────────────────────────────────────
  NCF            0.0030      0.0040      0.0048      
  LightGCN       0.0033      0.0045      0.0053      

  NDCG:
  Model          @5          @10         @20         
  ───────────────────────────────────────────────────
  NCF            0.0041      0.0067      0.0096      
  LightGCN       0.0044      0.0072      0.0102      

  Results saved to outputs/exp3_graph_results.json
```