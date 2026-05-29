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
    Early stopping at epoch 35 (best HitRate@10=0.0262)                                                                
NCF Training:  68%|████████▊    | 34/50 [13:50<06:30, 24.42s/epoch, hitrate=0.0210, train_loss=0.2924, val_loss=0.8025]
  Training chart saved to outputs/plots/ncf_training.png

  ── NCF Evaluation ──
  Metric      @5        @10       @20       
  ──────────────────────────────────────────
  Precision   0.0016    0.0016    0.0014    
  Recall      0.0081    0.0157    0.0278    
  HitRate     0.0081    0.0157    0.0278    
  MAP         0.0036    0.0046    0.0054    
  NDCG        0.0047    0.0072    0.0102    
  NCF model saved to outputs/models/ncf_best.pt
  Time: 843.0s

============================================================
  2/2: Training LightGCN
============================================================
  Building normalized adjacency matrix ...
  Adj matrix shape: (7963, 7963)
    Early stopping at epoch 78                                                                                         
LightGCN Training:  38%|████████████████▏                         | 77/200 [09:16<14:48,  7.23s/epoch, bpr_loss=0.4631]
  Training chart saved to outputs/plots/lightgcn_training.png

  ── LightGCN Evaluation ──
  Metric      @5        @10       @20       
  ──────────────────────────────────────────
  Precision   0.0016    0.0016    0.0014    
  Recall      0.0079    0.0161    0.0276    
  HitRate     0.0079    0.0161    0.0276    
  MAP         0.0036    0.0047    0.0055    
  NDCG        0.0046    0.0073    0.0102    
  LightGCN model saved to outputs/models/lightgcn_best.pt
  Time: 645.2s


============================================================
  EXPERIMENT 3: GRAPH MODEL COMPARISON SUMMARY
============================================================

  Precision:
  Model          @5          @10         @20         
  ───────────────────────────────────────────────────
  NCF            0.0016      0.0016      0.0014      
  LightGCN       0.0016      0.0016      0.0014      

  Recall:
  Model          @5          @10         @20         
  ───────────────────────────────────────────────────
  NCF            0.0081      0.0157      0.0278      
  LightGCN       0.0079      0.0161      0.0276      

  HitRate:
  Model          @5          @10         @20         
  ───────────────────────────────────────────────────
  NCF            0.0081      0.0157      0.0278      
  LightGCN       0.0079      0.0161      0.0276      

  MAP:
  Model          @5          @10         @20         
  ───────────────────────────────────────────────────
  NCF            0.0036      0.0046      0.0054      
  LightGCN       0.0036      0.0047      0.0055      

  NDCG:
  Model          @5          @10         @20         
  ───────────────────────────────────────────────────
  NCF            0.0047      0.0072      0.0102      
  LightGCN       0.0046      0.0073      0.0102      

  Results saved to outputs/exp3_graph_results.json
```