```
(torch) root@nnkmpjoblweenhmd-snow-7c84fdbd88-wfkk4:/data/coding/dm-final-explainable-rec# python experiments/exp1_baseline.py --models all
  Dataset: 4963 users, 3000 items
  Train: 106328 | Val: 4963 | Test: 4963
  Models to run: ['ncf', 'popularity', 'usercf']
  Device: cuda

============================================================
  Building Popularity baseline ...

  ── Popularity Evaluation ──
  Metric      @5        @10       @20       
  ──────────────────────────────────────────
  Precision   0.0008    0.0009    0.0009    
  Recall      0.0042    0.0093    0.0173    
  HitRate     0.0042    0.0093    0.0173    
  MAP         0.0019    0.0026    0.0032    
  NDCG        0.0025    0.0041    0.0062    
  Time: 0.7s

============================================================
  Building UserCF ...

  ── UserCF Evaluation ──
  Metric      @5        @10       @20       
  ──────────────────────────────────────────
  Precision   0.0021    0.0019    0.0017    
  Recall      0.0107    0.0193    0.0330    
  HitRate     0.0107    0.0193    0.0330    
  MAP         0.0056    0.0067    0.0076    
  NDCG        0.0069    0.0096    0.0130    
  Time: 128.1s

============================================================
  Training NCF ...
    Early stopping at epoch 36 (best HitRate@10=0.0288)                                                                
NCF Training:  70%|█████████    | 35/50 [14:24<06:10, 24.71s/epoch, hitrate=0.0254, train_loss=0.3014, val_loss=0.7309]
  Training chart saved to outputs/plots/ncf_training.png

  ── NCF Evaluation ──
  Metric      @5        @10       @20       
  ──────────────────────────────────────────
  Precision   0.0015    0.0015    0.0013    
  Recall      0.0073    0.0145    0.0264    
  HitRate     0.0073    0.0145    0.0264    
  MAP         0.0040    0.0050    0.0057    
  NDCG        0.0048    0.0071    0.0101    
  Time: 878.3s

============================================================
  EXPERIMENT 1: BASELINE COMPARISON SUMMARY
============================================================

  Precision:
  Model          @5          @10         @20         
  ───────────────────────────────────────────────────
  Popularity     0.0008      0.0009      0.0009      
  UserCF         0.0021      0.0019      0.0017      
  NCF            0.0015      0.0015      0.0013      

  Recall:
  Model          @5          @10         @20         
  ───────────────────────────────────────────────────
  Popularity     0.0042      0.0093      0.0173      
  UserCF         0.0107      0.0193      0.0330      
  NCF            0.0073      0.0145      0.0264      

  HitRate:
  Model          @5          @10         @20         
  ───────────────────────────────────────────────────
  Popularity     0.0042      0.0093      0.0173      
  UserCF         0.0107      0.0193      0.0330      
  NCF            0.0073      0.0145      0.0264      

  MAP:
  Model          @5          @10         @20         
  ───────────────────────────────────────────────────
  Popularity     0.0019      0.0026      0.0032      
  UserCF         0.0056      0.0067      0.0076      
  NCF            0.0040      0.0050      0.0057      

  NDCG:
  Model          @5          @10         @20         
  ───────────────────────────────────────────────────
  Popularity     0.0025      0.0041      0.0062      
  UserCF         0.0069      0.0096      0.0130      
  NCF            0.0048      0.0071      0.0101      

  Results saved to outputs/exp1_baseline_20260529_132514.json
```