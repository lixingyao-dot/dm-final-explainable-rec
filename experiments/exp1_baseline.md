```
(torch) root@nnkmpjoblweenhmd-snow-7c84fdbd88-wfkk4:/data/coding/dm-final-explainable-rec# python experiments/exp1_baseline.py
  Dataset: 4963 users, 3000 items
  Train: 106328 | Val: 4963 | Test: 4963
  Device: cuda

============================================================
  Training Popularity baseline ...

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
  Training UserCF ...
  Computing user similarities ...

  ── UserCF Evaluation ──
  Metric      @5        @10       @20       
  ──────────────────────────────────────────
  Precision   0.0003    0.0003    0.0003    
  Recall      0.0016    0.0032    0.0069    
  HitRate     0.0016    0.0032    0.0069    
  MAP         0.0008    0.0010    0.0013    
  NDCG        0.0010    0.0015    0.0024    
  Time: 136.9s

============================================================
  Training ItemCF ...
  Computing item similarities ...

  ── ItemCF Evaluation ──
  Metric      @5        @10       @20       
  ──────────────────────────────────────────
  Precision   0.0011    0.0009    0.0008    
  Recall      0.0054    0.0093    0.0159    
  HitRate     0.0054    0.0093    0.0159    
  MAP         0.0024    0.0029    0.0034    
  NDCG        0.0032    0.0044    0.0061    
  Time: 76.6s

============================================================
  Training NCF ...
    Early stopping at epoch 7                                                                                                                                                      
NCF Training:  12%|██████████▊                                                                               | 6/50 [01:22<10:02, 13.70s/epoch, train_loss=0.1334, val_loss=3.1468]

  ── NCF Evaluation ──
  Metric      @5        @10       @20       
  ──────────────────────────────────────────
  Precision   0.0023    0.0021    0.0018    
  Recall      0.0117    0.0214    0.0351    
  HitRate     0.0117    0.0214    0.0351    
  MAP         0.0050    0.0063    0.0072    
  NDCG        0.0067    0.0097    0.0131    
  Time: 92.6s

============================================================
  EXPERIMENT 1: BASELINE COMPARISON SUMMARY
============================================================

  Precision:
  Model          @5          @10         @20         
  ───────────────────────────────────────────────────
  Popularity     0.0008      0.0009      0.0009      
  UserCF         0.0003      0.0003      0.0003      
  ItemCF         0.0011      0.0009      0.0008      
  NCF            0.0023      0.0021      0.0018      

  Recall:
  Model          @5          @10         @20         
  ───────────────────────────────────────────────────
  Popularity     0.0042      0.0093      0.0173      
  UserCF         0.0016      0.0032      0.0069      
  ItemCF         0.0054      0.0093      0.0159      
  NCF            0.0117      0.0214      0.0351      

  HitRate:
  Model          @5          @10         @20         
  ───────────────────────────────────────────────────
  Popularity     0.0042      0.0093      0.0173      
  UserCF         0.0016      0.0032      0.0069      
  ItemCF         0.0054      0.0093      0.0159      
  NCF            0.0117      0.0214      0.0351      

  MAP:
  Model          @5          @10         @20         
  ───────────────────────────────────────────────────
  Popularity     0.0019      0.0026      0.0032      
  UserCF         0.0008      0.0010      0.0013      
  ItemCF         0.0024      0.0029      0.0034      
  NCF            0.0050      0.0063      0.0072      

  NDCG:
  Model          @5          @10         @20         
  ───────────────────────────────────────────────────
  Popularity     0.0025      0.0041      0.0062      
  UserCF         0.0010      0.0015      0.0024      
  ItemCF         0.0032      0.0044      0.0061      
  NCF            0.0067      0.0097      0.0131      

  Results saved to outputs/exp1_baseline_results.json
```