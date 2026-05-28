```
(torch) root@nnkmpjoblweenhmd-snow-7c84fdbd88-wfkk4:/data/coding/dm-final-explainable-rec# python experiments/exp2_semantic.py
============================================================
  Experiment 2: Semantic Contribution Analysis
  RQ2: Can review semantics improve recommendation?
============================================================
  Dataset: 4963 users, 3000 items
  Train: 106328 | Val: 4963 | Test: 4963
  Device: cuda

  Loading SBERT embeddings ...
  user_emb: (4963, 384), item_emb: (3000, 384)

============================================================
  1/4: Training NCF (baseline, no review semantics)
============================================================
    Early stopping at epoch 35                                                                                                                                                     
NCF Training:  68%|████████████████████████████████████████████████████████████▌                            | 34/50 [07:26<03:30, 13.14s/epoch, train_loss=0.2955, val_loss=0.8842]

  ── NCF Evaluation ──
  Metric      @5        @10       @20       
  ──────────────────────────────────────────
  Precision   0.0014    0.0013    0.0013    
  Recall      0.0071    0.0129    0.0254    
  HitRate     0.0071    0.0129    0.0254    
  MAP         0.0031    0.0039    0.0047    
  NDCG        0.0040    0.0059    0.0091    
  NCF model saved to outputs/models/ncf_best.pt
  Time: 458.4s

============================================================
  2/4: Training NCF+Review (alpha=0.1)
============================================================
    Early stopping at epoch 33 (val_loss=0.7545)                                                                                                                                   
NCF+Review Training:  64%|████████████████████████████████████████████████████▍                             | 32/50 [12:36<07:05, 23.64s/epoch, train_loss=0.2807, val_loss=0.7545]

  ── NCF+Review (alpha=0.1) Evaluation ──
  Metric      @5        @10       @20       
  ──────────────────────────────────────────
  Precision   0.0015    0.0014    0.0013    
  Recall      0.0073    0.0143    0.0252    
  HitRate     0.0073    0.0143    0.0252    
  MAP         0.0031    0.0041    0.0048    
  NDCG        0.0041    0.0064    0.0092    
  Time: 835.0s

============================================================
  3/4: Training NCF+Review (alpha=0.3)
============================================================
    Early stopping at epoch 33 (val_loss=0.7267)                                                                                                                                   
NCF+Review Training:  64%|████████████████████████████████████████████████████▍                             | 32/50 [10:01<05:38, 18.79s/epoch, train_loss=0.2837, val_loss=0.7267]

  ── NCF+Review (alpha=0.3) Evaluation ──
  Metric      @5        @10       @20       
  ──────────────────────────────────────────
  Precision   0.0013    0.0014    0.0012    
  Recall      0.0066    0.0141    0.0244    
  HitRate     0.0066    0.0141    0.0244    
  MAP         0.0028    0.0037    0.0044    
  NDCG        0.0037    0.0061    0.0087    
  Best NCF+Review model saved to outputs/models/ncf_review_best.pt
  Time: 671.4s

============================================================
  4/4: Training NCF+Review (alpha=0.5)
============================================================
    Early stopping at epoch 31 (val_loss=0.6271)                                                                                                                                   
NCF+Review Training:  60%|█████████████████████████████████████████████████▏                                | 30/50 [09:42<06:28, 19.42s/epoch, train_loss=0.3111, val_loss=0.6271]

  ── NCF+Review (alpha=0.5) Evaluation ──
  Metric      @5        @10       @20       
  ──────────────────────────────────────────
  Precision   0.0010    0.0012    0.0011    
  Recall      0.0050    0.0123    0.0220    
  HitRate     0.0050    0.0123    0.0220    
  MAP         0.0024    0.0034    0.0040    
  NDCG        0.0031    0.0054    0.0078    
  Time: 652.7s


============================================================
  EXPERIMENT 2: SEMANTIC CONTRIBUTION SUMMARY
============================================================

  Precision:
  Model                 @5          @10         @20         
  ──────────────────────────────────────────────────────────
  NCF                   0.0014      0.0013      0.0013      
  NCF+Review_alpha=0.1  0.0015      0.0014      0.0013      
  NCF+Review_alpha=0.3  0.0013      0.0014      0.0012      
  NCF+Review_alpha=0.5  0.0010      0.0012      0.0011      

  Recall:
  Model                 @5          @10         @20         
  ──────────────────────────────────────────────────────────
  NCF                   0.0071      0.0129      0.0254      
  NCF+Review_alpha=0.1  0.0073      0.0143      0.0252      
  NCF+Review_alpha=0.3  0.0066      0.0141      0.0244      
  NCF+Review_alpha=0.5  0.0050      0.0123      0.0220      

  HitRate:
  Model                 @5          @10         @20         
  ──────────────────────────────────────────────────────────
  NCF                   0.0071      0.0129      0.0254      
  NCF+Review_alpha=0.1  0.0073      0.0143      0.0252      
  NCF+Review_alpha=0.3  0.0066      0.0141      0.0244      
  NCF+Review_alpha=0.5  0.0050      0.0123      0.0220      

  MAP:
  Model                 @5          @10         @20         
  ──────────────────────────────────────────────────────────
  NCF                   0.0031      0.0039      0.0047      
  NCF+Review_alpha=0.1  0.0031      0.0041      0.0048      
  NCF+Review_alpha=0.3  0.0028      0.0037      0.0044      
  NCF+Review_alpha=0.5  0.0024      0.0034      0.0040      

  NDCG:
  Model                 @5          @10         @20         
  ──────────────────────────────────────────────────────────
  NCF                   0.0040      0.0059      0.0091      
  NCF+Review_alpha=0.1  0.0041      0.0064      0.0092      
  NCF+Review_alpha=0.3  0.0037      0.0061      0.0087      
  NCF+Review_alpha=0.5  0.0031      0.0054      0.0078      

  Results saved to outputs/exp2_semantic_results.json
  ```