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
    Early stopping at epoch 33 (best HitRate@10=0.0270)                                                                
NCF Training:  64%|████████▎    | 32/50 [13:01<07:19, 24.41s/epoch, hitrate=0.0197, train_loss=0.3059, val_loss=0.7349]
  Training chart saved to outputs/plots/ncf_training.png

  ── NCF Evaluation ──
  Metric      @5        @10       @20       
  ──────────────────────────────────────────
  Precision   0.0017    0.0017    0.0015    
  Recall      0.0087    0.0167    0.0292    
  HitRate     0.0087    0.0167    0.0292    
  MAP         0.0048    0.0058    0.0067    
  NDCG        0.0058    0.0083    0.0114    
  NCF model saved to outputs/models/ncf_best.pt
  Time: 793.9s

============================================================
  2/4: Training NCF+Review (alpha=0.1)
============================================================
    Early stopping at epoch 32 (best HitRate@10=0.0254)                                                                
NCF+Review Training:  62%|███▋  | 31/50 [47:39<29:12, 92.24s/epoch, hitrate=0.0208, train_loss=0.2828, val_loss=0.7406]
  Training chart saved to outputs/plots/ncf_review_training.png

  ── NCF+Review (alpha=0.1) Evaluation ──
  Metric      @5        @10       @20       
  ──────────────────────────────────────────
  Precision   0.0018    0.0017    0.0016    
  Recall      0.0089    0.0167    0.0310    
  HitRate     0.0089    0.0167    0.0310    
  MAP         0.0046    0.0056    0.0066    
  NDCG        0.0056    0.0082    0.0118    
  Time: 2931.3s

============================================================
  3/4: Training NCF+Review (alpha=0.3)
============================================================
    Early stopping at epoch 32 (best HitRate@10=0.0236)                                                                
NCF+Review Training:  62%|███▋  | 31/50 [47:37<29:11, 92.19s/epoch, hitrate=0.0204, train_loss=0.2924, val_loss=0.8150]
  Training chart saved to outputs/plots/ncf_review_training.png

  ── NCF+Review (alpha=0.3) Evaluation ──
  Metric      @5        @10       @20       
  ──────────────────────────────────────────
  Precision   0.0014    0.0014    0.0014    
  Recall      0.0071    0.0139    0.0274    
  HitRate     0.0071    0.0139    0.0274    
  MAP         0.0040    0.0049    0.0058    
  NDCG        0.0048    0.0070    0.0103    
  Best NCF+Review model saved to outputs/models/ncf_review_best.pt
  Time: 2930.0s

============================================================
  4/4: Training NCF+Review (alpha=0.5)
============================================================
    Early stopping at epoch 33 (best HitRate@10=0.0246)                                                                
NCF+Review Training:  64%|███▊  | 32/50 [49:04<27:36, 92.02s/epoch, hitrate=0.0228, train_loss=0.3020, val_loss=0.7194]
  Training chart saved to outputs/plots/ncf_review_training.png

  ── NCF+Review (alpha=0.5) Evaluation ──
  Metric      @5        @10       @20       
  ──────────────────────────────────────────
  Precision   0.0018    0.0016    0.0013    
  Recall      0.0089    0.0163    0.0254    
  HitRate     0.0089    0.0163    0.0254    
  MAP         0.0039    0.0048    0.0054    
  NDCG        0.0051    0.0075    0.0097    
  Time: 3016.6s


============================================================
  EXPERIMENT 2: SEMANTIC CONTRIBUTION SUMMARY
============================================================

  Precision:
  Model                 @5          @10         @20         
  ──────────────────────────────────────────────────────────
  NCF                   0.0017      0.0017      0.0015      
  NCF+Review_alpha=0.1  0.0018      0.0017      0.0016      
  NCF+Review_alpha=0.3  0.0014      0.0014      0.0014      
  NCF+Review_alpha=0.5  0.0018      0.0016      0.0013      

  Recall:
  Model                 @5          @10         @20         
  ──────────────────────────────────────────────────────────
  NCF                   0.0087      0.0167      0.0292      
  NCF+Review_alpha=0.1  0.0089      0.0167      0.0310      
  NCF+Review_alpha=0.3  0.0071      0.0139      0.0274      
  NCF+Review_alpha=0.5  0.0089      0.0163      0.0254      

  HitRate:
  Model                 @5          @10         @20         
  ──────────────────────────────────────────────────────────
  NCF                   0.0087      0.0167      0.0292      
  NCF+Review_alpha=0.1  0.0089      0.0167      0.0310      
  NCF+Review_alpha=0.3  0.0071      0.0139      0.0274      
  NCF+Review_alpha=0.5  0.0089      0.0163      0.0254      

  MAP:
  Model                 @5          @10         @20         
  ──────────────────────────────────────────────────────────
  NCF                   0.0048      0.0058      0.0067      
  NCF+Review_alpha=0.1  0.0046      0.0056      0.0066      
  NCF+Review_alpha=0.3  0.0040      0.0049      0.0058      
  NCF+Review_alpha=0.5  0.0039      0.0048      0.0054      

  NDCG:
  Model                 @5          @10         @20         
  ──────────────────────────────────────────────────────────
  NCF                   0.0058      0.0083      0.0114      
  NCF+Review_alpha=0.1  0.0056      0.0082      0.0118      
  NCF+Review_alpha=0.3  0.0048      0.0070      0.0103      
  NCF+Review_alpha=0.5  0.0051      0.0075      0.0097      

  Results saved to outputs/exp2_semantic_results.json
  ```