# Ablation Study Summary — Order Encoding (5 Configurations)

Instances (10): `pores_1, ibm32, bcspwr01, bcsstk01, bcspwr02, curtis54, will57, ash85, nos4, can__144` | Time limit: `20s`

| Configuration | Optimal (`BW==LB`) | Avg BW | Avg Base Clauses | Avg Enc (s) | Avg Sol (s) | Avg Total (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1_Naive** | 5/10 | 33.50 | 282,674 | 0.84 | 40.83 | 41.67 |
| **C2_+RCM** | 8/10 | 9.90 | 282,674 | 0.55 | 5.15 | 5.70 |
| **C3_+SymDeg** | 10/10 | 9.40 | 282,674 | 0.69 | 2.27 | 2.97 |
| **C4_+Seq2Hop** | 10/10 | 9.40 | 36,210 | 0.31 | 2.34 | 2.64 |
| **C5_+IncrAssump** | 10/10 | 9.40 | 36,210 | 0.33 | 4.06 | 4.39 |
