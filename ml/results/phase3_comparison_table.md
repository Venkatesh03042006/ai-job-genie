# Phase 3 vs Phase 2 Comparison (test split)

Split: `test` | matched_score threshold: `0.7` | queries: `434` | candidate JDs: `952`

Val triplet accuracy: pretrained=`0.5496` -> fine-tuned=`0.6409`

Hybrid alpha swept on val over [0.3, 0.5, 0.7] (selected by MRR): raw best alpha=`0.30`, normalized best alpha=`0.70` (final_score = alpha * tfidf_score + (1 - alpha) * sbert_score; "normalized" min-max scales each matcher's similarity row to [0, 1] per query before blending)

## Overall

| matcher | n_queries | n_candidates | P@1 | P@3 | P@5 | MRR |
|---|---|---|---|---|---|---|
| tfidf | 434 | 952 | 0.0622 | 0.0376 | 0.0341 | 0.1496 |
| sbert_pretrained | 434 | 952 | 0.0046 | 0.0161 | 0.0134 | 0.0559 |
| sbert_multi_qa_minilm_l6_cos_v1 | 434 | 952 | 0.0023 | 0.0092 | 0.0111 | 0.0404 |
| sbert_finetuned | 434 | 952 | 0.0138 | 0.0146 | 0.0101 | 0.0547 |
| hybrid_tfidf_sbert | 434 | 952 | 0.0207 | 0.0276 | 0.0258 | 0.1118 |
| hybrid_tfidf_sbert_normalized | 434 | 952 | 0.0346 | 0.0307 | 0.0300 | 0.1275 |

## By category

| matcher | category | n_queries | P@1 | P@3 | P@5 | MRR |
|---|---|---|---|---|---|---|
| tfidf | Database Administrator (DBA) | 31 | 0.0968 | 0.0430 | 0.0387 | 0.1856 |
| tfidf | Machine Learning (ML) Engineer | 26 | 0.1154 | 0.0769 | 0.0538 | 0.2267 |
| tfidf | Network Support Engineer | 26 | 0.0769 | 0.0256 | 0.0308 | 0.1495 |
| tfidf | AI Engineer | 25 | 0.0400 | 0.0267 | 0.0160 | 0.1117 |
| tfidf | Asst. Manager/ Manger (Administrative) | 23 | 0.0000 | 0.0290 | 0.0348 | 0.1027 |
| tfidf | Manager- Human Resource Management (HRM) | 23 | 0.0435 | 0.0290 | 0.0435 | 0.1281 |
| tfidf | DevOps Engineer | 21 | 0.0476 | 0.0476 | 0.0381 | 0.1514 |
| tfidf | Executive/ Senior Executive- Trade Marketing, Hygiene Products | 21 | 0.0476 | 0.0317 | 0.0286 | 0.1311 |
| tfidf | Full Stack Developer (Python,React js) | 21 | 0.0476 | 0.0159 | 0.0190 | 0.1288 |
| tfidf | Intern (Generative AI Engineering - 2D/3D Image Generation) | 21 | 0.0952 | 0.0317 | 0.0190 | 0.1455 |
| tfidf | Senior Software Engineer | 19 | 0.0526 | 0.0175 | 0.0105 | 0.1148 |
| tfidf | System Administrator (Operation & Maintenance of Server, Storage & Service Desk System) | 19 | 0.0000 | 0.0351 | 0.0421 | 0.1230 |
| tfidf | Data Science Engineer | 18 | 0.0000 | 0.0185 | 0.0333 | 0.0875 |
| tfidf | Executive/ Sr. Executive -IT | 17 | 0.0000 | 0.0588 | 0.0471 | 0.1443 |
| tfidf | Management Trainee - Mechanical | 17 | 0.0588 | 0.0392 | 0.0353 | 0.1489 |
| tfidf | Senior iOS Engineer | 17 | 0.0588 | 0.0196 | 0.0235 | 0.1180 |
| tfidf | Sr.Officer / Executive - Internal Audit | 11 | 0.1818 | 0.0606 | 0.0364 | 0.2304 |
| tfidf | Head of Internal Control & Compliance (ICC) - SEVP/DMD | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0721 |
| tfidf | Project Coordinator (Civil) | 10 | 0.2000 | 0.0667 | 0.0400 | 0.2488 |
| tfidf | Data Engineer | 9 | 0.1111 | 0.0370 | 0.0444 | 0.1891 |
| tfidf | HR Officer | 9 | 0.0000 | 0.0370 | 0.0222 | 0.1100 |
| tfidf | Mechanical Designer | 9 | 0.1111 | 0.0370 | 0.0444 | 0.1847 |
| tfidf | Mechanical Engineer | 9 | 0.0000 | 0.0370 | 0.0444 | 0.1291 |
| tfidf | Executive - VAT | 8 | 0.1250 | 0.0833 | 0.0500 | 0.2091 |
| tfidf | Marketing Officer | 5 | 0.2000 | 0.1333 | 0.1200 | 0.3869 |
| tfidf | Site Engineer | 4 | 0.2500 | 0.0833 | 0.0500 | 0.2992 |
| tfidf | Civil Engineer | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0483 |
| tfidf | Business Development Executive | 2 | 0.0000 | 0.0000 | 0.1000 | 0.1467 |
| sbert_pretrained | Database Administrator (DBA) | 31 | 0.0000 | 0.0215 | 0.0129 | 0.0625 |
| sbert_pretrained | Machine Learning (ML) Engineer | 26 | 0.0000 | 0.0128 | 0.0154 | 0.0816 |
| sbert_pretrained | Network Support Engineer | 26 | 0.0000 | 0.0385 | 0.0308 | 0.0952 |
| sbert_pretrained | AI Engineer | 25 | 0.0000 | 0.0000 | 0.0080 | 0.0389 |
| sbert_pretrained | Asst. Manager/ Manger (Administrative) | 23 | 0.0000 | 0.0000 | 0.0000 | 0.0191 |
| sbert_pretrained | Manager- Human Resource Management (HRM) | 23 | 0.0000 | 0.0145 | 0.0087 | 0.0230 |
| sbert_pretrained | DevOps Engineer | 21 | 0.0000 | 0.0000 | 0.0000 | 0.0169 |
| sbert_pretrained | Executive/ Senior Executive- Trade Marketing, Hygiene Products | 21 | 0.0000 | 0.0159 | 0.0095 | 0.0424 |
| sbert_pretrained | Full Stack Developer (Python,React js) | 21 | 0.0000 | 0.0317 | 0.0190 | 0.0858 |
| sbert_pretrained | Intern (Generative AI Engineering - 2D/3D Image Generation) | 21 | 0.0000 | 0.0000 | 0.0000 | 0.0204 |
| sbert_pretrained | Senior Software Engineer | 19 | 0.0000 | 0.0000 | 0.0000 | 0.0340 |
| sbert_pretrained | System Administrator (Operation & Maintenance of Server, Storage & Service Desk System) | 19 | 0.0000 | 0.0175 | 0.0316 | 0.0890 |
| sbert_pretrained | Data Science Engineer | 18 | 0.0556 | 0.0556 | 0.0333 | 0.1383 |
| sbert_pretrained | Executive/ Sr. Executive -IT | 17 | 0.0000 | 0.0000 | 0.0000 | 0.0246 |
| sbert_pretrained | Management Trainee - Mechanical | 17 | 0.0000 | 0.0000 | 0.0000 | 0.0130 |
| sbert_pretrained | Senior iOS Engineer | 17 | 0.0000 | 0.0196 | 0.0235 | 0.0766 |
| sbert_pretrained | Sr.Officer / Executive - Internal Audit | 11 | 0.0000 | 0.0000 | 0.0000 | 0.0033 |
| sbert_pretrained | Head of Internal Control & Compliance (ICC) - SEVP/DMD | 10 | 0.0000 | 0.0333 | 0.0200 | 0.0772 |
| sbert_pretrained | Project Coordinator (Civil) | 10 | 0.0000 | 0.0333 | 0.0400 | 0.0814 |
| sbert_pretrained | Data Engineer | 9 | 0.0000 | 0.0370 | 0.0222 | 0.0887 |
| sbert_pretrained | HR Officer | 9 | 0.0000 | 0.0000 | 0.0222 | 0.0367 |
| sbert_pretrained | Mechanical Designer | 9 | 0.0000 | 0.0370 | 0.0222 | 0.0647 |
| sbert_pretrained | Mechanical Engineer | 9 | 0.0000 | 0.0000 | 0.0000 | 0.0067 |
| sbert_pretrained | Executive - VAT | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0252 |
| sbert_pretrained | Marketing Officer | 5 | 0.0000 | 0.0667 | 0.0400 | 0.1452 |
| sbert_pretrained | Site Engineer | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0306 |
| sbert_pretrained | Civil Engineer | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0174 |
| sbert_pretrained | Business Development Executive | 2 | 0.5000 | 0.1667 | 0.1000 | 0.5227 |
| sbert_multi_qa_minilm_l6_cos_v1 | Database Administrator (DBA) | 31 | 0.0000 | 0.0000 | 0.0000 | 0.0160 |
| sbert_multi_qa_minilm_l6_cos_v1 | Machine Learning (ML) Engineer | 26 | 0.0000 | 0.0000 | 0.0000 | 0.0387 |
| sbert_multi_qa_minilm_l6_cos_v1 | Network Support Engineer | 26 | 0.0000 | 0.0128 | 0.0154 | 0.0658 |
| sbert_multi_qa_minilm_l6_cos_v1 | AI Engineer | 25 | 0.0000 | 0.0000 | 0.0080 | 0.0313 |
| sbert_multi_qa_minilm_l6_cos_v1 | Asst. Manager/ Manger (Administrative) | 23 | 0.0000 | 0.0145 | 0.0087 | 0.0199 |
| sbert_multi_qa_minilm_l6_cos_v1 | Manager- Human Resource Management (HRM) | 23 | 0.0000 | 0.0000 | 0.0087 | 0.0144 |
| sbert_multi_qa_minilm_l6_cos_v1 | DevOps Engineer | 21 | 0.0000 | 0.0000 | 0.0095 | 0.0177 |
| sbert_multi_qa_minilm_l6_cos_v1 | Executive/ Senior Executive- Trade Marketing, Hygiene Products | 21 | 0.0000 | 0.0159 | 0.0095 | 0.0469 |
| sbert_multi_qa_minilm_l6_cos_v1 | Full Stack Developer (Python,React js) | 21 | 0.0476 | 0.0317 | 0.0476 | 0.1361 |
| sbert_multi_qa_minilm_l6_cos_v1 | Intern (Generative AI Engineering - 2D/3D Image Generation) | 21 | 0.0000 | 0.0159 | 0.0095 | 0.0468 |
| sbert_multi_qa_minilm_l6_cos_v1 | Senior Software Engineer | 19 | 0.0000 | 0.0000 | 0.0000 | 0.0122 |
| sbert_multi_qa_minilm_l6_cos_v1 | System Administrator (Operation & Maintenance of Server, Storage & Service Desk System) | 19 | 0.0000 | 0.0000 | 0.0000 | 0.0123 |
| sbert_multi_qa_minilm_l6_cos_v1 | Data Science Engineer | 18 | 0.0000 | 0.0000 | 0.0000 | 0.0287 |
| sbert_multi_qa_minilm_l6_cos_v1 | Executive/ Sr. Executive -IT | 17 | 0.0000 | 0.0196 | 0.0235 | 0.0511 |
| sbert_multi_qa_minilm_l6_cos_v1 | Management Trainee - Mechanical | 17 | 0.0000 | 0.0000 | 0.0000 | 0.0031 |
| sbert_multi_qa_minilm_l6_cos_v1 | Senior iOS Engineer | 17 | 0.0000 | 0.0000 | 0.0118 | 0.0557 |
| sbert_multi_qa_minilm_l6_cos_v1 | Sr.Officer / Executive - Internal Audit | 11 | 0.0000 | 0.0000 | 0.0000 | 0.0025 |
| sbert_multi_qa_minilm_l6_cos_v1 | Head of Internal Control & Compliance (ICC) - SEVP/DMD | 10 | 0.0000 | 0.0333 | 0.0200 | 0.1060 |
| sbert_multi_qa_minilm_l6_cos_v1 | Project Coordinator (Civil) | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0144 |
| sbert_multi_qa_minilm_l6_cos_v1 | Data Engineer | 9 | 0.0000 | 0.0741 | 0.0444 | 0.1263 |
| sbert_multi_qa_minilm_l6_cos_v1 | HR Officer | 9 | 0.0000 | 0.0000 | 0.0222 | 0.0388 |
| sbert_multi_qa_minilm_l6_cos_v1 | Mechanical Designer | 9 | 0.0000 | 0.0000 | 0.0000 | 0.0132 |
| sbert_multi_qa_minilm_l6_cos_v1 | Mechanical Engineer | 9 | 0.0000 | 0.0370 | 0.0222 | 0.0406 |
| sbert_multi_qa_minilm_l6_cos_v1 | Executive - VAT | 8 | 0.0000 | 0.0000 | 0.0250 | 0.0495 |
| sbert_multi_qa_minilm_l6_cos_v1 | Marketing Officer | 5 | 0.0000 | 0.0667 | 0.0400 | 0.1612 |
| sbert_multi_qa_minilm_l6_cos_v1 | Site Engineer | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0068 |
| sbert_multi_qa_minilm_l6_cos_v1 | Civil Engineer | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0028 |
| sbert_multi_qa_minilm_l6_cos_v1 | Business Development Executive | 2 | 0.0000 | 0.0000 | 0.1000 | 0.1293 |
| sbert_finetuned | Database Administrator (DBA) | 31 | 0.0000 | 0.0108 | 0.0194 | 0.0650 |
| sbert_finetuned | Machine Learning (ML) Engineer | 26 | 0.0385 | 0.0256 | 0.0154 | 0.1080 |
| sbert_finetuned | Network Support Engineer | 26 | 0.0769 | 0.0385 | 0.0231 | 0.1405 |
| sbert_finetuned | AI Engineer | 25 | 0.0000 | 0.0400 | 0.0240 | 0.0770 |
| sbert_finetuned | Asst. Manager/ Manger (Administrative) | 23 | 0.0435 | 0.0145 | 0.0087 | 0.0731 |
| sbert_finetuned | Manager- Human Resource Management (HRM) | 23 | 0.0000 | 0.0145 | 0.0087 | 0.0318 |
| sbert_finetuned | DevOps Engineer | 21 | 0.0000 | 0.0000 | 0.0000 | 0.0068 |
| sbert_finetuned | Executive/ Senior Executive- Trade Marketing, Hygiene Products | 21 | 0.0000 | 0.0000 | 0.0000 | 0.0440 |
| sbert_finetuned | Full Stack Developer (Python,React js) | 21 | 0.0000 | 0.0000 | 0.0000 | 0.0341 |
| sbert_finetuned | Intern (Generative AI Engineering - 2D/3D Image Generation) | 21 | 0.0000 | 0.0000 | 0.0000 | 0.0132 |
| sbert_finetuned | Senior Software Engineer | 19 | 0.0000 | 0.0000 | 0.0000 | 0.0259 |
| sbert_finetuned | System Administrator (Operation & Maintenance of Server, Storage & Service Desk System) | 19 | 0.0526 | 0.0351 | 0.0211 | 0.1272 |
| sbert_finetuned | Data Science Engineer | 18 | 0.0000 | 0.0185 | 0.0111 | 0.0470 |
| sbert_finetuned | Executive/ Sr. Executive -IT | 17 | 0.0000 | 0.0000 | 0.0000 | 0.0092 |
| sbert_finetuned | Management Trainee - Mechanical | 17 | 0.0000 | 0.0000 | 0.0000 | 0.0063 |
| sbert_finetuned | Senior iOS Engineer | 17 | 0.0000 | 0.0196 | 0.0118 | 0.0421 |
| sbert_finetuned | Sr.Officer / Executive - Internal Audit | 11 | 0.0000 | 0.0000 | 0.0000 | 0.0130 |
| sbert_finetuned | Head of Internal Control & Compliance (ICC) - SEVP/DMD | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0728 |
| sbert_finetuned | Project Coordinator (Civil) | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0311 |
| sbert_finetuned | Data Engineer | 9 | 0.0000 | 0.0000 | 0.0000 | 0.0228 |
| sbert_finetuned | HR Officer | 9 | 0.0000 | 0.0000 | 0.0000 | 0.0049 |
| sbert_finetuned | Mechanical Designer | 9 | 0.0000 | 0.0370 | 0.0222 | 0.0392 |
| sbert_finetuned | Mechanical Engineer | 9 | 0.0000 | 0.0000 | 0.0000 | 0.0047 |
| sbert_finetuned | Executive - VAT | 8 | 0.0000 | 0.0417 | 0.0250 | 0.0737 |
| sbert_finetuned | Marketing Officer | 5 | 0.2000 | 0.0667 | 0.0400 | 0.2113 |
| sbert_finetuned | Site Engineer | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0060 |
| sbert_finetuned | Civil Engineer | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0351 |
| sbert_finetuned | Business Development Executive | 2 | 0.0000 | 0.1667 | 0.2000 | 0.2667 |
| hybrid_tfidf_sbert | Database Administrator (DBA) | 31 | 0.0000 | 0.0000 | 0.0065 | 0.0653 |
| hybrid_tfidf_sbert | Machine Learning (ML) Engineer | 26 | 0.0385 | 0.0513 | 0.0385 | 0.1509 |
| hybrid_tfidf_sbert | Network Support Engineer | 26 | 0.0000 | 0.0256 | 0.0385 | 0.1076 |
| hybrid_tfidf_sbert | AI Engineer | 25 | 0.0400 | 0.0400 | 0.0320 | 0.1349 |
| hybrid_tfidf_sbert | Asst. Manager/ Manger (Administrative) | 23 | 0.0435 | 0.0435 | 0.0348 | 0.1479 |
| hybrid_tfidf_sbert | Manager- Human Resource Management (HRM) | 23 | 0.0000 | 0.0435 | 0.0261 | 0.1010 |
| hybrid_tfidf_sbert | DevOps Engineer | 21 | 0.0000 | 0.0317 | 0.0190 | 0.0901 |
| hybrid_tfidf_sbert | Executive/ Senior Executive- Trade Marketing, Hygiene Products | 21 | 0.0000 | 0.0000 | 0.0095 | 0.0560 |
| hybrid_tfidf_sbert | Full Stack Developer (Python,React js) | 21 | 0.0952 | 0.0317 | 0.0286 | 0.1696 |
| hybrid_tfidf_sbert | Intern (Generative AI Engineering - 2D/3D Image Generation) | 21 | 0.0476 | 0.0159 | 0.0095 | 0.1008 |
| hybrid_tfidf_sbert | Senior Software Engineer | 19 | 0.0000 | 0.0000 | 0.0105 | 0.0737 |
| hybrid_tfidf_sbert | System Administrator (Operation & Maintenance of Server, Storage & Service Desk System) | 19 | 0.0000 | 0.0175 | 0.0105 | 0.0924 |
| hybrid_tfidf_sbert | Data Science Engineer | 18 | 0.0000 | 0.0556 | 0.0556 | 0.1472 |
| hybrid_tfidf_sbert | Executive/ Sr. Executive -IT | 17 | 0.0588 | 0.0392 | 0.0235 | 0.1670 |
| hybrid_tfidf_sbert | Management Trainee - Mechanical | 17 | 0.0588 | 0.0196 | 0.0235 | 0.1225 |
| hybrid_tfidf_sbert | Senior iOS Engineer | 17 | 0.0000 | 0.0000 | 0.0000 | 0.0550 |
| hybrid_tfidf_sbert | Sr.Officer / Executive - Internal Audit | 11 | 0.0000 | 0.0000 | 0.0545 | 0.1057 |
| hybrid_tfidf_sbert | Head of Internal Control & Compliance (ICC) - SEVP/DMD | 10 | 0.0000 | 0.0000 | 0.0200 | 0.0856 |
| hybrid_tfidf_sbert | Project Coordinator (Civil) | 10 | 0.0000 | 0.0333 | 0.0200 | 0.1247 |
| hybrid_tfidf_sbert | Data Engineer | 9 | 0.1111 | 0.0741 | 0.0444 | 0.1929 |
| hybrid_tfidf_sbert | HR Officer | 9 | 0.0000 | 0.0741 | 0.0444 | 0.1474 |
| hybrid_tfidf_sbert | Mechanical Designer | 9 | 0.0000 | 0.0741 | 0.0444 | 0.1449 |
| hybrid_tfidf_sbert | Mechanical Engineer | 9 | 0.0000 | 0.0000 | 0.0222 | 0.0395 |
| hybrid_tfidf_sbert | Executive - VAT | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0550 |
| hybrid_tfidf_sbert | Marketing Officer | 5 | 0.0000 | 0.0667 | 0.0400 | 0.1700 |
| hybrid_tfidf_sbert | Site Engineer | 4 | 0.0000 | 0.0000 | 0.0500 | 0.0902 |
| hybrid_tfidf_sbert | Civil Engineer | 3 | 0.0000 | 0.0000 | 0.0667 | 0.0934 |
| hybrid_tfidf_sbert | Business Development Executive | 2 | 0.0000 | 0.1667 | 0.1000 | 0.1839 |
| hybrid_tfidf_sbert_normalized | Database Administrator (DBA) | 31 | 0.0323 | 0.0108 | 0.0129 | 0.0965 |
| hybrid_tfidf_sbert_normalized | Machine Learning (ML) Engineer | 26 | 0.0385 | 0.0256 | 0.0308 | 0.1321 |
| hybrid_tfidf_sbert_normalized | Network Support Engineer | 26 | 0.0000 | 0.0256 | 0.0538 | 0.1308 |
| hybrid_tfidf_sbert_normalized | AI Engineer | 25 | 0.0400 | 0.0533 | 0.0400 | 0.1589 |
| hybrid_tfidf_sbert_normalized | Asst. Manager/ Manger (Administrative) | 23 | 0.0870 | 0.0580 | 0.0522 | 0.1938 |
| hybrid_tfidf_sbert_normalized | Manager- Human Resource Management (HRM) | 23 | 0.0000 | 0.0290 | 0.0348 | 0.1038 |
| hybrid_tfidf_sbert_normalized | DevOps Engineer | 21 | 0.0476 | 0.0476 | 0.0286 | 0.1375 |
| hybrid_tfidf_sbert_normalized | Executive/ Senior Executive- Trade Marketing, Hygiene Products | 21 | 0.0000 | 0.0000 | 0.0286 | 0.0820 |
| hybrid_tfidf_sbert_normalized | Full Stack Developer (Python,React js) | 21 | 0.0952 | 0.0794 | 0.0476 | 0.1965 |
| hybrid_tfidf_sbert_normalized | Intern (Generative AI Engineering - 2D/3D Image Generation) | 21 | 0.0952 | 0.0317 | 0.0190 | 0.1518 |
| hybrid_tfidf_sbert_normalized | Senior Software Engineer | 19 | 0.0526 | 0.0175 | 0.0105 | 0.1167 |
| hybrid_tfidf_sbert_normalized | System Administrator (Operation & Maintenance of Server, Storage & Service Desk System) | 19 | 0.0000 | 0.0351 | 0.0316 | 0.1017 |
| hybrid_tfidf_sbert_normalized | Data Science Engineer | 18 | 0.0556 | 0.0556 | 0.0333 | 0.1673 |
| hybrid_tfidf_sbert_normalized | Executive/ Sr. Executive -IT | 17 | 0.0588 | 0.0196 | 0.0118 | 0.1243 |
| hybrid_tfidf_sbert_normalized | Management Trainee - Mechanical | 17 | 0.0000 | 0.0000 | 0.0000 | 0.0667 |
| hybrid_tfidf_sbert_normalized | Senior iOS Engineer | 17 | 0.0000 | 0.0000 | 0.0118 | 0.0664 |
| hybrid_tfidf_sbert_normalized | Sr.Officer / Executive - Internal Audit | 11 | 0.0000 | 0.0303 | 0.0545 | 0.1063 |
| hybrid_tfidf_sbert_normalized | Head of Internal Control & Compliance (ICC) - SEVP/DMD | 10 | 0.0000 | 0.0333 | 0.0200 | 0.1033 |
| hybrid_tfidf_sbert_normalized | Project Coordinator (Civil) | 10 | 0.0000 | 0.0000 | 0.0400 | 0.1156 |
| hybrid_tfidf_sbert_normalized | Data Engineer | 9 | 0.1111 | 0.0370 | 0.0444 | 0.1993 |
| hybrid_tfidf_sbert_normalized | HR Officer | 9 | 0.0000 | 0.0370 | 0.0222 | 0.1002 |
| hybrid_tfidf_sbert_normalized | Mechanical Designer | 9 | 0.0000 | 0.0000 | 0.0000 | 0.0757 |
| hybrid_tfidf_sbert_normalized | Mechanical Engineer | 9 | 0.0000 | 0.0000 | 0.0222 | 0.1011 |
| hybrid_tfidf_sbert_normalized | Executive - VAT | 8 | 0.0000 | 0.0417 | 0.0250 | 0.1002 |
| hybrid_tfidf_sbert_normalized | Marketing Officer | 5 | 0.0000 | 0.1333 | 0.1200 | 0.2700 |
| hybrid_tfidf_sbert_normalized | Site Engineer | 4 | 0.2500 | 0.0833 | 0.0500 | 0.2758 |
| hybrid_tfidf_sbert_normalized | Civil Engineer | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0667 |
| hybrid_tfidf_sbert_normalized | Business Development Executive | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0729 |
