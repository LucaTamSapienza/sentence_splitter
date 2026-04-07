 Threshold   Precision    Recall        F1
--------------------------------------------
      |0.10      0.9782    0.9961    0.9869 <- BEST|
      |0.15      0.9811    0.9949    0.9879 <- BEST|
      |0.20      0.9833    0.9943    0.9888 <- BEST|
      |0.25      0.9845    0.9937    0.9891 <- BEST|
      |0.30      0.9863    0.9933    0.9898 <- BEST|
      |0.35      0.9885    0.9924    0.9905 <- BEST|
      |0.40      0.9901    0.9924    0.9913 <- BEST|
      |0.45      0.9906    0.9924    0.9915 <- BEST|
      |0.50      0.9913    0.9917    0.9915 <- BEST|
      |0.55      0.9920    0.9912    0.9916 <- BEST|
      |0.60      0.9923    0.9907    0.9915|
      |0.65      0.9936    0.9902    0.9919 <- BEST|
      |0.70      0.9942    0.9887    0.9914|
      |0.75      0.9948    0.9863    0.9905|
      |0.80      0.9957    0.9825    0.9890|
      |0.85      0.9964    0.9772    0.9867|
      |0.90      0.9969    0.9706    0.9835|

Best threshold selected: 0.65 (dev F1: 0.9919)

--- Final Evaluation on Test Sets ---
Dataset                                        P       R      F1   #Gold
--------------------------------------------------------------------
UD_English-EWT/en_ewt-ud-test.sent_split  0.9769  0.9379  0.9570    2077
UD_English-GUM/en_gum-ud-test.sent_split  0.9882  0.9699  0.9790    1464
UD_English-PUD/en_pud-ud-test.sent_split  0.9756  1.0000  0.9877    1000
UD_English-ParTUT/en_partut-ud-test.sent  0.9935  1.0000  0.9967     153
UD_Italian-ISDT/it_isdt-ud-test.sent_spl  0.9938  1.0000  0.9969     482
UD_Italian-MarkIT/it_markit-ud-test.sent  1.0000  0.9941  0.9971     340
UD_Italian-ParTUT/it_partut-ud-test.sent  1.0000  1.0000  1.0000     153
UD_Italian-VIT/it_vit-ud-test.sent_split  0.9609  0.9916  0.9760    1067
--------------------------------------------------------------------
MACRO AVG                                                 0.9863