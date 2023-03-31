1. teorico: idea di usare bayes point su classificatori rulesets
2. sperimentale: 
   a. interpretabilità migliorata => bisogna fare esperimenti
   b. accuratezza paragonabile a classificatori tradizionali (non interpretabili)

AQ => mediare altre ipotesi 
interpretabilità => ... (esempi)

altri dataset => confronti più approfonditi 
 - dataset artificiali


BO lo teniamo? => sì (su esann manco c'è)


ragionamenti su ipotesi generali e specifiche 
=> dipende da contesto di applicazione (e.g., in TTT non vorresti "FP".)



più bin aggiungiamo, più aumentiamo la complessità della regola. (disgiuunzioni)

il numero di congiunzioni ha a che fare con la generalità. 
con meno congiunzioni è più generale.

find-rs cerca meno complessa ma più specifica.
AQ ha tantissimi "bin", quindi la complessità della regola aumenta.


x0 == vhigh  ^ x1 in (whigh, high)

=> x0 == vhigh ^ x1 == vhigh
or x0 == vhigh ^ x1 == high

|V|^N (N=#features, |V|=#attributi in or)


