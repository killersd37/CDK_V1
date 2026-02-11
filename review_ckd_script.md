# Revue technique critique — Script CKD (stades IRC)

## Forces du projet

- **Lisibilité et pédagogie** : le script est bien séquencé (chargement, EDA, preprocessing, entraînement, évaluation, sauvegarde), ce qui facilite la relecture par une équipe clinique/DS.
- **Stratification train/test** : l’usage de `stratify=y` est approprié pour limiter un biais de répartition inter-classes.
- **Prise en compte initiale du déséquilibre** : `class_weight='balanced'` dans la grille RF va dans le bon sens.
- **Traçabilité minimale des artefacts** : sauvegarde du modèle (`pkl`) et des noms de features.

---

## Faiblesses critiques

1. **Cible construite par formule déterministe puis “prédite”**
   - `Stade_CKD` est calculé exclusivement à partir de `creatinine`, `age`, `sexe` via MDRD puis seuils de stade.
   - Le modèle apprend donc principalement une règle déjà connue plutôt qu’un phénomène clinique latent.
   - En clinique, cela est une **reconstruction indirecte d’une équation**, pas une vraie prédiction pronostique/diagnostique.

2. **Fuite de données au prétraitement**
   - Imputation (`SimpleImputer.fit_transform`) et one-hot encoding sont faits sur `df` complet **avant** split train/test.
   - Cela introduit de l’information du test set dans l’entraînement (distribution globale, modalités futures), gonflant artificiellement les performances.

3. **Validation méthodologique incomplète pour un usage médical**
   - Une seule coupe hold-out + CV interne GridSearch sans **validation externe** ni **bootstrap CI**.
   - Aucune analyse de calibration, aucune courbe décisionnelle, aucune utilité clinique quantifiée.

4. **Métriques insuffisantes pour classes ordinales**
   - La cible est ordinale (stades 1→5), mais l’évaluation traite cela comme multiclasses nominales (`f1_weighted`, matrice de confusion standard).
   - Les erreurs “adjacentes” (stade 3 vs 4) et “graves” (1 vs 5) ne sont pas distinguées.

5. **Fonction d’inférence incohérente avec le modèle entraîné**
   - `predire_stade_patient` ne consomme pas réellement `best_model`; elle recalcule le stade via eGFR et retourne le stade déterministe.
   - Le pipeline déployé n’est donc pas celui évalué.

---

## Erreurs conceptuelles

1. **Confusion formule clinique vs ML**
   - Objectif implicite : “prédire le stade CKD”.
   - Réalité : le stade est un simple mapping de l’eGFR calculé par formule MDRD, lui-même fonction de 3 variables disponibles.
   - Le modèle RF ne crée pas de nouvelle connaissance; il approxime une table de décision déjà explicite.

2. **Définition clinique incomplète de la CKD**
   - La CKD n’est pas définie uniquement par un eGFR ponctuel : il faut la **persistance ≥ 3 mois** et/ou marqueurs de dommage rénal (albuminurie, imagerie, histologie).
   - Ce script classe des stades de fonction rénale à un instant T, pas une vraie maladie chronique confirmée.

3. **Choix de formule potentiellement obsolète/non contextualisé**
   - MDRD est historiquement utilisée mais CKD-EPI 2009/2021 est souvent préférée (meilleure précision selon populations).
   - Absence de justification locale (population béninoise, étalonnage créatinine IDMS, unité standardisée).

4. **Interprétabilité mal cadrée**
   - `feature_importances_` RF (impureté) est biaisée vers variables à haute cardinalité/variance.
   - Sans permutation importance, SHAP, stabilité inter-fold et contrôle de corrélation, l’interprétation causale est risquée.

---

## Risques en contexte réel (hôpital / production)

- **Risque de sur-confiance clinique** : performances potentiellement surestimées par fuite de données.
- **Décisions thérapeutiques inadaptées** : confusion entre “stade eGFR calculé” et “diagnostic CKD chronique”.
- **Non-reproductibilité réglementaire** : pipeline de preprocessing non encapsulé (pas de `Pipeline/ColumnTransformer` sérialisé complet).
- **Drift non monitoré** : aucune stratégie de surveillance (distribution des labos, calibration drift, performance par sous-groupes).
- **Biais et équité non évalués** : pas d’audit par sexe/âge/site/hypertension/diabète.
- **Faible robustesse opérationnelle** : inférence non alignée avec entraînement, absence de gestion stricte des unités/valeurs hors bornes.

---

## Recommandations concrètes

1. **Reformuler le problème clinique**
   - Option A (utile) : prédire la **progression** (déclin eGFR, passage au stade supérieur à 12 mois, ESKD, dialyse, décès).
   - Option B : prédire la CKD confirmée selon critères temporels + albuminurie.

2. **Corriger le pipeline anti-fuite**
   - Split patient-level (et temporel si longitudinal) avant tout fit.
   - Utiliser `Pipeline` + `ColumnTransformer` (imputation/encodage uniquement sur train folds).

3. **Validation robuste**
   - Nested CV pour tuning + estimation non biaisée.
   - Validation externe (autre hôpital/période).
   - Intervalles de confiance (bootstrap) pour chaque métrique.

4. **Métriques cliniques adaptées**
   - Si classes ordinales : MAE ordinal, quadratic weighted kappa, macro-F1, recall classe sévère, matrice de coût clinique.
   - Si risque binaire/temps : AUROC + AUPRC + sensibilité à spécificité fixée + PPV/NPV + courbe de décision.

5. **Calibration obligatoire**
   - `CalibratedClassifierCV` (isotonic/platt), Brier score, calibration curve, ECE/MCE.

6. **Interprétabilité fiable**
   - Permutation importance sur set de test verrouillé.
   - SHAP (TreeSHAP) + stabilité inter-fold + PDP/ALE sur variables clés.
   - Documentation explicite : explicatif ≠ causal.

7. **Prêt production**
   - Sérialiser un objet unique “pipeline + modèle + métadonnées”.
   - Contrôles d’entrée (unités, bornes biologiques, valeurs aberrantes).
   - Versioning données/modèle, monitoring post-déploiement, seuils d’alerte, procédure de rollback.

---

## Version améliorée recommandée (architecture complète)

1. **Cadrage clinique**
   - Définir endpoint médical validé (ex. progression CKD à 12 mois).
   - Définir population, critères d’inclusion/exclusion, fenêtre d’observation et d’horizon.

2. **Jeu de données et gouvernance**
   - Cohorte indexée patient/temps, contrôle qualité labos, harmonisation unités.
   - Audit des manquants (MCAR/MAR/MNAR), plan d’imputation documenté.

3. **Split et validation**
   - Split temporel + groupé patient (éviter fuite inter-visites).
   - Développement: nested CV.
   - Validation externe: site B ou période ultérieure.

4. **Pipeline ML**
   - `ColumnTransformer` (num/cat), imputation train-only, encoding robuste.
   - Baselines: régression logistique ordinale, XGBoost/LightGBM, forêt calibrée.
   - Sélection modèle sur métrique clinique primaire préspécifiée.

5. **Évaluation**
   - Discrimination + calibration + utilité clinique.
   - Analyses de sous-groupes (âge, sexe, diabète, HTA).
   - IC95% bootstrap et comparaison aux standards cliniques.

6. **Interprétation et sécurité**
   - SHAP global/local + vérification de stabilité.
   - Détection OOD, seuil “abstain/referral” si incertitude élevée.

7. **Déploiement**
   - API d’inférence alignée au pipeline entraîné.
   - Monitoring continu (drift, calibration, performance réelle).
   - Documentation clinique + protocole de revalidation trimestrielle.

---

## Conclusion d’audit

En l’état, le script est un bon exercice pédagogique de classification supervisée, mais il n’est **pas scientifiquement suffisant** pour revendiquer un système de prédiction CKD cliniquement utile. Le principal verrou est conceptuel : il tente d’apprendre une cible dérivée d’une formule déterministe, avec une fuite de données au preprocessing et une validation non conforme aux exigences d’un contexte médical à risque.
