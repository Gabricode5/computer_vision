# Rapport d'audit — Compétences C6 / C7 / C8

**Projet :** `computer_vision` — Détection de posture & analyse biomécanique par LLM local
**Date :** 2026-05-21
**Évaluateur :** Claude Sonnet 4.6

---

## Vue d'ensemble du projet

Le projet est une application de vision par ordinateur composée de :
- **MediaPipe** (pose estimation + 33 keypoints corporels)
- **OpenCV** (Haar cascade pour détection de visages)
- **Ollama + llama3.2:3b** ("BOB") — LLM local pour analyse biomécanique
- **FastAPI** (backend REST)
- **Streamlit + streamlit-webrtc** (frontend temps réel)
- **Docker** (containerisation du backend)

Fichiers sources : 5 fichiers `.py` utiles, 1 `Dockerfile`, 1 `requirement.txt`, 1 `Readme.md`.

---

## C6 — Veille technique & réglementaire

### Preuves trouvées

| Critère | Statut | Preuve / Absence |
|---|---|---|
| Sources techniques référencées | ❌ ABSENT | Aucun arXiv, paper, lien doc officielle, leaderboard |
| Processus de veille documenté | ❌ ABSENT | Pas d'ADR, pas de journal, pas de changelog |
| État de l'art cité et appliqué | ⚠️ IMPLICITE | MediaPipe est utilisé sans justification écrite |
| Veille réglementaire | ❌ ABSENT | Zéro mention RGPD, AI Act, CNIL, biométrie |
| Recommandations formulées et tracées | ❌ ABSENT | Aucune |

### Analyse détaillée

**État de l'art implicite, non documenté.** Le projet choisit MediaPipe (Google, 2020), solution effectivement de référence pour la pose estimation légère sur CPU. Mais ce choix n'est justifié nulle part. Il n'y a aucune trace écrite comparant MediaPipe à OpenPose, MoveNet, YOLOv8-Pose ou MMPose. La décision d'utiliser llama3.2:3b via Ollama — un modèle 3 milliards de paramètres pour de l'analyse biomécanique — n'est pas non plus argumentée.

**Lacune réglementaire critique.** Le projet collecte et traite en temps réel :
- Des images de corps humains (33 keypoints corporels)
- Des détections de visages (`app.py:36-57`)

Ces traitements constituent des **données biométriques** au sens du RGPD (Art. 9) et de l'AI Act. Or pas une seule ligne de documentation ne mentionne :
- La base légale du traitement
- Les obligations CNIL pour la biométrie
- L'AI Act (système d'IA à risque limité ou élevé selon le contexte d'usage)
- La durée de conservation, la minimisation des données
- Les mentions légales ou politique de confidentialité

### Verdict C6 : NON VALIDÉ

Aucun processus de veille documenté, aucune source référencée, silence total sur la réglementation biométrique alors que le sujet l'impose directement.

---

## C7 — Identification de services IA existants

### Preuves trouvées

| Critère | Statut | Preuve / Absence |
|---|---|---|
| Besoin fonctionnel documenté | ❌ ABSENT | Readme = guide d'installation uniquement |
| Candidats IA identifiés | ❌ ABSENT | Aucune liste d'alternatives |
| Tableau comparatif | ❌ ABSENT | Aucun fichier benchmark, aucune matrice |
| Critères d'évaluation définis | ❌ ABSENT | Aucun critère (FPS, précision, licence, taille) |
| Justification du choix final | ❌ ABSENT | Choix posé sans argumentation |
| Alternatives écartées avec raisons | ❌ ABSENT | Aucune |
| Plan de repli documenté | ❌ ABSENT | Aucun |

### Analyse détaillée

**Deux services IA sont intégrés mais jamais benchmarkés.** Le projet utilise :

1. **MediaPipe Pose** pour la détection de posture (`pose.py:6`)
2. **Ollama / llama3.2:3b** pour l'analyse LLM (`agent.py:26`)

Pour MediaPipe, les alternatives pertinentes à benchmarker auraient été : MoveNet (Google TFLite, plus rapide), YOLOv8-Pose (Ultralytics, meilleure précision), MMPose (académique), OpenPose (historique mais lourd). Aucune mention.

Pour le LLM, les alternatives auraient pu inclure : mistral:7b, phi3, gemma:2b (plus léger), ou des APIs cloud (OpenAI, Anthropic) avec analyse coût/latence/confidentialité. Aucune mention.

Le `Readme.md` se limite à 5 commandes de lancement, sans cahier des charges, sans contraintes formulées (latence cible, précision requise, contrainte matérielle, budget, licence open-source imposée ou non).

### Verdict C7 : NON VALIDÉ

Deux services IA sont effectivement intégrés et fonctionnels, ce qui démontre une capacité d'implémentation. Mais l'absence totale de processus de benchmark, de comparatif, de critères et de justification ne permet pas de valider la compétence d'*identification et de sélection argumentée*.

---

## C8 — Paramétrage d'un service IA

### Preuves trouvées

| Critère | Statut | Preuve / Absence |
|---|---|---|
| Intégration d'un service IA | ✅ OUI | MediaPipe + Ollama intégrés et fonctionnels |
| Paramètres configurés selon doc officielle | ✅ OUI | Paramètres MediaPipe corrects (`pose.py:23-30`) |
| Configuration externalisée | ❌ ABSENT | Tout hardcodé, pas de `.env` ni `.env.example` |
| Gestion des erreurs et cas limites | ✅ PARTIEL | try/except présents, timeout configuré |
| Documentation de l'intégration | ⚠️ MINIMAL | Quelques commentaires, pas de docstrings complètes |
| Tests ou validation automatisée | ❌ ABSENT | Zéro fichier de test |
| Connecteurs isolés (séparation des responsabilités) | ⚠️ PARTIEL | FastAPI séparé, mais couplage fort dans `app.py` |

### Analyse détaillée

#### Points forts

**Intégration MediaPipe correcte.** Les paramètres sont définis selon la documentation officielle (`pose.py:23-30`) :

```python
self.pose = self.mpPose.Pose(
    static_image_mode=self.mode,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=self.upBody,
    smooth_segmentation=self.smooth,
    min_detection_confidence=self.detectionCon,
    min_tracking_confidence=self.trackCon
)
```

Les imports forcés (`pose.py:5-7`) pour contourner un bug Python 3.12 témoignent d'une résolution de problème réelle.

**Intégration Ollama conforme.** Le payload REST respecte l'API Ollama (`agent.py:25-29`) avec `model`, `prompt`, `stream: False`.

**Gestion des erreurs robuste :**
- Timeout HTTP de 25s (`app.py:73`)
- `try/except` avec `HTTPException` (`agent.py:31-35`)
- Message d'erreur utilisateur lisible (`app.py:77`)
- Fallback propre si `streamlit-webrtc` absent (`app.py:11-21`)
- Vérification `cascade.empty()` (`app.py:32-33`)
- Vérification image nulle après décodage (`app.py:133-134`)

**Architecture frontend/backend.** La séparation FastAPI / Streamlit est pertinente. Le `Dockerfile` déploie le backend de manière autonome.

#### Points faibles

**Zéro externalisation de la configuration.** L'URL Ollama est hardcodée deux fois :
- `agent.py:8` : `OLLAMA_URL = "http://127.0.0.1:11434/api/generate"`
- `app.py:73` : `http://127.0.0.1:8000`

Le nom du modèle `"llama3.2:3b"` est hardcodé (`agent.py:26`). Changer de port ou de modèle impose de modifier le code source. Il n'existe aucun `.env`, `.env.example`, ou fichier de configuration.

**Aucun test automatisé.** Pas un seul fichier `test_*.py`. Pas de test unitaire sur `generer_description()`, pas de test du endpoint `/analyze`, pas de test de la classe `poseDetector`.

**Module orphelin non documenté.** `face_detector.py` définit une fonction `visualize()` utilisant l'API MediaPipe Tasks (`face_detector.py:32-75`), mais l'application finale utilise un Haar cascade OpenCV à la place (`app.py:28-57`). Ce fichier est mort, sans explication.

**Couplage UI/logique.** La fonction `call_bob()` dans `app.py:66-77` fait un appel HTTP direct depuis le fichier UI. Aucune classe `Client` ou couche service n'isole cet appel.

**`main.py` quasi-vide.** Deux lignes (`print(mp.__version__)` + un commentaire) sans utilité fonctionnelle.

### Verdict C8 : PARTIELLEMENT VALIDÉ

L'intégration technique est réelle et fonctionnelle. Les paramètres des deux services IA (MediaPipe, Ollama) sont correctement configurés. La gestion des erreurs de base est présente. Mais l'absence de configuration externalisée, l'absence totale de tests, un module mort non expliqué et un couplage UI/service non résolu constituent des lacunes trop importantes pour une validation complète.

---

## Synthèse exécutive

| Compétence | Verdict | Score estimé |
|---|---|---|
| **C6** — Veille technique & réglementaire | ❌ NON VALIDÉ | ~0/20 |
| **C7** — Identification de services IA existants | ❌ NON VALIDÉ | ~2/20 |
| **C8** — Paramétrage d'un service IA | ⚠️ PARTIELLEMENT VALIDÉ | ~11/20 |

---

## Recommandations prioritaires

### C6

1. Rédiger un document `docs/veille_technologique.md` citant au minimum 3 sources (paper MediaPipe BlazePose, doc Ollama, un benchmark pose estimation type Papers With Code).
2. Ajouter une section RGPD/biométrie au README expliquant la base légale et les mesures de protection des données.

### C7

1. Créer un fichier `docs/benchmark_services_ia.md` avec un tableau comparatif (MediaPipe vs MoveNet vs YOLOv8-Pose ; llama3.2:3b vs mistral:7b vs gemma:2b).
2. Définir et documenter les critères de sélection (latence cible, précision, fonctionnement offline, licence).

### C8

1. Créer un `.env.example` exposant `OLLAMA_URL`, `AGENT_URL`, `OLLAMA_MODEL`.
2. Supprimer ou documenter `face_detector.py` (module orphelin).
3. Ajouter au minimum 3 tests : `test_generer_description()`, `test_analyze_endpoint()`, `test_poseDetector_init()`.
