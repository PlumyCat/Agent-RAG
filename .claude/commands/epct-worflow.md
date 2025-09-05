# EPCT - MCP Python Workflow

**Explore → Plan → Code → Test** workflow optimisé pour les serveurs MCP Python.

## Processus EPCT MCP

### 🔍 **Explore** (Analyse parallèle)
- **Codebase** : structure serveur, outils existants, handlers
- **Protocole MCP** : messages supportés, schémas JSON  
- **Tests** : couverture actuelle, cas manquants
- **Dépendances** : versions MCP SDK, compatibilité
- **Documentation** : README, docstrings, exemples

### 📋 **Plan** (Planification détaillée)
Produire un plan concret d'implémentation :
- **Objectif** : fonctionnalité à ajouter/modifier
- **Impact** : outils affectés, breaking changes
- **Architecture** : modifications serveur/handlers
- **Tests** : stratégie de test et validation
- **Étapes** : séquence d'implémentation incrémentale

### 💻 **Code** (Implémentation)
Implémenter de manière incrémentale :
- **Schémas** : définition JSON des outils
- **Handlers** : logique métier des outils MCP
- **Validation** : inputs/outputs selon protocole
- **Error handling** : gestion robuste des erreurs
- **Logging** : traces pour debugging

### 🧪 **Test** (Validation & Loop)
Exécuter et itérer jusqu'au succès :
- **Tests unitaires** : chaque outil individuellement
- **Tests MCP** : communication avec inspector
- **Tests intégration** : Claude Code + serveur MCP
- **Quality checks** : black, mypy, couverture
- **Loop until green** : répéter corrections jusqu'au succès

## Critères de Done

✅ **Fonctionnalité** 
- Outil MCP fonctionne comme spécifié
- Messages conformes au protocole MCP
- Validation inputs/outputs correcte

✅ **Qualité**
- Tests passent (>80% couverture)
- Code formaté (black, isort)  
- Type hints validés (mypy)
- Pas d'erreurs de linting

✅ **Intégration**
- Serveur démarre sans erreur
- Inspector MCP valide la communication
- Claude Code peut utiliser le serveur
- Documentation mise à jour

## Automatisation

Utilise les formatters/linters automatiquement pendant le développement et invoque le **python-test-runner** subagent proactivement pour valider chaque étape.

En cas de complexité élevée, utilise le mode **thinking** pour l'analyse et la planification.