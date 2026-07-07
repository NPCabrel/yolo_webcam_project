# Architecture du Système

## Vue d'ensemble

┌─────────────────────────────────────────────────────────────────┐
│ YOLO Security System │
├─────────────────────────────────────────────────────────────────┤
│ │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐ │
│ │ WebcamStream│───▶│YOLODetector │───▶│ DistanceAlarm │ │
│ │ (hardware) │ │ (model) │ │ (business) │ │
│ └─────────────┘ └─────────────┘ └─────────────────┘ │
│ │ │ │ │
│ ▼ ▼ ▼ │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ Main Loop (orchestrator) │ │
│ │ - read frame → detect → check alarm → save screenshot │ │
│ └──────────────────────────────────────────────────────────┘ │
│ │ │ │ │
│ ▼ ▼ ▼ │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐ │
│ │ UI │ │ Screenshot │ │ Utils │ │
│ │ (display) │ │ Manager │ │ (helpers) │ │
│ └─────────────┘ └─────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

## Flux de données

1. **WebcamStream** capture une frame
2. **YOLODetector** détecte les objets
3. **DistanceAlarm** vérifie si des personnes sont trop proches
4. **ScreenshotManager** sauvegarde l'image si déclenché
5. **UI** affiche le résultat

## Design Patterns utilisés

- **Singleton**: Configuration globale
- **Factory**: Création des composants
- **Strategy**: Plugins de détection (futur)
- **Observer**: Pour les événements (alarme, screenshot)