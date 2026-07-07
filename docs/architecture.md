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

1. **WebcamStream** capture a frame
2. **YOLODetector** detecte  objets
3. **DistanceAlarm** verify if the personnes aren't far awy
4. **ScreenshotManager** save the image if detected
5. **UI** show the result

## Design Patterns utilisés

- **Singleton**: globla Configurationx
- **Factory**: Creation of composants
- **Strategy**: Plugins of detection (futur)
- **Observer**: for  events (alarm, screenshot)