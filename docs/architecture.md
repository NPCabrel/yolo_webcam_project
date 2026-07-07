# Architecture du Système

architecture_figma_v2.png

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




# UML Diagrams

| Diagram	       |      File	                     |       Description |
| ----------------|----------------------------------|-------------------- |
| Class	        |       class_diagram.puml	       |      Code structure |
| Sequence	     |     sequence_diagram.puml	    |         Execution flow |
| State	          |    state_diagram.puml	        |     State machine |
| Use Case	       |   usecase_diagram.puml	        |     Use cases |
| Deployment	    |      deployment_diagram.puml	|         System architecture |
| Components	    |      component_diagram.puml	|         Modules |
| Data Flow	       |   dataflow_diagram.puml	    |         Data flow |
| Activity	        |  activity_diagram.puml	    |         Business logic |
| Object	        |      object_diagram.puml	    |         Runtime example |
| Communication	    |  communication_diagram.puml	|     Component interaction |