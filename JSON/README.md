# JSON files
The GUI of cedar uses JSON files to save and load its architectures. To make this repository usable from the cedar files it also loads from these files. Some of the files in this directory are therefore architectures as saved by cedar. 

## Cedar architectures
- mental_imagery_extended.json : This architecture is the original architecture as specified in [A Neural Dynamic Architecture That Autonomously Builds Mental Models](https://www.ini.rub.de/upload/file/1535484449_7a08dac58161aae9652b/kounatidou_05_14.pdf) with additional cardinal directions added. 
- mental_imagery_extended_recording.json : This architecture is the same as the one above with the difference that cedar will save outputs of some modules to disk.
- test_architecture.json : This is a small test architecture with the sole purpose to test if the loading of cedar architectures works correctly. 

## Experiment templates
cedar uses so called experiment files to run repeatable experiments. The _2PremisesTemplate.json_, _3PremisesTemplate.json_ and _4PremisesTemplate.json_ files are templates for such experiments where the spatial relations and the objects in a scene have to be set, but the rest is fixed. 

## Additional files
The _mental_imagery_plot_widget.json_ file defines how to plot the spatial reasoning architecture during a run in the GUI.
