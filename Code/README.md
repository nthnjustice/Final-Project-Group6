# Code

## IMPORTANT STEP!

The shapefile (a composite file type made up of a bunch of other files) for vhf_locations_1997to2012_pip is too large to store on Git (zipping did not solve the problem). Therefore it's stored on Google Drive and will need to be downloaded in order to run CleanSheepData.py. However, all other scripts will run independent of CleanSheepData.py because the cleaned data output of this code is already available. 

## Instructions for retrieving this data:
1) Download the zip file: https://drive.google.com/file/d/1sJgyUdxDZGlJnzHmzlLJ2Dj3xlK7wyo5/view?usp=sharing
2) Extract the contents into Final-Project-Group6 > Code > Data > Sheep > Source
3) All contents of this zip should be in the same level as study_sheep_1995to2015.csv and vhf_locations_1997to2012.csv

## Execution

All scripts in Preprocessing and Modeling.py can be run individually (following the instructions above for CleanSheepData.py) or all at once using Main.py.

### NOTE when using Main.py

The imported scripts are not proper modules and so the current working directory must be ~Code aka ~Final-Project-Group6/Code aka same level as Main.py

## Execution Order

Order doesn't technically matter because both source and clean data are both available. If recreating my process, run the preprocessing script in the order listed in Main.py, followed by Modeling.py.

## Descriptions

### Preprocessing > CleanHerdData.py

Takes compiled_data_1970to2015.csv, hc_bighorn_herds_latlon.shp, and pneumonia_status_x.csv in Data > Herd > Source and outputs cleaned versions in Data > Herd > Clean.
Output files are formatted to be compiled into a model inputs table.

### Preprocessing > CleanSheepData.py

Takes vhf_locations_1970to2012.csv, study_sheep_1995to2015, and vhf_locations_1970to2012_pip.shp (read instructions above) in Data > Sheep > Clean and outputs cleaned versions in Data > Sheep > Clean. Output files are formatted to be compiled into a model inputs table.

### Preprocessing > CleanTranslocationData.py

Take translocation_events_1997to2016 in Data > Translocation > Source and outputs cleaned version in Data > Translocation > CLean. Output file is formatted to be compiled into a model inputs table. 

### Preprocessing > BuildInputData.py

Uses the cleaned datasets configured by the scripts above to construct a final dataset to be used in the modeling process and outputs results in Data > Input > inputs.csv.

### Main.py

Runs all of the scripts above, but read top instructions regarding importing large shapefile for CleanSheepData.py and for how the working directory should be set.