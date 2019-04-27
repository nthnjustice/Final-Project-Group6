# import dependencies
import math
import numpy as np
import pandas as pd
import geopandas as gpd

pd.options.mode.chained_assignment = None

####################################################################################################
# initialize model inputs table with herd and pneumonia status data
####################################################################################################

# load cleaned herd and pneumonia status data
herds = gpd.read_file('Data/Herd/Clean/hc_bighorn_herds_latlon.shp')
pnStatus = pd.read_csv('Data/Herd/Clean/pneumonia_status_1995to2015.csv')

# initialize storage for unique herd names and years in study
names = herds['NM_Label'].tolist()
years = [str(i) for i in range(1995, 2016)]

# change dataframe indices
herds.set_index('NM_Label', inplace=True)
pnStatus.set_index('Pop', inplace=True)

# purpose: reclassify pneumonia status to binary variables
# input: pneumonia status
# output: 0 if status is healthy, 1 otherwise
def classifyStatus(status):
  if status == 'Healthy':
    return 0
  elif status == 'Outbreak':
    return 1
  elif status == 'Infected - Lambs':
    return 1
  elif status == 'Infected - All Ages':
    return 1
  elif status == 'No Population':
    return np.NaN
  else:
    return np.NaN

# initialize storage for output object
inputs = {'Herd': [], 'Year': [], 'Status': [], 'StatusClass': [], 'Area': [], 'Perimeter': []}

# loop through all unique herds and years
for i in names:
  for j in years:
    # populate the output object with information about year, herd geography, and pneumonia status
    inputs['Herd'].append(i)
    inputs['Year'].append(j)
    inputs['Status'].append(pnStatus.loc[i, j])
    inputs['StatusClass'].append(classifyStatus(pnStatus.loc[i, j]))
    inputs['Area'].append(herds.loc[i, 'AREA'])
    inputs['Perimeter'].append(herds.loc[i, 'PERIMETER'])

# convert output object to dataframe
inputs = pd.DataFrame(inputs)

####################################################################################################
# add compiled population data to model inputs table
####################################################################################################

# load cleaned compiled data
compiled = pd.read_csv('Data/Herd/Clean/compiled_data_1970to2015.csv')

# filter compiled population data by herds and years in study
compiled = compiled[compiled['Pop'].isin(names)]
compiled = compiled[compiled['year'].isin(years)]

# initialize storage for compiled population data to append to output object
populations = {'PopTot': [], 'PopAdults': [], 'Ewes': [], 'Rams': [], 'Lambs': []}

# loop through all unique herds and years
for i in names:
  for j in years:
    # subset the compiled population information for the herd and year in focus
    subset = compiled[compiled['Pop'] == i]
    subset = subset[subset['year'] == int(j)]
    subset.reset_index(drop=True, inplace=True)

    if subset.shape[0] == 1:
      # append compiled popualtion data matching the herd and year in focus
      populations['PopTot'].append(subset.loc[0, 'PopTot'])
      populations['PopAdults'].append(subset.loc[0, 'PopAdults'])
      populations['Ewes'].append(subset.loc[0, 'Ewes'])
      populations['Rams'].append(subset.loc[0, 'Rams'])
      populations['Lambs'].append(subset.loc[0, 'Lambs'])
    else:
      # append default values
      populations['PopTot'].append(np.NaN)
      populations['PopAdults'].append(np.NAN)
      populations['Ewes'].append(np.NaN)
      populations['Rams'].append(np.NaN)
      populations['Lambs'].append(np.NaN)

# append compiled population data to output object
inputs['PopTot'] = populations['PopTot']
inputs['PopAdults'] = populations['PopAdults']
inputs['Ewes'] = populations['Ewes']
inputs['Rams'] = populations['Rams']
inputs['Lambs'] = populations['Lambs']

####################################################################################################
# add sheep demographic data to model inputs table
####################################################################################################

# load cleaned sheep demographic data
sheep = pd.read_csv('Data/Sheep/Clean/study_sheep_1995to2015.csv')

# initialize storage for sheep demographic data to append to output object
translocated = {'NonResident': [], 'RT': [], 'T': []}

# loop through all unique herds and years
for i in names:
  for j in years:
    # subset the sheep demography data for the herd and year in focus
    subset = sheep[sheep['Herd'] == i]
    subset = subset[subset['Year'] == int(j)]
    subset.reset_index(drop=True, inplace=True)

    if subset.shape[0] == 1:
      # append sheep demography matching the herd and year in focus
      translocated['NonResident'].append(subset.loc[0, 'NonResident'])
      translocated['RT'].append(subset.loc[0, 'RT'])
      translocated['T'].append(subset.loc[0, 'T'])
    else:
      # append default values
      translocated['NonResident'].append(0)
      translocated['RT'].append(0)
      translocated['T'].append(0)

# append sheep demographic data to output object
inputs['NonResident'] = translocated['NonResident']
inputs['RT'] = translocated['RT']
inputs['T'] = translocated['T']

####################################################################################################
# add translocation event data to model inputs table
####################################################################################################

# load cleaned translocation event data
events = pd.read_csv('Data/Translocation/Clean/translocation_events_1997to2016.csv')

# copy year column and change index
events['Year2'] = events['Year']
events.set_index('Year', inplace=True)

# initialize storage for translocation event data to append to output object
translocations = {'Translocation': [], 'TypeBoth': [], 'TypeSupp': [], 'TypeVac': []}

# loop through all unique herds and years
for i in names:
  for j in years:
    j = int(j)
    if j in events['Year2'].tolist():
      # append translocation data if an event occurred during the year in focus
      translocations['Translocation'].append(1)
      translocations['TypeBoth'].append(events.loc[j, 'Both'])
      translocations['TypeSupp'].append(events.loc[j, 'Supplement'])
      translocations['TypeVac'].append(events.loc[j, 'Vacant'])
    else:
      # append default values
      translocations['Translocation'].append(0)
      translocations['TypeBoth'].append(0)
      translocations['TypeSupp'].append(0)
      translocations['TypeVac'].append(0)

# append translocation event data to output object
inputs = pd.concat([inputs, pd.DataFrame(translocations)], axis=1)

####################################################################################################
# add vhf location data to model inputs table
####################################################################################################

# load cleaned vhf location data
vhf = pd.read_csv('Data/Sheep/Clean/vhf_locations_1997to2012_pip.csv')

# initialize storage for vhf location data to append to output object
visitors = {'Visitors': [], 'VisEwes': [], 'VisRams': [], 'VisR': [], 'VisRT': [], 'VisT': []}

# loop through all unique herds and years
for i in names:
  for j in years:
    # subset all vhf location records matching the herd and year in focus
    subset = vhf[vhf['Herd'] == i]
    subset = subset[subset['Year'] == int(j)]
    subset.reset_index(drop=True, inplace=True)

    if subset.shape[0] == 1:
      # append vhf location information for the herd and year in focus
      visitors['Visitors'].append(subset.loc[0, 'Visitors'])
      visitors['VisEwes'].append(subset.loc[0, 'VisEwes'])
      visitors['VisRams'].append(subset.loc[0, 'VisRams'])
      visitors['VisR'].append(subset.loc[0, 'VisR'])
      visitors['VisRT'].append(subset.loc[0, 'VisRT'])
      visitors['VisT'].append(subset.loc[0, 'VisT'])
    else:
      # append default values
      visitors['Visitors'].append(np.NAN)
      visitors['VisEwes'].append(np.NAN)
      visitors['VisRams'].append(np.NAN)
      visitors['VisR'].append(np.NAN)
      visitors['VisRT'].append(np.NAN)
      visitors['VisT'].append(np.NAN)

# append vhf location data to output object
inputs = pd.concat([inputs, pd.DataFrame(visitors)], axis=1)

####################################################################################################
# polish model inputs table
####################################################################################################

# remove records with no pneumonia health status in any of the years
inputs.dropna(subset=['StatusClass'], inplace=True)
inputs.reset_index(drop=True, inplace=True)

# impute missing population data using herd averages across all years with data
for i in range(inputs.shape[0]):
  if math.isnan(inputs.loc[i, 'PopTot']) is True:
    subset = inputs[inputs['Herd'] == inputs.loc[i, 'Herd']]
    subset.dropna(subset=['PopTot'], inplace=True)
    inputs.loc[i, 'PopTot'] = int(round(subset['PopTot'].mean()))
    inputs.loc[i, 'Ewes'] = int(round(subset['Ewes'].mean()))
    inputs.loc[i, 'Rams'] = int(round(subset['Rams'].mean()))
    inputs.loc[i, 'Lambs'] = int(round(subset['Rams'].mean()))
    inputs.loc[i, 'PopAdults'] = inputs.loc[i, 'Ewes'] + inputs.loc[i, 'Rams']

# save model inputs table
inputs.to_csv('Data/Model/Input/inputs.csv', index=False)
