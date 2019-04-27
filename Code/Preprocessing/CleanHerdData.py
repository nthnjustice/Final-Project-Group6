# import dependencies
import math
import pandas as pd
import geopandas as gpd

####################################################################################################
# wrangle herd spatial-polygon data
####################################################################################################

# load herd shapefile in latlon projection
herds = gpd.read_file('Data/Herd/Source/hc_bighorn_herds_latlon.shp')

# build list of columns to drop
toDrop = herds.columns[2:4].tolist()
toDrop.extend(herds.columns[6:8].tolist())
toDrop.extend(herds.columns[9:27].tolist())

# drop meaningless columns
herds.drop(toDrop, axis=1, inplace=True)
herds.reset_index(drop=True, inplace=True)

# purpose: crosswalk used to standardize herd names
# input: herd name
# output: mapped herd name if crosswalk evaluates, original parameter otherwise
def formatHerdName(name):
  # build crosswalk to relate ambiguous herd names to standardized values
  map = {
    'Asotin': ['ASOTIN'],
    'Bear Creek': ['BearCreek', 'BEARCR'],
    'Big Canyon': ['BigCanyon', 'BIGCANYON'],
    'Black Butte': ['BlackButte', 'BLACKBUTTE'],
    'Imnaha': ['IMNAHA'],
    'Lower Hells Canyon': ['LHC', 'LowerHellsCanyon', 'LowerHells'],
    'Lostine': ['LOSTINE'],
    'Mountain View': ['MountainView', 'MTNVIEW'],
    'Muir Creek': ['Muir', 'MuirCreek', 'MUIR'],
    'Myers Creek': ['Myers', 'MyersCreek', 'MYERS'],
    'Redbird': ['REDBIRD'],
    'Sheep Mountain': ['SheepMountain', 'SHEEPMTN'],
    'Upper Hells Canyon ID': ['UHC-ID', 'UHCID'],
    'Upper Hells Canyon OR': ['UHC-OR', 'UHCOR'],
    'Upper Saddle': ['UpperSaddle', 'UHC'],
    'Wenaha': ['WENAHA']
  }

  # check if herd name can be mapped
  for k, v in map.items():
    if name in v:
      return k

  return name

# standardize herd names
herds['NM_Label'] = [formatHerdName(i) for i in herds['NM_Label']]

# sort dataframe by herd name
herds.sort_values(by=['NM_Label'], inplace=True)
herds.reset_index(drop=True, inplace=True)

# save cleaned herd data
herds.to_file('Data/Herd/Clean/hc_bighorn_herds_latlon.shp')

# initialize herd copy with multi-part polygons unlisted into single polygons
herdsExp = herds.explode()

# save cleaned exploded-herd data
herdsExp.to_file('Data/Herd/Clean/hc_bighorn_herds_latlon.geojson', 'GeoJSON')

####################################################################################################
# wrangle pneumonia status data
####################################################################################################

# load aggregated 1995-2011 pneumonia status data
pn95To11 = pd.read_csv('Data/Herd/Source/pneumonia_status_1995to2011.csv')
# load supplemental yearly pneumonia status data
pn12 = pd.read_csv('Data/Herd/Source/pneumonia_status_2012.csv')
pn13 = pd.read_csv('Data/Herd/Source/pneumonia_status_2013.csv')
pn14 = pd.read_csv('Data/Herd/Source/pneumonia_status_2014.csv')
pn15 = pd.read_csv('Data/Herd/Source/pneumonia_status_2015.csv')

# sort dataframe to facilitate merging
pn95To11.sort_values(by=['Pop'], inplace=True)
pn95To11.reset_index(drop=True, inplace=True)

# rename year-based columns
col = 2
for i in range(1995, 2012):
  pn95To11.columns.values[col] = i
  col += 1

# purpose: append single-year pneumonia status dataframe as a column to another dataframe
# input: dataframe to gain column, single-year pneumonia status dataframe, label for new column
# output: merge of the first two parameters
def appendPnYear(df, year, label):
  # sort dataframe to facilitate merging
  year.sort_values(by=['NAME'], inplace=True)
  # store column number containing pneumonia status
  col = year.columns.values[2]
  # merge dataframes
  df = pd.concat([df, year[col]], axis=1)
  # rename new column
  df.columns.values[df.shape[1] - 1] = label

  return df

# initialize storage for all years of pneumonia status
pnStatus = pn95To11

# call function to append each single-year pneumonia status
pnStatus = appendPnYear(pnStatus, pn12, '2012')
pnStatus = appendPnYear(pnStatus, pn13, '2013')
pnStatus = appendPnYear(pnStatus, pn14, '2014')
pnStatus = appendPnYear(pnStatus, pn15, '2015')

# standardize herd names
pnStatus['Pop'] = [formatHerdName(i) for i in pnStatus['Pop']]
pnStatus.reset_index(drop=True, inplace=True)

# drop meaningless column
pnStatus.drop(['XID'], axis=1, inplace=True)

# sort dataframe by herd name
pnStatus.sort_values(by=['Pop'], inplace=True)
pnStatus.reset_index(drop=True, inplace=True)

# replace values with meaningful strings
pnStatus[pnStatus == 'HEALTHY'] = 'Healthy'
pnStatus[pnStatus == 'INVASION'] = 'Outbreak'
pnStatus[pnStatus == 'INFECTED'] = 'Infected - Lambs'
pnStatus[pnStatus == 'INFECTED_A'] = 'Infected - All Ages'
pnStatus[pnStatus == 'NOPOP'] = 'No Population'

# handle corner-cases where value string has extra space
pnStatus[pnStatus == 'INFECTED '] = 'Infected - Lambs'
pnStatus[pnStatus == 'INFECTED_A '] = 'Infected - All Ages'

# enforce column labels as strings
pnStatus.columns = [str(i) for i in pnStatus.columns.values]

# save cleaned pneumonia status data
pnStatus.to_csv('Data/Herd/Clean/pneumonia_status_1995to2015.csv', index=False)

####################################################################################################
# wrangle compiled data
####################################################################################################

# load compiled data
compiled = pd.read_csv('Data/Herd/Source/compiled_data_1970to2015.csv')

# drop meaningless columns
compiled.drop(['Record', 'r_est', 'r', 'PopEst', 'TotCount'], axis=1, inplace=True)

# add new column summing population counts
compiled['PopTot'] = compiled['Lambs'] + compiled['Rams'] + compiled['Ewes']

# add new column summing adult population counts
compiled['PopAdults'] = compiled['Rams'] + compiled['Ewes']

# standardize herd names
compiled['Pop'] = [formatHerdName(i) for i in compiled['Pop']]
compiled.reset_index(drop=True, inplace=True)

# sort dataframe by herd name and year
compiled.sort_values(by=['Pop', 'year'], inplace=True)
compiled.reset_index(drop=True, inplace=True)

# enforce column labels as strings
compiled.columns = [str(i) for i in compiled.columns.values]

# save cleaned compiled data
compiled.to_csv('Data/Herd/Clean/compiled_data_1970to2015.csv', index=False)
