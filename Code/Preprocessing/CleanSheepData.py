# import dependencies
import pandas as pd
import geopandas as gpd
import utm

pd.options.mode.chained_assignment = None

####################################################################################################
# wrangle study sheep data
####################################################################################################

# load study sheep data
sheep = pd.read_csv('Data/Sheep/Source/study_sheep_1995to2015.csv')

# build list of columns to drop
toDrop = ['Frequency', 'DateLastSeen', 'Comments', 'ToothAge']

# drop meaningless columns
sheep.drop(toDrop, axis=1, inplace=True)
sheep.reset_index(drop=True, inplace=True)

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
sheep['Herd'] = [formatHerdName(i) for i in sheep['Herd']]

# append new year column parsed from date
sheep['Year'] = [int(sheep.loc[i, 'EntryDate'].split('/')[2]) for i in range(sheep.shape[0])]

# initialize object to store residency information summarized by year and herd
sheepDict = {
  'Herd': [], 'Year': [], 'NonResident': [], 'RT': [], 'T': []
}

# populate residency information summarized by year and herd
for i in sheep['Herd'].unique().tolist():
  for j in sheep['Year'].unique().tolist():
    # build subset of herd and year in focus
    subset = sheep[sheep['Herd'] == i]
    subset = subset[subset['Year'] == j]

    # update object
    sheepDict['Herd'].append(i)
    sheepDict['Year'].append(j)

    # initialize boolean indicators for non-residency status
    RT = False
    T = False

    # update boolean indicators if non-residency status exists
    if 'RT' in subset['Source'].tolist():
      RT = True

    if 'T' in subset['Source'].tolist():
      T = True

    # update object based on presence/absence of non-residency status
    if RT is True or T is True:
      sheepDict['NonResident'].append(1)
    else:
      sheepDict['NonResident'].append(0)

    if RT is True:
      sheepDict['RT'].append(1)
    else:
      sheepDict['RT'].append(0)

    if T is True:
      sheepDict['T'].append(1)
    else:
      sheepDict['T'].append(0)

# convert residency information object to dataframe
sheep = pd.DataFrame(sheepDict)

# save cleaned study sheep data
sheep.to_csv('Data/Sheep/Clean/study_sheep_1995to2015.csv', index=False)

####################################################################################################
# wrangle vhf location data
####################################################################################################

# load vhf location data
vhf = pd.read_csv('Data/Sheep/Source/vhf_locations_1997to2012.csv', low_memory=False)

# build list of columns to to drop
toDrop = ['TYPE', 'TIME', 'FREQ', 'LOCATION']
toDrop.extend(vhf.columns[19:24].tolist())
toDrop.extend(vhf.columns[28:41].tolist())

# drop meaningless columns
vhf.drop(toDrop, axis=1, inplace=True)
vhf.reset_index(drop=True, inplace=True)

# standardize herd names
vhf['Herd'] = [formatHerdName(i) for i in vhf['Herd']]

# purpose: checks if string can be converted to numeric value
# input: string to evaluate
# output: True if string can be converted to numeric value, False otherwise
def asNumeric(val):
  return str(val).replace('.', '', 1).isdigit()

# purpose: drop rows with invalid coordinate values, convert coordinate values to floats
# input: dataframe with UTM coordinates, name of UTM coordinate column to clean
# output: copy of first parameter with cleaned UTM coordinate column
def cleanUtmCoordinates(df, coord):
  # drop rows where coordinate value is null
  df = df[pd.notnull(df[coord])]
  df.reset_index(drop=True, inplace=True)

  # initialize list of coordinate values that can't be converted to numeric
  invalid = set([df.loc[i, coord] for i in range(df.shape[0]) if not asNumeric(df.loc[i, coord])])

  # drop rows where coordinate value can't be converted to numeric
  for i in invalid:
    df = df[df[coord] != i]
    df.reset_index(drop=True, inplace=True)

  # convert coordinate values to floats
  df[coord] = [float(df.loc[i, coord]) for i in range(df.shape[0])]

  # drop rows where coordinate values exceed UTM bounds
  df = df[df[coord] > 0]
  df.reset_index(drop=True, inplace=True)

  return df

# call function to clean UTM coordinate columns
vhf = cleanUtmCoordinates(vhf, 'UTME')
vhf = cleanUtmCoordinates(vhf, 'UTMN')

# purpose: transform coordinate from UTM to latlon projection
# input: dataframe with UTM coordinates, row index of dataframe to transform,
  # label of UTME coordinate, label of UTMN coordinate,
  # latlon coordinate to return where 0 == lat and 1 == lon
# output: UTM coordinate in latlon projection
def convertUtmToLatlon(df, i, utmE, utmN, coord):
  return utm.to_latlon(df.loc[i, utmE], df.loc[i, utmN], 11, 'U')[coord]

# build lists of UTM coordinates transformed to latlon projection
lat = [convertUtmToLatlon(vhf, i, 'UTME', 'UTMN', 0) for i in range(vhf.shape[0])]
lon = [convertUtmToLatlon(vhf, i, 'UTME', 'UTMN', 1) for i in range(vhf.shape[0])]

# assign transformed coordinates
vhf['Lat'] = lat
vhf['Lon'] = lon

# handle corner-case and drop entry with impractical coordinates
vhf = vhf[vhf['Lat'] != vhf['Lat'].min()]
vhf.reset_index(drop=True, inplace=True)

# save cleaned vhf location data
vhf.to_csv('Data/Sheep/Clean/vhf_locations_1997to2012.csv', index=False)

####################################################################################################
# wrangle vhf polygon-in-point location data
####################################################################################################

# load spatial vhf location data
vhfPip = gpd.read_file('Data/Sheep/Source/vhf_locations_1997to2012_pip.shp')
# reload study sheep data
sheep = pd.read_csv('Data/Sheep/Source/study_sheep_1995to2015.csv')

# subset vhf records that have demographic records
vhfPip = vhfPip[vhfPip['ID'].isin(sheep['Animal ID'])]
vhfPip.reset_index(drop=True, inplace=True)

# append columns for sex and residency status to all vhf records
for i in range(vhfPip.shape[0]):
  # subset demographic information for sheep in focus
  subset = sheep[sheep['Animal ID'] == vhfPip.loc[i, 'ID']]
  subset.reset_index(drop=True, inplace=True)

  # assign demographic information
  vhfPip.loc[i, 'Sex'] = subset['Sex'][0]
  vhfPip.loc[i, 'Source'] = subset['Source'][0]

# initialize storage for output object
vhfPipDict = {
  'Herd': [], 'Year': [], 'Visitors': [], 'VisEwes': [], 'VisRams': [],
  'VisR': [], 'VisRT': [], 'VisT': []
}

# loop through each unique herd and valid year
for i in vhfPip['NM_Label'].unique():
  for j in range(1997, 2013):
    # subset vhf records matching herd and year in focus
    subset = vhfPip[vhfPip['NM_Label'] == i]
    subset = subset[subset['Year'] == j]
    subset.reset_index(drop=True, inplace=True)

    # append herd and year information to output object
    vhfPipDict['Herd'].append(i)
    vhfPipDict['Year'].append(j)

    if subset.shape[0] > 0:
      # isolate vhf records for sheep not native to the herd in focus (visitors)
      visitors = subset[subset['Herd'] != i]

      # initialize count variables
      numVisitors = len(visitors['ID'].unique())
      ewes = 0
      rams = 0
      r = 0
      rt = 0
      t = 0

      # loop through all unique visitors in the herd and year in focus
      for k in visitors['ID'].unique():
        # subset the demographic information for the visitor in focus
        match = sheep[sheep['Animal ID'] == k]
        match.reset_index(drop=True, inplace=True)

        # increment appropriate demographic counters
        if match['Sex'][0] == 'M':
          rams += 1
        else:
          ewes += 1

        if match['Source'][0] == 'R':
          r += 1
        elif match['Source'][0] == 'RT':
          rt += 1
        else:
          t += 1

      # append information about visitors to the herd and year in focus
      vhfPipDict['Visitors'].append(len(visitors['ID'].unique()))
      vhfPipDict['VisEwes'].append(ewes)
      vhfPipDict['VisRams'].append(rams)
      vhfPipDict['VisR'].append(r)
      vhfPipDict['VisRT'].append(rt)
      vhfPipDict['VisT'].append(t)

    else:
      # append default values to output object
      vhfPipDict['Visitors'].append(0)
      vhfPipDict['VisEwes'].append(0)
      vhfPipDict['VisRams'].append(0)
      vhfPipDict['VisR'].append(0)
      vhfPipDict['VisRT'].append(0)
      vhfPipDict['VisT'].append(0)

# convert output object to dataframe
vhfPip= pd.DataFrame(vhfPipDict)

# save cleaned vhf location data
vhfPip.to_csv('Data/Sheep/Clean/vhf_locations_1997to2012_pip.csv', index=False)
