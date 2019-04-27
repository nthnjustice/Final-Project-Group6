# import dependencies
import pandas as pd

####################################################################################################
# wrangle translocation event data
####################################################################################################

# load translocation event data
events = pd.read_csv('Data/Translocation/Source/translocation_events_1997to2016.csv')

# add new year column parsed from Date
events['Year'] = [int(events.loc[i, 'Date'].split('-')[2]) for i in range(events.shape[0])]

# format year column
for i in range(events.shape[0]):
  if events.loc[i, 'Year'] > 96:
    events.loc[i, 'Year'] += 1900
  else:
    events.loc[i, 'Year'] += 2000

# enforce year values as strings
events['Year'] = [str(events.loc[i, 'Year']) for i in range(events.shape[0])]

# add new column summing adult population counts
events['Adults'] = events['Males'] + events['Females']

# add new column summing lamb population counts
events['Lambs'] = events['Male lambs'] + events['Female lambs']

# purpose: classify translocation event type after aggregating all instances within a year
# input: year subset of translocation events
# output: vacant if only vacant translocation event occurred in the year, supplement if
  # only supplement translocation event occurred in the year, both if
  # both vacant and supplement translocation events occurred in the year
def classifyTranslocation(df):
  vac = False
  sup = False

  for i in df['Type']:
    if i == 'Vacant habitat':
      vac = True
    elif i == 'Supplement':
      sup = True

  if vac is True and sup is False:
    return 'Vacant'
  elif vac is False and sup is True:
    return 'Supplement'
  elif vac is True and sup is True:
    return 'Both'

# initialize object to store translocation events information summarized by year
eventsDict = {
  'Year': [], 'Total': [], 'Adults': [], 'Males': [], 'Females': [], 'Lambs': [], 'Type': []
}

# populate translocation events information summarized by year
for i in events['Year'].unique():
  subset = events[events['Year'] == i]
  eventsDict['Year'].append(i)
  eventsDict['Total'].append(subset['Total'].sum())
  eventsDict['Adults'].append(subset['Adults'].sum())
  eventsDict['Males'].append(subset['Males'].sum())
  eventsDict['Females'].append(subset['Females'].sum())
  eventsDict['Lambs'].append(subset['Lambs'].sum())
  eventsDict['Type'].append(classifyTranslocation(subset))

# re-initialize new translocation events dataframe
events = pd.DataFrame(eventsDict)

# append one-hot encoded columns for translocation event type
events = pd.concat([events, pd.get_dummies(events['Type'])], axis=1)

# enforce year values as strings
events['Year'] = [str(events.loc[i, 'Year']) for i in range(events.shape[0])]

# save cleaned translocation event data
events.to_csv('Data/Translocation/Clean/translocation_events_1997to2016.csv', index=False)
