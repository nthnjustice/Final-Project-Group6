####################################################################################################
# NOTES:
# 1) This file will run everything for the project
# 2) Most will run quickly, except for CleanSheepData.py
# 3) Please see README file in the Code directory to get instructions for downloading and
  # unzipping the data for vhf_locations_1997to2012_pip otherwise the script CleanSheepData.py
  # will not run. However, all other scripts will execute independent of this one (cleaned data
  # is already available).
# 4) The Preprocessing and Modeling scripts are not proper modules so the working directory for
  # every script in this project must be "~Code"  aka ""~Final-Project-Group6/Code"
####################################################################################################

from Preprocessing import CleanHerdData
from Preprocessing import CleanSheepData
from Preprocessing import CleanTranslocationData
from Preprocessing import BuildInputData
import Modeling
