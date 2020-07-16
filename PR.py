#==================================IMPORTING NEEDED PACKAGES============================
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
#%matplotlib inline #for jupyter notes



#==========================================DATA========================================
# MODELYEAR e.g. 2014
# MAKE e.g. Acura
# MODEL e.g. ILX
# VEHICLE CLASS e.g. SUV
# ENGINE SIZE e.g. 4.7
# CYLINDERS e.g 6
# TRANSMISSION e.g. A6
# FUEL CONSUMPTION in CITY(L/100 km) e.g. 9.9
# FUEL CONSUMPTION in HWY (L/100 km) e.g. 8.9
# FUEL CONSUMPTION COMB (L/100 km) e.g. 9.2
# CO2 EMISSIONS (g/km) e.g. 182 --> low --> 0

df = pd.read_csv('FuelConsumptionCo2.csv')
# take a look at the dataset
print(df.head())

#===========================Selecting the features for the project===============================
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)
#plotting the emission and engine size
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()



#=====================================TRAIN/TEST SPLIT======================================
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]



#==================================POLYNOMIAL REGRESSION====================================





