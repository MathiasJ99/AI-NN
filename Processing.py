#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class dataProcessing:
    def __init__(self,filename):
        self.filename = filename


    def Correlation(self, sheet_name): #produces a correlation matrix of given columns in sheetname
        FEHData = pd.read_excel(self.filename, sheet_name)
        df = pd.DataFrame(FEHData)
        correlation_matrix = df.corr()
        return(correlation_matrix.to_string())# Display the correlation matrix

    
    def Standardise(self, sheet_name): 
        FEHData = pd.read_excel(self.filename, sheet_name)
        df = pd.DataFrame(FEHData) # creates data frame
        #print("##########") ## debugging
        df_standardised = df.apply(lambda column: 0.8 * (((column - column.min())/(column.max() - column.min())) + 0.1)) #applies standardisation to data frame
        return(df_standardised)

