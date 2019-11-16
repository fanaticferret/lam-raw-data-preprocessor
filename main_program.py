# In[ ]:


# Import some basic libraries

import pandas as pd
import numpy as np
from math import sqrt

from scipy.stats.mstats import mode, gmean, hmean

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.image as mpimg

import datetime


# In[ ]:


def initialize(df, interval):
    
    try:
    
        df.columns = ["Identifier", "Year", "DOY", "Hour", "Minute", "Second", "Hundredth of a second", "Length (m)", "Lane", "Direction", "Vehicle type", "Speed (km/h)", "Status", "Occupancy", "Interval", "Queue start"]
    
    except ValueError:
    
        print("The dataframe is empty!")
        
        return df
    
    df = df.reset_index(drop=True)
    
    df = df.drop(["Identifier", "Second", "Hundredth of a second", "Status", "Vehicle type", "Lane", "Queue start", "Length (m)", "Occupancy", "Interval"], axis = 1)
    
    df.DOY = df.DOY.astype(dtype = 'float32')
    
    df = create_date(df)
    
    df = parse_datetime(df)
    
    df = inward_outward(df)
    
    df, row = resample(df, interval)
    
    df = compute_density(df, row)
    
    print("Your dataframe has been initialized succesfully.")
    
    # Plot to check for missing flow rates.
    
    # ax = df.plot(y = ['Flow rate (dir. 1)', 'Flow rate (dir. 2)'], color = ['darkseagreen', 'steelblue', 'rebeccapurple'], use_index=True)
            
    # ax.legend(['Flow rate (dir. 1)','Flow rate (dir. 2)'])

    return df


# In[ ]:


def fetch_data(first, last, yr, row2, row6):
    
    D = pd.DataFrame()
    
    for i in range (int(first), int(last) + 1):
        
        try:
            
            small_D = pd.read_csv("https://aineistot.liikennevirasto.fi/lam/rawdata/{}/{}/lamraw_{}_{}_{}.csv".format(yr, row6, row2, yr[-2:], str(i)), sep = ";", header = None)
        
            D = D.append(small_D)
            
        except:
            
            print("The day {} of year {} was missing.".format(i, yr))
            
            pass
        
    return D


# In[ ]:


def mega_collector():
    
    row = input("Please specify a range of years that you are interested in, separated by a comma.\n")
    
    row2 = input("Which sensor are you interested in?\n")
    
    row5 = input("Please specify the resampling interval to be used.\n")
    
    row6 = input("Please input the area code, one of the following:\n01 Uusimaa\n02 Varsinais-Suomi\n03 Kaakkois-Suomi\n04 Pirkanmaa\n08 Pohjois-Savo\n10 Etelä-Pohjanmaa\n12 Pohjois-Pohjanmaa\n14 Lappi\n")
    
    years = row.split(',')
    
    first_df = pd.DataFrame()
    
    i = int(years[0])
    
    while i < int(years[1]) + 1:
        
        if check_if_leap_year(i):
            
            array = [[1, 31], [32, 60], [61, 91], [92, 121], [122, 152], [153, 182], [183, 213], [214, 244], [245, 274], [275, 305], [306, 335], [336, 366]]
            
        else:
            
            array = [[1, 31], [32, 59], [60, 90], [91, 120], [121, 151], [152, 181], [182, 212], [213, 243], [244, 273], [274, 304], [305, 334], [335, 365]]
        
        # range(len(array))
        
        for j in range(len(array)):
                     
            first = array[j][0]
                     
            last = array[j][1]
                     
            yr = str(i)
                     
            second_df = fetch_data(first, last, yr, row2, row6)
                     
            second_df = initialize(second_df, row5)
            
            if second_df.empty:
                
                pass
            
            else:
            
                first_df = first_df.append(second_df)
                     
        i += 1
                     
    row3 = input("Do you want to save the new megaframe in a csv? (Yes or No)\n")
    
    if str(row3) == 'Yes':
        
        row4 = input("Please input the name in form SENSOR-FIRST_YEAR-LAST_YEAR-RESAMPLING_INTERVAL (minus .csv).\n")
        
        first_df.to_csv("{}.csv".format(str(row4)))
        
    else:
        
        pass
                     
    return first_df


# In[ ]:


def concatenate():
    
    row = input("Specify the csv:s to be concatenated by their filenames, separated by a comma.")
    
    csvs = row.split(',')
    
    strng = "{}".format(csvs[0])

    snsr = strng[0:3]
    
    first_csv = pd.read_csv("{}".format(csvs[0]), index_col = 0, header = [0, 1])
    
    i = 1
    
    while i < len(csvs):
        
        next_csv = pd.read_csv("{}".format(csvs[i]), index_col = 0, header = [0, 1])
        
        first_csv = first_csv.append(next_csv)
        
        i += 1
        
    first_csv.index = pd.to_datetime(first_csv.index)
    
    return first_csv, snsr


# In[ ]:


def check_if_leap_year(year):

    if year % 4 != 0:
        truth = False
        
    elif year % 100 != 0:
        truth = True
    
    elif year % 400 != 0:
        truth = False
        
    else:
        truth = True
            
    return truth


# In[ ]:


def create_date(df):
    
    d = df['DOY'] - 1
    
    string = "20{}".format(str(int(float(df.iloc[0, 0]))))
    
    date = pd.Series()
    
    td = pd.to_timedelta(d.values, unit = 'days')
    
    nampi1 = np.array(td.values)
    
    nampi2 = np.full(shape = (nampi1.size,), fill_value = np.datetime64('{}-01-01'.format(string)))
    
    nampi2 += nampi1
    
    df['Date'] = pd.Series(nampi2)
    
    df['Date'] += pd.to_timedelta(df.Hour.values, unit = 'h')
    
    df['Date'] += pd.to_timedelta(df.Minute.values, unit = 'm')
    
    df = df.drop(['DOY'], axis = 1)
    
    return df


# In[ ]:


def parse_datetime(df):
    
    df["Year"] = df.Date.apply(lambda x: x.year)
    df["Day of week"] = df.Date.apply(lambda x: x.dayofweek)
    
    return df


# In[ ]:


def inward_outward(df):
    
    boolean1 = df['Direction'] == 1
    
    boolean2 = df['Direction'] == 2
    
    df['Speed (dir. 1)'] = boolean1 * df['Speed (km/h)']
    
    df['Speed (dir. 2)'] = boolean2 * df['Speed (km/h)']
    
    return df


# In[ ]:


def resample(df, row):

    df = df.set_index('Date')
    
    df_resampled = pd.DataFrame(data = {'Hour': df['Hour']}, index = df.index)
    
    df_resampled = df_resampled.resample(rule='{}T'.format(row), how=['median'])

    df_resampled = df_resampled.interpolate(method='polynomial', order=1)
    
    # Inward-outward counts
    
    # Dir. 1

    cnt2 = pd.DataFrame({'Date': df.index, 'count': np.nan})
    
    cnt2 = cnt2.set_index('Date')
    
    cnt2.loc[df['Speed (dir. 1)'] > 0, 'count'] = df.loc[df['Speed (dir. 1)'] > 0, 'Speed (dir. 1)']
    
    cnt2 = cnt2.resample('{}T'.format(row), how=['count'])

    # Dir. 2
    
    cnt3 = pd.DataFrame({'Date': df.index, 'count': np.nan})
    
    cnt3 = cnt3.set_index('Date')
    
    cnt3.loc[df['Speed (dir. 2)'] > 0, 'count'] = df.loc[df['Speed (dir. 2)'] > 0, 'Speed (dir. 2)']
    
    cnt3 = cnt3.resample('{}T'.format(row), how=['count'])
    
    # Inward-outward velocities
    
    # Dir. 1
    
    cnt4 = pd.DataFrame({'Date': pd.to_datetime(df.index), 'count': np.nan})
    
    cnt4 = cnt4.set_index('Date')
    
    cnt4.loc[df['Speed (dir. 1)'] > 0, 'count'] = df.loc[df['Speed (dir. 1)'] > 0, 'Speed (dir. 1)']
    
    cnt4 = cnt4.resample('{}T'.format(row)).apply(space_mean_speed)
    
    # Dir. 2
    
    cnt5 = pd.DataFrame({'Date': pd.to_datetime(df.index), 'count': np.nan})
    
    cnt5 = cnt5.set_index('Date')
    
    cnt5.loc[df['Speed (dir. 2)'] > 0, 'count'] = df.loc[df['Speed (dir. 2)'] > 0, 'Speed (dir. 2)']
    
    cnt5 = cnt5.resample('{}T'.format(row)).apply(space_mean_speed)
    
    # Assigning inward-outward & total flow rates

    df_resampled["Flow rate (dir. 1)", "median"] = cnt2.values
    
    df_resampled["Flow rate (dir. 1)", "median"] = df_resampled["Flow rate (dir. 1)", "median"].interpolate(method='polynomial', order=1)
    
    df_resampled["Flow rate (dir. 2)", "median"] = cnt3.values
    
    df_resampled["Flow rate (dir. 2)", "median"] = df_resampled["Flow rate (dir. 2)", "median"].interpolate(method='polynomial', order=1)
    
    # Assigning inward-outward speeds
    
    df_resampled['Speed (dir. 1)', 'median'] = cnt4.values
    
    df_resampled['Speed (dir. 1)', 'median'] = df_resampled['Speed (dir. 1)', 'median'].interpolate(method='polynomial', order=1)
    
    df_resampled['Speed (dir. 2)', 'median'] = cnt5.values
    
    df_resampled['Speed (dir. 2)', 'median'] = df_resampled['Speed (dir. 2)', 'median'].interpolate(method='polynomial', order=1)
    
    return df_resampled, row


# In[ ]:


def space_mean_speed(array):
    
    array = array[np.logical_not(np.isnan(array))]
    
    avg = np.mean(array)
    
    try:
    
        var = np.sum([(array - avg) **2]) / len(array)
    
    except ZeroDivisionError:
        
        return float(0)
    
    sms = (avg / 2) + np.sqrt(((avg ** 2) / 4) - var)
    
    return sms


# In[ ]:


def compute_density(df, row):
    
    a = np.array(df['Flow rate (dir. 1)', 'median'], dtype=float)
    
    b = np.array(df['Speed (dir. 1)', 'median'] * (int(row)/60), dtype=float)
    
    c = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    
    d = np.array(df['Flow rate (dir. 2)', 'median'], dtype=float)
    
    e = np.array(df['Speed (dir. 2)', 'median'] * (int(row)/60), dtype=float)
    
    f = np.divide(d, e, out=np.zeros_like(d), where=e!=0)
    
    df['Density (dir. 1)', 'median'] = pd.Series(c).values
    
    df['Density (dir. 2)', 'median'] = pd.Series(f).values
    
    return df


# In[ ]:


D = mega_collector()
