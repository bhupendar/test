# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:33:43 2018

@author: bsingh
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 13:06:06 2018

@author: bhupendarsingh
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#reading the data
df = pd.read_csv('KCPD_Crime_Data_2017.csv')


#### Various operations on Dataframes
#Print a concise summary of a DataFrame 
df.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 132139 entries, 0 to 132138
Data columns (total 24 columns):
Report_No            132139 non-null int64
Reported_Date        132139 non-null object
Reported_Time        132139 non-null object
From_Date            131844 non-null object
From_Time            131744 non-null object
To_Date              49634 non-null object
To_Time              49140 non-null object
Offense              132139 non-null int64
IBRS                 131044 non-null object
Description          132139 non-null object
Beat                 131293 non-null float64
Address              132115 non-null object
City                 132101 non-null object
Zip Code             132139 non-null int64
Rep_Dist             131132 non-null object
Area                 131132 non-null object
DVFlag               132139 non-null object
Invl_No              132139 non-null int64
Involvement          132139 non-null object
Race                 114090 non-null object
Sex                  114090 non-null object
Age                  71014 non-null float64
Firearm Used Flag    132139 non-null object
Location             132139 non-null object
dtypes: float64(2), int64(4), object(18)
memory usage: 24.2+ MB

#column labels 
df.columns

Index(['Report_No', 'Reported_Date', 'Reported_Time', 'From_Date', 'From_Time',
       'To_Date', 'To_Time', 'Offense', 'IBRS', 'Description', 'Beat',
       'Address', 'City', 'Zip Code', 'Rep_Dist', 'Area', 'DVFlag', 'Invl_No',
       'Involvement', 'Race', 'Sex', 'Age', 'Firearm Used Flag', 'Location'],
      dtype='object')

#Return the dtypes in the DataFrame 
df.dtypes

Report_No              int64
Reported_Date         object
Reported_Time         object
From_Date             object
From_Time             object
To_Date               object
To_Time               object
Offense                int64
IBRS                  object
Description           object
Beat                 float64
Address               object
City                  object
Zip Code               int64
Rep_Dist              object
Area                  object
DVFlag                object
Invl_No                int64
Involvement           object
Race                  object
Sex                   object
Age                  float64
Firearm Used Flag     object
Location              object
dtype: object

#Return a Numpy representation of the DataFrame
df.values

array([[170086272, '10/05/2017', '10:00', ..., nan, 'N',
        '400 W 58 ST\nKANSAS CITY 64113\n(39.022397, -94.59269)'],
       [170026074, '04/12/2017', '13:01', ..., nan, 'N',
        '3500 PROSPECT AV\nKANSAS CITY 64127\n'],
       [170003559, '01/15/2017', '02:35', ..., 23.0, 'N',
        'N NW GREEN HILLS RD\nKANSAS CITY 64152\n'],
       ...,
       [170103604, '12/10/2017', '19:46', ..., nan, 'N',
        '11600 E US\nHW KANSAS CITY 64133\n'],
       [170023896, '04/04/2017', '17:11', ..., 29.0, 'N',
        '18 ST and PROSPECT AV\nKANSAS CITY 64127\n'],
       [170057331, '06/28/2017', '13:37', ..., nan, 'N',
        '13100 HOLMES RD\nKANSAS CITY 64145\n(38.889239, -94.586045)']],
      dtype=object)
      
#The index (row labels) of the DataFrame
df.index

RangeIndex(start=0, stop=132139, step=1)

#This function returns the first n rows for the object based on position.
df.head(2)

   Report_No Reported_Date Reported_Time   From_Date From_Time     To_Date  \
0  170086272    10/05/2017         10:00  10/04/2017     16:00  10/05/2017   
1  170026074    04/12/2017         13:01  03/20/2017     11:30  04/11/2017   

  To_Time  Offense IBRS           Description  \
0   10:00      690  23H    Stealing All Other   
1   00:45      630  23C  Stealing Shoplifting   

                         Rep_Dist Area DVFlag  \
0                        ...                            PJ4582  MPD      U   
1                        ...                            PJ2875  EPD      U   

   Invl_No Involvement Race  Sex  Age Firearm Used Flag  \
0        1         SUS    U    U  NaN                 N   
1        1         VIC  NaN  NaN  NaN                 N   

                                            Location  
0  400 W 58 ST\nKANSAS CITY 64113\n(39.022397, -9...  
1              3500 PROSPECT AV\nKANSAS CITY 64127\n  

[2 rows x 24 columns]


#This function returns last n rows from the object based on position.
df.tail(2)

        Report_No Reported_Date Reported_Time   From_Date From_Time To_Date  \
132137  170023896    04/04/2017         17:11  04/04/2017     17:11     NaN   
132138  170057331    06/28/2017         13:37  06/28/2017     13:30     NaN   

       To_Time  Offense IBRS           Description  \
132137     NaN      403  13A  Agg Assault - Domest   
132138     NaN      630  23C  Stealing Shoplifting   

                         Rep_Dist Area  \
132137                        ...                            PJ1525  EPD   
132138                        ...                            PJ7403  SPD   

       DVFlag  Invl_No Involvement Race Sex   Age Firearm Used Flag  \
132137      U        1         ARR    B   F  29.0                 N   
132138      U        2         SUS    W   M   NaN                 N   

                                                 Location  
132137         18 ST and PROSPECT AV\nKANSAS CITY 64127\n  
132138  13100 HOLMES RD\nKANSAS CITY 64145\n(38.889239...  

[2 rows x 24 columns]

#Return a tuple representing the dimensionality of the DataFrame.
df.shape

(132139, 24)

#about df
type(df)

pandas.core.frame.DataFrame

#number of unique elements in the object
df.nunique()

Report_No            52554
Reported_Date          365
Reported_Time         1440
From_Date              728
From_Time             1436
To_Date                495
To_Time               1310
Offense                101
IBRS                    56
Description            116
Beat                   186
Address              15396
City                    38
Zip Code               185
Rep_Dist              6868
Area                     9
DVFlag                   3
Invl_No                 30
Involvement              3
Race                     7
Sex                      4
Age                     98
Firearm Used Flag        2
Location             23769
dtype: int64

#Return the ftypes (indication of sparse/dense and dtype) in DataFrame.
df.ftypes

Report_No              int64:dense
Reported_Date         object:dense
Reported_Time         object:dense
From_Date             object:dense
From_Time             object:dense
To_Date               object:dense
To_Time               object:dense
Offense                int64:dense
IBRS                  object:dense
Description           object:dense
Beat                 float64:dense
Address               object:dense
City                  object:dense
Zip Code               int64:dense
Rep_Dist              object:dense
Area                  object:dense
DVFlag                object:dense
Invl_No                int64:dense
Involvement           object:dense
Race                  object:dense
Sex                   object:dense
Age                  float64:dense
Firearm Used Flag     object:dense
Location              object:dense
dtype: object

#Return counts of unique dtypes in this object.
df.get_dtype_counts()

float64     2
int64       4
object     18
dtype: int64

#Return counts of unique ftypes in this object.
df.get_ftype_counts()

float64:dense     2
int64:dense       4
object:dense     18
dtype: int64

#Exclude Object datatypes
df.select_dtypes(exclude=['object']).head(5)

   Report_No  Offense   Beat  Zip Code  Invl_No   Age
0  170086272      690  221.0     64113        1   NaN
1  170026074      630  332.0     64127        1   NaN
2  170003559     1850  422.0     64152        1  23.0
3  170001089     1849  122.0     64106        1   NaN
4  170007467      670  324.0     64127        1  38.0

#Include Object datatypes
df.select_dtypes(include=['object']).head(5)

  Reported_Date Reported_Time   From_Date From_Time     To_Date To_Time IBRS  \
0    10/05/2017         10:00  10/04/2017     16:00  10/05/2017   10:00  23H   
1    04/12/2017         13:01  03/20/2017     11:30  04/11/2017   00:45  23C   
2    01/15/2017         02:35  01/15/2017     02:35         NaN     NaN  35B   
3    01/05/2017         11:40  01/05/2017     11:40         NaN     NaN  35A   
4    01/30/2017         13:04  01/27/2017     21:25  01/27/2017   22:30  23D   

            Description                                      Address  \
0    Stealing All Other                                 400  W 58 ST   
1  Stealing Shoplifting                            3500  PROSPECT AV   
2  Possession of Drug E  N GREEN HILLS RD and NW OLD TIFFANY SPRINGS   
3  Possession/Sale/Dist                              1100  TROOST AV   
4  Stealing from Buildi                                4800  E 24 ST   

          City Rep_Dist Area DVFlag Involvement Race  Sex Firearm Used Flag  \
0  KANSAS CITY   PJ4582  MPD      U         SUS    U    U                 N   
1  KANSAS CITY   PJ2875  EPD      U         VIC  NaN  NaN                 N   
2  KANSAS CITY   PP0269  NPD      U         ARR    B    M                 N   
3  KANSAS CITY   PJ1046  CPD      U         VIC  NaN  NaN                 N   
4  KANSAS CITY   PJ2012  EPD      U         VIC    B    F                 N   

                                            Location  
0  400 W 58 ST\nKANSAS CITY 64113\n(39.022397, -9...  
1              3500 PROSPECT AV\nKANSAS CITY 64127\n  
2           N NW GREEN HILLS RD\nKANSAS CITY 64152\n  
3  1100 TROOST AV\nKANSAS CITY 64106\n(39.10068, ...  
4  4800 E 24 ST\nKANSAS CITY 64127\n(39.08209, -9... 

#Quickly retrieve single value at passed column and index
df.get_values()

array([[170086272, '10/05/2017', '10:00', ..., nan, 'N',
        '400 W 58 ST\nKANSAS CITY 64113\n(39.022397, -94.59269)'],
       [170026074, '04/12/2017', '13:01', ..., nan, 'N',
        '3500 PROSPECT AV\nKANSAS CITY 64127\n'],
       [170003559, '01/15/2017', '02:35', ..., 23.0, 'N',
        'N NW GREEN HILLS RD\nKANSAS CITY 64152\n'],
       ...,
       [170103604, '12/10/2017', '19:46', ..., nan, 'N',
        '11600 E US\nHW KANSAS CITY 64133\n'],
       [170023896, '04/04/2017', '17:11', ..., 29.0, 'N',
        '18 ST and PROSPECT AV\nKANSAS CITY 64127\n'],
       [170057331, '06/28/2017', '13:37', ..., nan, 'N',
        '13100 HOLMES RD\nKANSAS CITY 64145\n(38.889239, -94.586045)']],
      dtype=object)
      
#Return a list representing the axes of the DataFrame.      
df.axes

[RangeIndex(start=0, stop=132139, step=1),
 Index(['Report_No', 'Reported_Date', 'Reported_Time', 'From_Date', 'From_Time',
        'To_Date', 'To_Time', 'Offense', 'IBRS', 'Description', 'Beat',
        'Address', 'City', 'Zip Code', 'Rep_Dist', 'Area', 'DVFlag', 'Invl_No',
        'Involvement', 'Race', 'Sex', 'Age', 'Firearm Used Flag', 'Location'],
       dtype='object')]

#Number of axes / array dimensions
df.ndim

2

#Return an int representing the number of elements in this object.
df.size

3171336

#True if NDFrame is entirely empty [no items], meaning any of the axes are of length 0.
df.empty

False

#Attempt to infer better dtypes for object columns.
df.infer_objects().dtypes

Report_No              int64
Reported_Date         object
Reported_Time         object
From_Date             object
From_Time             object
To_Date               object
To_Time               object
Offense                int64
IBRS                  object
Description           object
Beat                 float64
Address               object
City                  object
Zip Code               int64
Rep_Dist              object
Area                  object
DVFlag                object
Invl_No                int64
Involvement           object
Race                  object
Sex                   object
Age                  float64
Firearm Used Flag     object
Location              object
dtype: object

#Detect missing values.
df.isna().head(5)

Report_No  Reported_Date  Reported_Time  From_Date  From_Time  To_Date  \
0      False          False          False      False      False    False   
1      False          False          False      False      False    False   
2      False          False          False      False      False     True   
3      False          False          False      False      False     True   
4      False          False          False      False      False    False   

   To_Time  Offense   IBRS  Description    ...     Rep_Dist   Area  DVFlag  \
0    False    False  False        False    ...        False  False   False   
1    False    False  False        False    ...        False  False   False   
2     True    False  False        False    ...        False  False   False   
3     True    False  False        False    ...        False  False   False   
4    False    False  False        False    ...        False  False   False   

   Invl_No  Involvement   Race    Sex    Age  Firearm Used Flag  Location  
0    False        False  False  False   True              False     False  
1    False        False   True   True   True              False     False  
2    False        False  False  False  False              False     False  
3    False        False   True   True   True              False     False  
4    False        False  False  False  False              False     False  

[5 rows x 24 columns]

#Detect existing (non-missing) values.
df.notna().head(5)

   Report_No  Reported_Date  Reported_Time  From_Date  From_Time  To_Date  \
0       True           True           True       True       True     True   
1       True           True           True       True       True     True   
2       True           True           True       True       True    False   
3       True           True           True       True       True    False   
4       True           True           True       True       True     True   

   To_Time  Offense  IBRS  Description    ...     Rep_Dist  Area  DVFlag  \
0     True     True  True         True    ...         True  True    True   
1     True     True  True         True    ...         True  True    True   
2    False     True  True         True    ...         True  True    True   
3    False     True  True         True    ...         True  True    True   
4     True     True  True         True    ...         True  True    True   

   Invl_No  Involvement   Race    Sex    Age  Firearm Used Flag  Location  
0     True         True   True   True  False               True      True  
1     True         True  False  False  False               True      True  
2     True         True   True   True   True               True      True  
3     True         True  False  False  False               True      True  
4     True         True   True   True   True               True      True  

[5 rows x 24 columns]

df.loc[4].at['Report_No'] #df.head(5)
170007467

#Transpose index and columns
df.T

#Generates descriptive statistics that summarize the central tendency,
#dispersion and shape of a dataset’s distribution, excluding NaN values.
df.describe()

          Report_No        Offense           Beat       Zip Code  \
count  1.321390e+05  132139.000000  131293.000000  132139.000000   
mean   1.700392e+08     996.372131     328.665626   65068.210445   
std    5.539807e+05     645.601640     195.853709    5764.338055   
min    1.000808e+08     101.000000       0.000000   20619.000000   
25%    1.700239e+08     630.000000     212.000000   64114.000000   
50%    1.700588e+08     702.000000     315.000000   64127.000000   
75%    1.700836e+08    1401.000000     422.000000   64133.000000   
max    1.717068e+08    3079.000000    9514.000000   99999.000000   

             Invl_No           Age  
count  132139.000000  71014.000000  
mean        1.267446     37.493142  
std         1.026722     15.185495  
min         1.000000     18.000000  
25%         1.000000     26.000000  
50%         1.000000     34.000000  
75%         1.000000     47.000000  
max        30.000000    507.000000  

#Describing all columns of a DataFrame regardless of data type.
df.describe(include='all')

           Report_No Reported_Date Reported_Time   From_Date From_Time  \
count   1.321390e+05        132139        132139      131844    131744   
unique           NaN           365          1440         728      1436   
top              NaN    02/21/2017         12:00  02/21/2017     12:00   
freq             NaN           567           561         523      4610   
mean    1.700392e+08           NaN           NaN         NaN       NaN   
std     5.539807e+05           NaN           NaN         NaN       NaN   
min     1.000808e+08           NaN           NaN         NaN       NaN   
25%     1.700239e+08           NaN           NaN         NaN       NaN   
50%     1.700588e+08           NaN           NaN         NaN       NaN   
75%     1.700836e+08           NaN           NaN         NaN       NaN   
max     1.717068e+08           NaN           NaN         NaN       NaN   

           To_Date To_Time        Offense    IBRS      Description  \
count        49634   49140  132139.000000  131044           132139   
unique         495    1310            NaN      56              116   
top     06/18/2017   08:00            NaN     13B  Property Damage   
freq           286    1999            NaN   15567            12206   
mean           NaN     NaN     996.372131     NaN              NaN   
std            NaN     NaN     645.601640     NaN              NaN   
min            NaN     NaN     101.000000     NaN              NaN   
25%            NaN     NaN     630.000000     NaN              NaN   
50%            NaN     NaN     702.000000     NaN              NaN   
75%            NaN     NaN    1401.000000     NaN              NaN   
max            NaN     NaN    3079.000000     NaN              NaN   

                 Rep_Dist    Area  DVFlag  \
count                  ...                    131132  131132  132139   
unique                 ...                      6868       9       3   
top                    ...                    PJ3601     EPD       U   
freq                   ...                      1641   35166  101664   
mean                   ...                       NaN     NaN     NaN   
std                    ...                       NaN     NaN     NaN   
min                    ...                       NaN     NaN     NaN   
25%                    ...                       NaN     NaN     NaN   
50%                    ...                       NaN     NaN     NaN   
75%                    ...                       NaN     NaN     NaN   
max                    ...                       NaN     NaN     NaN   

              Invl_No Involvement    Race     Sex           Age  \
count   132139.000000      132139  114090  114090  71014.000000   
unique            NaN           3       7       4           NaN   
top               NaN         VIC       B       M           NaN   
freq              NaN       68526   45962   52857           NaN   
mean         1.267446         NaN     NaN     NaN     37.493142   
std          1.026722         NaN     NaN     NaN     15.185495   
min          1.000000         NaN     NaN     NaN     18.000000   
25%          1.000000         NaN     NaN     NaN     26.000000   
50%          1.000000         NaN     NaN     NaN     34.000000   
75%          1.000000         NaN     NaN     NaN     47.000000   
max         30.000000         NaN     NaN     NaN    507.000000   

       Firearm Used Flag                            Location  
count             132139                              132139  
unique                 2                               23769  
top                    N  11600 E US\nHW KANSAS CITY 64133\n  
freq              121106                                1381  
mean                 NaN                                 NaN  
std                  NaN                                 NaN  
min                  NaN                                 NaN  
25%                  NaN                                 NaN  
50%                  NaN                                 NaN  
75%                  NaN                                 NaN  
max                  NaN                                 NaN  

[11 rows x 24 columns]

#Including only string columns in a DataFrame description.
df.describe(include=np.object)

       Reported_Date Reported_Time   From_Date From_Time     To_Date To_Time  \
count         132139        132139      131844    131744       49634   49140   
unique           365          1440         728      1436         495    1310   
top       02/21/2017         12:00  02/21/2017     12:00  06/18/2017   08:00   
freq             567           561         523      4610         286    1999   

          IBRS      Description            Address         City Rep_Dist  \
count   131044           132139             132115       132101   131132   
unique      56              116              15396           38     6868   
top        13B  Property Damage  11600  E US 40 HW  KANSAS CITY   PJ3601   
freq     15567            12206               1473       131519     1641   

          Area  DVFlag Involvement    Race     Sex Firearm Used Flag  \
count   131132  132139      132139  114090  114090            132139   
unique       9       3           3       7       4                 2   
top        EPD       U         VIC       B       M                 N   
freq     35166  101664       68526   45962   52857            121106   

                                  Location  
count                               132139  
unique                               23769  
top     11600 E US\nHW KANSAS CITY 64133\n  
freq                                  1381 

#Including only numeric columns in a DataFrame description.
df.describe(include=np.number)

          Report_No        Offense           Beat       Zip Code  \
count  1.321390e+05  132139.000000  131293.000000  132139.000000   
mean   1.700392e+08     996.372131     328.665626   65068.210445   
std    5.539807e+05     645.601640     195.853709    5764.338055   
min    1.000808e+08     101.000000       0.000000   20619.000000   
25%    1.700239e+08     630.000000     212.000000   64114.000000   
50%    1.700588e+08     702.000000     315.000000   64127.000000   
75%    1.700836e+08    1401.000000     422.000000   64133.000000   
max    1.717068e+08    3079.000000    9514.000000   99999.000000   

             Invl_No           Age  
count  132139.000000  71014.000000  
mean        1.267446     37.493142  
std         1.026722     15.185495  
min         1.000000     18.000000  
25%         1.000000     26.000000  
50%         1.000000     34.000000  
75%         1.000000     47.000000  
max        30.000000    507.000000 

#Excluding string columns in a DataFrame description.
df.describe(exclude=np.object)

          Report_No        Offense           Beat       Zip Code  \
count  1.321390e+05  132139.000000  131293.000000  132139.000000   
mean   1.700392e+08     996.372131     328.665626   65068.210445   
std    5.539807e+05     645.601640     195.853709    5764.338055   
min    1.000808e+08     101.000000       0.000000   20619.000000   
25%    1.700239e+08     630.000000     212.000000   64114.000000   
50%    1.700588e+08     702.000000     315.000000   64127.000000   
75%    1.700836e+08    1401.000000     422.000000   64133.000000   
max    1.717068e+08    3079.000000    9514.000000   99999.000000   

             Invl_No           Age  
count  132139.000000  71014.000000  
mean        1.267446     37.493142  
std         1.026722     15.185495  
min         1.000000     18.000000  
25%         1.000000     26.000000  
50%         1.000000     34.000000  
75%         1.000000     47.000000  
max        30.000000    507.000000 

#Generates descriptive statistics that summarize the central tendency,
#dispersion and shape of a dataset’s distribution, excluding NaN values.
df['Report_No'].describe()

count    1.321390e+05
mean     1.700392e+08
std      5.539807e+05
min      1.000808e+08
25%      1.700239e+08
50%      1.700588e+08
75%      1.700836e+08
max      1.717068e+08
Name: Report_No, dtype: float64

df['Offense'].describe()

count    132139.000000
mean        996.372131
std         645.601640
min         101.000000
25%         630.000000
50%         702.000000
75%        1401.000000
max        3079.000000
Name: Offense, dtype: float64

df['Reported_Date'].describe()
count         132139
unique           365
top       02/21/2017
freq             567
Name: Reported_Date, dtype: object

df['Reported_Time'].describe()
count     132139
unique      1440
top        12:00
freq         561
Name: Reported_Time, dtype: object

df['Description'].describe()
count              132139
unique                116
top       Property Damage
freq                12206
Name: Description, dtype: object

df['Age'].describe()
count    71014.000000
mean        37.493142
std         15.185495
min         18.000000
25%         26.000000
50%         34.000000
75%         47.000000
max        507.000000
Name: Age, dtype: float64

df['Area'].describe()
count     131132
unique         9
top          EPD
freq       35166
Name: Area, dtype: object

df['City'].describe()
count          132101
unique             38
top       KANSAS CITY
freq           131519
Name: City, dtype: object

df['Zip Code'].describe()
count    132139.000000
mean      65068.210445
std        5764.338055
min       20619.000000
25%       64114.000000
50%       64127.000000
75%       64133.000000
max       99999.000000
Name: Zip Code, dtype: float64

#Detect missing values for an array-like object.
df['Report_No'].isnull().sum()
0

df['Offense'].isnull().sum()
0

df['Reported_Date'].isnull().sum()
0

df['Reported_Time'].isnull().sum()
0

df['Age'].isnull().sum()
61125

df['Area'].isnull().sum()
1007

df['City'].isnull().sum()
38

df['Zip Code'].isnull().sum()
0

#count all columns including null
dataset = pd.read_csv('KCPD_Crime_Data_2017.csv')
nnull_columns=df.columns[df.notnull().any()]
df[nnull_columns].notnull().sum()

Report_No            132139
Reported_Date        132139
Reported_Time        132139
From_Date            131844
From_Time            131744
To_Date               49634
To_Time               49140
Offense              132139
IBRS                 131044
Description          132139
Beat                 131293
Address              132115
City                 132101
Zip Code             132139
Rep_Dist             131132
Area                 131132
DVFlag               132139
Invl_No              132139
Involvement          132139
Race                 114090
Sex                  114090
Age                   71014
Firearm Used Flag    132139
Location             132139
dtype: int64

#count null columns
dataset = pd.read_csv('KCPD_Crime_Data_2017.csv')
null_columns=df.columns[df.isnull().any()]
df[null_columns].isnull().sum()

From_Date      295
From_Time      395
To_Date      82505
To_Time      82999
IBRS          1095
Beat           846
Address         24
City            38
Rep_Dist      1007
Area          1007
Race         18049
Sex          18049
Age          61125
dtype: int64

#only City column null and other columns associated with it
print(df[df["City"].isnull()][null_columns])

From_Date From_Time     To_Date To_Time IBRS   Beat  \
917            NaN       NaN         NaN     NaN  120    NaN   
3609    03/06/2017     11:20         NaN     NaN  13B    NaN   
6012    10/26/2017     13:18         NaN     NaN  90Z  999.0   
7439    10/26/2017     14:00         NaN     NaN  35A  999.0   
9767           NaN       NaN         NaN     NaN  23F    NaN   
11909   10/26/2017     13:18         NaN     NaN  35A  999.0   
16190          NaN       NaN         NaN     NaN  240    NaN   
20597          NaN       NaN         NaN     NaN  280    NaN   
20747          NaN       NaN         NaN     NaN  26F    NaN   
21534   11/05/2017     23:50         NaN     NaN  13B  543.0   
23573          NaN       NaN         NaN     NaN  120    NaN   
35386          NaN       NaN         NaN     NaN  280    NaN   
44632   03/06/2017     11:20         NaN     NaN  520    NaN   
47828          NaN       NaN         NaN     NaN  NaN    NaN   
47838   03/06/2017     11:20         NaN     NaN  520    NaN   
48595   10/26/2017     13:18         NaN     NaN  90Z  999.0   
49584          NaN       NaN         NaN     NaN  11A    NaN   
51049   03/06/2017     11:20         NaN     NaN  13B    NaN   
56748   10/18/2017     14:45         NaN     NaN  35A  999.0   
63447   04/01/2017     23:30  04/02/2017   05:00  23D  999.0   
64814   02/07/2017     09:00         NaN     NaN  520    NaN   
67005   01/28/2017     17:42         NaN     NaN  520  999.0   
67163   10/18/2017     14:45         NaN     NaN  35A  999.0   
69347          NaN       NaN         NaN     NaN  120    NaN   
76031   02/07/2017     09:00         NaN     NaN  520    NaN   
76249   11/05/2017     23:50         NaN     NaN  13B  543.0   
78408   10/26/2017     13:18         NaN     NaN  90Z  999.0   
79903          NaN       NaN         NaN     NaN  NaN    NaN   
85418          NaN       NaN         NaN     NaN  23F    NaN   
91533   10/26/2017     13:18         NaN     NaN  35A  999.0   
98830   04/01/2017     23:30  04/02/2017   05:00  23D  999.0   
102531  10/26/2017     13:18         NaN     NaN  35A  999.0   
104691         NaN       NaN         NaN     NaN  90Z    NaN   
112200         NaN       NaN         NaN     NaN  90Z    NaN   
112648         NaN       NaN         NaN     NaN  26F    NaN   
115510         NaN       NaN         NaN     NaN  11A    NaN   
118008         NaN       NaN         NaN     NaN  240    NaN   
128609         NaN       NaN         NaN     NaN  120    NaN   

                         Address City Rep_Dist  Area Race  Sex   Age  
917                          NaN  NaN      NaN   NaN  NaN  NaN   NaN  
3609                         NaN  NaN      NaN   NaN    B    F  23.0  
6012      E 19 ST S and HAZEL AV  NaN   PJX007  OSPD    W    M  23.0  
7439           8300  E TRUMAN RD  NaN   PJX007  OSPD  NaN  NaN   NaN  
9767                         NaN  NaN      NaN   NaN    W    M  46.0  
11909     E 19 ST S and HAZEL AV  NaN   PJX007  OSPD    W    M  23.0  
16190                        NaN  NaN      NaN   NaN    B    F  53.0  
20597                        NaN  NaN      NaN   NaN    W    M  21.0  
20747                        NaN  NaN      NaN   NaN    B    F  36.0  
21534       11800  BLUE RIDGE BL  NaN   PJ7293   SPD    W    M  30.0  
23573                        NaN  NaN      NaN   NaN    B    F  48.0  
35386                        NaN  NaN      NaN   NaN    U    U   NaN  
44632                        NaN  NaN      NaN   NaN    B    M  52.0  
47828                        NaN  NaN      NaN   NaN    W    F  41.0  
47838                        NaN  NaN      NaN   NaN    B    F  23.0  
48595     E 19 ST S and HAZEL AV  NaN   PJX007  OSPD    W    F  19.0  
49584                        NaN  NaN      NaN   NaN    B    F  21.0  
51049                        NaN  NaN      NaN   NaN    B    M  52.0  
56748          8300  E TRUMAN RD  NaN   PJX007  OSPD    W    F   NaN  
63447            700  E WHITE DR  NaN   PJCX01  OSPD    U    U   NaN  
64814                        NaN  NaN      NaN   NaN  NaN  NaN   NaN  
67005   MARSH AV and E TRUMAN RD  NaN   PJX007  OSPD    U    U   NaN  
67163          8300  E TRUMAN RD  NaN   PJX007  OSPD  NaN  NaN   NaN  
69347                        NaN  NaN      NaN   NaN    W    M  54.0  
76031                        NaN  NaN      NaN   NaN    W    M  28.0  
76249       11800  BLUE RIDGE BL  NaN   PJ7293   SPD    W    F  41.0  
78408     E 19 ST S and HAZEL AV  NaN   PJX007  OSPD  NaN  NaN   NaN  
79903                        NaN  NaN      NaN   NaN    U    U   NaN  
85418                        NaN  NaN      NaN   NaN    U    U   NaN  
91533     E 19 ST S and HAZEL AV  NaN   PJX007  OSPD  NaN  NaN   NaN  
98830            700  E WHITE DR  NaN   PJCX01  OSPD    W    M  56.0  
102531    E 19 ST S and HAZEL AV  NaN   PJX007  OSPD    W    F  19.0  
104691                       NaN  NaN      NaN   NaN    U    U   NaN  
112200                       NaN  NaN      NaN   NaN    U    U   NaN  
112648                       NaN  NaN      NaN   NaN    B    F  35.0  
115510                       NaN  NaN      NaN   NaN    B    M  52.0  
118008                       NaN  NaN      NaN   NaN    B    M  53.0  
128609                       NaN  NaN      NaN   NaN    B    M  56.0  

#print all null columns
print(df[df.isnull().any(axis=1)][null_columns].head())

    From_Date From_Time     To_Date To_Time IBRS   Beat  \
0  10/04/2017     16:00  10/05/2017   10:00  23H  221.0   
1  03/20/2017     11:30  04/11/2017   00:45  23C  332.0   
2  01/15/2017     02:35         NaN     NaN  35B  422.0   
3  01/05/2017     11:40         NaN     NaN  35A  122.0   
5  01/14/2017     06:38  01/14/2017   06:40  240  635.0   

                                       Address         City Rep_Dist Area  \
0                                 400  W 58 ST  KANSAS CITY   PJ4582  MPD   
1                            3500  PROSPECT AV  KANSAS CITY   PJ2875  EPD   
2  N GREEN HILLS RD and NW OLD TIFFANY SPRINGS  KANSAS CITY   PP0269  NPD   
3                              1100  TROOST AV  KANSAS CITY   PJ1046  CPD   
5                        4300  N CORRINGTON AV  KANSAS CITY   PC1130  SCP   

  Race  Sex   Age  
0    U    U   NaN  
1  NaN  NaN   NaN  
2    B    M  23.0  
3  NaN  NaN   NaN  
5  NaN  NaN   NaN 

#only City column null and other columns associated with it
null_columns=df.columns[df.isnull().any()]
print(df[df["Address"].isnull()][null_columns])

         From_Date From_Time To_Date To_Time IBRS  Beat Address City Rep_Dist  \
917            NaN       NaN     NaN     NaN  120   NaN     NaN  NaN      NaN   
3609    03/06/2017     11:20     NaN     NaN  13B   NaN     NaN  NaN      NaN   
9767           NaN       NaN     NaN     NaN  23F   NaN     NaN  NaN      NaN   
16190          NaN       NaN     NaN     NaN  240   NaN     NaN  NaN      NaN   
20597          NaN       NaN     NaN     NaN  280   NaN     NaN  NaN      NaN   
20747          NaN       NaN     NaN     NaN  26F   NaN     NaN  NaN      NaN   
23573          NaN       NaN     NaN     NaN  120   NaN     NaN  NaN      NaN   
35386          NaN       NaN     NaN     NaN  280   NaN     NaN  NaN      NaN   
44632   03/06/2017     11:20     NaN     NaN  520   NaN     NaN  NaN      NaN   
47828          NaN       NaN     NaN     NaN  NaN   NaN     NaN  NaN      NaN   
47838   03/06/2017     11:20     NaN     NaN  520   NaN     NaN  NaN      NaN   
49584          NaN       NaN     NaN     NaN  11A   NaN     NaN  NaN      NaN   
51049   03/06/2017     11:20     NaN     NaN  13B   NaN     NaN  NaN      NaN   
64814   02/07/2017     09:00     NaN     NaN  520   NaN     NaN  NaN      NaN   
69347          NaN       NaN     NaN     NaN  120   NaN     NaN  NaN      NaN   
76031   02/07/2017     09:00     NaN     NaN  520   NaN     NaN  NaN      NaN   
79903          NaN       NaN     NaN     NaN  NaN   NaN     NaN  NaN      NaN   
85418          NaN       NaN     NaN     NaN  23F   NaN     NaN  NaN      NaN   
104691         NaN       NaN     NaN     NaN  90Z   NaN     NaN  NaN      NaN   
112200         NaN       NaN     NaN     NaN  90Z   NaN     NaN  NaN      NaN   
112648         NaN       NaN     NaN     NaN  26F   NaN     NaN  NaN      NaN   
115510         NaN       NaN     NaN     NaN  11A   NaN     NaN  NaN      NaN   
118008         NaN       NaN     NaN     NaN  240   NaN     NaN  NaN      NaN   
128609         NaN       NaN     NaN     NaN  120   NaN     NaN  NaN      NaN   

       Area Race  Sex   Age  
917     NaN  NaN  NaN   NaN  
3609    NaN    B    F  23.0  
9767    NaN    W    M  46.0  
16190   NaN    B    F  53.0  
20597   NaN    W    M  21.0  
20747   NaN    B    F  36.0  
23573   NaN    B    F  48.0  
35386   NaN    U    U   NaN  
44632   NaN    B    M  52.0  
47828   NaN    W    F  41.0  
47838   NaN    B    F  23.0  
49584   NaN    B    F  21.0  
51049   NaN    B    M  52.0  
64814   NaN  NaN  NaN   NaN  
69347   NaN    W    M  54.0  
76031   NaN    W    M  28.0  
79903   NaN    U    U   NaN  
85418   NaN    U    U   NaN  
104691  NaN    U    U   NaN  
112200  NaN    U    U   NaN  
112648  NaN    B    F  35.0  
115510  NaN    B    M  52.0  
118008  NaN    B    M  53.0  
128609  NaN    B    M  56.0  

After assessing the output it seems that these 24 removed can be dropped


----------------------------------------------------under work---------------------------------------------

#1 - Find and Replace
#2 - Label Encoding
#3 - One Hot Encoding
#4 - Custom Binary Encoding

obj_df = df.select_dtypes(include=['object']).copy()
obj_df.head()

null_columns=obj_df.columns[obj_df.isnull().any()]
obj_df[null_columns].isnull().sum()

obj_df["Address"].dropna()

null_columns=obj_df.columns[obj_df.isnull().any()]
obj_df[null_columns].isnull().sum()
#1 - Find and Replace





