#==============================================================================
# The purpose of this document is to create data tables that will later be 
# joined together to form the complete dataset for the project
# Each section is separated by the method, topic and source of data. The 
# sections are separated using '##'
#
#
#
#
#
#
#
#
#
#
#==============================================================================


from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import json
import string
import re


###############################################################################

#number of years being used in the analysis DECIDE IF YOU WANT til 2020 or 2021

List_of_state_names =['Alabama', 'Alaska', 'Arizona', 'Arkansas',
       'California', 'Colorado', 'Connecticut', 'Delaware',
       'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
       'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
       'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
       'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
       'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
       'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
       'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
       'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
       'West Virginia', 'Wisconsin', 'Wyoming']
List_of_state_ab = ['AK','AL','AR','AZ','CA','CO','CT','DC','DE','FL','GA','HI','IA','ID','IL','IN','KS','KY','LA','MA','MD','ME','MI','MN','MO','MS','MT','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VA','VT','WA','WI','WV','WY']
long_to_short_dic = {'Alaska':'AK','Alabama':'AL','Arkansas':'AR','Arizona':'AZ','California':'CA','Colorado':'CO','Connecticut':'CT','District of Columbia':'DC','Delaware':'DE','Florida':'FL','Georgia':'GA','Hawaii':'HI','Iowa':'IA','Idaho':'ID','Illinois':'IL','Indiana':'IN','Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Massachusetts':'MA','Maryland':'MD','Maine':'ME','Michigan':'MI','Minnesota':'MN','Missouri':'MO','Mississippi':'MS','Montana':'MT','North Carolina':'NC','North Dakota':'ND','Nebraska':'NE','New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM','Nevada':'NV','New York':'NY','Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Virginia':'VA','Vermont':'VT','Washington':'WA','Wisconsin':'WI','West Virginia':'WV','Wyoming':'WY'}
short_to_long_dic = {'AK':'Alaska','AL':'Alabama','AR':'Arkansas','AZ':'Arizona','CA':'California','CO':'Colorado','CT':'Connecticut','DC':'District of Columbia','DE':'Delaware','FL':'Florida','GA':'Georgia','HI':'Hawaii','IA':'Iowa','ID':'Idaho','IL':'Illinois','IN':'Indiana','KS':'Kansas','KY':'Kentucky','LA':'Louisiana','MA':'Massachusetts','MD':'Maryland','ME':'Maine','MI':'Michigan','MN':'Minnesota','MO':'Missouri','MS':'Mississippi','MT':'Montana','NC':'North Carolina','ND':'North Dakota','NE':'Nebraska','NH':'New Hampshire','NJ':'New Jersey','NM':'New Mexico','NV':'Nevada','NY':'New York','OH':'Ohio','OK':'Oklahoma','OR':'Oregon','PA':'Pennsylvania','RI':'Rhode Island','SC':'South Carolina','SD':'South Dakota','TN':'Tennessee','TX':'Texas','UT':'Utah','VA':'Virginia','VT':'Vermont','WA':'Washington','WI':'Wisconsin','WV':'West Virginia','WY':'Wyoming'}
# assigning a list to use for urls
# Remove 2021 if you decide to change the nuber of years
year = ['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
num_years = len(year)

############################################################################### 

# Loading the Electric vehicle population data
EV_pull = pd.read_csv("C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5622 Machine Learning/CSCI5662 Project/EV_population_data.csv")

# Checking the data using head()

# print(EV_pull.head()) #uncomment to execute

# Grouping rows by State then year, and calculating the count of 
# observations in each group, naming count column 'Label_ref'

EV_1 = EV_pull.groupby(['State', 'Model Year']).size().reset_index(name='Label_ref')
print(EV_1)

# Checking the class of EV_1
print(type(EV_1))

# Dropping rows with Model Year < 2008
EV_1.drop(EV_1[EV_1['Model Year']<2008].index, inplace = True)

# checking type of EV_1
print(type(EV_1))
#print(EV_1)

# checking the maximum value of Label_ref
print(EV_1['Label_ref'].max())

# checking mean of Label_ref
print(EV_1['Label_ref'].mean())

# Setting markers for dicretizing the Label_ref = Count of EV vehicles per state per year
low_marker = 600
medium_marker = 15000

# creating discretized column 'Label' 
conditions = [
    (EV_1['Label_ref'] <= low_marker),
    (EV_1['Label_ref'] > low_marker) & (EV_1['Label_ref'] <= medium_marker),
    (EV_1['Label_ref'] > medium_marker)
    ]
# Assigning respective values based on condition
values = ['Low', 'Medium', 'High']

#Adding column 'Label' based on conditions and values above
EV_1['Label'] = np.select(conditions, values)

EV_1['State'] = EV_1['State'].replace(short_to_long_dic)
EV_1.rename(columns = {'Model Year':'Year'}, inplace = True)

print(EV_1.head())

#Since there is an issue in merging this dataframe with the others, we need to check
# the class of elements in bothe state and year column
# bothe should be strings

print(type(EV_1.iloc[3]['State']))
print(type(EV_1.iloc[3]['Year']))

EV_1['Year'] = EV_1['Year'].map(str)


###############################################################################

#Energy Consumption data per state 2008-2020

# Loading csv file for Energy consumption of each state till 2020 in Billion Btu
E_state_pull = pd.read_csv("C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5622 Machine Learning/CSCI5662 Project/EnergyConsumptionByState.csv")
E_state_pull.info()

# Keeping only the columns and years that are required
E_state_pull = E_state_pull[['State', 'MSN', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']]

# Dropping rows where MSN != TETCB
E_state_pull.drop(E_state_pull[E_state_pull['MSN'] != 'TETCB'].index, inplace = True)
print(E_state_pull.head())

# Changing the structure of the dataframe
State_energy_con = pd.melt(E_state_pull, id_vars = ['State'], value_vars=['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020'], var_name='Year', value_name='EnergyC_addunit')
State_energy_con


# Removing 'US' from the dataframe
State_energy_con.drop(State_energy_con[State_energy_con['State'] == 'US'].index, inplace = True)

State_energy_con['State'] = State_energy_con['State'].replace(short_to_long_dic)
State_energy_con.head()
print(State_energy_con.dtypes)

###############################################################################

# Education attainment table, Bachelor's degree or higher by state 2008-2021


#creating sub urls to join in the function    
url_1 = 'https://fred.stlouisfed.org/release/tables?rid=330&eid=391444&od='
url_2 = '-01-01#'   

def get_fred_education_data():
    states=[]
    education_level = []
    Year = []
    
    for i in range(num_years):
        url = url_1+year[i]+url_2

        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        var = soup.find_all('span', class_ = 'fred-rls-elm-nm') 
        for state in var:
            s = state.get_text()
            #print(s)
            states.append(s)    #Adding each state to the list of states
            Year.append(year[i])
        #print(len(states))
        
        ed = soup.find_all('td', class_ = 'fred-rls-elm-vl-td')
        for element in ed:
           t = element.get_text().strip()
           #print(t)
           education_level.append(t)     #Adding each text item to the list of education level percentage
        #Creating new list because multiple lines had the'td' tag and only every third value is needed
        edu_level = education_level[0::3]  
         
    final = [states,edu_level, Year]    
    return(final)
        
    

education = get_fred_education_data()


education_data = pd.DataFrame()
education_data['State']=education[0]
education_data['Percentage_pop_with_bachelors'] = education[1]
education_data['Year']=education[2]
education_data['Percentage_pop_with_bachelors'] = education_data['Percentage_pop_with_bachelors'].astype(float)
print(education_data.dtypes)
print(education_data.tail())

###############################################################################

#Average salary by state from 2008-2021


salary_url_pre = 'https://fred.stlouisfed.org/release/tables?rid=249&eid=259515&od='
salary_url_post = '-01-01#'

def get_salary_data():
    
    #defining vectors to add data to
    state_name_inc = []
    median_income_temp = []
    Year_inc = []

    for i in range(num_years):
        
        url = salary_url_pre+year[i]+salary_url_post
        #print(url)
        res = requests.get(url)
        #print(res)
        
        soup_median_income = BeautifulSoup(res.content, 'html.parser')
        print(soup_median_income.find_all('td', class_ = 'fred-rls-elm-nm-cntnr fred-rls-elm-nm-td'))
        find_state_name = soup_median_income.find_all('td', class_ = 'fred-rls-elm-nm-cntnr fred-rls-elm-nm-td')
        for state_name in find_state_name:
            #print(state_name.get_text().strip())
            state_name_inc.append(state_name.get_text().strip())
            Year_inc.append(year[i])
            
        #print(state_name_inc)
        #print(len(state_name_inc))
        
        find_inc = soup_median_income.find_all('td', class_ = 'fred-rls-elm-vl-td')    
        for inc in find_inc:
            to_text = inc.get_text().strip()
            to_text = re.sub(',','',to_text)
            median_income_temp.append(to_text)
            #print(to_text)
        #print(median_income_temp) 
        #print(len(median_income_temp))
        median_inc = median_income_temp[0::3] 
        #print(len(median_inc))
        
         
    final_inc = [state_name_inc, Year_inc, median_inc]
    return final_inc

med_inc = get_salary_data()
print(med_inc)

median_salary_data = pd.DataFrame()
median_salary_data['State'] = med_inc[0]
median_salary_data['Year'] = med_inc[1]
median_salary_data['Median_income'] = med_inc[2]
median_salary_data.dtypes
median_salary_data['Median_income'] = median_salary_data['Median_income'].astype(float)
median_salary_data.dtypes
median_salary_data.drop(median_salary_data[median_salary_data['State']=='The United States'].index, inplace = True)
print(median_salary_data)



###############################################################################

#Age distribution by state: Adults 19-25, Adults 26-34, Adults 35-54


year
old_pop_data= pd.read_csv('C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5622 Machine Learning/CSCI5662 Project/pop_data_2018.csv', skiprows=2)
print(old_pop_data.shape)


def get_pop_data():
    file_1 = 'C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5622 Machine Learning/CSCI5662 Project/pop_data_'
    frames =[]
    for i in range(num_years):
        filename= file_1+year[i]+'.csv'
        
        doc = pd.read_csv(filename, skiprows=(2))
        #print(doc.shape)
        
        doc = doc[['Location', 'Adults 19-25', 'Adults 26-34', 'Adults 35-54']]
               
        print(doc.shape)
        
        delete_later =[]
        
        for j in range(len(doc['Location'])):
            
            
            if doc.iloc[j]['Location'] in List_of_state_names:
                delete_later.append(1)
            else:
                delete_later.append(0)
            
        
        doc['Check'] = delete_later
        doc.drop(doc[doc['Check'] != 1].index, inplace=True)

        doc = doc[['Location',  'Adults 19-25',  'Adults 26-34',  'Adults 35-54']]
        mul = len(doc[['Location']])
        Year_pop_data =[year[i]]*mul
        doc['Year'] = Year_pop_data
        doc['Year'] = doc['Year'].astype(object)
        doc.rename(columns={'Location':'State'}, inplace = True)
        
        frames.append(doc)
    return pd.concat(frames)
        

        #doc.shape
Population_data = get_pop_data()
print(Population_data)
print(Population_data.shape)
print(Population_data.dtypes)


###############################################################################

#Getting News data about 'EV', "Electric Vehicles' and 'renewable energy'

filename = 'Headlines_for_analysis.csv'
this_file = open(filename, 'w')

write_stuff = 'Label,Date,Source,Title,Description\n'
this_file.write(write_stuff)
this_file.close()
q_topics = ['electric vehicle', 'renewable energy', 'energy']
end_point = 'https://newsapi.org/v2/everything'

for topic in q_topics:
    
    URL_dic = {'apiKey': 'e814794a812449d1b7c7d485b5e4525a',
              'q': topic,
              'pageSize': 100,
              'sortBy' : 'top',
              'totalRequests': 100
              }
    response = requests.get(end_point, URL_dic)
    print(response)
    json_text = response.json()
    print(json_text)
    
    
    this_file = open(filename, 'a')
    LABEL = topic
    for item in json_text['articles']:
        print(item, "\n\n")
        
        #Source
        Source = item['source']['name']
        print(Source)
        
        #Date
        A_date = item['publishedAt']
        B_date = A_date.split('T')
        date = B_date[0]
        print(date)
        
        #Title
        title = item['title']
        title = title.lower()
        title = re.sub(r'\d+', '', title)
        title = re.sub('—',' ', str(title))
        table = title.maketrans('','',string.punctuation)
        title = title.translate(table)
        title = re.sub('\n|\r', '', title)
        title = re.sub(r' +', ' ', title)
                    
        title = title.strip()
        print(title)
    
        description = item['description']
        print(description,'\n')
        description = str(description)
        description = description.lower()
        description = re.sub(r'\d+', '', description)
        description = re.sub(r'[,‘’.—;@%_#?!&$/()="<>:\-\']+', ' ', description, flags = re.IGNORECASE)
        description = re.sub('\n|\r', '', description)
        description = re.sub(r' +', ' ', description)
                    
        title = title.strip()
        print(title)
        
        write_stuff = str(LABEL)+','+str(date)+','+str(Source)+','+str(title)+','+str(description)+'\n'
        print(write_stuff)
        
        this_file.write(write_stuff)
    
    this_file.close()
    

text_df = pd.read_csv(filename, error_bad_lines = False, encoding = 'cp1252')
text_df.head()

text_df.columns

text_df = text_df.dropna()
text_df.drop_duplicates(inplace=True)
print(len(text_df['Description']))

description_list = []  ## USE THIS AS THE SNIPPET
label_list = []


ARM_description_list = []
for i in range(len(text_df['Description'])):
    listname = 'list'+str(i)
    
    x = (text_df.iloc[i]['Description'])
    #print([x])
    ARM_description_list.append([x])

## Use this list for ARM    
print(ARM_description_list)

###############################################################################

## Total employment in each state from 2008-2020


# To get LineCode the below code will return all the possible line codes. 
# We need to select the total employment Line Code
get_line_code = 'http://apps.bea.gov/api/data?&UserID=21B05CEF-CC78-40C4-BF59-752A897A0DAF&method=GetParameterValuesFiltered&datasetname=Regional&TargetParameter=LineCode&TableName=CAEMP25N&ResultFormat=JSON'
response = requests.get(get_line_code)
print(response.content)


# Creating an endpoint for the API call
endpoint_jobs = 'https://apps.bea.gov/api/data'

# Creating  a BEA API dictionary to access tables
bea_API_dic = {'TableName' : 'CAEMP25N', 
'GeoFips' : 'STATE',
'Year' : '2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020',
'datasetname' : 'Regional',
'LineCode' : '10',
'method' : 'GetData',
'UserID' : '21B05CEF-CC78-40C4-BF59-752A897A0DAF',
'ResultFormat' : 'json'
    }


## Creating csv file to add data to
job_filename = 'tot_employment.csv'
file = open(job_filename, 'w')

# Adding headers to the csv file, note the \n in the end
write_this = 'State,Year,num_jobs_tot\n'
file.write(write_this)
file.close()


## Getting data from API
response = requests.get(endpoint_jobs,bea_API_dic)
print(response)
json_t = response.json()
print(json_t)  ########## Use this for before data cleaning

## Saving json file to computer
save_file = open('employement.json', 'w')
json.dump(json_t, save_file, indent =4)  #this saves the file in a pretty format
save_file.close()

## Extracting relevant information from files
file = open(job_filename, 'a')

for item in json_t['BEAAPI']['Results']['Data']:
    #print(item, '\n\n')
    State = item['GeoName']
    #print(State)
    
    Year = item['TimePeriod']
    
    #Removing commas cause it can fuck up the csv file
    num_jobs_total = item['DataValue']
    num_jobs_total = re.sub(',','', num_jobs_total)
        
    #writing each observation in the csv file
    write_new = str(State)+','+str(Year)+','+str(num_jobs_total)+'\n'
    print(write_new)
    file.write(write_new)
    
file.close()


# Converting csv to pandas dataframe
tot_jobs = pd.read_csv(job_filename)
tot_jobs.head()

# Removing all State variable values that are irrelevant 
delete_later_2 = []
for j in range(len(tot_jobs['State'])):
    if tot_jobs.iloc[j]['State'] in List_of_state_names:
        delete_later_2.append(1)
    else:
        delete_later_2.append(0)
        
tot_jobs['Crap'] = delete_later_2

tot_jobs.drop(tot_jobs[tot_jobs['Crap'] != 1].index, inplace =True)

tot_jobs.head()

tot_jobs = tot_jobs[['State', 'Year', 'num_jobs_tot']]
tot_jobs.shape
tot_jobs['Year'] = tot_jobs['Year'].astype(object)
tot_jobs.dtypes
print(tot_jobs)
# Since all the values from this dataframe did not show up in the main dataframe along with the others
# Henche checking if there is an issue in the Key = State, Year
print(tot_jobs.iloc[3]['State'])
print(len(tot_jobs['Year']))
#checking the class of the element in the YEar column
print(type(tot_jobs.iloc[3]['Year']))
#returns int. This is why the merge function was not working for this dataframe

#converting int to str using map function
tot_jobs['Year'] = tot_jobs['Year'].map(str)    
print(type(tot_jobs.iloc[3]['Year']))
#returns string. Good

###############################################################################


# Statewise average temperature data from 2008-2021

#def get_state_temp():
    
file_end = '_State_av_temp.csv'
temp_data = []   
for i in range(num_years):
    year_vec =[]
    temp_filename = year[i]+file_end
    
    temp_1 = pd.read_csv(temp_filename, skiprows=(3))
    print(temp_1.columns)
    temp_1 = temp_1[['Location', 'Value']]
    year_vec = [year[i]]*len(temp_1['Location'])
    temp_1['Year'] = year_vec
    temp_1.rename(columns = {'Location':'State','Value':'Av_temperature'}, inplace = True)
    print(temp_1.head())
    temp_data.append(temp_1)

temperature_data = pd.concat(temp_data)  
temperature_data.dtypes
        
    
    
###############################################################################

# Getting GDP per capita by state from 2008-2021

#Unit: Millions of dollar

gdp_1 = pd.read_csv('GDP_by_State.csv', skiprows=(3))
gdp_1.head()

gdp_1 = gdp_1[['GeoName', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']]
gdp_1.head()

#gdp_1.drop(gdp_1[gdp_1['GeoName']=='United States *'].index, inplace =True)
#gdp_1.head()

delete_later_3 =[]
for i in range(len(gdp_1['GeoName'])):
    if gdp_1.iloc[i]['GeoName'] in List_of_state_names:
        delete_later_3.append(1)
    else:
        delete_later_3.append(0)
        
gdp_1['Crap'] = delete_later_3
gdp_1.drop(gdp_1[gdp_1['Crap']!=1].index, inplace =True)
gdp_1.columns

gdp_1 = gdp_1[['GeoName', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
       '2015', '2016', '2017', '2018', '2019', '2020', '2021']]
gdp_1.columns

GDP_data = pd.melt(gdp_1, id_vars=['GeoName'], value_vars=['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021'], var_name='Year', value_name='GDP')
GDP_data.head()

GDP_data.rename(columns={'GeoName':'State'}, inplace = True)
GDP_data.columns
GDP_data.dtypes



#GDP_data['GDP'] = GDP_data['GDP'].astype(float)
#GDP_data.dtypes



###############################################################################

# Here all the data collected will be combined.
# Checking once again if all the data frames are in the desired format

print('The shape of Energy consumption dataframe is:', State_energy_con.shape)
print(State_energy_con.columns)
print('The shape of Bachelore attainment dataframe is:', education_data.shape)
print(education_data.columns)
print('The shape of Population dataframe is:', Population_data.shape)
print(Population_data.columns)
print('The shape of Total jobs dataframe is:', tot_jobs.shape)
print(tot_jobs.columns)
print('The shape of Average temperature dataframe is:', temperature_data.shape)
print(temperature_data.columns)
print('The shape of Electric vehicle registration dataframe is:', EV_1.shape)
print(EV_1.columns)
print('The shape of Median salary dataframe is:', median_salary_data.shape)
print(median_salary_data.columns)
print('The shape of GDP dataframe is:', GDP_data.shape)
print(GDP_data.columns)

# It can be seen that the shapes of the dataframes are different. This is an issue. 
# The Bachelor attainment, Average temp and Population data have data from 2008-2021, whereas
# Energy consumption and total jobs dataframe have data from 2008-2020
# The Electric vehicle registration data is much smaller than the rest. 
# This is because Electric vehicles sales have just skyrocketed


pd_1 = pd.merge(education_data,Population_data, how = 'outer', on=['State', 'Year'])
pd_1.head()
pd_2 = pd.merge(pd_1,State_energy_con, how = 'outer', on=['State', 'Year'])
pd_2.tail()
pd_2.dtypes
pd_3 = pd.merge(pd_2,tot_jobs, how = 'outer', on=['State', 'Year'])
pd_3.columns
pd_4 = pd.merge(pd_3,temperature_data, how = 'outer', on=['State', 'Year'])
pd_4.columns
pd_5 = pd.merge(pd_4, median_salary_data, how = 'outer', on=['State', 'Year'])
pd_5.columns
pd_6 = pd.merge(pd_5, GDP_data, how = 'outer', on = ['State', 'Year'])
pd_6.columns
pd_6.shape
pd_7 = pd.merge(pd_6, EV_1, how='outer',  on=['State', 'Year'])
pd_7.columns

pd_7.to_csv('C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5622 Machine Learning/CSCI5662 Project/Initial_EV_dataframe.csv')

# At this point, it was found that the EV sales data was inaccurate
# Using numerous other sources and some common sense another set for EV sales was fabricated
# https://www.copilotsearch.com/posts/states-with-the-most-electric-vehicles/
EV_sales = pd.read_csv("C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5622 Machine Learning/CSCI5662 Project/EV_data_fab.csv")
EV_sales.head()

print(type(EV_sales.iloc[2]['Year']))

# Changing Year from int to str

EV_sales['Year'] = EV_sales['Year'].map(str)

print(type(EV_sales.iloc[2]['Year']))

low_marker = 800
medium_marker = 2000

# creating discretized column 'Label' 
conditions = [
    (EV_sales['Number of registrations'] <= low_marker),
    (EV_sales['Number of registrations'] > low_marker) & (EV_sales['Number of registrations'] <= medium_marker),
    (EV_sales['Number of registrations'] > medium_marker)
    ]
# Assigning respective values based on condition
values = ['Low', 'Medium', 'High']

#Adding column 'Label' based on conditions and values above
EV_sales['Label'] = np.select(conditions, values)

pd_x = pd.merge(pd_6, EV_sales, how='outer',  on=['State', 'Year'])
pd_x.columns

pd_x.to_csv('C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5622 Machine Learning/CSCI5662 Project/ElectricVehiclePred.csv')




