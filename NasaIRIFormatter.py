from datetime import date,timedelta
import pandas as pnd
import numpy as np


def is_unique(s):
    a = s.to_numpy()
    return (a[0] == a).all()

names=[]
datafolder="data\\"
with open(datafolder+"ISTA_2017.txt") as file:
    for line in file.readlines()[1:59]:
       names.append(line[line.find(' ')+1:].strip().replace(" ", ""))

    
df=pnd.read_csv(datafolder+'ISTA_2017.txt', skiprows=61,header=None,delim_whitespace=True)
df.columns=names
##print(df)

df_size=len(df.index)
print(df_size)
height_start=100
height_end=300
parameters_to_check=[names[5],names[14],names[15],names[26],names[36],names[41],names[46],names[51],names[52],names[53],names[55],names[57]]
constant_parameters=['Year','Month','Day','DOY','Hour']
changing_parameters=[]

#print(parameters_to_check)
is_parameter_constant=True

#----Analyze Section--------------
for parameter in parameters_to_check:

    is_parameter_constant=True
    
    print("Checking",parameter,"'s interchange according to hour and height ")

    for hour in range(0,df_size,39): #height points for each hour=39

        sc_start=(height_start//50-2)+hour
        sc_end=(height_end//50-2)+hour
        
        hour_height_section=df[parameter][sc_start:sc_end]

        if is_unique(hour_height_section):
            print(parameter,"is same for the dataframe elements between index ",sc_start,"and",sc_end)
        else:
            is_parameter_constant=False
            print(parameter,"has changed for the dataframe elements between index ",sc_start,"and",sc_end)
            
    if is_parameter_constant:
        print(parameter,"is constant and can be reduced to 1 value per hour")
        constant_parameters.append(parameter)
    else :
        print(parameter,"is changing and will be extendend as a feature according to height")
        changing_parameters.append(parameter)

print("CONSTANT PARAMETERS (WITH TIME PARAMETERS) ARE : ", constant_parameters)
print("CHANGING PARAMETERS ARE : ", changing_parameters)


#-------Manipulation Section----------------
print("STARTING TO MANIPULATE...")

constant_frame=df[constant_parameters].copy()
changing_frame=df[changing_parameters].copy()

new_data_constant_list =[]
new_data_changing_list =[]

new_column_names =[]

for parameter in changing_parameters:
    for height in range(height_start,height_end+1,50):
        new_column_names.append(parameter+'_'+str(height)+'km')

print("NEW PARAMETERS WILL BE : ", constant_parameters+new_column_names)

for hour in range(0,len(constant_frame.index),39):
    new_data_constant_list.append(constant_frame.iloc[[hour]].values.flatten().tolist())

new_constant_frame=pnd.DataFrame(data=np.array(new_data_constant_list),columns=constant_parameters)

print("CONSTANT DATAFRAME IS GENERATED")

#for changing parameters

for index in range(0,len(changing_frame.index),39):

    sc_start=(height_start//50-2)+index
    sc_end=(height_end//50-2)+index
    row_list=[]

    for parameter in changing_parameters:
        for sub_index in range(sc_start,sc_end+1):
            row_list.append(changing_frame[parameter][sub_index])

    new_data_changing_list.append(row_list)


new_changing_frame=pnd.DataFrame(data=np.array(new_data_changing_list),columns=new_column_names)

print("CHANGING DATAFRAME IS GENERATED")


result_frame = pnd.concat([new_constant_frame, new_changing_frame], axis=1)
print("New Data Frame With New Features ")
print(result_frame)

result_frame.to_csv(datafolder+"ISTA_2017_EDITED.txt", encoding='utf-8', index=False,sep='\t')

print("Saved the file.")












