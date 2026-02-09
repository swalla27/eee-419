# pandas examples
  
from pandas import DataFrame, read_csv
import pandas as pd

# let's create some data so we can us pandas to export it
# majors:   the majors represented by students in the class
# students: how many students with each major
# cols:     the column headings for our dataframe

majors = ['CS','CSE','EE','Physics','Chemistry']   # majors
students = [15, 15, 35, 3, 2]                      # count of students
cols = [ 'major', 'count' ]                        # column headings

# zip them together to create a list of tuples
enrolled = list(zip(majors,students))

# print what we have so far
print(majors)
print(students)
print(enrolled)

# create a data frame
student_df = DataFrame( data = enrolled, columns = cols )
print('\n\nThe original dataframe')
print(student_df)

# write a CSV file
student_df.to_csv('enrollment.csv',index=False,header=True)

# can we read it back in?
check = read_csv('enrollment.csv')
print('\n\nRead the data back in...')
print(type(check))
print(check)

# export to Excel
check.to_excel('check.xlsx',sheet_name='first_sheet',index=False)

# read it back in
get_back = pd.read_excel('check.xlsx',0)
print('\nread it back in from excel')
print(get_back)

# what are the types of the fields?
print('\ncheck the data types of the fields')
print(student_df.dtypes)

# get information about the dataframe
print("\nDataframe info")
print(student_df.info())

# get more info
print('\nmore information...')
print(student_df['count'].describe())

# check for unique values
print('\nunique values of counts')
print(student_df['count'].unique())

# find the min and max values
print("\nMaximum and minimum count values")
print(student_df['count'].max())
print(student_df['count'].min())

# print the first few rows
print('\nFirst few rows')
print(student_df.head(2))

# print the first few rows a different way
print('\nFirst few rows by index')
print(student_df[:2])

# print the last few rows
print('\nLast few rows')
print(student_df.tail(2))

# now let's sort the data
sort_major = check.sort_values(['major'],ascending=True)
print('\ntype of sorted data')
print(type(sort_major))

print('\nthe data sorted by major')
print(sort_major)

# now let's sort the data
sort_major = check.sort_values(['major'],ascending=False)
print('\nthe data sorted by major descending')
print(sort_major)

# rename a column
print('\nrename columns')
check.columns=['major','num_students']
print(check)

# add a column
print('\nadd a column with a constant value')
check['space_available'] = 10
print(check)

# add a column of data
print('\nadd a column of different values')
check['room size'] = [100,200,300,400,500]
print(check)

print('\nchange a single value')
check.at[3,'major'] = 'BIO'
print(check)
# put it back!
check.at[3,'major'] = 'Physics'

# change the value in a column
print('\nchange all values in a column')
check['room size'] -= 50
print(check)

# delete a column
print('\ndelete a column')
del check['space_available']
print(check)

# select columns to print
print('\nprint only selected columns')
print(check[['major','room size']])

# grouping with the same value
print('\ngroup by number of students')
num_stu = check.groupby('num_students')
print(num_stu.sum())

new_majors = ['MATH','AERO','ME']   # majors
new_students = [6, 9, 5]            # count of students

# zip them together to create a list of tuples
new_enrolled = list(zip(new_majors,new_students))

# print what we have so far
print('\nNew dataframe')
print(new_majors)
print(new_students)
print(new_enrolled)

# create a data frame
new_student_df = DataFrame( data = new_enrolled, columns = cols )
print('\na new dataframe')
print(new_student_df)

# now combine two data frames
print('\nand combine it with the original')
combo = pd.concat([student_df,new_student_df])
print(combo)

# what is the size of the dataframe?
print('\nthe shape of this dataframe is')
print(combo.shape)

# find things by their index
print("\nprint rows with index 0")
print(combo.loc[0])

# reset the indices
print("\nreset the indices")
new_combo = combo.reset_index()
print(new_combo)

# reset without the "index" column
print("\nreset indices w/o index column in place")
combo.reset_index(inplace=True,drop=True)
print(combo)

# more ways to find things
print('\nget index 3 of the major column')
print(combo.at[3,'major'])

print('\nspecify third row and second column indices')
print(combo.iat[2,1])

print('\nfourth row, second element')
print(combo.loc[3].iat[1])

print('\nuse a filter')
print(combo[combo['count'] > 10])

# find correlations
print('\ncorrelation between columns')
col_1 = check['num_students']
col_2 = check['room size']
correlation = col_1.corr(col_2)
print(correlation)

print('\ncomplete correlation matrix')
print(check.corr())

# find covariances
print('\ncovariance between columns')
covariance = col_1.cov(col_2)
print(covariance)

print('\ncomplete covariance matrix')
print(check.cov())
