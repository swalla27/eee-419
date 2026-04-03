# pandas visualization examples
# author: sdm

from pandas import DataFrame, read_csv             # bring in the packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

majors = ['CS','CSE','EE','Physics','Chemistry']   # majors
students = [15, 12, 35, 3, 2]                      # count of students
a_grade = [5,5,20,1,0]                             # students with As
b_grade = [8,4,10,1,1]                             # students with Bs
c_grade = [1,2,2,1,0]                              # students with Cs
d_grade = [1,1,2,0,1]                              # students with Ds
e_grade = [0,0,1,0,0]                              # students with Es

# column headings
cols = [ 'major', 'count', 'a_grade', 'b_grade', 'c_grade',
         'd_grade', 'e_grade']
grades = cols[2:]

# zip them together to create a list of tuples
enrolled = list(zip(majors,students,a_grade,b_grade,c_grade,d_grade,e_grade))
student_df = DataFrame( data = enrolled, columns = cols )
student_df.set_index('major',inplace=True,drop=False)
print(student_df)

# generate plots - a pie chart
student_df['count'].plot(kind='pie',y='major',
                         labels=student_df['major'],autopct='%1.1f%%')
plt.title('number of students per major')
plt.axis('off')
plt.show()

# generate plots - an exploding pie chart with a shadow
explode_list = np.zeros(len(student_df),float)
explode_list[-3:] = [.1,.2,.3]
student_df['count'].plot(kind='pie',y='major',shadow=True,explode=explode_list,
                         startangle=0,
                         labels=student_df['major'],autopct='%1.1f%%')
plt.title('percentage of students per major: exploding slices')
plt.axis('off')
plt.show()

# generate plots - a historgram
student_df.plot(kind='hist',y='count',
                bins=np.arange(0,student_df['count'].max()+5,5))
plt.xlabel('number of students')
plt.title('histogram of students per major')
plt.show()

# generate plots - legend automatically added if > 1 line!
student_df.plot(kind='line')
plt.xlabel('major')
plt.ylabel('number of students')
plt.title('students per major')
plt.show()

# generate a line chart for EE major
ee_grades = student_df.loc['EE',grades]
ee_grades.plot(kind='line')
plt.ylabel('number of students')
plt.xlabel('grades')
plt.title('grade distribution for EE students')
plt.show()

# create an area plot for grades per major
# and transpose so axes are swapped!
grades_df = student_df[grades].transpose()     # extract just the grades
grades_df.plot(kind='area')                    # create the plot
plt.ylabel('number of students')
plt.xlabel('grade')
plt.title('grade distribution - stacked')
plt.show()

# create an area chart that is not stacked
grades_df.plot(kind='area',stacked=False,alpha=0.75)
plt.ylabel('number of students')
plt.xlabel('grade')
plt.title('grade distribution - unstacked')
plt.show()

# read in a dataframe for use with bar charts
gifts_df = read_csv('gifts.csv',index_col=0)
print("\n\nthe new dataframe")
plt.title('grade distribution')
print(gifts_df)

# create a bar chart for one of the items
gifts_df.loc['toys'].plot(kind='bar',rot=0)    # rot rotates x-axis labels
plt.ylabel('count')
plt.title('toys per year')
plt.show()

# create a bar chart for all the items at once
gifts_df.plot(kind='bar',rot=0)
plt.ylabel('count')
plt.title('gifts per year')
plt.show()

# create a bar chart for all the items at once and stack them
gifts_df.plot(kind='bar',rot=0,stacked=True)
plt.ylabel('count')
plt.title('gifts per year - stacked')
plt.show()

# create a horizontal bar chart for all the items at once and stack them
gifts_df.plot(kind='barh',rot=0,stacked=True)
plt.xlabel('count')
plt.title('gifts per year - horizontal stack')
plt.show()

# create a box and whiskers plot
gifts_df.plot(kind='box')
plt.ylabel('gifts per year')
plt.title('box chart for gifts per year')
plt.show()

# swap the axes so it makes more sense
gifts_tp_df = gifts_df.transpose()
gifts_tp_df.plot(kind='box')
plt.ylabel('gifts per year')
plt.title('box chart for gift types per year')
plt.show()

# show the graph horizontally
gifts_tp_df.plot(kind='box',vert=False)
plt.xlabel('gifts per year')
plt.title('box chart for gift types per year horizontally')
plt.show()

# create subplots for the graphs!
fig = plt.figure()
ax0 = fig.add_subplot(1,2,1)      # add to first position of 1x2 display
ax1 = fig.add_subplot(1,2,2)      # add to second position of 1x2 display
gifts_tp_df.plot(kind='box',ax=ax0,figsize=(10,4))
gifts_tp_df.plot(kind='box',vert=False,ax=ax1)
ax0.set_ylabel('gifts per year')
ax0.set_title('gift types per year')
ax1.set_xlabel('gifts per year')
ax1.set_title('gift types per year')
fig.suptitle('vertical vs horizontal box charts')
plt.show()

# add a row that sums the others; transpose so it's a column
print("\n\nadd a total row, transpose, and add a column")
gifts_df.loc['Yearly Total'] = gifts_df.sum()
gifts_tp_df = gifts_df.transpose()
gifts_tp_df['Year Num'] = list(range(1,len(gifts_tp_df)+1))
print(gifts_tp_df)

# create a scatter plot
gifts_tp_df.plot(kind='scatter',x='Year Num',y='Yearly Total')
plt.title('scatter plot of gifts vs year')
plt.show()

# create a bubble plot after creating normalized weights
weights = gifts_tp_df['Yearly Total'] - gifts_tp_df['Yearly Total'].min()
weights /= ( gifts_tp_df['Yearly Total'].max() - 
             gifts_tp_df['Yearly Total'].min() )
weights += 1        # in case of 0!
weights *= weights  # get larger differences
weights *= 100      # make visible
gifts_tp_df.plot(kind='scatter',x='Year Num',y='Yearly Total',s=weights)
plt.title('scatter plot of gifts vs year')
plt.show()

# use seaborn with pandas - create a new data frame with sample data
clean_data = np.arange(0.0,50.0)
noise = 5 * np.random.randn(len(clean_data))
data = clean_data + noise
sample_num = range(0,50)

sample_df = DataFrame(data = zip(sample_num,data),columns=['sample','value'])

ax = sn.regplot(x=sample_df.index,y='value',data=sample_df)
plt.xlabel('sample number')
plt.title('regression line with 95% confidence level')
plt.show()
