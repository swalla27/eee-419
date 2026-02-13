# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 13 February 2026

# Project 1 Problem 1

# I did not use AI at all to complete this assignment

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#####################
##### Problem 1 #####
#####################

# First, read the heart data csv into a df. 
df = pd.read_csv('projects/heart1.csv')

# Next, print characteristics of the data to the terminal, such as the header, some stats, correlation and covariance matrices.
print('\nFirst several rows of these data:')
print(df.head())

print('\nStatistical measures for each column:')
print(df.describe())

print('\nCovariance matrix for these data:')
print(df.cov())

print('\nCorrelation matrix for these data:')
print(df.corr())

# This will make the pair plot. Code is taken from the provided example.
sns.set_theme(style='whitegrid', context='notebook')
sns.pairplot(df,height=1.5)
plt.show()

#############################################
##### Analysis of variable correlations #####
#############################################

"""
Allow me to begin by stating that I have a bachelor's degree in Biomedical Engineering and I have worked as an EMT for a year, and so I am very well educated on this
topic. My opinions reflect the current medical advice with respect to heart disease, and I will probably explain this in more detail than necessary.

Within this dataset, the strongest positive predictor for the presence or absence of heart disease is thal, which corresponds to the presence or absence of certain areas 
to the heart muscle which exhibit diminished blood flow. This can occur due to previous heart attacks, which cause irreversible damage to the myocardium.

However, exercise induced angina (eia), EKG ST segment depression (opst), EKG ST segment slope (dests), and number of major vessels colored by fluoroscopy (nmvcf) are also
strong predictors for the presence or absence of heart disease. I struggle to understand the purpose of addressing the problem this way, because many of those
things are just diagnostic criteria for heart disease. A physician would make this diagnosis themselves based on this information, and the purpose of making that
diagnosis is to warn patients about the risk for heart attack and stroke. Heart disease is merely a way for a doctor to say 'You are high risk and you need to change things
about your life if you want to avoid heart attacks and strokes'. 

The strongest negative predictor for the presence or absence of heart disease is mhr, which is the maximum heart rate achieved. I would expect this to be related 
to fitness levels and age, because patients tolerate high heart rates more easily in youth, and physical fitness will also affect heart rate. That is to say,
I believe this variable to be a side effect of physical fitness and not the cause of heart disease in and of itself.

Age: It would appear that blood pressure increases with age and maximum heart rate decreases. According to the correlation values in that matrix, age is a weak 
predictor of heart disease.

Sex: According to this article by Harvard medical school, the prevalence of heart disease is lower in women. Based on that information, I will assume that +1 in this column
means the patient is male and 0 means they are female. 
https://www.health.harvard.edu/heart-health/the-heart-disease-gender-gap

Chest Pain Type: Diagnosing heart problems with chest pain is really an acute emergency, and I would recommend going to the ER if you experience sharp, radiating, and 
crushing chest pain. Obviously, if a patient ever uses those adjectives I just mentioned, that is cause for serious concern and an EKG at minimum. I would assume that 
higher numbers are closer to those adjectives I mentioned, because those are indications of a medical emergency that requires immediate attention.

Resting blood pressure: Higher resting blood pressure has been proven to increase the risk of heart attacks, this is already well established. That correlation value 
seems way too low, in my opinion. The mechanism through which this happens is that higher blood pressures place more strain on vessels, which can lead
to aneurysms in the abdominal aorta or blood clots occluding the coronary arteries (the definition of a heart attack).

Serum cholesterol: Yes, cholesterol levels are correlated with the probability of heart attacks. A higher serum cholesterol should predict heart disease pretty well.
I would expect this variable to be highly correlated with diet and sodium intake, where sodium intake and the lack of fruit and vegetables will drastically increase
the risk of adverse outcomes. 

Fasting blood sugar: This measure is more closely tied with diabetes, and not with heart disease. However, there is certainly an overlap in Americans that have both
Type II diabetes and heart disease simultaneously. I would expect a higher fasting blood sugar to imply the patient has become insensitive to insulin, which is the
definition of Type II diabetes. I am surprised this correlation is negative, that doesn't seem quite right to me.

Resting EKG results: Well yes, this is the definition of how you diagnose heart attacks. Certain observations in an EKG mean that someone is having a heart attack
right now, and the EKG can also tell a physician information about how healthy the heart is. Obviously, since the correlation is positive, that means that higher
values correspond to concerning observations in the EKG, for example ST segment elevation.

Maximum heart rate achieved: If the patient is unable to reach a very high heart rate, that would imply some kind of damage to the myocardium. This is correlated with
activity levels and age, and I would expect those variables to also be strong predictors of heart attack probability. 

Exercise induced angina: Yes, angina means chest pain, and I would be very concerned about this if my patient said they have exercise induced angina. The specific adjectives
they choose to describe their pain are very important though, and I am concerned this format of looking at the data is incomplete.

ST segment depression: I am not familiar with this terminology, but I do know that ST segment elevation is an indication of an acute myocardial infarction, which means
they are having a heart attack right now. I will take your word for it, observing this is associated positively with heart disease.

ST segment slope: Fast changes in the EKG are not necessarily bad, but perhaps this situation is different. I would recommend asking a doctor to interpret your EKG results, 
and this way of looking at things seems overly simplistic.

Number of major vessels colored by fluoroscopy: Well, the positive correlation here is surprising. I would expect a greater number of vessels illuminated by 
fluoroscopy to indicate that most of the heart is being perfused properly... But also, if someone has any of those major vessels occluded, that would constitute
a medical emergency, in which case why are we wasting time taking this data?

Thallium: This is my first time hearing about this test, but it sounds like they are using thallium to illuminate where metabolism is taking place in the heart.
A dim region would indicate that metabolism is not happening as much as surrounding regions, which would indicate some type of damage. Again, that is literally
the definition of heart disease. There is no 'prediction' happening here, we're done.
"""