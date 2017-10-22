from pandas import DataFrame
import json
from pprint import pprint
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Read in training data
with open('classified/classified_preprints.json') as json_data:
    preprints = json.load(json_data)

with open('classified/classified_share_data.json') as json_data:
    sharedata = json.load(json_data)

# Extract relevant features
rows = []
testTitles = []
testSubjects = []

'''
# Use only first category
for work in preprints[:400]:
    rows.append({'title': work["abstract"], 'subject': work["subjects"][0]})
    testTitles.append(work['title'])
    testSubjects.append(work['subjects'][0])

for work in sharedata[:400]:
    rows.append({'title': work["abstract"], 'subject': work["subjects"][0]})
    testTitles.append(work['title'])
    testSubjects.append(work['subjects'][0])
'''

# Use all categories
for work in preprints[:200]:
    for subject in work['subjects']:
        rows.append({'title': work["abstract"], 'subject': subject})
    testTitles.append(work['title'])
    testSubjects.append(work['subjects'])

for work in sharedata[:200]:
    for subject in work['subjects']:
        rows.append({'title': work["abstract"], 'subject': subject})
    testTitles.append(work['title'])
    testSubjects.append(work['subjects'])

# Use Panda to transform into a dataframe
df = DataFrame(rows)

# Train Classifier
cv = CountVectorizer(ngram_range=(1, 2))

counts = cv.fit_transform(df['title'].values)

classifier = MultinomialNB()
targets = df['subject'].values
classifier.fit(counts, targets)


# Read in data in need of classification
with open('unclassified_projects.json') as json_data:
    un = json.load(json_data)

# Format unclassified data
unclassifiedRows = []
for work in un:
    unclassifiedRows.append(work['title'] + ' ' + work['description'])

unclassifiedCounts = cv.transform(unclassifiedRows)

f = open('output.txt', 'w')
f.truncate()
# Make predictions
predictions = classifier.predict(unclassifiedCounts)
for i in range(1, len(unclassifiedRows)):
    f.write(un[i]['title'] + ': ' + predictions[i] + '\n')

'''
# Predict the subject of the titles we inputted
testCounts = cv.transform(testTitles)
testPrediction = classifier.predict(testCounts)
'''

'''
#Test only one subject
print(numpy.mean(testPrediction == testSubjects))

'''

'''
# Test when running with multiple inputs
count = 0.0
for i in range(len(testPrediction)):
    if testPrediction[i] in testSubjects[i]:
        count = count + 1

print(count / len(testPrediction))
'''