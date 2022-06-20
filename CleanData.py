#!/usr/bin/env python3

import csv
import pandas as pd
import re
import json

def main():
    howToUse()
    fileName = getFileName()
    cleanFile(fileName)
    congrats()

def howToUse():
    print("*** Begin Cleaning! ***\n")
    
def getFileName():
    name = "TweetData.csv"
    print("*** Getting File ***\n")
    return name
    
def cleanFile(fileName):
    newOutName = "CLEAN_" + fileName
    print("*** Cleaning File ***\n")
    with open(fileName) as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
    
        for row in csvReader:
            outputFile = open(newOutName, "a")
            #writer = csv.writer(outputFile, escapechar=' ', quoting=csv.QUOTE_NONE)
            writer = csv.writer(outputFile)
              
            #strip quotes and single quotes from all
            count = 0
            for field in row:
                tempField = str(row[count]).lower()
                
                if(count == 16):
                	tempField = re.sub("\d", "", tempField)
                	tempField = tempField.encode("ascii", "ignore").decode()
                	tempField = re.sub("http\S+", "", tempField)
                	#print(tempField)
                	
                tempField = tempField.replace('@','')
                tempField = tempField.replace("&amp",'')
                
                tempField = negation(tempField)
                
                if(count == 2 or count == 3):
                	tempField = tempField.encode("ascii", "ignore").decode()
                	tempField = tempField
                tempField = tempField.replace(":",'')
                tempField = tempField.replace("\n","")
                
                if(count == 16):
                	tempField = iterateField(tempField)
                	
                row[count] = tempField
                count = count + 1
            
            writer.writerow(row)
            outputFile.close()
            
def negation(field):
	field = field.replace("won't",'will not')
	field = field.replace("isn't",'is not')
	field = field.replace("aren't",'are not')
	field = field.replace("can't",'can not')
	field = field.replace("couldn't",'could not')
	field = field.replace("didn't",'did not')
	field = field.replace("won't",'will not')
	field = field.replace("doesn't",'does not')
	field = field.replace("don't",'do not')
	field = field.replace("hadn't",'had not')
	field = field.replace("hasn't",'has not')
	field = field.replace("won't",'will not')
	field = field.replace("haven't",'have not')
	field = field.replace("shouldn't",'should not')
	field = field.replace("weren't",'were not')
	field = field.replace("wouldn't",'would not')
	field = field.replace("would've", 'would have')
	return field

def removeRepeats(word):
	newString = ''.join([j for i,j in enumerate(word) if j not in word[:i]])
	return newString

def iterateField(field):
	newField = ""
	for word in field.split():
		replacement = slangReplace(word)
		
		if(replacement != 0):
			newField = newField + replacement + " "
		else:
			newField = newField + word + " "
	
	#print(newField)
	return newField

def slangReplace(word):
	f = open('myslang.json')
	data = json.load(f)
	lowerWord = str(word).lower()

	try:	
		replacement = data[lowerWord]
	except KeyError:
		replacement = 0
	
	isInt = isinstance(replacement, int)
	if(isInt is False and "-or-" in replacement):
		replacement = grabFirstOption(replacement)
		
	return replacement
	
def grabFirstOption(replacement):
	isInt = isinstance(replacement, int)
	word = "-or-"
	first = ""
	if(isInt is False):
		if(word in replacement):
			first = replacement.split("-")[0]
	return first

def congrats():
    print("*** Dance Time ***\n")
    print("Your Dataset should be clean now!")
    print("")
if __name__ == "__main__":
    main()
