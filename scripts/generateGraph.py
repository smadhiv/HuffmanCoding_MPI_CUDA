resourceUsed = []
architectureName = []
runTime = dict()
avgTime = dict()
finalAvgTime = dict()

def getHeaderInformation(name):
  print ("processing file " + file)
  with open(file, 'r') as fileText:
    text = fileText.readlines()
    text = [i.strip() for i in text if i.strip() != '']
    architectureName.extend(text[0].split()[-1:])
    resourceUsed.extend(text[1].split()[-1:])
    compressionFileSize = str(text[2].split()[-1:])
    return text, compressionFileSize

def calculateRunTimeAvg(text, compressionFileSize):
  tempTime = []
  for index in list(range(3, len(text))):
    tempList = text[index].split()
    if tempList[0] == 'FileSize:':
      runTime[compressionFileSize] = tempTime
      avgTime[compressionFileSize] = round(sum(tempTime)/len(tempTime), 3)
      compressionFileSize = str(tempList[1])
      tempTime = []
    elif tempList[0] == 'Time':
      tempTime.append(float(str(tempList[2]).replace(":", ".")))
    else:
      break
  runTime[compressionFileSize] = tempTime
  avgTime[compressionFileSize] = round(sum(tempTime)/len(tempTime), 3)
  compressionFileSize = str(tempList[1])
  finalAvgTime[str(text[0].split()[-1:])] = avgTime

def processFile(name):
  text, compressionFileSize = getHeaderInformation(name)
  calculateRunTimeAvg(text, compressionFileSize)
  
import os
os.chdir('./test/')
fileList = os.listdir('.')
for file in fileList:
  processFile(file)
print (architectureName)
print (finalAvgTime)

    
  


