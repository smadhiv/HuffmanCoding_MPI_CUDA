resourceUsed = []
architectureName = []
runTime = dict()
avgTime = dict()

import os
os.chdir('./test/')
fileList = os.listdir('.')
for file in fileList:
  print ("processing file " + file)
  with open(file, 'r') as fileText:
    text = fileText.readlines()
    text = [i.strip() for i in text if i.strip() != '']
    architectureName.extend(text[0].split()[-1:])
    resourceUsed.extend(text[1].split()[-1:])
    compressionFileSize = str(text[2].split()[-1:])
    tempTime = []
    for index in list(range(3, len(text))):
      tempList = text[index].split()
      if tempList[0] == 'FileSize:':
        runTime[compressionFileSize] = tempTime
        avgTime[compressionFileSize] = round(sum(tempTime)/len(tempTime), 3)
        compressionFileSize = str(tempList[1])
        tempTime = []
      elif tempList[0] == 'Time':
        tempTime.append(float(tempList[2][:-4] + '.' + tempList[2][-3:]))
      else:
        break
    runTime[compressionFileSize] = tempTime
    avgTime[compressionFileSize] = round(sum(tempTime)/len(tempTime), 3)
    compressionFileSize = str(tempList[1])
    print (architectureName)
    print (avgTime)

    
  


