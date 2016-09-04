resourceList = []
runTime = dict()
avgTime = dict()
architectureList = []
numProcsSet = set()
fileSizeSet = set()
Serial = []
CUDA = []
MPI = []
CUDAMPI = []
CUDA_speedup = []
MPI_speedup = []
CUDAMPI_speedup = []
 
def getHeaderInformation(name):
  """get resource and architecture information of  the run"""
  print ("processing file " + file)
  with open(file, 'r') as fileText:
    text = fileText.readlines()
    text = [i.strip() for i in text if i.strip() != '']
    archType = (text[0].split()[1])
    architectureList.append(archType)
    resourceList.append(text[1].split()[1])
    return text, archType

def calculateRunTimeAvg(text, archType):
  """calculate the average run times for MPI and non-MPI runs"""
  flag = 0
  lineNum = 2
  fileSizeSet.add(int(text[lineNum].split()[1][:-2]))
  compressionFileSize = archType + ':' + str(text[lineNum].split()[1]) + ':'
  if str(text[3].split()[0]) == 'MPIPROCS:':
    lineNum += 1
    flag = 1
    numberOfProcesses = str(text[lineNum].split()[1])
  lineNum += 1
  if flag == 0:
    nonMPIRun(text, archType, compressionFileSize)
  else:
    mpiRunTag = compressionFileSize + numberOfProcesses + ':'
    numProcsSet.add(int(numberOfProcesses))
    MPIRun(text, archType, compressionFileSize, mpiRunTag)
    
def nonMPIRun(text, archType, compressionFileSize):
  """calculate the average run times for non-MPI runs"""
  tempTime = []
  for index in list(range(3, len(text))):
    tempList = text[index].split()
    if tempList[0] == 'FileSize:':
      runTime[compressionFileSize] = tempTime
      avgTime[str(compressionFileSize)] = round(sum(tempTime)/len(tempTime), 3)
      fileSizeSet.add(int(tempList[1][:-2]))
      compressionFileSize = archType + ':' + str(tempList[1]) + ':'
      tempTime = []
    elif tempList[0] == 'Time':
      tempTime.append(float(str(tempList[2]).replace(":", ".")))    
    else:
      break
  runTime[compressionFileSize] = tempTime
  avgTime[compressionFileSize] = round(sum(tempTime)/len(tempTime), 3)

  
def MPIRun(text, archType, compressionFileSize, mpiRunTag):
  """calculate the average run times for MPI runs"""
  tempTime = []
  for index in list(range(4, len(text))):
    tempList = text[index].split()
    if tempList[0] == 'MPIPROCS:':
      runTime[mpiRunTag] = tempTime
      avgTime[mpiRunTag] = round(sum(tempTime)/len(tempTime), 3)
      mpiRunTag = compressionFileSize + str(tempList[1]) + ':'
      numProcsSet.add(int(tempList[1]))
      tempTime = []
    elif tempList[0] == 'Time':
      tempTime.append(float(str(tempList[2]).replace(":", ".")))    
    else:
      compressionFileSize = archType + ':' + str(tempList[1]) + ':'
  runTime[mpiRunTag] = tempTime
  avgTime[mpiRunTag] = round(sum(tempTime)/len(tempTime), 3)
 
def getAvgTimeForEach():
  """generate lists of average time from dictionary""" 
  fileSizeList = sorted(fileSizeSet)
  numProcessList = sorted(numProcsSet)
  if 'Serial' in architectureList:
    for size in fileSizeList:
      Serial.append(avgTime['Serial' + ':' + str(size) + 'MB' + ':'])
      
  if 'CUDA' in architectureList:
    for size in fileSizeList:
      CUDA.append(avgTime['CUDA' + ':' + str(size) + 'MB' + ':'])

  if 'CUDAMPI' in architectureList:
    for size in fileSizeList:
      temp = []
      CUDAMPI.append(temp)
      for num in numProcessList:
        temp.append(avgTime['CUDAMPI' + ':' + str(size) + 'MB' + ':' + str(num) + ':'])
        
  if 'MPI' in architectureList:
    for size in fileSizeList:
      temp = []
      MPI.append(temp)
      for num in numProcessList:
        temp.append(avgTime['MPI' + ':' + str(size) + 'MB' + ':' + str(num) + ':'])

def getSpeedup():
  """generate lists of sppedup from run time""" 
  numProcessList = sorted(numProcsSet)
  
  if 'CUDA' in architectureList:
    for num in list(range(0, len(Serial))):
      CUDA_speedup.append(round(Serial[num] / CUDA[num], 3))

  if 'CUDAMPI' in architectureList:
    for num in list(range(0, len(Serial))):
      temp = []
      CUDAMPI_speedup.append(temp)
      for index in list(range(0, len(numProcessList))):
        temp.append(round(Serial[num]/CUDAMPI[num][index], 3))
        
  if 'MPI' in architectureList:
    for num in list(range(0, len(Serial))):
      temp = []
      MPI_speedup.append(temp)
      for index in list(range(0, len(numProcessList))):
        temp.append(round(Serial[num]/MPI[num][index], 3))

    
import os
os.chdir('../logs/')
fileList = os.listdir('.')
for file in fileList:
  text, archType = getHeaderInformation(file)
  calculateRunTimeAvg(text, archType)
getAvgTimeForEach()
getSpeedup()
fileSizeList = sorted(fileSizeSet)
numProcessList = sorted(numProcsSet)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure(1)
plt.title('CUDA and MPI')
plt.grid(True)
plt.xlabel('Processes')
plt.ylabel('Speedup')
for num in list(range(0, len(fileSizeSet))):
  plt.plot(numProcessList, CUDAMPI_speedup[num], marker='o', label=str(fileSizeList[num]) + 'MB')
plt.legend(loc=4)
plt.savefig('CUDAMPI.png', bbox_inches='tight')

plt.figure(2)
plt.title('MPI')
plt.grid(True)
plt.xlabel('Processes')
plt.ylabel('Speedup')
for num in list(range(0, len(fileSizeSet))):
  plt.plot(numProcessList, MPI_speedup[num], marker='o', label=str(fileSizeList[num]) + 'MB')
plt.legend(loc=4)
plt.savefig('MPI.png', bbox_inches='tight')
