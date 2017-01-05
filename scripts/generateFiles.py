import os
#import matplotlib.pyplot as plt
import random

# variables
KB = 1024
MB = KB * 1024
GB = MB * 1024

with open('64MB', 'wb') as fout:
    fout.write(os.urandom(64 * MB))
print "64"

with open('128MB', 'wb') as fout:
    fout.write(os.urandom(128 * MB))
print "128"

with open('256MB', 'wb') as fout:
    fout.write(os.urandom(256 * MB))
print "256"

with open('512MB', 'wb') as fout:
    fout.write(os.urandom(512 * MB))
print "512"
	
with open('768MB', 'wb') as fout:
    fout.write(os.urandom(786 * MB))
print "786"

with open('1024MB', 'wb') as fout:
    fout.write(os.urandom(1024 * MB))
print "1024"

with open('1280MB', 'wb') as fout:
    fout.write(os.urandom(1280 * MB))
print "1280"

with open('1536MB', 'wb') as fout:
    fout.write(os.urandom(1536 * MB))
print "1536"
	
with open('1792MB', 'wb') as fout:
    fout.write(os.urandom(1792 * MB))
print "1792"

with open('2048MB', 'wb') as fout:
    fout.write(os.urandom(2048 * MB))
print "2048"

with open('2304MB', 'wb') as fout:
    fout.write(os.urandom(2304 * MB))
print "2304"

