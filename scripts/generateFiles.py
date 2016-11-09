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
	
# # objects
# axis = [0] * 256
# freq = [0] * 256

# # fill axis
# for i in xrange(0, 256):
	# axis[i] = i

# randomly generate and save files
#for i in range(0, 128 * MB):
	# num = random.randint(0, 255)
	# freq[num] = freq[num] + 1

	# conditional writes
	# if i < 64 * MB:
		# fout_64MB.write(chr(random.randint(0, 255)))
	# if i < 128 * MB:
		# fout_128MB.write(chr(random.randint(0, 255)))
	# if i < 256 * MB:
		# fout_256MB.write(chr(random.randint(0, 255)))
	# if i < 512 * MB:
		# fout_512MB.write(chr(random.randint(0, 255)))
	# if i < 768 * MB:
		# fout_768MB.write(chr(random.randint(0, 255)))
	# if i < 1024 * MB:
		# fout_1024MB.write(chr(random.randint(0, 255)))
	# if i < 1280 * MB:
		# fout_1280MB.write(chr(random.randint(0, 255)))
	# if i < 1536 * MB:
		# fout_1536MB.write(chr(random.randint(0, 255)))
	# if i < 1792 * MB:
		# fout_1792MB.write(chr(random.randint(0, 255)))
	# if i < 2048 * MB:
		# fout_2048MB.write(chr(random.randint(0, 255)))

# plot
# plt.bar(axis, freq)
# plt.axis([0, 255, 0, 5000])
# plt.show()
