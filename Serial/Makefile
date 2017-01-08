all: 
	gcc -o ../bin/compress compress.c ../include/serialFunctions.c
	gcc -o ../bin/decompress decompress.c ../include/serialFunctions.c

clean:
	if [ -a ../bin/compress ]; then rm -f ../bin/compress; fi;
	if [ -a ../bin/decompress ]; then rm -f ../bin/decompress; fi;

