# This program performs huffman compression using CUDA and MPI. Pure GPU, MPI as well  as pure serial versions are also incuded.
# All executables should have two arguments inputFile and outputFile.
# MPI-CUDA, CUDA as well as MPI implementations scales linearly and only limitation is  the  maximum file size that the file system supports.
# For decompression, use same number of MPI processes as the one used for compression. For pure serial version and pure GPU version use deompress under Serial folder. For MPI and MPICUDA version use the  MPIDecompress in MPI folder for decompression.

