#!/usr/bin/env python

import sys
import struct

fileA = open( sys.argv[1], 'r' )
fileB = open( sys.argv[2], 'r' )

numDiff = 0
maxDiff = 0.0
bufSize = 1024

fileA.read(56)
fileB.read(56)

while True:
  
  strfA = fileA.read(bufSize)
  if strfA == "":
    break
  
  strfB = fileB.read(bufSize)
  if len(strfA) != len(strfB):
    print 'The files have different sizes!'
    sys.exit(1)
        
  count = len(strfA) / 4
  bufA = struct.unpack( '%uf' % count, strfA )
  bufB = struct.unpack( '%uf' % count, strfB )
  
  for valA, valB in zip(bufA, bufB):   
      
    if valA != valB:
      numDiff += 1
      maxDiff = max( maxDiff, abs( valA - valB ) )
      
if fileB.read(1) != "":
  print 'File A is shorter than file B!'

print 'Differences: %u' % numDiff
print 'Max difference: %f' % maxDiff

fileA.close()
fileB.close()
