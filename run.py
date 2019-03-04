import os
import sys

logfile = './test.txt'
mode = 'F'
batchsize = 256
test_bs = 256
depth = 2
runtime = 10

cmd = 'python3 train.py --batch-size %d --test-batch-size %d --mode %s --depth %d --log %s' % (batchsize,test_bs,mode,depth,logfile)
os.system(cmd)
