import csv
import glob
imgdir='waldo_original'
paths = glob.glob('{}/*.jpg'.format(imgdir))
anns = '{}/annotations.csv'.format(imgdir)


f = open(anns,'r')



