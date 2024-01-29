#!/usr/bin/env python

#Adrien Anthore, 29 Jan 2024
#main.py
#main python file


import numpy as np

import os
import configparser
import sys

from tqdm import tqdm

from astropy.io import fits

from make_cat import crea_dendrogram
from corr_cat import *

if not os.path.exists("./dendrocat"):
	os.makedirs("./dendrocat")

def read_config_file(file_path):
	config = configparser.ConfigParser()
	config.read(file_path)
	return config['Parameters']
	
if len(sys.argv) != 2:
	print("Usage: python3 main.py <config_file_path>")
	sys.exit(1)

config_file_path = sys.argv[1]
parameters = read_config_file(config_file_path)

survey = parameters.get('SURVEY')
path = parameters.get('PATH')
res = float(parameters.get('RESOLUTION'))
sensi = float(parameters.get('SENSITIVITY'))
R1 = float(parameters.get('R1'))
P1 = float(parameters.get('P1'))
R2 = float(parameters.get('R2'))
P2 = float(parameters.get('P2'))


if __name__ == "__main__":

	nbsou = 0
	f = open("full_"+survey+"_DencdroCat.txt", "w")
	f.write("_RAJ2000\t_DECJ2000\tSpeakTot\tMaj\tMin\tPA\tflag\n")

	list_fits = os.listdir(path)

	print("Detection phase:")
	for fits_file in list_fits:
		print("File:", fits_file)
		crea_dendrogram(path+fits_file, sensi)

	print("Cleaning phase:")
	TTSC = open("TTSC_"+survey+".txt", 'w')
	TTSC.close()
	n = len(list_fits)
	
	list_cat = []
	
	for ind, f1 in enumerate(list_fits):
		print("\nStep:\t%d/%d"%(ind+1, n))
		print("Curent:",f1)
		flag=False
		temp_list = list_fits[:ind] + list_fits[ind+1:] 
		cat1 = "./dendrocat/"+f1.split("/")[-1][:-5]+"_DendCat.txt"
		
		list_cat.append(cat1)
		
		clean_cat(cat1, res, R1, P1, R2, P2, survey)
		ctot = np.loadtxt(cat1, skiprows=1)

		print("Check overlaps...")
		for f2 in temp_list:
			fits1 = path + f1
			fits2 = path + f2

			cat2 = "./dendrocat/"+f2.split("/")[-1][:-5]+"_DendCat.txt"

			if check_overlap(fits1,fits2):
				print(f2,"overlaps!")
				flag=True
				c2 = np.loadtxt(cat2, skiprows=1)
				overlap_c2, residual_cat = get_overlap_sources(c2, fits1)
				
				print("Cleaning overlapping region:")
				if list_fits.index(f2) > ind:
					overlap_c2 = clean_overlap(overlap_c2, res, R1, P1, R2, P2, survey)
				
				ctot = np.vstack((ctot, overlap_c2))
				
				
				f_temp = open(cat2, "w")
				if list_fits.index(f2) > ind:
					np.savetxt(f_temp, residual_cat, fmt=['%.6f', '%.6f', '%.6e', '%.3f', '%.3f', '%.1f'], header="_RAJ2000\t_DECJ2000\tSpeakTot\tMaj\tMin\tPA\n" , delimiter='\t', comments='')
				else:
					np.savetxt(f_temp, residual_cat, fmt=['%.6f', '%.6f', '%.6e', '%.3f', '%.3f', '%.1f', '%i'], header="_RAJ2000\t_DECJ2000\tSpeakTot\tMaj\tMin\tPA\tflag\n" , delimiter='\t', comments='')
				f_temp.close()
				
		if flag:
			ctot = flux_NMS(ctot, res)
			
		f_temp = open(cat1, "w")
		np.savetxt(f_temp, ctot, fmt=['%.6f', '%.6f', '%.6e', '%.3f', '%.3f', '%.1f', '%i'], delimiter='\t')
		f_temp.close()

	for cat_name in tqdm(list_cat):
		cat = np.loadtxt(cat_name, skiprows=1)
		nbsou += cat.shape[0]
		np.savetxt(f, cat, fmt=['%.6f', '%.6f', '%.6e', '%.3f', '%.3f', '%.1f', '%i'], delimiter='\t')


	f.close()
	
print("END")
print(nbsou, "sources catalogued")
