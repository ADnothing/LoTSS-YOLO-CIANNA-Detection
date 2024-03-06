#!/usr/bin/env python

#Adrien Anthore, 06 Mar 2024
#main.py
#main python file
#call: python3 -W "ignore" main.py [-c] <config file>
#If "-c" is specified, will start from a previous run.


import numpy as np

import os, configparser, sys, time, argparse
from multiprocessing import Pool

from tqdm import tqdm

from make_cat import crea_dendrogram
from corr_cat import check_overlap, get_overlap_sources, third_NMS

c = False

def parse_arguments():
	parser = argparse.ArgumentParser(description='Create catalog of detection with AstroDendro package.')
	parser.add_argument('-c', '--continue', action='store_true', help='Continue your progression, will read done.txt')
	parser.add_argument('config_file', metavar='config_file_path', type=str, help='Path to config file')
	
	return parser.parse_args()

def read_config_file(file_path):
	config = configparser.ConfigParser()
	config.read(file_path)
	return config['Parameters']

def run(fits_file):
	global c
	flag = 1
	fits_file = path + fits_file

	with open("done.txt", "r") as done_file:
		if fits_file in done_file.read():
			flag = 0

	if fits_file[-5:] != ".fits":
		flag = 0

	if flag:
		crea_dendrogram(fits_file, [res, sensi, R1, P1, R2, P2])

	with open("done.txt", "a") as done_file:
		done_file.write(fits_file + "\n")

def initialisation():
	global c
	args = parse_arguments()
	config_file_path = args.config_file
	c = vars(args)["continue"]
	
	if not(os.path.exists("./dendrocat/")):
		os.makedirs("./dendrocat/")
	
	if not c:
		done = open("done.txt", "w")
		done.close()
	
		for f in os.listdir("./dendrocat/"):
			os.remove("./dendrocat/" + f)
	else:
		if not(os.path.exists("./done.txt")):
			print("No progress saved.")
			done = open("done.txt", "w")
			done.close()
		
	parameters = read_config_file(config_file_path)
	global survey, path, res, sensi, R1, P1, R2, P2
	survey = parameters.get('SURVEY')
	path = parameters.get('PATH')
	res = float(parameters.get('RESOLUTION'))
	sensi = float(parameters.get('SENSITIVITY'))
	R1 = float(parameters.get('R1'))
	P1 = float(parameters.get('P1'))
	R2 = float(parameters.get('R2'))
	P2 = float(parameters.get('P2'))



if __name__ == "__main__":

	initialisation()

	nbsou = 0
	f = open("full_"+survey+"_DencdroCat.txt", "w")
	f.write("#_RAJ2000\t_DECJ2000\tSpeakTot\tSdensity\tMaj\tMin\tPA\tflag\n")

	list_fits = os.listdir(path)

	print("Detection:")
	with Pool(processes=12) as pool:
		pool.map(run, list_fits)
	"""for fi in list_fits:
		print(fi)
		crea_dendrogram(path+fi, [res, sensi, R1, P1, R2, P2])"""
		
	print("Overlap management:")
	list_cat = []
	for ind, f1 in enumerate(list_fits):
		flag=False
		temp_list = list_fits[:ind] + list_fits[ind+1:] 
		cat1 = "./dendrocat/"+f1[:-5]+"_DendCat.txt"
		
		list_cat.append(cat1)
		
		ctot = np.loadtxt(cat1, comments='#')

		for f2 in temp_list:
			fits1 = path + f1
			fits2 = path + f2

			cat2 = "./dendrocat/"+f2[:-5]+"_DendCat.txt"

			if check_overlap(fits1,fits2):
				flag=True
				c2 = np.loadtxt(cat2, comments='#')
				overlap_c2, residual_cat = get_overlap_sources(c2, fits1)
				
				ctot = np.vstack((ctot, overlap_c2))
				
				f_temp = open(cat2, "w")
				np.savetxt(f_temp, residual_cat, fmt=['%.6f', '%.6f', '%.6e', '%.6e', '%.3f', '%.3f', '%.1f', '%i'], header="_RAJ2000\t_DECJ2000\tSpeakTot\tSdensity\tMaj\tMin\tPA\tflag\n" , delimiter='\t', comments='#')
				f_temp.close()
				
		if flag:
			ctot = third_NMS(ctot, res)
			
		f_temp = open(cat1, "w")
		np.savetxt(f_temp, ctot, fmt=['%.6f', '%.6f', '%.6e', '%.6e', '%.3f', '%.3f', '%.1f', '%i'], delimiter='\t')
		f_temp.close()

	for cat_name in tqdm(list_cat):
		cat = np.loadtxt(cat_name, comments='#')
		nbsou += cat.shape[0]
		np.savetxt(f, cat, fmt=['%.6f', '%.6f', '%.6e', '%.6e', '%.3f', '%.3f', '%.1f', '%i'], delimiter='\t')
		
	f.close()
	
	print("END")
	print(nbsou, "sources catalogued")
