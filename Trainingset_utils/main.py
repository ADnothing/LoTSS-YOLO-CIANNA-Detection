#!/usr/bin/env python

#Adrien Anthore, 09 Apr 2024
#main.py
#main python file
#call: python3 -W "ignore" main.py [-c] <config file>
#If "-c" is specified, will start from a previous run.


import numpy as np

import os, configparser, sys, time, argparse
from multiprocessing import Pool
import concurrent.futures

from tqdm import tqdm

from make_cat import crea_dendrogram
from corr_cat import third_NMS

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

	flag = 1
	f1 = path + fits_file

	with open("done.txt", "r") as done_file:
		if f1 in done_file.read():
			flag = 0

	if f1[-5:] != ".fits":
		flag = 0

	if flag:
		crea_dendrogram(f1, [res, sensi, R1, P1, R2, P2])

		with open("done.txt", "a") as done_file:
			done_file.write(f1 + "\n")

def overlap(fits_file):

	flag = 1
	temp_list = [f2 for f2 in list_fits if f2 != fits_file]
	
	with open("NMSed.txt", "r") as done_file:
		if fits_file in done_file.read():
			flag = 0
	
	if fits_file[-5:] != ".fits":
		flag = 0
		
	if flag:
		third_NMS(fits_file, temp_list, path, res)

		with open("NMSed.txt", "a") as done_file:
			done_file.write(fits_file + "\n")
	

def initialisation():

	global c
	args = parse_arguments()
	config_file_path = args.config_file
	c = vars(args)["continue"]
	
	if not(os.path.exists("./dendrocat/")):
		os.makedirs("./dendrocat/")
		
	if not(os.path.exists("./Catalogs/")):
		os.makedirs("./Catalogs/")
		
	if not(os.path.exists("./raw/")):
		os.makedirs("./raw/")
	
	if not c:
		done = open("done.txt", "w")
		done.close()
		NMSed = open("NMSed.txt", "w")
		NMSed.close()
	
		for f in os.listdir("./dendrocat/"):
			os.remove("./dendrocat/" + f)
	else:
		if not(os.path.exists("./done.txt")):
			print("No dendrocat saved.")
			done = open("done.txt", "w")
			done.close()
			
		if not(os.path.exists("./NMSed.txt")):
			print("No NMS saved.")
			done = open("NMSed.txt", "w")
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
	
	global list_fits
	list_fits = os.listdir(path)



if __name__ == "__main__":

	initialisation()

	print("Detection:")
	with Pool(processes=10) as pool:
		pool.map(run, list_fits)
	"""for fi in list_fits:
		print(fi)
		crea_dendrogram(path+fi, [res, sensi, R1, P1, R2, P2])"""
		
	print("Overlap management:")
	with Pool(processes=10) as pool:
		pool.map(overlap, list_fits)
	
	print("END")
