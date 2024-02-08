#!/usr/bin/env python

#Adrien Anthore, 08 Feb 2024
#main.py
#main python file


import numpy as np

import os, configparser, sys

from tqdm import tqdm

from astropy.io import fits
from astroquery.xmatch import XMatch
from astroquery.vizier import Vizier
from astropy.table import Table, vstack
import astropy.units as u
Vizier.ROW_LIMIT = -1

from make_cat import crea_dendrogram
from corr_cat import *
from CrossMatch import NN_Xmatch

if os.path.exists("./dendrocat/"):
	for f in os.listdir("./dendrocat/"):
		os.remove("./dendrocat/"+f)
else:
	os.makedirs("./dendrocat/")
	
if os.path.exists("./TTSC/"):
	for f in os.listdir("./TTSC/"):
		os.remove("./TTSC/"+f)
else:
	os.makedirs("./TTSC/")

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

	print("Detection:")
	for fits_file in list_fits:
		print("File:", fits_file)
		crea_dendrogram(path+fits_file, sensi)
		print("Cleaning:")
		cat = "./dendrocat/"+fits_file[:-5]+"_DendCat.txt"
		clean_cat(cat, res, R1, P1, R2, P2)

	print("Check overlaps:")
	n = len(list_fits)
	list_cat = []
	for ind, f1 in enumerate(list_fits):
		print("\nStep:\t%d/%d"%(ind+1, n))
		print("Curent:",f1)
		flag=False
		temp_list = list_fits[:ind] + list_fits[ind+1:] 
		cat1 = "./dendrocat/"+f1[:-5]+"_DendCat.txt"
		
		list_cat.append(cat1)
		
		ctot = np.loadtxt(cat1, skiprows=1)

		print("Check overlaps...")
		for f2 in temp_list:
			fits1 = path + f1
			fits2 = path + f2

			cat2 = "./dendrocat/"+f2[:-5]+"_DendCat.txt"

			if check_overlap(fits1,fits2):
				print(f2,"overlaps!")
				flag=True
				c2 = np.loadtxt(cat2, skiprows=1)
				overlap_c2, residual_cat = get_overlap_sources(c2, fits1)
				
				ctot = np.vstack((ctot, overlap_c2))
				
				f_temp = open(cat2, "w")
				np.savetxt(f_temp, residual_cat, fmt=['%.6f', '%.6f', '%.6e', '%.3f', '%.3f', '%.1f', '%i'], header="_RAJ2000\t_DECJ2000\tSpeakTot\tMaj\tMin\tPA\tflag\n" , delimiter='\t', comments='')
				f_temp.close()
				
		if flag:
			ctot = flux_NMS(ctot, res)
			
		f_temp = open(cat1, "w")
		np.savetxt(f_temp, ctot, fmt=['%.6f', '%.6f', '%.6e', '%.3f', '%.3f', '%.1f', '%i'], delimiter='\t')
		f_temp.close()

	print("Overlap NMS:")
	for cat_name in tqdm(list_cat):
		cat = np.loadtxt(cat_name, skiprows=1)
		nbsou += cat.shape[0]
		np.savetxt(f, cat, fmt=['%.6f', '%.6f', '%.6e', '%.3f', '%.3f', '%.1f', '%i'], delimiter='\t')


	f.close()
	
	print("Re-injection process:")
	list_TTSC = []
	for ind, f1 in enumerate(list_fits):
	
		flag=False
		temp_list = list_fits[:ind] + list_fits[ind+1:]
		
		ttsc1 = "./TTSC/TTSC_"+f1[:-5]+".txt" 
		
		list_TTSC.append(ttsc1)
		
		ttsctot = np.loadtxt(ttsc1)
		for f2 in temp_list:
			fits1 = path + f1
			fits2 = path + f2

			ttsc2 = "./TTSC/TTSC_"+f2[:-5]+".txt"

			if check_overlap(fits1,fits2):

				flag=True
				ttsc_temp = np.loadtxt(ttsc2, skiprows=1)
				overlap_ttsc, residual_cat = get_overlap_sources(ttsc_temp, fits1)
		
				ttsctot = np.vstack((ttsctot, overlap_ttsc))
				
				f_temp = open(ttsc2, "w")
				np.savetxt(f_temp, residual_cat)
				f_temp.close()
			
		if flag:
			ttsctot = flux_NMS(ttsctot, res)
				
		f_temp = open(ttsc1, "w")
		np.savetxt(f_temp, ttsctot)
		f_temp.close()

	reinjected=0
	for f in list_fits:
		print(f)
		ttsc = "./TTSC/TTSC_"+f[:-5]+".txt"
		ttsc = np.loadtxt(ttsc)
		
		input_table = Table()
		input_table["ra"] = ttsc[:,0]
		input_table["dec"] = ttsc[:,1]
		input_table["flux"] = ttsc[:,2]
		input_table["maj"] = ttsc[:,3]
		input_table["min"] = ttsc[:,4]
		input_table["pa"] = ttsc[:,5]
				
		hdul = fits.open(path+f)
		side = np.max(hdul[0].data.shape)*hdul[0].header["CDELT2"]
		r = np.sqrt(2)*side/2
				
		RA = hdul[0].header["CRVAL1"]
		DEC = hdul[0].header["CRVAL2"]
				
		DESI = Vizier.query_region(SkyCoord(ra=RA, dec=DEC, unit='deg', frame='icrs'), radius=r*u.deg, catalog="VII/292/north")[0]
			
		mask_ra = np.logical_and(DESI['RAJ2000'] >= input_table['ra'].min(), DESI['RAJ2000'] <= input_table['ra'].max())
		mask_dec = np.logical_and(DESI['DEJ2000'] >= input_table['dec'].min(), DESI['DEJ2000'] <= input_table['dec'].max())
		DESI = DESI[np.logical_and(mask_ra, mask_dec)]
			
		table_low = input_table[input_table['flux'] < 5]
		table_high = input_table[input_table['flux'] >= 5]
					
		high_X_DESI, _ = NN_Xmatch(table_high, DESI, 6*u.arcsec, "ra", "dec", 'RAJ2000', 'DEJ2000', sort_test_by="flux")
		high_X_Allwise = XMatch.query(cat1=table_high, cat2='vizier:II/328/allwise', max_distance=6*u.arcsec, colRA1='ra', colDec1='dec', colRA2='RAJ2000', colDec2='DEJ2000')
		
		l_DESI = len(DESI)
		DESI_X_Allwise = XMatch.query(cat1=DESI[:int(l_DESI/5)], cat2='vizier:II/328/allwise', max_distance=1*u.arcsec, colRA1='RAJ2000', colDec1='DEJ2000', colRA2='RAJ2000', colDec2='DEJ2000')
		for i in range(4):
			temp = XMatch.query(cat1=DESI[int(l_DESI*(i+1)/5):int(l_DESI*(i+2)/5)], cat2='vizier:II/328/allwise', max_distance=1*u.arcsec, colRA1='RAJ2000', colDec1='DEJ2000', colRA2='RAJ2000', colDec2='DEJ2000')
			DESI_X_Allwise = vstack([DESI_X_Allwise, temp])
					
		low_X, _ = NN_Xmatch(table_low, DESI_X_Allwise, 3*u.arcsec, "ra", "dec", 'RAJ2000_1', 'DEJ2000_1', sort_test_by="flux")
					
		high1 = np.array([high_X_DESI["ra"], high_X_DESI["dec"], high_X_DESI["flux"], high_X_DESI["maj"], high_X_DESI["min"], high_X_DESI["pa"]]).T
		high2 = np.array([high_X_Allwise["ra"], high_X_Allwise["dec"], high_X_Allwise["flux"], high_X_Allwise["maj"], high_X_Allwise["min"], high_X_Allwise["pa"]]).T
		low = np.array([low_X["ra"], low_X["dec"], low_X["flux"], low_X["maj"], low_X["min"], low_X["pa"]]).T

		high = np.vstack((high1, high2))
		high = np.hstack((high, np.ones((high.shape[0],1))))
		low = np.hstack((low, 2*np.ones((low.shape[0],1))))
		matched = np.vstack((high, low))
				
		matched = flux_NMS(matched, 6)
					
		nbsou+=len(matched)
		reinjected+=len(matched)
					
		f = open("full_LoTSS_DencdroCat.txt", "a")
		np.savetxt(f, matched, fmt=['%.6f', '%.6f', '%.6e', '%.3f', '%.3f', '%.1f', '%i'],  delimiter='\t')
		f.close()
		
	print(reinjected,"sources have been reinjected by cross matching.")
	print("END")
	print(nbsou, "sources catalogued")
