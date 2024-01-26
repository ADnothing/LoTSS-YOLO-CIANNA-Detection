#!/usr/bin/env python

#Adrien Anthore, 26 jan 2023
#Env: Python 3.6.13
#corr_cat.py

import numpy as np

from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord

import os
from tqdm import tqdm


def update_progress(progress):
	"""
	Function to update the progress bar in the console.
	"""
	bar_length = 100  # Number of characters in the progress bar
	filled_length = int(bar_length * progress)
	bar = '#' * filled_length + '-' * (bar_length - filled_length)
	percentage = int(progress * 100)
	print(f'\rProgress : |{bar}| {percentage}% ', end='')
	
#===============================================================================================================

def Rexcl(flux, P1, R1, P2, R2):
	"""
	Rejection radius function:
	Around bright sources, we often find artifacts due to interferometry.
	To reject those artifacts, we defined a radius that is a function of
	the integrated flux of the source studied. As the extent of the artifacts
	seems to scale very fast with the integrated flux, we chose an exponential
	law which is parameterized by typical integrated flux values "P" (low and high)
	with their expected rejection radius "R". The parameters P and R are
	chosen from guesses by analyzing the fields studied.

	Args:
		flux (float): Integrated flux value for which the rejection radius is calculated.
		P1 (float): Typical low integrated flux (sources with no artifacts).
		R1 (float): Typical radius of exclusion for the low integrated flux.
		P2 (float): Typical high integrated flux (sources with lot of artifacts).
		R2 (float): Typical radius of exclusion for high integrated flux sources.

	Returns:
		float: Rejection radius calculated based on the input parameters.
	"""

	c = (R2-R1*np.exp(P2-P1))/(1-np.exp(P2-P1))
	k = (R1-c)*np.exp(-P1)
	
	return k*np.exp(np.log10(flux)) + c
	
#===============================================================================================================
	
def clean_cat(name_file, res, R1, P1, R2, P2, survey):
	"""
	Clean the catalog generated with crea_dendrogram to suppress multiple detections
	as well as supposedly false detections.
	This process results in the writing of 2 files:
		- The catalog cleaned of a typical field (overwriting the input file).
		- The "To Test Sources Catalog" (TTSC) containing all rejected sources that could be True detections.
		
	The name of the TTSC is: TTSC_{survey}.txt

	Args:
		name_file (str): File path of the catalog generated with crea_dendrogram.
		res (float): Resolution of the instrument.
		R1 (float): Parameter R1 for rejection radius calculation.
		P1 (float): Parameter P1 for rejection radius calculation.
		R2 (float): Parameter R2 for rejection radius calculation.
		P2 (float): Parameter P2 for rejection radius calculation.
		survey (str): Name of the survey.

	Returns:
		None.

	Information about the number of sources cleaned and sources to test are printed at the end.
	"""
	excl = 0 #Number of exclusion (is displayed for the user)
	
	data = np.loadtxt(name_file, skiprows=1)
	data = data[np.argsort(-data[:,2])]
	
	i = 0 #progress indice
	excl_array = [] #excluded sources
	while i<data.shape[0]-1:

		flux = data[i,2]
		R = Rexcl(flux, P1, R1, P2, R2)
			
		c1 = SkyCoord(data[i,0]*u.deg, data[i,1]*u.deg, frame='icrs')
		c2 = SkyCoord(data[i+1:,0]*u.deg, data[i+1:,1]*u.deg, frame='icrs')
		sep_array = c1.separation(c2)
			
		#Here is the exclusion condition,
		#A source to be excluded must :
		#Be closer than the resolution to the current source
		#or be inside the rejection radius with a flux lower than 100 mJy
		mask1 = sep_array.arcsecond<res
		mask2 = np.logical_and(data[i+1:,2]<1e2,sep_array.arcsecond < R)
		excl_ind = np.where(np.logical_or(mask2, mask1))[0] + i + 1
			
			
		for ind in sorted(excl_ind, reverse=True):
				
			excl_source = data[ind, :]
			excl_array.append(excl_source)
				
			data = np.delete(data, ind, 0)
			excl+=1
		
		i+=1
		
		progress = (i + 1) / (data.shape[0]-1)
		update_progress(progress)
	
	
		
	data_pos = SkyCoord(data[:,0]*u.deg, data[:,1]*u.deg, frame='icrs')
	totest=0
	for source in excl_array:
				
		excl_pos = SkyCoord(source[0]*u.deg, source[1]*u.deg, frame='icrs')
		sep_array = excl_pos.separation(data_pos)
				
		if source[2]>1 and not(any(sep_array.arcsecond<res)):
			TTSC = open("TTSC_"+survey+".txt", 'a')
			np.savetxt(TTSC, [source], delimiter='\t')
			TTSC.close()
			totest+=1
	
	final_data = data[data[:,2]>2e-1]
	excl += len(data)-len(final_data)
	
	TTSC = open("TTSC_"+survey+".txt", 'a')
	np.savetxt(TTSC, data[data[:,2]<2e-1], delimiter='\t')
	TTSC.close()
	totest += len(data)-len(final_data)
	
	f = open(name_file, 'w')
	if final_data.shape[1] == 6:
		np.savetxt(f, np.hstack((final_data,np.zeros((final_data.shape[0],1)))), fmt=['%.6f', '%.6f', '%.6e', '%.3f', '%.3f', '%.1f', '%i'], header="_RAJ2000\t_DECJ2000\tSpeakTot\tMaj\tMin\tPA\tflag\n", delimiter='\t', comments='')
	else:
		np.savetxt(f, final_data, fmt=['%.6f', '%.6f', '%.6e', '%.3f', '%.3f', '%.1f', '%i'], header="_RAJ2000\t_DECJ2000\tSpeakTot\tMaj\tMin\tPA\tflag\n", delimiter='\t', comments='')
	f.close()
	
	print("\nNumber of exclusions: %d"%excl)
	print("%d sources to test"%totest)
	
#===============================================================================================================

def check_overlap(file1, file2):
	"""
	Check if two FITS files have spatial overlap.

	Args:
		file1 (str): File path of the first FITS file.
		file2 (str): File path of the second FITS file.

	Returns:
		bool: True if the two files have spatial overlap, False otherwise.

	This function checks whether two FITS files have spatial overlap. It computes
	the diagonal distance of the images and compares it with the separation between
	their central coordinates. If the separation is less than or equal to the diagonal
	distance, it indicates that the images overlap.
	"""

	hdul1 = fits.open(file1)
	hdul2 = fits.open(file2)

	side1 = np.max(hdul1[0].data.shape)*hdul1[0].header["CDELT2"]
	d1 = np.sqrt(2)*side1/2
	
	side2 = np.max(hdul2[0].data.shape)*hdul2[0].header["CDELT2"]
	d2 = np.sqrt(2)*side2/2

	ra1 = hdul1[0].header["CRVAL1"]
	dec1 = hdul1[0].header["CRVAL2"]
	c1 = SkyCoord(ra1, dec1, unit='deg',frame='icrs')


	ra2 = hdul2[0].header["CRVAL1"]
	dec2 = hdul2[0].header["CRVAL2"]
	c2 = SkyCoord(ra2, dec2, unit='deg',frame='icrs')

	sep = c1.separation(c2).deg

	if sep<=d1+d2:
		return True
	else:
		return False
		
#===============================================================================================================
		
def get_overlap_sources(cat, field_fits):
	"""
	Extract sources from a catalog that overlap and do not overlap with a field.

	Args:
		cat (numpy.ndarray): Input catalog containing source positions.
		field_fits (str): File path of the FITS file representing the field.

	Returns:
		tuple: Two numpy arrays - the first containing sources that overlap with the field,
			and the second containing sources that do not overlap.

	This function extracts sources from a catalog based on whether they overlap with
	a specified field represented by a FITS file. It calculates the diagonal distance
	of the field and compares it with the separation between the center of the field
	and the positions of sources in the catalog.
	"""
	
	field_hdul = fits.open(field_fits)
	
	side = np.max(field_hdul[0].data.shape)*field_hdul[0].header["CDELT2"]
	d = np.sqrt(2)*side/2
	
	Cra = field_hdul[0].header["CRVAL1"]
	Cdec = field_hdul[0].header["CRVAL2"]
	center = SkyCoord(Cra, Cdec, unit='deg',frame='icrs')
	
	Sra = cat[:, 0]
	Sdec = cat[:, 1]
	Spos = SkyCoord(Sra, Sdec, unit='deg',frame='icrs')
	
	sep = center.separation(Spos).deg
	
	mask_inferior_d = sep <= d
	mask_residual = ~mask_inferior_d
	
	cat_overlaped = cat[mask_inferior_d]
	cat_residual = cat[mask_residual]
	
	return cat_overlaped, cat_residual
	
#===============================================================================================================
	
def clean_overlap(data, res, R1, P1, R2, P2, survey):
	"""
	Clean a catalog by removing multiple detections and excluding artifacts
	around bright sources based on spatial overlap.
	This function has the same process as "clean_cat" but is 
	specific for the case when we clean overlapping region.
	The expected data in output is typically the output "cat_overlaped"
	from the function "get_overlap_sources".

	Args:
		data (numpy.ndarray): Input catalog containing source information.
		res (float): Resolution of the instrument.
		R1 (float): Parameter R1 for rejection radius calculation.
		P1 (float): Parameter P1 for rejection radius calculation.
		R2 (float): Parameter R2 for rejection radius calculation.
		P2 (float): Parameter P2 for rejection radius calculation.
		survey (str): Name of the survey.

	Returns:
		numpy.ndarray: Cleaned catalog containing sources that pass the cleaning criteria.
	"""
	
	excl = 0 #Number of exclusion (is displayed for the user)
	
	data = data[np.argsort(-data[:,2])]
	
	i = 0 #progress indice
	excl_array = [] #excluded sources
	while i<data.shape[0]-1:

		flux = data[i,2]
		R = Rexcl(flux, P1, R1, P2, R2)
			
		c1 = SkyCoord(data[i,0]*u.deg, data[i,1]*u.deg, frame='icrs')
		c2 = SkyCoord(data[i+1:,0]*u.deg, data[i+1:,1]*u.deg, frame='icrs')
		sep_array = c1.separation(c2)
			
		#Here is the exclusion condition,
		#A source to be excluded must :
		#Be closer than 2 arcsecond to the current source
		#or be inside the rejection radius with a flux lower than 100 mJy
		mask1 = sep_array.arcsecond<res
		mask2 = np.logical_and(data[i+1:,2]<1e2,sep_array.arcsecond < R)
		excl_ind = np.where(np.logical_or(mask2, mask1))[0] + i + 1
			
			
		for ind in sorted(excl_ind, reverse=True):
				
			excl_source = data[ind, :]
			excl_array.append(excl_source)
				
			data = np.delete(data, ind, 0)
			excl+=1
		
		i+=1
		
		progress = (i + 1) / (data.shape[0]-1)
		update_progress(progress)
	
	
		
	data_pos = SkyCoord(data[:,0]*u.deg, data[:,1]*u.deg, frame='icrs')
	totest=0
	for source in excl_array:
				
		excl_pos = SkyCoord(source[0]*u.deg, source[1]*u.deg, frame='icrs')
		sep_array = excl_pos.separation(data_pos)
				
		if source[2]>1 and not(any(sep_array.arcsecond<res)):
			TTSC = open("TTSC_"+survey+".txt", 'a')
			np.savetxt(TTSC, [source], delimiter='\t')
			TTSC.close()
			totest+=1
			
			
	
	final_data = data[data[:,2]>2e-1]
	excl += len(data)-len(final_data)
	
	TTSC = open("TTSC_"+survey+".txt", 'a')
	np.savetxt(TTSC, data[data[:,2]<2e-1], delimiter='\t')
	TTSC.close()
	totest += len(data)-len(final_data)
		
			
	print("\nNumber of exclusions: %d"%excl)
	print("%d sources to test"%totest)
			
		
	if final_data.shape[1] == 6:
		return np.hstack((final_data,np.zeros((final_data.shape[0],1))))
	else:
		return final_data
		
#===============================================================================================================
		
def flux_NMS(cat, reject):
	"""
	Perform a Non-Maximum Suppression (NMS) process on a catalog.
	The parameter taken into account for the suppresion is the integrated flux of the sources.

	Args:
		cat (numpy.ndarray): Input catalog containing source information.
		reject (float): Separation threshold for exclusion.

	Returns:
		numpy.ndarray: Catalog after the Third NMS process.
	"""

	excl = 0

	cat = cat[np.argsort(-cat[:,2])]
	i = 0
	print("Third NMS process...")
	while i<cat.shape[0]-1:

		c1 = SkyCoord(cat[i,0]*u.deg, cat[i,1]*u.deg, frame='icrs')
		c2 = SkyCoord(cat[i+1:,0]*u.deg, cat[i+1:,1]*u.deg, frame='icrs')

		sep_array = c1.separation(c2)

		excl_ind = np.where(sep_array.arcsecond < reject)[0] + i + 1

		for ind in sorted(excl_ind, reverse=True):
			cat = np.delete(cat, ind, 0)
			excl+=1

		i+=1

		progress = (i + 1) / (cat.shape[0]-1)
		update_progress(progress)

	print("\nNumber of exclusions : %d\n"%excl)

	return cat
