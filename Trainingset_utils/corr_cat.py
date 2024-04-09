#!/usr/bin/env python

#Adrien Anthore, 09 Apr 2024
#Env: Python 3.6.7
#corr_cat.py

import numpy as np

from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord

import os
from tqdm import tqdm

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
		P2 (float): Typical high integrated flux (sources with a lot of artifacts).
		R2 (float): Typical radius of exclusion for high integrated flux sources.

	Returns:
		float: Rejection radius calculated based on the input parameters.
	"""

	c = (R2-R1*np.exp(P2-P1))/(1-np.exp(P2-P1))
	k = (R1-c)*np.exp(-P1)
	
	return k*np.exp(np.log10(flux)) + c
	
#===============================================================================================================
	
def clean_cat(data, min_val, res, R1, P1, R2, P2):
	"""
	Clean the catalog by removing overlapping sources, detected artifacts and sources with low significance.
	All the remove sources are returned and to be tested with IR/optical counter parts.

	Args:
		data (numpy.ndarray): Array containing catalog data, with columns representing:
			- Column 0: Right Ascension (RA) in degrees
			- Column 1: Declination (DEC) in degrees
			- Column 2: Flux of the source
			- Column 3: Surface flux density
			- Column 4: Major axis of the source
			- Column 5: Minor axis of the source
			- Column 6: Position angle of the source
		min_val (float): Minimum pixel value for detection.
		res (float): Resolution of the instrument.
		R1 (float): Parameter for cleaning the catalog.
		P1 (float): Parameter for cleaning the catalog.
		R2 (float): Parameter for cleaning the catalog.
		P2 (float): Parameter for cleaning the catalog.

	Returns:
		final_data (numpy.ndarray): Cleaned catalog data containing sources.
		final_ttsc (numpy.ndarray): Excluded sources to be match with optical/IR counter parts.
	"""
	data = data[np.argsort(-data[:,2])]
	
	i = 0 #progress indice
	excl_array = [] #excluded sources
	while i<data.shape[0]-1:

		flux = data[i,2]
		R = Rexcl(flux, P1, R1, P2, R2)
			
		c1 = SkyCoord(data[i,0]*u.deg, data[i,1]*u.deg, frame='icrs')
		c2 = SkyCoord(data[i+1:,0]*u.deg, data[i+1:,1]*u.deg, frame='icrs')
		sep_array = c1.separation(c2)
			
		
		mask1 = sep_array.arcsecond<res
		mask2 = np.logical_and(data[i+1:,2] < 0.05*flux, sep_array.arcsecond < R)
		excl_ind = np.where(np.logical_or(mask2, mask1))[0] + i + 1
			
		for ind in sorted(excl_ind, reverse=True):
				
			excl_source = data[ind, :]
			excl_array.append(excl_source)
				
			data = np.delete(data, ind, 0)
		
		i+=1
		
	data_pos = SkyCoord(data[:,0]*u.deg, data[:,1]*u.deg, frame='icrs')
	
	ttsc = []
	for source in excl_array:
				
		excl_pos = SkyCoord(source[0]*u.deg, source[1]*u.deg, frame='icrs')
		sep_array = excl_pos.separation(data_pos)
				
		
		if not(any(sep_array.arcsecond<res)):
			ttsc.append(source)
		
	ttsc = np.array(ttsc)
	
	S_threshold = 10*(min_val*1e3)
	
	mask = np.logical_and(data[:, 3] > 1.2e-2, data[:, 2] > S_threshold)

	final_data = data[mask]
	if len(ttsc)>0:
		final_ttsc = np.vstack([ttsc, data[np.logical_not(mask)]])
	else:
		final_ttsc = data[np.logical_not(mask)]
		
	
	return final_data, final_ttsc
	
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
	d1 = side1/2
	
	side2 = np.max(hdul2[0].data.shape)*hdul2[0].header["CDELT2"]
	d2 = side2/2

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
			and the second contains sources that do not overlap.

	This function extracts sources from a catalog based on whether they overlap with
	a specified field represented by a FITS file. It calculates the diagonal distance
	of the field and compares it with the separation between the center of the field
	and the positions of sources in the catalog.
	"""
	
	field_hdul = fits.open(field_fits)
	
	side = np.max(field_hdul[0].data.shape)*field_hdul[0].header["CDELT2"]
	d = 1.1*(side/2)
	
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
		
def third_NMS(f1, list_fits, path, reject, col=7):
	"""
	Apply a Non-Max Suppression (NMS) on catalogs from images that have an overlap.
	The method use for the suppression is a Nearest Neigbourg suppression.
	
	Args:
		f1 (str): File name of the first FITS file.
		list_fits (list): List of file names of FITS files to compare with.
		path (str): Path to the FITS files.
		reject (float): Threshold distance for rejecting redundant sources, in arcseconds.
		col (int, optional): Index of the column in the catalog containing the confidence score. Default is 7.

	The resulting catalog from the image in the fits f1 is saved as a text file in the 'Catalogs' directory.
	"""
	
	fits1 = path + f1
	cat1 = "./dendrocat/"+f1[:-5]+"_DendCat.txt"
	c1 = np.loadtxt(cat1, comments='#')
		
	for f2 in tqdm(list_fits):
		
		fits2 = path + f2
		cat2 = "./dendrocat/"+f2[:-5]+"_DendCat.txt"
			
		if check_overlap(fits1,fits2):
			
			overlap_c1, residual_c1 = get_overlap_sources(c1, fits2)
			
			c2 = np.loadtxt(cat2, comments='#')
			overlap_c2, residual_c2 = get_overlap_sources(c2, fits1)
					
			excl_ind = []
				
			for idx, source in enumerate(overlap_c1):
					
				pos = SkyCoord(source[0]*u.deg, source[1]*u.deg, frame='icrs')
				pos_test = SkyCoord(overlap_c2[:,0]*u.deg, overlap_c2[:,1]*u.deg, frame='icrs')
					
				sep_array = pos.separation(pos_test)
					
				test_ind = np.where(sep_array.arcsecond < reject)[0]
					
				if any(source[col] >= overlap_c2[test_ind,col]):
					excl_ind.append(idx)
						
			for ind in sorted(excl_ind, reverse=True):
				overlap_c1 = np.delete(overlap_c1, ind, 0)
						
					
			c1 = np.vstack((residual_c1, overlap_c1))
			
		
	f = open("./Catalogs/"+f1[:-5]+"_DendCat.txt", "w")
	np.savetxt(f, c1, fmt=['%.6f', '%.6f', '%.6e', '%.6e', '%.3f', '%.3f', '%.1f', '%.3e', '%i'], delimiter='\t', header="_RAJ2000\t_DECJ2000\tSpeakTot\tSdensity\tMaj\tMin\tPA\tRMS\tflag")
	f.close()

#===============================================================================================================
		
def clean_redundancy(cat, reject, col=7):
	"""
	Remove redundant sources from a catalog based on a separation threshold.

	Args:
		cat (numpy.ndarray): Input catalog containing source positions and other information.
		reject (float): Threshold distance for rejecting redundant sources, in arcseconds.
		col (int, optional): Index of the column use to sort the catalog. Default is 7.

	Returns:
		numpy.ndarray: Catalog with redundant sources removed.
	"""
	
	cat = cat[np.argsort(-cat[:,col])]
		
	i = 0
	while i<cat.shape[0]-1:

		c1 = SkyCoord(cat[i,0]*u.deg, cat[i,1]*u.deg, frame='icrs')
		c2 = SkyCoord(cat[i+1:,0]*u.deg, cat[i+1:,1]*u.deg, frame='icrs')

		sep_array = c1.separation(c2)

		excl_ind = np.where(sep_array.arcsecond < reject)[0] + i + 1

		for ind in sorted(excl_ind, reverse=True):
			cat = np.delete(cat, ind, 0)
			
		i+=1
		
	return cat
