#!/usr/bin/env python

#Adrien Anthore, 29 Jan 2024
#Env: Python 3.6.7
#CrossMatch.py

import warnings

import numpy as np

from astropy import units as u

from numba import njit

@njit
def match_coord(cat2cross, refcoord, sep):
	"""
	Perform coordinate matching between two catalogs.

	Args:
		cat2cross (numpy.ndarray): Catalog to cross-match.
		refcoord (numpy.ndarray): Reference catalog for cross-matching.
		sep (float): Separation threshold for matching in radians.

	Returns:
		Tuple[numpy.ndarray, numpy.ndarray]: Matched indices in 'cat2cross' and corresponding indices in 'refcoord'.

	This function performs coordinate matching between two catalogs using the Haversine formula.
	It compares the celestial coordinates of sources in 'cat2cross' with the coordinates in 'refcoord'.
	If the separation between a source in 'cat2cross' and any source in 'refcoord' is within the specified
	separation threshold ('sep'), the indices of the matched sources in 'cat2cross' and 'refcoord' are returned.
	"""
	
	matched_idx = []
	ctp_idx = []
	
	refcoord = np.ascontiguousarray(refcoord)
	
	for i in range(cat2cross.shape[0]):
		if refcoord.shape[0] > 0:
		
			source = cat2cross[i,:]
			
			lon1 = source[0]*np.pi/180.
			lat1 = source[1]*np.pi/180.
			lon2 = refcoord[:,0]*np.pi/180.
			lat2 = refcoord[:,1]*np.pi/180.
			
			sdlon = np.sin(lon2 - lon1)
			cdlon = np.cos(lon2 - lon1)
			
			slat1 = np.sin(lat1)
			slat2 = np.sin(lat2)
			clat1 = np.cos(lat1)
			clat2 = np.cos(lat2)
			
			num1 = clat2*sdlon
			num2 = clat1*slat2 - slat1*clat2*cdlon
			denominator = slat1*slat2 + clat1*clat2*cdlon
			
			separation = np.arctan2(np.hypot(num1, num2), denominator)
			
			idxs = np.argsort(separation)
			separation = separation[idxs]
			
			ind = 0
			for j in idxs:
				if j in ctp_idx:
					ind+=1
			
			if separation[ind] <= sep:
				matched_idx.append(i)
				ctp_idx.append(ind)
				
		else:
			break
				
	return matched_idx, ctp_idx
	
#===============================================================================================================

def NN_Xmatch(cat_test, cat_ref, maxsep, colRA_test, colDec_test, colRA_ref, colDec_ref, sort_test_by=""):
	"""
	Perform nearest-neighbor cross-matching between two catalogs.

	Args:
		cat_test (astropy.table.Table): Catalog to be tested.
		cat_ref (astropy.table.Table): Reference catalog for cross-matching.
		maxsep (Quantity): Maximum separation allowed for matching. (dimension of an angle)
		colRA_test (str): Column name containing Right Ascension in the catalog to be tested.
		colDec_test (str): Column name containing Declination in the catalog to be tested.
		colRA_ref (str): Column name containing Right Ascension in the reference catalog.
		colDec_ref (str): Column name containing Declination in the reference catalog.
		sort_test_by (str, optional): Column name for sorting the catalog to be tested. Defaults to "".

	Returns:
		Tuple[astropy.table.Table, astropy.table.Table]: Matched sub-catalogs from the test and reference catalogs.

	This function performs nearest-neighbor cross-matching between a catalog and a reference catalog.
	The catalogs are assumed to be Astropy Tables, and the matching is done based on Right Ascension (RA) and Declination (Dec).
	The maximum separation allowed for a match is specified by the 'maxsep' parameter.
	"""
	
	RA1 = np.array(cat_test[colRA_test])
	DEC1 = np.array(cat_test[colDec_test])
	
	RA2 = np.array(cat_ref[colRA_ref])
	DEC2 = np.array(cat_ref[colDec_ref])
	
	sep = maxsep.to(u.radian)
	
	if len(sort_test_by)>0:
		try:
			sorting_col = cat_test[sort_test_by]
			RA1 = RA1[np.argsort(-sorting_col)]
			DEC1 = DEC1[np.argsort(-sorting_col)]
		except KeyError:
			warnings.warn(f"WARNING: {sort_test_by} is not a column from tested catalog.\n Sorting canceled.")
			
	test = np.vstack([RA1, DEC1]).T
	ref = np.vstack([RA2, DEC2]).T
	
	test_idx, ref_idx = match_coord(test, ref, sep.value)
	
	return cat_test[test_idx], cat_ref[ref_idx]
