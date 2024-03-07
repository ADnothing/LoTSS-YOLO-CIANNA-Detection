print("\t"*3,"Testing the current environment\n")
print("="*80)

#Python check
print("\t"*3,"Python")
import sys
py_v = sys.version.split(" ")[0]
if not(int(py_v.split(".")[1]) >= 6 and int(py_v.split(".")[1]) <= 9):
	print("WARNING")
	print(f"You are using Python {py_v}")
	print("We recomand using Python 3.6.7 or an equivalent.")
else:
	print("Good version of Python")
print("="*80)

#NumPy check
print("\t"*3,"NumPy")
try:
	import numpy
	print("NumPy checked!")
	if numpy.__version__ != '1.19.2':
		print("WARNING")
		print(f"You are using NumPy {numpy.__version__}")
		print("Make sure it is compatible with NumPy 1.19.2")
except:
	print("Failed to import NumPy!")
print("="*80)

#Astropy check
print("\t"*3,"Astropy")
try:
	import astropy
	print("astropy checked!")
	if astropy.__version__ != '4.1':
		print("WARNING")
		print(f"You are using Astropy {astropy.__version__}")
		print("Make sure it is compatible with Astropy 4.1")
except:
	print("Failed to import Astropy!")
print("="*80)

#astrodendro check
print("\t"*3,"astrodendro")
try:
	import astrodendro
	print("astrodendro checked!")
	if astrodendro.__version__ != '0.2.0':
		print("WARNING")
		print(f"You are using astrodendro {astrodendro.__version__}")
		print("Make sure it is compatible with astrodendro 0.2.0")
except:
	print("Failed to import astrodendro!")
print("="*80)

#Numba check
print("\t"*3,"Numba")
try:
	import numba
	print("numba checked!")
	if numba.__version__ != '0.53.1':
		print("WARNING")
		print(f"You are using Numba {numba.__version__}")
		print("Make sure it is compatible with Numba 0.53.1")
except:
	print("Failed to import Numba!")
print("="*80)

#astroquery check
print("\t"*3,"astroquery")
try:
	import astroquery
	print("astroquery checked!")
	if astroquery.__version__ != '0.4.6':
		print("WARNING")
		print(f"You are using astroquery {astroquery.__version__}")
		print("Make sure it is compatible with astroquery 0.4.6")
except:
	print("Failed to import astroquery!")
print("="*80)

#tqdm check
print("\t"*3,"tqdm")
try:
	import tqdm
	print("tqdm checked!")
	if tqdm.__version__ != '4.62.3':
		print("WARNING")
		print(f"You are using tqdm {tqdm.__version__}")
		print("Make sure it is compatible with tqdm 4.62.3")
except:
	print("Failed to import tqdm!")
print("="*80)

print("\n","\t"*3,"Test terminated!")
