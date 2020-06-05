# Limpy
This is a python package for Line Intensity Mapping code for python. 

#Module()s
lcp.py: luminosity for Cplus lines. This module can do following things:
	i) read files files of halo list and SFR rate
	ii) Calculates halo mass to SFR rate following mean and/or scatter relation. 
	iii) Calculate halo mass to Cplus luminosity (halomass---> SFR rate --> CII luminosity)
	ii) It has some in build plotting modules. 
	


1) Install the package. 

i) Download the package.
ii) Do "cd Limpy"
iii) Then install it by command "python setup.py install"
iv) Next, add the PYTHONPATH path of Limpy to your .bashrc. 

	You can do something like this:
	"vi .bashrc"
	export PYTHONPATH="${PYTHONPATH}:/Users/anirbanroy/Documents/Limpy/"
	
You are all set. 

2) Run the code
you can see some examples I executed through a Jupyter notebook script. 

	i) cd "PATH TO LIMPY/notebook/
	ii) jupyter-notebook test.ipynb
	
you can run now...
