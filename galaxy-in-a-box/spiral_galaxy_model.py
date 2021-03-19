import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib
import math
import os
import sys
from sympy import exp
from astropy.io import fits
from matplotlib.colors import LogNorm
import copy
import csv
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from timer import Timer # Here we call the 'timer.py' script added in the folder
import Galaxy_clusters as cd
import Galaxycluster_emission as ce

### Calling the imported timer function to control the time of calculations
#   Can be commented (if you do so, consider commenting it also in the preamble)
t = Timer()
t.start()

### Predefining plot settings
params={
	'axes.labelsize'	   : 24,
	'axes.linewidth'	   : 1.5,
	'lines.markeredgewidth': 1.5,
	'font.size'			   : 24,
	'legend.fontsize'	   : 24,
	'xtick.labelsize'	   : 24,
	'ytick.labelsize'	   : 24,
	'xtick.major.size'	   : 12,
	'xtick.minor.size'	   : 8,
	'ytick.major.size'	   : 12,
	'ytick.minor.size'	   : 8,
	'savefig.dpi'		   : 300,
	'font.family'		   : 'serif',
	'font.serif'		   : 'Times',
	'text.usetex'		   : True,
	'xtick.direction'      : 'out',     # direction: in, out, or inout
	'xtick.minor.visible'  : True,   	# visibility of minor ticks on x-axis
	'ytick.direction'      : 'out',     # direction: in, out, or inout
	'ytick.minor.visible'  : True,    	# visibility of minor ticks on x-axis
	'xtick.top'			   : True,
	'ytick.right'		   : True,
}
plt.rcParams.update(params)


######################################################
### Creating galaxy spatial and mass distributions ###
######################################################

def spatial_mass_dist(
	NC,                                # Number of clusters
	mmin,                              # Min. cluster mass
	mmax):                             # Max. cluster mass

	spatial_setup = {}
	for line in open("spatial_setup.dat","r").readlines():
		spatial_setup[line.split()[0]]=float(line.split()[1])

	alpha = spatial_setup['alpha']
	A1    = spatial_setup['A1']
	B1    = spatial_setup['B1']
	N1    = spatial_setup['N1']
	A2    = spatial_setup['A2']
	B2    = spatial_setup['B2']
	N2    = spatial_setup['N2']
	S2    = spatial_setup['S2']
	h0    = spatial_setup['h0']

	### Mass distribution ###
	MassDist  = []
	massrange = np.arange(mmin,mmax,1)
	MassDist  = massrange**(alpha)

	MassDistribution = (MassDist/max(MassDist))*mmax
	mass = []
	while len(mass) < (NC):
		MassDist = np.random.choice(MassDistribution,size = 1)
		if MassDist > mmin:
			mass.append(MassDist)
	new_MassDistribution = [[i[0]] for i in mass]


	### Initial parameters for spatial distribution ###
	# Ringermacher & Mead 2009
	mu = 0                             # centered on function spiral value
	Nd = int(NC)                       # disk cluster number
	Nb = int(NC*(1/6)) 	               # boulge cluster number
	l  = 10 				           # number of stars per point value of function

	beta   = 1.0/h0

	### Creating exponential distribution ###
	R    = np.arange(0,10,0.0001)
	rexp = beta*np.exp(-(R*beta))
	m    = np.random.choice(rexp,int(Nd/(2))) # radially follows an exponential distribution - randomly chosen

	### Generating spiral arm shape given parameters ###
	no  = m*2.8*math.pi/(max(m)) #values of phi with designated highest phi value
	Rad = rexp*2.8*math.pi/(max(rexp))
	n   = no[no<(1.85*math.pi)]

	X = []
	Y = []

	for phi in n:
		r     = A1/(math.log(B1*math.tan((phi)/(2*N1)))) #actual values for spiral arm randomly chosen
		sigma = 1/(2+0.5*phi) #spread from spiral value following normal distribution
		rx    = np.random.normal(mu,sigma,1)
		ry    = np.random.normal(mu,sigma,1)
		x     = r*math.cos(phi)+rx
		y     = r*math.sin(phi)+ry
		X.append(x)
		Y.append(y)

	X.extend(np.negative(X)[::-1])
	Y.extend(np.negative(Y)[::-1])
	X = (np.array(np.array(X))).flatten()
	Y = (np.array(np.array(Y))).flatten()
	X = X[::-1]
	new_Y = [[i] for i in Y]
	new_X = [[i] for i in X]
	SpatialArray = np.append(new_X,new_Y,1)

	u  = no[no>(1.85*math.pi)]
	X1 = []
	Y1 = []

	for phi in u:
		r     = A2/(math.log(B2*math.tan((phi)/(2*N2)))) + S2
		sigma = 1/(2+0.5*phi) #spread from spiral value following normal distribution
		rx    = np.random.normal(mu,sigma,1)
		ry    = np.random.normal(mu,sigma,1)
		x     = r*math.cos(phi)+rx
		y     = r*math.sin(phi)+ry
		X1.append(x)
		Y1.append(y)

	X1.extend(np.negative(X1)[::-1])
	Y1.extend(np.negative(Y1)[::-1])
	X1 = (np.array(np.array(X1))).flatten()
	Y1 = (np.array(np.array(Y1))).flatten()
	X1 = X1[::-1]
	new_Y1 = [[i] for i in Y1]
	new_X1 = [[i] for i in X1]
	SpatialArray1 = np.append(new_X1,new_Y1,1)

	### Combining Spatial and Mass Distributions into one array ###
	SpatialArray     = np.append(SpatialArray, SpatialArray1, 0)
	SpatialMassArray = np.append(SpatialArray, new_MassDistribution,1)
	SpatialX         = SpatialMassArray[:, 0]
	SpatialY         = SpatialMassArray[:, 1]
	Mass             = (SpatialMassArray[:, 2]).flatten()


	return Mass, SpatialX, SpatialY

mass, X, Y = spatial_mass_dist(NC = 1e4, mmin = 1e4, mmax = 1e6)

filename = 'galaxycluster_emission.csv'
if os.path.exists(filename):
	os.remove(filename)

cluster_setup = {}
for line in open("cluster_setup.dat","r").readlines():
	cluster_setup[line.split()[0]]=float(line.split()[1])

# For iterations uncomment these lines (and comment the ones that are currently active, besides 'Mcm')
# SFE = [,] # Here provide all of the values which you want to run the model over
# IMF = [,] # Here provide all of the values which you want to run the model over
# tff = [,] # Here provide all of the values which you want to run the model over

SFE     = [cluster_setup['SFE']]
IMF     = [cluster_setup['imf']]
tff     = [cluster_setup['tff']]

Mcm     = cluster_setup['Mcm']
Mcm_gal = np.array(mass)


for n in tff:
	for j in IMF:
		for s in SFE:
			for i in Mcm_gal:
				f = open("cluster_setup.dat")
				fout = open("cluster_setup_change.dat", "wt")
				for line in f:
					fout.write(line.replace(str(Mcm), str(i)))
					for line in f:
						fout.write(line.replace(str(SFE), str(s)))
						for line in f:
							fout.write(line.replace(str(IMF),str(j)))
							for line in f:
								fout.write(line.replace(str(tff),str(n)))
				f.close()
				fout.close()
				outputfile = "gal_emission"+"_tff="+str(n)+"_imf="+str(j)+"_SFE="+str(s)+".csv"
				newname    = "distributions_Mcm="+str(i)+"_tff="+str(n)+"_imf="+str(j)+"_SFE="+str(s)+".npy"
				distribution = cd.Mod_distribution()
				distribution.calc(output = 0)
				os.rename("distribution.npy",newname)
				g = open("image_setup.dat")
				gout = open("image_setup_change.dat", "wt")
				for line in g:
					gout.write(line.replace('distribution.npy', newname))
				g.close()
				gout.close()
				template = ce.Mod_Template()
				template.main(output = 0,FILE = outputfile)
				try:
					os.remove(newname)
				except OSError:
					pass


			im = []
			mass = []
			N = []

			with open(outputfile, 'r') as file:
				reader = csv.reader(file, delimiter='\t')
				for row in reader:
					im.append(float(row[0]))
					mass.append(float(row[1]))
					N.append(float(row[2]))

			mass = [[i] for i in mass]
			ims  = [[i] for i in im]
			N    = [[i] for i in N]

			comb  = np.append(ims,mass,1)
			comb  = np.append(comb,N,1)
			comb1 = []

			while (len(comb1)+len(comb)) < 10000:
				comb1.extend(comb)

			for i in comb:
				comb1.extend([i])
				if len(comb1) == 10000:
					break

			#print(np.amax(X))
			t.stop()

			dims = (1401,1401) #total grid dimensions i.e. galaxy size in pixels each pixel is 17.4pc
			Galaxyarray = np.zeros(dims)
			for i in range(0,len(X)):
				R = (comb1[i][1]/(np.pi*144))**(1/2)

				if 2*R > 17.4:
					dim  = int(2*R/17.4)
					d    = comb1[i][0]/(dim**2)
					data = np.zeros((dim,dim))
					data.fill(d)
				else:
					d    = comb1[i][0]
					data = np.zeros((3,3))
					data[1,1] = d

				x = X[i]*56
				y = Y[i]*56

				Galaxyarray[int((x+(dims[0]-len(data))/2)):int((x+(dims[0]+len(data))/2)),int((y+(dims[1]-len(data))/2)):int((y+(dims[1]+len(data))/2))]+=data

			config = {}
			f = open('image_setup_change.dat','r')
			for line in f.readlines():
				config[line.split()[0]]=line.split()[1]

			# Parameters relating to new image
			dist       = float(config['bob'])   # distance to cluster in pc
			pixel_size = float(config['psize']) # pixel size in arcsec
			resolution = float(config['beam'])  # resolution of new image
			dim_pix    = int(config['dim'])     # image size in pixels
			npix_beam  = 2.*np.pi*(resolution/2./(2.*np.log(2.))**0.5)**2 / pixel_size**2   # number of pixels per beam

			beam = Gaussian2DKernel(resolution/pixel_size/(2.*(2.*np.log(2.))**0.5))
			im_obs = convolve(Galaxyarray, beam, boundary='extend')/npix_beam

			im_obs[im_obs==0] = 1e-100
			rnge = pixel_size*dim_pix/2.        # 1. arcmin -> kpc: Comment this one

			# Comment the the lines below until 'hdu.writeto' if you're not interested in producing a fits file
			half_im = dim_pix / 2
			header = fits.Header()
			header['BMAJ'] = resolution / 3600.
			header['BMIN'] = resolution / 3600.
			header['BPA'] = 0.0
			header['BTYPE'] = 'Intensity'
			header['BUNIT'] = 'JY KM/S /BEAM '
			header['EQUINOX'] = 2.000000000000E+03
			header['CTYPE1'] = 'RA---SIN'
			header['CRVAL1'] = 0.0
			header['CDELT1'] =  pixel_size/3600.
			header['CRPIX1'] =  half_im
			header['CUNIT1'] = 'deg     '
			header['CTYPE2'] = 'DEC--SIN'
			header['CRVAL2'] = 0.0
			header['CDELT2'] =  pixel_size/3600.
			header['CRPIX2'] =  half_im
			header['CUNIT2'] = 'deg     '
			header['RESTFRQ'] =   9.879267000000E+11
			header['SPECSYS'] = 'LSRK    '
			hdu = fits.PrimaryHDU(im_obs, header=header)
			hdu.writeto("Gal_Template"+"_tff="+str(n)+"_imf="+str(j)+"_SFE="+str(s)+".fits", overwrite = True)

			print ("Peak intensity in image is %6.5f Jy km/s/beam" %(im_obs.max()))

			### Image plotting ###
			plt.figure(figsize=[15,12])
			my_cmap = copy.copy(matplotlib.cm.get_cmap('bone')) # copy the default cmap
			my_cmap.set_bad([0,0,0])

			plt.imshow(
			im_obs,
			interpolation='nearest',
			cmap = my_cmap,
			norm = LogNorm(vmin=1e-8, vmax=np.amax(im_obs)),
			extent=(rnge,-rnge,-rnge,rnge)      # 2. arcsec -> kpc: Comment the extent
			)

			cbar = plt.colorbar()
			cbar.set_label('Jy km s$^{-1}$ beam$^{-1}$')
			plt.xlabel('Offset (arcmin)')
			plt.ylabel('Offset (arcmin)')
			#plt.xticks(np.arange(0,1401,175))  # 4. arcmin -> kpc: Uncomment this one
			#plt.yticks(np.arange(0,1401,175))  # 5. arcmin -> kpc: Uncomment this one
			#labels=[12,9,6,3,0,-3,-6,-9,-12]   # 6. arcmin -> kpc: Uncomment this one
			#ax.set_xticklabels(labels)         # 7. arcmin -> kpc: Uncomment this one
			#ax.set_yticklabels(labels)		    # 8. arcmin -> kpc: Uncomment this one
			plt.tight_layout(pad=0.1)
			plt.savefig("Gal_Template"+"_tff="+str(n)+"_imf="+str(j)+"_SFE="+str(s)+".pdf",bbox_inches='tight')
