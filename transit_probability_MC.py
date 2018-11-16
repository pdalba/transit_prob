"""
Name
----
transit_probability_MC.py

Description
-----------
The purpose of this script is estimate the transit probabilities for a sample of exo-
planets based on several of the physical/orbital parameters. The transit probability is
estimated through a Monte Carlo simulation, where parameter values are drawn from either
normal or skew normal distributions based on the parameter uncertainties. The result is a
probability distribution for the transit probability. 

This script can also load in observing baselines for TESS, and reduce the geometric 
transit probability by a factor of the orbital period divided by the observing baseline.

This script was written in Python v.2.7.10. For more information on running this script,
please see Dalba et al. (2018) on arXiv or in PASP.

Input
-----
You must have the appropriate data (e.g., parameter values and uncertainties) saved in a
separately file before running this code. I have provided an example file for reference.

Output
------
For each exoplanet in your list, the script will create a directory and save plots of 
each parameter's probability distribution. It will also save the transit probability 
distribution for each planet. It will also create a few files containing summary informa-
tion for the whole run. 


Author
------
Paul A. Dalba, PhD
Department of Earth Sciences
UC Riverside 
2018
pdalba -at- ucr -dot- edu
"""
#-----------------------------------------------------------------------------------------
#Import various math, science, and plotting packages.
imprt = 0
while imprt==0:
	import numpy, scipy
	import commands 
	import matplotlib
	import matplotlib.pyplot as plt
	from matplotlib.pyplot import *
	from matplotlib import colors, cm
	from matplotlib.pylab import *
	from matplotlib.font_manager import FontProperties
	from mpl_toolkits.mplot3d import Axes3D
	import os, glob
	import sys
	from scipy import stats, signal
	from scipy import optimize
	from scipy.optimize import curve_fit
	from scipy.optimize import fsolve
	from numpy import *
	import pickle
	imprt = 1 
close('all')
#-----------------------------------------------------------------------------------------


#Load in exoplanet data. See "Processed_NASA_Exoplanet_Archive_Data.txt" as an example.
# This script is currently set up to take in exactly that format of data. 
#-----------------------------------------------------------------------------------------
RV_data_filename = 'Processed_NASA_Exoplanet_Archive_Data.txt'
RV_dtype = [('id',int),('name','S50'),('a',float),('a_err1',float),('a_err2',float),\
	('ecc',float),('ecc_err1',float),('ecc_err2',float),('w',float),('w_err1',float),\
	('w_err2',float),('Rp',float),('Rp_err1',float),('Rp_err2',float),('Rstar',float),\
	('Rstar_err1',float),('Rstar_err2',float),('tranflag',int),('ra',float),\
	('dec',float),('elon',float),('elat',float)]
RV_data = loadtxt(RV_data_filename,dtype=RV_dtype,delimiter=',',skiprows=1)

#Load in orbital period data. Here, this is stored in the same data file as the RV data.
# You can load in the orbit
P = loadtxt('Processed_NASA_Exoplanet_Archive_Data.txt',skiprows=1,usecols=[22],\
	delimiter=',')

#Depending on your initial accessing/processing of the exoplanet data, you may have empty
# data entries. In the case of my example data set, empty values were filled with -999.0
# upon downloading from the NASA Exoplanet Archive. This block simply looped through the
# data and removed any planets with a -999.0 from further analysis. If you have no planets
# to exclude, comment out this block.
n_excluded = 0
del_list = array([])
for i in range(size(RV_data)):
	if -999.0 in RV_data[i]:
		n_excluded += 1
		del_list = append(del_list,i)
		print RV_data['name'][i]
	if RV_data['w'][i] < 0.: 
		RV_data['w'][i] += 360.	
print ''
print str(n_excluded)+' planets were excluded from the full sample'
print ''
RV_data = delete(RV_data,del_list)	
P = delete(P,del_list)	
#-----------------------------------------------------------------------------------------


#Load information about TESS observations. The tau parameter is the length of TESS obser-
# vational baseline. It has the same units as the orbital period above (days, in the case
# of the example) and its order in the file should match the order of exoplanets in the
# RV_data and P arrays (after any exclusions have been made). The example file contains
# additional information about the exoplanets, including the camera number in which it was
# observed in each sector (if any). The single column being read in here (tau) is just the
# total observing time in all sectors in both cycles. 
#-----------------------------------------------------------------------------------------
tau = loadtxt(data_path+'TESS_Sectors_of_Observation.csv',skiprows=1,usecols=[5],\
	delimiter=',')
#-----------------------------------------------------------------------------------------


#Set conversion factors for physical units
#-----------------------------------------------------------------------------------------
Rsun = 6.96e8    #m
RJ   = 6.9911e7  #m
AU   = 1.496e11  #m
#-----------------------------------------------------------------------------------------


#Define the functions that create distributions
#-----------------------------------------------------------------------------------------
#This function takes in the skew normal parameters, the values of the physical parameter
# at the given percentiles, and the given percentiles. It evaluates the CDF of a skew
# normal distribution at the physical parameter values and finds the residual between
# those percentiles and the given percentiles.
def skewnorm_residuals(p,perc_values,perc):
	return perc - scipy.stats.skewnorm.cdf(perc_values,p[0],loc=p[1],scale=p[2])

#This function takes in physical parameter value and uncertainty and returns draws from
# a normal distribution. It also takes in the bounds of the parameter (given its 
# physical meaning) and it will iteratively re-draw values until all draws fall in the 
# acceptable bounds
def norm_draws(loc,scale,n,bounds):
	#Make the initial draw
	draws = random.normal(loc=loc,scale=scale,size=n) 
	#Find the rejects and redraw them
	rejects = where((draws<bounds[0])|(draws>bounds[1]))[0]
	while size(rejects) > 0:
		draws[rejects] = random.normal(loc=loc,scale=scale,size=size(rejects))
		rejects = where((draws<bounds[0])|(draws>bounds[1]))[0]
	return draws

#This function takes in functional parameters for the skew normal distribution and then
# draws from that distribution
def skewnorm_draws(alpha,loc,scale,n,bounds):
	#Make the initial draw
	draws = scipy.stats.skewnorm.rvs(alpha,loc=loc,scale=scale,size=n)
	#Find the rejects and redraw them
	rejects = where((draws<bounds[0])|(draws>bounds[1]))[0]
	while size(rejects) > 0:
		draws[rejects] = scipy.stats.skewnorm.rvs(alpha,loc=loc,scale=scale,\
			size=size(rejects))
		rejects = where((draws<bounds[0])|(draws>bounds[1]))[0]	
	return draws

#This function fits the physical parameter and its asymmetric errors to a skew normal
# distribution. Here, pname is the string name of the physical parameter.
def fit_skew_normal(x,plus,minus,pname,save_path):
	#Guess the functional parameters
	loc = x + 0.5*(plus-minus)
	scale = 0.5*(plus+minus)
	alpha = (plus-minus)/(plus+minus)
	p0 = array([alpha,loc,scale])
	if plus > minus:
		pbounds = ((0.,-inf,0.),(inf,inf,inf),)
	else:
		pbounds = ((-inf,-inf,-inf),(0.,inf,inf),)
	#Now set the percentile values and their percentiles
	perc_values = array([x-minus,x,x+plus])
	perc = array([0.159,0.5,0.841])
	#Now run the LS minimization
	res = scipy.optimize.least_squares(skewnorm_residuals,p0,args=(perc_values,perc),\
		ftol=1e-15,xtol=1e-15,gtol=1e-15,bounds=pbounds) 
	#Save the result of the fit to file
	file = open(save_path+pname+'_skewnorm_fit.txt','w')
	file.write(str(res))
	file.close()
	#If the fit was not successful, create a file that indicates this failure
	if not res.success:
		file = open(save_path+pname+'_skewnorm_fit_failed.info','w')
		file.close()
	return res			

#Define a function that plots the distribution of random draws from a skew normal for 
# easy inspection later. Here, pname is the string name of the physical parameter.
def plot_skewnorm(draws,pname,save_path,x,plus,minus):
	figure()
	hist(draws,bins=50,normed=True)
	ylabel('f('+pname+')',fontsize='x-large')
	xlabel(pname,fontsize='x-large')
	axvline(percentile(draws,[15.9]),c='r',label='15.8%')
	axvline(percentile(draws,[50,]),c='c',label='50.0%')
	axvline(percentile(draws,[84.1]),c='lime',label='84.1%')
	legend(loc='best')
	title('DATA = ['+str(round(x-minus,4))+', '+str(round(x,4))+', '+\
		str(round(x+plus,4))+']',fontsize='large')
	savefig(save_path+pname+'_skewnorm_dist.pdf',bbox_inches='tight')
	close()
	return None
#-----------------------------------------------------------------------------------------


#Begin a loop over the planets to determine the PDFs of their transit probability
#-----------------------------------------------------------------------------------------
#Create storage arrays for transit probability statistics
pt_mean,pt_median,pt_std = zeros(size(RV_data)),zeros(size(RV_data)),zeros(size(RV_data))
pt_16,pt_50,pt_84 = zeros(size(RV_data)),zeros(size(RV_data)),zeros(size(RV_data))
pt_err1, pt_err2 = zeros(size(RV_data)),zeros(size(RV_data))

#Also store the stats of the "fitted" parameter distributions
Rstar_med, Rstar_err1, Rstar_err2 = zeros(size(RV_data)), zeros(size(RV_data)), \
	 zeros(size(RV_data))
Rp_med, Rp_err1, Rp_err2 = zeros(size(RV_data)), zeros(size(RV_data)),\
	 zeros(size(RV_data))
a_med, a_err1, a_err2 = zeros(size(RV_data)), zeros(size(RV_data)),\
	 zeros(size(RV_data))
ecc_med, ecc_err1, ecc_err2 = zeros(size(RV_data)), zeros(size(RV_data)),\
	 zeros(size(RV_data))
w_med, w_err1, w_err2 = zeros(size(RV_data)), zeros(size(RV_data)),\
	 zeros(size(RV_data))

#The amount of parameter draws is chosen by the user. Too small will not full sample the 
# distribution of transit probability values, but too large will be computationally pro-
#hibitive. I found one million to be a reasonable balance.
n_draws = int(1e6)
N_transits_draws = zeros(n_draws)
for i in range(size(RV_data)):
	#Make a directory with the name of the planet and its ID (if necessary). This path
	# information can be changed in whatever you see fit.
	dir_name = str(RV_data['id'][i]).zfill(3)+'-'+\
		RV_data['name'][i].split('"')[-2].replace(' ','_')	
	this_planet_path = dir_name+'/'
	#Check to see this is path already exists, if so, do not overwrite it
	all_dir = glob.glob('*/')
	make_dir = True
	for j in range(size(all_dir)):
		if dir_name in all_dir[j]: 
			make_dir = False
			break	
	if make_dir: os.system('mkdir '+this_planet_path)
	
	#Stellar radius
	Rstar_bounds = [-inf,inf]
	if RV_data['Rstar_err1'][i] == RV_data['Rstar_err2'][i]:
		#Normal distribution
		Rstar_draws = norm_draws(RV_data['Rstar'][i],RV_data['Rstar_err1'][i],n_draws,\
			Rstar_bounds)
	else:
		#Skew normal distribution.	
		skew_fit = fit_skew_normal(RV_data['Rstar'][i],RV_data['Rstar_err1'][i],\
			RV_data['Rstar_err2'][i],'Rstar',this_planet_path)
		#Now draw from that skew norm distribution	
		Rstar_draws = skewnorm_draws(skew_fit.x[0],skew_fit.x[1],skew_fit.x[2],n_draws,\
			Rstar_bounds)
		#Also save a histogram of the distribution for easy viewing
		plot_skewnorm(Rstar_draws,'Rstar',this_planet_path,RV_data['Rstar'][i],\
			RV_data['Rstar_err1'][i],RV_data['Rstar_err2'][i])
	#Find and store the errs for these draws
	Rstar_med[i], Rstar_err1[i], Rstar_err2[i] = percentile(Rstar_draws,50.), \
		percentile(Rstar_draws,84.1)-percentile(Rstar_draws,50.), \
		percentile(Rstar_draws,50.)-percentile(Rstar_draws,15.9)	

	#Planet radius
	Rp_bounds = [-inf,inf]
	if RV_data['Rp_err1'][i] == RV_data['Rp_err2'][i]:
		#Normal distribution
		Rp_draws = norm_draws(RV_data['Rp'][i],RV_data['Rp_err1'][i],n_draws,Rp_bounds)
	else:
		#Skew normal distribution.	
		skew_fit = fit_skew_normal(RV_data['Rp'][i],RV_data['Rp_err1'][i],\
			RV_data['Rp_err2'][i],'Rp',this_planet_path)
		#Now draw from that skew norm distribution	
		Rp_draws = skewnorm_draws(skew_fit.x[0],skew_fit.x[1],skew_fit.x[2],n_draws,\
			Rp_bounds)
		#Also save a histogram of the distribution for easy viewing
		plot_skewnorm(Rp_draws,'Rp',this_planet_path,RV_data['Rp'][i],\
			RV_data['Rp_err1'][i],RV_data['Rp_err2'][i])
	#Find and store the errs for these draws
	Rp_med[i], Rp_err1[i], Rp_err2[i] = percentile(Rp_draws,50.), \
		percentile(Rp_draws,84.1)-percentile(Rp_draws,50.), \
		percentile(Rp_draws,50.)-percentile(Rp_draws,15.9)	

	#Orbital semi-major axis
	a_bounds = [-inf,inf]
	if RV_data['a_err1'][i] == RV_data['a_err2'][i]:
		#Normal distribution
		a_draws = norm_draws(RV_data['a'][i],RV_data['a_err1'][i],n_draws,a_bounds)
	else:
		#Skew normal distribution.	
		skew_fit = fit_skew_normal(RV_data['a'][i],RV_data['a_err1'][i],\
			RV_data['a_err2'][i],'a',this_planet_path)
		#Now draw from that skew norm distribution	
		a_draws = skewnorm_draws(skew_fit.x[0],skew_fit.x[1],skew_fit.x[2],n_draws,\
			a_bounds)
		#Also save a histogram of the distribution for easy viewing
		plot_skewnorm(a_draws,'a',this_planet_path,RV_data['a'][i],\
			RV_data['a_err1'][i],RV_data['a_err2'][i])
	#Find and store the errs for these draws
	a_med[i], a_err1[i], a_err2[i] = percentile(a_draws,50.), \
		percentile(a_draws,84.1)-percentile(a_draws,50.), \
		percentile(a_draws,50.)-percentile(a_draws,15.9)	

	#Orbital eccentricity
	ecc_bounds = [-inf,inf]
	if RV_data['ecc_err1'][i] == RV_data['ecc_err2'][i]:
		#Normal distribution
		ecc_draws = norm_draws(RV_data['ecc'][i],RV_data['ecc_err1'][i],n_draws,ecc_bounds)
	else:
		#Skew normal distribution.	
		skew_fit = fit_skew_normal(RV_data['ecc'][i],RV_data['ecc_err1'][i],\
			RV_data['ecc_err2'][i],'ecc',this_planet_path)
		#Now draw from that skew norm distribution	
		ecc_draws = skewnorm_draws(skew_fit.x[0],skew_fit.x[1],skew_fit.x[2],n_draws,\
			ecc_bounds)
		#Also save a histogram of the distribution for easy viewing
		plot_skewnorm(ecc_draws,'ecc',this_planet_path,RV_data['ecc'][i],\
			RV_data['ecc_err1'][i],RV_data['ecc_err2'][i])
	#Find and store the errs for these draws
	ecc_med[i], ecc_err1[i], ecc_err2[i] = percentile(ecc_draws,50.), \
		percentile(ecc_draws,84.1)-percentile(ecc_draws,50.), \
		percentile(ecc_draws,50.)-percentile(ecc_draws,15.9)	

	#Longitude of periastron
	w_bounds = [-inf,inf]
	if RV_data['w_err1'][i] == RV_data['w_err2'][i]:
		#Normal distribution
		w_draws = norm_draws(RV_data['w'][i],RV_data['w_err1'][i],n_draws,w_bounds)
	else:
		#Skew normal distribution.	
		skew_fit = fit_skew_normal(RV_data['w'][i],RV_data['w_err1'][i],\
			RV_data['w_err2'][i],'w',this_planet_path)
		#Now draw from that skew norm distribution	
		w_draws = skewnorm_draws(skew_fit.x[0],skew_fit.x[1],skew_fit.x[2],n_draws,\
			w_bounds)
		#Also save a histogram of the distribution for easy viewing
		plot_skewnorm(w_draws,'w',this_planet_path,RV_data['w'][i],\
			RV_data['w_err1'][i],RV_data['w_err2'][i])
	#Find and store the errs for these draws
	w_med[i], w_err1[i], w_err2[i] = percentile(w_draws,50.), \
		percentile(w_draws,84.1)-percentile(w_draws,50.), \
		percentile(w_draws,50.)-percentile(w_draws,15.9)	

	#Calculate the transit probability using the draws from each parameter distribution.
	# See the text of the Dalba et al. 2018 for the explanation of this equation. For any
	# unrealistic probabilities (i.e., <0 or >1), assign the edge value.
	p_transit_draws = (Rstar_draws*Rsun+Rp_draws*RJ)*(1.+ecc_draws*cos((pi/2.)-\
		(w_draws*pi/180.)))/(a_draws*AU*(1.-ecc_draws*ecc_draws))
	#p_transit_draws = (Rstar_draws*Rsun+Rp_draws*RJ)/(a_draws*AU)
	p_transit_draws[where(p_transit_draws>1.)[0]]=1.
	p_transit_draws[where(p_transit_draws<0.)[0]]=0.

	
	#In this step, account for the TESS observational baseline by multiplying by a factor
	# of the observational baseline divided by the orbital period. Simply comment the 
	# next few line out to skip this. This factor is only multiplied if the observation
	# baseline reduces the probability of seeing a transit. Otherwise, the geometric 
	# probability is not reduced.
	time_factor = tau[i]/P[i]
	#Limit the tau/P penalty to unity (it can't improve the geometric transit prob.)
	if time_factor < 1.:
		p_transit_draws *= time_factor
		
	#To calculate the N distribution (number of RV exoplanets that transit), keep a run-
	# ning sum of all of the transit draws. This is the expected value problem as des-
	# cribed in the Dalba et al. 2018 paper. 
	N_transits_draws += p_transit_draws	
	
	#Save the figure of p_transit_draws for this exoplanet
 	figure()
 	hist(p_transit_draws,bins=50,normed=True,label='Draws')
 	ylabel('f(p_transit_tess)',fontsize='x-large')
 	xlabel('p_transit_tess',fontsize='x-large')
 	perc_values = percentile(p_transit_draws,[15.9,50.,84.1])
 	x, err1, err2 = perc_values[1], perc_values[2]-perc_values[1], perc_values[1]-\
 		perc_values[0]
 	title('p_transit_tess = '+str(round(x,5))+' +'+str(round(err1,5))+' / -'+\
 		str(round(err2,5)),fontsize='x-large')
 	if xlim()[1] > 1.: xlim(0,1)
 	x_range = linspace(xlim()[0],xlim()[1],1000)
 	norm_pdf = scipy.stats.norm.pdf(x_range,loc=mean(p_transit_draws),\
 		scale=std(p_transit_draws))
 	plot(x_range,norm_pdf,c='r',lw=2,label='Normal PDF')
 	legend(loc='best')
 	savefig(this_planet_path+'p_transit_v4.pdf',bbox_inches='tight')			 
 	close()
	
	#Save the statistics for this distribution 
	pt_mean[i],pt_median[i],pt_std[i] = mean(p_transit_draws), median(p_transit_draws),\
		std(p_transit_draws)
	pt_16[i],pt_50[i],pt_84[i] = perc_values
	pt_err1[i], pt_err2[i] = pt_84[i]-pt_50[i], pt_50[i]-pt_16[i]
		
	#End of loop over all exoplanets	
	print 'Finished '+RV_data['name'][i]+'.......'+str(i+1)+' of '+str(size(RV_data))
#-----------------------------------------------------------------------------------------


#Save the distribution of N_transits_draws as a binary pickle file (optional)
#-----------------------------------------------------------------------------------------
with open('N_transits_draws.pickle','wb') as pckl:
	pickle.dump(N_transits_draws,pckl)
pckl.close()	
#-----------------------------------------------------------------------------------------


#Generate histogram for N_transits (optional)
#-----------------------------------------------------------------------------------------
N_16, N_50, N_84 = percentile(N_transits_draws,[15.9,50.,84.1])
figure()
hist(N_transits_draws,bins=100,normed=True)
ylabel('f(N_transits)',fontsize='x-large')
xlabel('N_transits',fontsize='x-large')
axvline(N_16,c='r')
axvline(N_50,c='r')
axvline(N_84,c='r')
title('N_transits = '+str(round(N_50,1))+' +'+str(round(N_84-N_50,1))+' / -'+\
	str(round(N_50-N_16,1)), fontsize='x-large')
xlim(N_50 - 8*(N_50-N_16),N_50+8*(N_84-N_50))	
savefig('N_transits.pdf',bbox_inches='tight')
#-----------------------------------------------------------------------------------------


#Save statistics for all planets, including their RA/DEC and ecliptic coords (optional)
#-----------------------------------------------------------------------------------------
save_file = 'transit_probability_statistics.txt'
save_header = 'ID,name,pt_tess_mean,pt_tess_median,pt_tess_std,pt_tess_16,pt_tess_50,'+\
	'pt_tess_84,pt_tess_err1,pt_tess_err2,tranflag,ra,dec,elon,elat'
save_dtype = [('v1',float),('v2','S50'),('v3',float),('v4',float),('v5',float),\
	('v6',float),('v7',float),('v8',float),('v9',float),('v10',float),('v11',float),\
	('v12',float),('v13',float),('v14',float),('v15',float)]	
save_data = zeros(size(RV_data),dtype=save_dtype)
#Fill the array
save_data['v1'] = RV_data['id']
save_data['v2'] = RV_data['name']
save_data['v3'], save_data['v4'], save_data['v5'] = pt_mean, pt_median, pt_std
save_data['v6'], save_data['v7'], save_data['v8'] = pt_16, pt_50, pt_84
save_data['v9'], save_data['v10'], save_data['v11'] = pt_err1, pt_err2, RV_data['tranflag']
save_data['v12'], save_data['v13'] = RV_data['ra'], RV_data['dec']
save_data['v14'], save_data['v15'] = RV_data['elon'], RV_data['elat']
save_format = ['%4.1i','%20.20s','%.10e','%.10e','%.10e','%.10e','%.10e','%.10e','%.10e',\
	'%.10e','%.10e','%.10e','%.10e','%.10e','%.10e']
savetxt(save_file,save_data,delimiter=',',header=save_header,fmt=save_format)	
#-----------------------------------------------------------------------------------------


