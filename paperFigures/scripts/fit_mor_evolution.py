## created on Oct 22th, 2021

import numpy as np
from scipy.interpolate import interp1d
import scipy

import pandas as pd

from astropy.table import Table, vstack
from astropy.io.fits import getdata

from collections import defaultdict
from kllr import *
import linmix

def main():
    run     = 'gauss003_rhod_02Lstar'
    prefix  = 'lbd_gt5_%s_err'%run
    outpath = '/data/des61.a/data/johnny/Buzzard/Buzzard_v2.0.0/y3_rm/output/to_heidi/'
    # fname   = outpath+'buzzard_v2_lambda_gt20_mu-star_rhod.fits'
    fname   = outpath+'buzzard_v2_lambda_gt5_small_mu-star_rhod.fits'
    cat = Table(getdata(fname))
    
    ## Pivot points

    Mp = 1.#10.**15.5#np.nanmedian(cat['M200_true'])
    Mup= 1.#1./10**10#np.nanmedian(cat['MU'])

    print('Pivot Points')
    print(r'M200: %.2e $M_{\odot}$'%Mp)
    print(r'$\mu_{\star}: %.2e M_{\odot}$'%(Mup*1.0e10))
    
    mask = (cat['MU_%s'%run]>0.)&(cat['LAMBDA_CHISQ']>=5.)

    zcls = np.array(cat['redshift'][mask])
    m200 = np.array(cat['M200'][mask]/Mp)

    mu    = get_mustar_log(cat['MU_%s'%run][mask]/Mup)
    muerr = get_mustar_log(cat['MU_ERR_JK_%s'%run][mask]/Mup)

    mut   = get_mustar_log(cat['MU_TRUE_%s'%run][mask]/Mup)
    muterr= get_mustar_log(cat['MU_TRUE_ERR_JK_%s'%run][mask]/Mup)

    ng    = np.array(cat['Ngals_%s'%run][mask])
    ngt   = np.array(cat['Ngals_true_%s'%run][mask])
    
    ## Selecting Fiting variables
    x = m200
    y1= mu
    y2= mut
    
    y1err = muerr
    y2err = muterr
    
    ## Bining the sample
    zbins = np.nanpercentile(zcls,np.linspace(0,100,16))
    # zbins = np.linspace(0.1,0.7,16)
    zkeys, zmed = makeBin(zcls,zbins)
    zlabel = [r'%.3f < z < %.3f'%(zl,zh) for zl,zh in zip(zbins[:-1],zbins[1:])]

    
    ## Fiting the mass-observable relation
    outs = []
    print('Evolution Fit')
    for idx,zli in zip(zkeys,zlabel):

        mor = mass_observable_relation()
        print('zbin: %s'%zli)
        mor.add_dataset('observed',x[idx],y1[idx],y1err[idx],y_label='mu_obs')
        mor.add_dataset('true'    ,x[idx],y2[idx],y2err[idx],y_label='mu_tru')

        #mor.add_dataset('observed',x[idx],y1[idx],y_label='mu_obs')
        #mor.add_dataset('true'    ,x[idx],y2[idx],y_label='mu_tru')

        mor.fit_linmix('observed',is_log=True,nbins=16,percentile=[16,84])
        mor.fit_linmix('true'    ,is_log=True,nbins=16,percentile=[16,84])

        outs.append(mor)
        print('\n')
    
    print('Saving results: %s'%prefix)
    save_mor_results(prefix,outs)
    print('done')

def get_mustar_log(x):
    return x

class mass_observable_relation:
    """This class perfom the fit of the mass-observable relation
    
    Parameters:
    x: indepedent variable (Cluster Mass)
    y: depedent varibale (Observable)
    y_err: error on the indepedent variable
    dataset_name: key name for the loaded dataset; used to save the temp files
    
    Functions:
    __init__()
    
    add_dataset(name,x,y,y_err=None)
    fit(name,is_log=False,bins=None,nbins=16)
    
    """
    def __init__(self):
        print('Welcome Mass-Observable Relation Fit')
        self.data  = defaultdict(dict)
    
    def add_dataset(self,name,x,y,y_err=None,y_label='y',truncate=None):
        if truncate is not None:
            yl, yh = truncate
            yc = y.copy()
            mask = (yc>yl)&(yc<yh)
            self.data[name]['delta'] = mask
            y = np.where(y<yl,yl,y)
            y = np.where(y>yh,yh,y)
            
        else:
            self.data[name]['delta'] = np.full(y.shape,True)
            
        self.data[name]['x'] = np.array(x)
        self.data[name]['y'] = np.array(y)
        
        self.data[name]['log_x']     = np.log10(np.array(x))
        self.data[name]['log_y']     = np.log10(np.array(y))
        
        if y_err is not None:
            self.data[name]['y_err']     = y_err
            self.data[name]['log_y_err'] = y_err/np.array(y)
        else:
            self.data[name]['y_err']     = None
            self.data[name]['log_y_err'] = None
        
        self.data[name]['ylabel']    = y_label
        return self
    
    def fit_kllr(self,name,is_log=False,bins=None,nbins=11,percentile=[16,84],
                 kernel_type = 'gaussian', kernel_width = 0.45,fast_calc=False,
                 xrange=None, nBootstrap=100, verbose = True):
        
        ## load variables
        x, y, y_err,_ = self.get_variable(name,is_log=is_log)
        
        ## set y_err to none
        #y_err = None
        
        ## get nbins percentile
        if nbins is not None: bins = np.nanpercentile(x,np.linspace(0,100,nbins))
        
        lm = kllr_model(kernel_type = kernel_type, kernel_width = kernel_width)
        out = lm.fit(x, y, y_err, xrange, bins, nBootstrap, fast_calc, verbose)
        
        # out =  xv, y_exp, intercept_exp, slope_exp, scatter_exp
        
        ## save standard output
        ## format: x, y, y-, y+, slope, slope+, slope-, scatter, scatter-, scatter+
        self.data[name]['kllr'] = get_output_variables(*out[:5])

    def fit_linmix(self,name, nbins=11, bins=None, is_log=False, K=3, percentile=[16,84],
                   nchains=12, silent=True, maxiter=50000):
        x, y, y_err, delta = self.get_variable(name,is_log=is_log)
        
        if ((delta == False).any())&(y_err is not None):
            y_err[~delta] = 0.#np.nanmedian(y_err[delta])
            
        lm = linmix.LinMix(x, y, ysig=y_err, delta=delta, K=K, nchains=nchains)
        lm.run_mcmc(silent=silent,maxiter=maxiter)
        
        alpha, beta = lm.chain[:]['alpha'],lm.chain[:]['beta']
        scatter     = np.sqrt(lm.chain[:]['sigsqr'])

        # extend results along x-vec
        ## get nbins percentile
        if bins is None: bins = np.nanpercentile(x,np.linspace(0,100,nbins))
        
        ## compute out the fitted lines
        xbins= np.tile(bins,(len(alpha),1))
        ybins= func_line(xbins,alpha[:,np.newaxis],beta[:,np.newaxis])
        
        #sscatter= np.std(ybins-np.median(ybins,0))
        bbeta  = np.tile(beta,(len(bins),1)).T
        sscater= np.tile(scatter,(len(bins),1)).T
        aalpha = np.tile(alpha,(len(bins),1)).T
        
        ## save standard output
        ## format: x, y, y-, y+, slope, slope+, slope-, scatter, scatter-, scatter+
        self.data[name]['linmix'] = get_output_variables(bins,ybins,aalpha,bbeta,sscater)      
#         line = alpha[:,np.newaxis] + beta[:,np.newaxis]*bins
#         median = np.nanpercentile(line,50,axis=0)
#         scm = np.nanpercentile(line,16,axis=0)-median
#         scp = np.nanpercentile(line,84,axis=0)-median
#         scc = 0.5*(scp-scm)
        
#         self.data[name]['linmix']['scatter' ] = scc
#         self.data[name]['linmix']['scatter+'] = scp
#         self.data[name]['linmix']['scatter-'] = scm
        
    def fit_emcee(self, name, nwalkers=64, nsteps=1000, discard=100,
                  nbins=11,bins=None, is_log=False, percentile=[2.5,97.5]):
        x, y, y_err, delta = self.get_variable(name,is_log=is_log)
        if ((delta == False).any())&(y_err is not None):
            print('here')
            y_err[~delta] = 0.#np.nanmedian(y_err[delta])
        
        #y_err   = 1/np.sqrt(10**y) 
        mychain = run_emcee(x,y,y_err,nsteps=nsteps,nwalkers=nwalkers, discard=int(discard))
        out     = get_emcee_results(mychain, p=percentile) ## results in self.data[name]['emcee']
        self.data[name]['emcee'] = out
        self.data[name]['emcee']['chain'] = mychain
        
    def get_variable(self,name,is_log=False):
        myvars = ['x','y','y_err','delta']
        
        if is_log: myvars = ['log_%s'%vi for vi in myvars]
        x = self.data[name][myvars[0]][:]
        y = self.data[name][myvars[1]][:]
        y_err = self.data[name][myvars[2]]
        delta = self.data[name]['delta']
        return x,y,y_err,delta
    
def get_output_variables(x,y,intercept,slope,scatter):
    percentile2Sigma =[2.5,97.5]
    percentile1Sigma =[16,84]
    output_Data = dict()
    output_Data['x']  = x
    output_Data['y']  = np.nanpercentile(y, 50, 0)
    output_Data['y-'] = np.nanpercentile(y, percentile1Sigma[0], 0)
    output_Data['y+'] = np.nanpercentile(y, percentile1Sigma[1], 0)
    output_Data['y--'] = np.nanpercentile(y, percentile2Sigma[0], 0)
    output_Data['y++'] = np.nanpercentile(y, percentile2Sigma[1], 0)

    output_Data['slope']  = np.nanpercentile(slope, 50, 0)
    output_Data['slope-'] = np.nanpercentile(slope, percentile1Sigma[0], 0)
    output_Data['slope+'] = np.nanpercentile(slope, percentile1Sigma[1], 0)
    output_Data['slope--'] = np.nanpercentile(slope, percentile2Sigma[0], 0)
    output_Data['slope++'] = np.nanpercentile(slope, percentile2Sigma[1], 0)

    output_Data['scatter']  = np.nanpercentile(scatter, 50, 0)
    output_Data['scatter-'] = np.nanpercentile(scatter, percentile1Sigma[0], 0)
    output_Data['scatter+'] = np.nanpercentile(scatter, percentile1Sigma[1], 0)
    output_Data['scatter--'] = np.nanpercentile(scatter, percentile2Sigma[0], 0)
    output_Data['scatter++'] = np.nanpercentile(scatter, percentile2Sigma[1], 0)
    
    output_Data['intercept']  = np.nanpercentile(intercept, 50, 0)
    output_Data['intercept-'] = np.nanpercentile(intercept, percentile1Sigma[0], 0)
    output_Data['intercept+'] = np.nanpercentile(intercept, percentile1Sigma[1], 0)
    output_Data['intercept--'] = np.nanpercentile(intercept, percentile2Sigma[0], 0)
    output_Data['intercept++'] = np.nanpercentile(intercept, percentile2Sigma[1], 0)
    return output_Data

def func_line(x,alpha,slope):
    return alpha+slope*x

def makeBin(variable, xedges):
    xbins = (xedges[1:]+xedges[:-1])/2
    indices = [ np.where((variable >= xedges[i]) & (variable <= xedges[i + 1]))[0] for i in range(len(xedges)-1)]
    return indices, xbins

import pickle 

def save_mor_results(key,outs,root='/data/des61.a/data/johnny/Buzzard/mor_data/'):
    fname = root+'/%s.obj'%key
    file_pi = open(fname, 'w') 
    pickle.dump(outs, file_pi)
    file_pi.close()
    
def load_mor_results(key,root='/data/des61.a/data/johnny/Buzzard/mor_data/'):
    fname = root+'/%s.obj'%key
    filehandler = open(fname, 'r') 
    p2 = pickle.load(filehandler)
    filehandler.close()
    return p2

### emcee ####
from scipy import odr
import emcee

x0_prior_min = np.array([0.2,1.25,1.6,0.10])
x0_prior_max = np.array([0.4,1.75,2.5,0.25])

def run_emcee(x,y,yerr,nsteps=1000,nwalkers=100,discard=100):
    #guess,gerr = fit_poly_2nd(x,y,1/np.sqrt(10**y))
    guess,gerr = fit_poly_2nd(x,y)

    ndim= 4
    pos = [guess + gerr*np.random.randn(ndim) for i in range(nwalkers)]
    print(pos[0])
    #pos = list(np.array([np.random.uniform(xmin,xmax, nwalkers) for xmin, xmax in zip(x0_prior_min,x0_prior_max)]).T)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
    sampler.run_mcmc(pos, nsteps)
    
    samples = sampler.chain[:, discard:, :].reshape((-1, ndim)).copy()
    sampler.reset()
    return samples

def qmodel(theta,x):
    a, b, c = theta
    return a*x**2 + b*x + c
    
def lnlike(theta, x, y, yerr):
    a, b, c, ln_sig = theta
    model = a*x**2 + b*x + c
    inv_sigma2 = 1.0/(yerr**2 + ln_sig**2)
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior(theta):
    a, b, c, ln_sig = theta
    if (0. < ln_sig):
        return 0.
    return -np.inf

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

def get_emcee_results(samples,p=[2.5,97.5]):
    pars = list(map(lambda v: (v[1], v[2], v[0]), zip(*np.nanpercentile(samples, [p[0], 50, p[1]],axis=0))))
    mydict  = dict()
    mypars  = ['beta2','beta1','alpha','scatter']
    for li, ppi in zip(mypars,pars):
        for mi,pi in zip(['','+','-'],ppi):
            mydict[li+mi] = pi
    return mydict

quad_model = odr.Model(qmodel)
def fit_poly_2nd(x,y,yerr=None,guess=[0.3,0.7,1.]):
    # Create a RealData object
    data = odr.RealData(x, y, sy=yerr)

    # Set up ODR with the model and data.
    model = odr.ODR(data, quad_model, beta0=guess, maxit=100)

    # Run the regression.
    out  = model.run()
    
    #print fit parameters and 1-sigma estimates
    popt = out.beta
    perr = out.sd_beta
    
    dof  = len(y)-1
    sig2 = np.sqrt(np.sum((y - qmodel(popt,x))** 2) / dof)
    
    popt = np.append(popt,sig2)
    perr = np.append(perr,np.array([0.075]))
    print('fit parameter 1-sigma error')
    for i in range(len(popt)):
        print('%i: %.5f \t %.5f'%(i, popt[i],perr[i]))
        
    return popt, perr

### Plot Functions
def plot_par_evolution(zmed,outs,var,name,xlabel=None,ax=None,zp=0.35,color='b',color2='k',label=''):
    if ax is None: ax = plt.axes()
    
    to_fit = []
    for i in range(len(outs)):
        self = outs[i]
        zi = zmed[i]
        
        lm = self.data[name]['emcee']            
        median = np.mean(lm[var])
        lower_quartile, upper_quartile = np.mean(lm[var+'-']),np.mean(lm[var+'+'])
        si = 0.5*(upper_quartile-lower_quartile)
            
        to_fit.append([zi,median,si,lower_quartile,upper_quartile])

    to_fit = np.array(to_fit).T
    
    p = ax.plot(to_fit[0], to_fit[1], 'o', color=color,label=label)
    ax.vlines(to_fit[0], to_fit[3], to_fit[4], color=p[0].get_color())

    eta = np.log((1+to_fit[0])/(1+zp))
    lm1 = linmix.LinMix(eta,to_fit[1], ysig=to_fit[2], K=3)
    lm1.run_mcmc(silent=True, maxiter=10000)
    
    plot_linmix(np.linspace(0.2,0.65,31),lm1,zp=zp,ax=ax,color=color2)
    if xlabel is None: xlabel=var
    ax.set_ylabel(xlabel,fontsize=24)
    
def plot_linmix(z,lm1,zp=0.4,ax=None,color='k',label=''):
    if ax is None: ax = plt.axes()
    
    eta = np.log((1+z)/(1+zp))
    m1 = np.median(lm1.chain['alpha'][:,np.newaxis]     + lm1.chain['beta'][:,np.newaxis]*eta,axis=0)
    q1 = np.nanpercentile(lm1.chain['alpha'][:,np.newaxis] + lm1.chain['beta'][:,np.newaxis]*eta,16,axis=0)
    q2 = np.nanpercentile(lm1.chain['alpha'][:,np.newaxis] + lm1.chain['beta'][:,np.newaxis]*eta,84,axis=0)
    
    p = ax.plot(z,m1,color=color,label=label)
    ax.fill_between(z,q1,q2,alpha=0.3,color=p[0].get_color())


if __name__ == "__main__":
    main()