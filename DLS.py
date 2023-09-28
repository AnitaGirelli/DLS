import numpy as np
import glob, os
import pandas as pd
import matplotlib.pyplot as plt
from inspect import signature
from scipy.optimize import curve_fit as fit

class DLS_class:
    '''  Class to load data, create a dataframe, analyse data  '''

    def __init__(self,dirname='',wavelength = 660e-9,n0 = 1.33 ):
        
        self.dirname = dirname
        self.wavelength = wavelength
        self.n0 = n0
        self.df = pd.DataFrame( )
        self.ave_df = pd.DataFrame( )
        
        
    def load_data(self):
        """ Load the data in a given folder and save it in a dataframe """

        os.chdir(self.dirname)
        for fname in glob.glob("*.dat"):    
            lines = []
            i = 0
            
            try:
                with open (fname, 'rt') as file:     # Open file for reading text data.
                    for line in file:                # For each line, stored as line,
                        lines.append(line)           # add its contents to lines.
                        
                        index = line.find("Scattering angle")
                        if index != -1:             # If something was found,
                            #theta_index = len(lines)
                            theta = float(line[18:].strip()) # scattering angle, deg

                        index = line.find("Duration")
                        if index != -1:             # If something was found,
                            duration = float(line[14:].strip()) # duration, s

                        index = line.find("Viscosity")
                        if index != -1:             # If something was found,
                            n = float(line[18:].strip()) # viscosity, mPas
                            
                        index = line.find("Temperature")
                        if index != -1:             # If something was found,
                            temp = float(line[17:].strip()) # temperature, K

                        index = line.find("Laser intensity")
                        if index != -1:             # If something was found,
                            laser_intensity = float(line[22:].strip()) # laser intensity, mW

                        index = line.find("Average Count rate  A")
                        if index != -1:             # If something was found,
                            av_cra = float(line[29:].strip()) # average count rate A, kHz

                        index = line.find("Average Count rate  B")
                        if index != -1:             # If something was found,
                            av_crb = float(line[29:].strip()) # average count rate B, kHz
                            
                        index = line.find("Intercept")
                        if index != -1:             # If something was found,
                            intercept = float(line[11:].strip()) # average count rate B, kHz
                            
                        index = line.find("g2")
                        if index != -1:             # If something was found,
                            g2_index = len(lines)   # row index for lag time / g2 data

                        index = line.find("Count Rate History")
                        if index != -1:             # If something was found,
                            cr_index = len(lines)   # row index for count rate history
                    
                    
                    t, g2 = np.genfromtxt(fname,delimiter=None,autostrip=True,unpack=True,skip_header=g2_index,skip_footer=len(lines)-cr_index) # lag time / g2 data
                    #time, cra, crb = np.genfromtxt(fname,delimiter=None,autostrip=True,unpack=True,skip_header=cr_index,skip_footer=0) # count rate history
        
                    self.df=self.df.append(  
                            {"Scattering angle" :               theta, 
                                "q" :                          (4*np.pi*self.n0*np.sin(np.round(theta)/2*np.pi/180)/self.wavelength), # scattering vector, 1/m
                                "Duration" :                    duration,
                                "Temperature" :                 int(np.round(temp)),
                                "Laser intensity" :             laser_intensity,
                                "Average Count rate  A":        av_cra,
                                "Average Count rate  B":        av_crb,
                                "Intercept":                    intercept,
                                "g2":                           g2,
                                "t":                            t,
                                "filename":                     fname},
                             ignore_index=True)
                  
            except ValueError:
                print(f'The file {fname} is not a g2 file.')

        
    def average_data(self, plot=True):
        """ Average the measurements at the same temperature and angle
            Args:
                plot: choose to plot or not the experimental g2 functions for each temperature
        """
        temperatures = sorted(set(self.df['Temperature']))
        
        for tn,temperature in enumerate(temperatures):
            angles = sorted(set(self.df[self.df['Temperature']==temperature]['Scattering angle']))
            colore = plt.cm.jet(np.linspace(0,1,len(angles)))

            if plot:    
                plt.figure()
                plt.title(f'T='+str(temperature)+' K')
                plt.grid(ls=':', c='gray', alpha=0.7)
                plt.xscale('log')
                plt.xlabel('t (s)')
                plt.ylabel('$g_2$-1')

            for n, angle in enumerate(angles):
                df_selected = self.df[(self.df['Scattering angle']==angle)&(self.df['Temperature']==temperature)]

                try:
                    len_g2 = []
                    for index in df_selected.index:
                        len_g2.append(len(df_selected['g2'][index]))

                    len_g2_final = np.min(len_g2)
                    g2_all = np.zeros([len_g2_final, len(df_selected.index)])
                    
                    g2 = np.zeros(len_g2_final)
                    dg2 = np.zeros(len_g2_final)
                    t = np.zeros(len_g2_final)
                    
                    for i,index in enumerate(df_selected.index):   
                        g2_all[:,i]=df_selected['g2'][index][:len_g2_final]

                    if len(df_selected.index) < 2:
                        g2 = g2_all
                        dg2 = g2_all*0.1

                    else:
                        g2 = np.mean(g2_all,axis=1)
                        dg2 = np.std(g2_all,axis=1)

                    t = df_selected['t'][index][:len_g2_final]
                    
                    if plot:
                        plt.errorbar(t, g2, dg2, c=colore[n], label=f'{angle:.0f}')
                        
                    self.ave_df=self.ave_df.append(  
                                    {   "Scattering angle":             angle,
                                        "q" :                          (4*np.pi*self.n0*np.sin(np.round(angle)/2*np.pi/180)/self.wavelength), # scattering vector, 1/m
                                        "Temperature" :                 df_selected['Temperature'].mean(),
                                        "Laser intensity" :             df_selected['Laser intensity'].mean(),
                                        "Average Count rate  A":        df_selected["Average Count rate  A"].mean(),
                                        "Average Count rate  B":        df_selected["Average Count rate  B"].mean(),
                                        "Intercept":                    df_selected["Intercept"].mean(),
                                        "g2":                           g2,
                                        "dg2":                          dg2,
                                        "t":                            t,},
                                    ignore_index=True)
                except ValueError:
                    print(temperature, angle, df_selected['filename'])#len_g2, df_selected.index)

            if plot:
                plt.legend(frameon=False, ncols=2, fontsize=8, title='angle')



    def plot_fit_g2(self, function, tnorm=10, p0=False, plot=True, boundaries=False):
        """ Fit the g2 functions
        Args:
            function: function to use for the fit (exponential, stretched_exponential, double_stretched_exponential, double_exponential)
            tnorm: number of datapoint to average to get the intercept of the g2
            p0: parameter initial guesses for the fit
            plot: choose to plot or not the g2 functions with their fits for each temperature
        """
        temperatures = sorted(set(self.ave_df['Temperature']))
        colors_temp = plt.cm.coolwarm(np.linspace(0,1,len(temperatures)))
        Dconst = np.zeros(len(temperatures))
        
        fig1 = plt.figure()
        npar = len(signature(function).parameters)
       
        if not p0:
            p0 = np.ones(npar-1)

        if not boundaries:
            boundaries = (np.zeros(npar-1),np.ones(npar-1)*100)

        for tn,temperature in enumerate(temperatures):
        
            df_selected = self.ave_df[self.ave_df['Temperature']==temperature]
            
            q = df_selected['q']
            q_fit = []
            nqvals = len(set(df_selected['q']))
            colors = plt.cm.jet(np.linspace(0,1,nqvals))
       
            all_popt=[]

            if plot:
                plt.figure()
            
            angles=(df_selected['Scattering angle'])
            
            
            for qv, angle in enumerate(angles):   

                i = df_selected[df_selected['Scattering angle']==angle].index[0]

                y =   df_selected['g2'][i] /np.mean(df_selected['g2'][i][:tnorm])
                dy = df_selected['dg2'][i] /np.mean(df_selected['g2'][i][:tnorm])
                x = df_selected['t'][i]
                
                try:
                    # fit the g2 functions
                    popt, pcov = fit(function, xdata=x, ydata=y, sigma=dy, p0=p0, bounds=boundaries)
                    q_fit.append(df_selected['q'][i])
                    all_popt += [popt]

                except RuntimeError:
                    print("Couldn't fit !!")

                if plot:
                    plt.errorbar(x, y, dy, c=colors[qv], marker='o', ls='', mec='black', mew=.3, label=f'{angle:.0f}')
                    plt.plot(x, function(x,*popt), c='r', ls='--',lw=1)
                    plt.grid(ls=':', c='gray', alpha=0.7)    
                    plt.xscale('log')
                    plt.xlabel('t (s)')
                    plt.ylabel('$g_2$-1')
                    plt.title(f'T={temperature:.0f} K')
                    plt.legend(frameon=False, fontsize=8, title='angle', ncols=2)
 
            # plot Gamma vs q^2 and fit it linearly
            all_popt = np.array(all_popt)
            plt.figure()
            plt.figure(fig1)

            Gamma = 1/all_popt[:,1]
            q_fit = np.array(q_fit)

            D, pcov = fit(self.tau_fit, xdata=q_fit**2, ydata=Gamma)
            plt.plot(q_fit**2, Gamma, color=colors_temp[tn],marker='o', ls='', mec='black', mew=.3, label=f'{temperature:.0f}')
            plt.plot(q_fit**2, self.tau_fit(q_fit**2,D), color='red', ls='dashed', lw=1)
            plt.grid(ls=':', c='gray', alpha=0.7)       
            plt.xlabel('$q^2$ (m$^{-2}$)')
            plt.ylabel('$\Gamma$ (s$^{-1}$)')
            Dconst[tn] = D
            plt.legend(title='T (K)')
            
        return (np.array(all_popt), np.array(q), np.array(Dconst), np.array(temperatures))
    
    
    ############################################################################
    # fit functions:
    @staticmethod
    def gauss_function(x, a, x0, sigma):
            return a*np.exp(-(x-x0)**2/(2*np.abs(sigma)**2))
    
    @staticmethod   
    def tau_fit(x,a):
        return a*x
    
    @staticmethod
    def tau_fit_offset(x,a,b):
        return a*x+b
    
    @staticmethod
    def exponential(x,a,b):
        return np.abs(a)*np.exp(-2*x/(np.abs(b)))
    
    @staticmethod
    def stretched_exponential(x,a,b,c):
        return np.abs(a)*np.exp(-2*(x/np.abs(b))**c)

    @staticmethod
    def double_stretched_exponential(x,a1,b1,c1,a2,b2,c2):
        return np.abs(a1)*np.exp(-2*(x/np.abs(b1))**c1) + np.abs(a2)*np.exp(-2*(x/np.abs(b2))**c2)
    
    @staticmethod
    def double_exponential(x,a1,b1,a2,b2,c2):
        return np.abs(a1)*np.exp(-2*x/np.abs(b1)) + np.abs(a2)*np.exp(-2*(x/np.abs(b2))**c2)
    
    @staticmethod
    def arrhenius(x,a,b):
        return -a*x+b
    
    @staticmethod
    def power_law(x,a,b,c): # power law fit: X(T) = X0*(T/Ts-1))^(-gamma)
        return a*(x/b-1)**(-c)


    ############################################################################

        
        
            
            