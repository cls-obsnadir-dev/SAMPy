#!/usr/bin/python
# -*- coding: utf-8 -*-
#sampy.py

"""
Anno Domini 02/02/2021 S. DINARDO fecit :   Creation, version 1.00

                            #####################################
                            #                                   #
                            #    Classes, functions             #
                            #    and data constants which       #
                            #    allow to retrack               #
                            #    SAR altimetry waveforms        #
                            #    with SAMOSA and SAMOSA+        #
                            #    retracker                      #
                            #####################################

Classes :

  - SAMOSA  -> class to retrack SAR altimetry waveform by SAMOSA and SAMOSA+ retracker

Functions :

  - initialize_epoch -> function to provide the guess epoch
  - compute_ThNEcho ->  function to provide the Thermal Noise from Echo

### TO DO LIST  ### ->->->

References:

REF1: Ray C., Martin-Puig C., Clarizia M.P., Ruffini G., Dinardo S., Gommenginger G., Benveniste J., (2014):
      SAR Altimeter Backscattered Waveform Model, IEEE Transactions on Geoscience and Remote Sensing,
      vol. 53, no. 2, pp. 911-919, Feb. 2015, https://doi.org/10.1109/TGRS.2014.2330423.
REF2: Dinardo S.,  Fenoglio l., Buchhaupt C., Becker, M., Scharroo R., Fernandes M.J., Benveniste J., (2017):
      Coastal SAR and PLRM Altimetry in German Bight and West Baltic Sea. Advances in Space Research. 62. https://doi.org/10.1016/j.asr.2017.12.018.
REF3: Dinardo, Salvatore, (2020). Techniques and Applications for Satellite SAR Altimetry over water, land and ice, 56.
      Technische Universitat, Darmstadt, https://doi.org/10.25534/tuprints-00011343, Ph.D. Thesis, ISBN 978-3-935631-45-7

"""

## Loading Libraries
## -----------------------

import numpy as np
import os
import scipy.optimize

## -----------------------

# -------------------------------------------------------------------------
# =========================================================================================
# ============   D E F I N I T I O N  OF  F U N C T I O N S   ========================
# -------------------------------------------------------------------------
# Add here functions which are common and used by all the classes in the library

# ----------------------------------------------------------------------
def initialize_epoch(data,tau,Raw_Elevation,CST,size_half_block=10):

    """

       Function -> initialize_epoch(data,tau,Raw_Elevation,CST,size_half_block)
                   Function providing the first-guess epoch

          Input :

                data -> waveform data matrix (dimensions are rangeXrecords)
                tau ->  time for each sample of the waveforms in data
                Raw_Elevation -> Orbit Height minus one-way tracker range
                CST -> structure of constant (speedlight in CST.c0)
                size_half_block -> half size of the moving window (10 records generally)
    """

    dr = CST.c0 / 2 * np.mean(np.diff(tau))
    DX = Raw_Elevation / dr
    DX[ np.where(np.isnan(DX)) ] = 0

    if np.shape(data)[1] > 30e3 :

        threshold_pos = 5000  / dr
        threshold_neg = -400  / dr

    else :

        threshold_pos = 9000  / dr
        threshold_neg = -400  / dr

    DX[ np.where(DX>threshold_pos) ] = threshold_pos
    DX[ np.where(DX<threshold_neg) ] = threshold_neg

    DX=max(DX)-DX
    DN=np.around(max(DX)).astype(np.int).item()

    REGG = np.zeros((np.shape(data)[0] + DN, np.shape(data)[1]), dtype=np.float32)

    for i in np.arange( np.shape(data)[1]) :

        REGG[  np.around(DX[i]).astype(np.int).item() : np.around(DX[i]).astype(np.int).item() + np.shape(data)[0], i ]=data[:, i] / max(data[:, i])

    COR_O = np.zeros( np.shape(data) , dtype=np.float32)

    for i in np.arange( np.shape(REGG)[1]) :

        fi = max(0, i - size_half_block)
        la = min( np.shape(REGG)[1], i + size_half_block+1)

        block = REGG[:, fi: la].astype(np.float64)

        block = np.delete(block, np.where(np.isnan(np.sum(block, axis=0))), axis=1)

        tmp = np.prod(block, axis=1)
        COR = tmp / max(tmp)
        COR_O[:, i]=COR[ np.around(DX[i]).astype(np.int).item() : np.around(DX[i]).astype(np.int).item()+ np.shape(data)[0] ]

    del REGG

    epoch0 = tau[np.argmax(COR_O,axis=0)]
    del COR_O
    return epoch0

# ----------------------------------------------------------------------

def compute_ThNEcho(data,NstartNoise,NendNoise):

    """

       Function -> compute_ThNEcho(data,NstartNoise,NendNoise)
                   Function providing the Thermal Noise computed from the waveform

          Input :
                data -> waveform data matrix (dimensions are rangeXrecords)
                NstartNoise -> value of the range gate from which to start the noise window (counting from 1)
                NendNoise -> value of the range gate at which to stop the noise window (counting from 1)
    """

    NstartNoise=int(NstartNoise)
    NendNoise=int(NendNoise)

    data = np.sort(data[0:np.shape(data)[0] // 2,:], axis=0)
    data[np.where(data <= 0)] = np.nan

    if  NstartNoise-1<0:

        NstartNoise=0

    if  NendNoise>np.shape(data)[0] // 2:

        NendNoise=np.shape(data)[0] // 2

    ThNEcho=np.nanmedian(data[NstartNoise-1:NendNoise,:],axis=0)
    ThNEcho[ np.where(np.isnan(ThNEcho)) ]= np.nanmedian(ThNEcho)
    return ThNEcho

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# ===================   END  OF  F UN C T I O N S   ===============================
# =========================================================================================

# -------------------------------------------------------------------------
# ===================   START OF THE CLASSES   ===============================
# =========================================================================================

    # -------------------------------------------------------------------------------------------
    #                           SAMOSA CLASS
    # -------------------------------------------------------------------------------------------

# ===============================================================================================

class SAMOSA:

    """

    Class -> SAMOSA(self)

        Input :

    Class Public Methods :

            - retrack_SAMOSA() -> method to retrack a SAR waveform by SAMOSA or SAMOSA+ retracker

    """

    # ----------------------------------------------------------------------
    # ------------- Initialization------------------------------------------

    def __init__(self,CST,RDB,OPT,LUT):

        """

         Private Method -> SAMOSA Class's Init

             Input :

                    - CST : structure with fields: c0 (lightspeed in m/sec), R_2 ( Reference Ellipsoid  Earth Radius in m),
                            f_e (Reference Ellipsoid Earth Flatness), gamma_3_4 (gamma function value at 3/4)
                    - RDB : structure with fields: Np_burst (number of pulses in a burst), Npulse (number of range gates per pulse),
                            PRF_SAR (Pulse Repetition Frequency in SAR mode, in Hz), BRI (Burst Repetition Interval in sec),
                            f_0 (carrier frequency in Hz), Bs (Sampled Bandwidth) , theta_3x (antenna 3dB aperture along track in radiant),
                            theta_3y (antenna 3dB aperture across track in radiant),
                    - OPT : structure with fields: method (optimization solver method), ftol (exit tolerance on f), xtol (exit tolerance on x)
                            diff_step (relative step size for the finite difference approximation of the Jacobian) max_nfev (maximum number of function evaluations)
                            gtol (exit tolerance on gradient norm of f), loss (loss function) => see scipy.optimize.least_squares for details
                    - LUT : structure with fields: F0 (filename of the F0 LUT), F1 (filename of the F1 LUT),
                            alphap_noweight (filename of the alphap LUT in case no weighting), alphap_weight (filename of the alphap LUT in case of weighting),
                            alphapower_noweight (filename of the alpha power LUT in case no weighting).
                            All the LUT files must be in a folder named auxi and located in the same folder as sampy.py

         """

        print('\n  Initialiating the Class ...')
        folder=os.path.dirname(__file__) + os.sep + "auxi" + os.sep

        if not hasattr(LUT,'F0'):
            print('  Error: LUT Attribute F0 not given in F0')
            self.sucess = False

        if not hasattr(LUT,'F1'):
            print('  Error: LUT Attribute F1 not given in F1')
            self.sucess = False

        if not hasattr(LUT,'alphap_noweight'):
            print('  Error: LUT Attribute alphap_noweight not given in alphap_noweight')
            self.sucess = False

        if not hasattr(LUT,'alphap_weight'):
            print('  Error: LUT Attribute alphap_weight not given in alphap_weight')
            self.sucess = False

        if not hasattr(LUT,'alphapower_noweight'):
            print('  Error: LUT Attribute alphapower_noweight not given in alphapower_noweight')
            self.sucess = False

        if not hasattr(LUT,'alphapower_weight'):
            print('  Error: LUT Attribute alphapower_weight not given in alphapower_weight')
            self.sucess = False

        if os.path.isfile(folder+ os.path.sep +LUT.F0):

            self.F0_LUT=np.genfromtxt(folder+ os.path.sep +LUT.F0, dtype='float', comments='#', delimiter=None)

        else:
            print("  Fatal Error: file " +folder+ os.path.sep +LUT.F0 + ' does not exist... exiting the class ')
            self.sucess = False

        if os.path.isfile(folder+ os.path.sep +LUT.F1):

            self.F1_LUT=np.genfromtxt(folder+ os.path.sep +LUT.F1, dtype='float', comments='#', delimiter=None)

        else:
            print("  Fatal Error: file " +folder+ os.path.sep +LUT.F1 + ' does not exist... exiting the class ')
            self.sucess = False

        if os.path.isfile(folder+ os.path.sep +LUT.alphap_noweight):

            self.alphap_LUT_NoWght = np.genfromtxt(folder + os.path.sep + LUT.alphap_noweight, dtype='float', comments='#', delimiter=',')

        else:
            print("  Fatal Error: file " +folder+ os.path.sep +LUT.alphap_noweight + ' does not exist... exiting the class ')
            self.sucess = False

        if os.path.isfile(folder+ os.path.sep +LUT.alphap_weight):

            self.alphap_LUT_Wght = np.genfromtxt(folder + os.path.sep + LUT.alphap_weight, dtype='float', comments='#', delimiter=',')

        else:
            print("  Fatal Error: file " +folder+ os.path.sep +LUT.alphap_weight + ' does not exist... exiting the class ')
            self.sucess = False

        if os.path.isfile(folder+ os.path.sep +LUT.alphapower_noweight):

            self.alphapower_LUT_NoWght = np.genfromtxt(folder + os.path.sep + LUT.alphapower_noweight, dtype='float', comments='#', delimiter=',')

        else:
            print("  Fatal Error: file " +folder+ os.path.sep +LUT.alphapower_noweight + ' does not exist... exiting the class ')
            self.success = False

        if os.path.isfile(folder+ os.path.sep +LUT.alphapower_weight):

            self.alphapower_LUT_Wght = np.genfromtxt(folder + os.path.sep + LUT.alphapower_weight, dtype='float', comments='#', delimiter=',')

        else:
            print("  Fatal Error: file " +folder+ os.path.sep +LUT.alphapower_weight + ' does not exist... exiting the class ')
            self.success = False

        if not hasattr(CST,'c0'):
            print('  Error: CST Attribute lightspeed not given in c0')
            self.sucess = False

        if not hasattr(CST,'R_e'):
            print('  Fatal Error: CST Attribute Earth Radius not given in R_e')
            self.sucess = False

        if not hasattr(CST,'f_e'):
            print('  Fatal Error: CST Attribute Earth Flatness not given in f_e')
            self.sucess = False

        if not hasattr(CST,'gamma_3_4'):
            print('  Fatal Error: CST Attribute Gamma Function Value at 3/4 not given in gamma_3_4')
            self.sucess = False

        if not hasattr(RDB,'Np_burst'):
            print('  Fatal Error: RDB Attribute Number of pulses per burst not given in Np_burst')
            self.sucess = False

        if not hasattr(RDB,'PRF_SAR'):
            print('  Fatal Error: RDB Attribute SAR PRF not given in PRF_SAR')
            self.sucess = False

        if not hasattr(RDB,'BRI'):
            print('  Fatal Error: RDB Attribute Burst Repetition Interval not given in BRI')
            self.sucess = False

        if not hasattr(RDB,'f_0'):
            print('  Fatal Error: RDB Attribute Carrier Frequency not given in f_0')
            self.sucess = False

        if not hasattr(RDB,'Bs'):
            print('  Fatal Error: RDB Attribute Sampled Bandwidth not given in Bs')
            self.sucess = False

        if not hasattr(RDB,'theta_3x'):
            print('  Fatal Error: RDB Attribute 3dB antenna aperture along-track not given in theta_3x')
            self.sucess = False

        if not hasattr(RDB,'theta_3y'):
            print('  Fatal Error: RDB Attribute 3dB antenna aperture across-track not given in theta_3y')
            self.sucess = False

        if not hasattr(OPT,'method'):
            print('  Fatal Error: OPT Attribute method not given in method')
            self.sucess = False

        if not hasattr(OPT,'ftol'):
            print('  Fatal Error: OPT Attribute function exit tolerance not given in ftol')
            self.sucess = False

        if not hasattr(OPT,'xtol'):
            print('  Fatal Error: OPT Attribute x exit tolerance not given in xtol')
            self.sucess = False

        if not hasattr(OPT,'gtol'):
            print('  Fatal Error: OPT Attribute gradient exit tolerance not given in gtol')
            self.sucess = False

        if not hasattr(OPT,'diff_step'):
            print('  Fatal Error: OPT Attribute difference step not given in diff_step')
            self.sucess = False

        if not hasattr(OPT,'max_nfev'):
            print('  Fatal Error: OPT Attribute max evaluated function number not given in max_nfev')
            self.sucess = False

        if not hasattr(OPT,'loss'):
            print('  Fatal Error: OPT Attribute loss function not given in loss')
            self.sucess = False

        self.CST=CST
        self.RDB=RDB
        self.OPT=OPT
        self.CONF = None

        self.RDB.PRI_SAR = 1./self.RDB.PRF_SAR
        self.RDB.lambda_0 = self.CST.c0 / self.RDB.f_0
        self.RDB.dfa = self.RDB.PRF_SAR / self.RDB.Np_burst
        self.CST.ecc_e = np.sqrt((2. - self.CST.f_e)* self.CST.f_e) # Earth Eccentricty
        self.CST.b_e = self.CST.R_e* np.sqrt(1. - self.CST.ecc_e**2)

        self.max_model=1
        self.sucess=True
        print('  Class initialized with success \n')

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    def __Generate_SamosaDDM(self,epoch_ns,SWH,tau,l,GEO):

        """
         Private Method -> __Generate_SamosaDDM

             Input :
                    - self : class self
                    - epoch_ns : input epoch given in nanoseconds
                    - SWH : input SWH in meter
                    - tau : time array  (giving the time of each range gate of the waveform, tau=0 is given at the reference gate)
                    - l : input Doppler Beam index
                    - GEO : structure with fields ... (see method Retrack_Samosa)

             Output:

                    - DDM (Delay-Doppler Map)

        """

        epoch_sec = epoch_ns * 1e-9    ### epoch (convert back in second)

        earth_radius = np.sqrt(self.CST.R_e**2.0 * (np.cos(np.deg2rad(GEO.LAT)))**2 + self.CST.b_e** 2.0 *(np.sin(np.deg2rad(GEO.LAT)))**2)

        tau=tau-epoch_sec              ### tau and epoch are both given in seconds
        Dk = (tau*self.RDB.Bs)
        kappa = (1. + GEO.Height/earth_radius)

        alpha_x = 8. * np.log(2.)  / (GEO.Height**2 * self.RDB.theta_3x**2)
        alpha_y = 8. * np.log(2.)  / (GEO.Height**2 * self.RDB.theta_3y**2)

        if self.CONF.wf_weighted :

            if self.CONF.step == 1 :

                ind = np.argmin(abs(self.alphap_LUT_Wght[:, 0] - SWH))
                alpha_p = self.alphap_LUT_Wght[:, 1][ind]
                Lx = self.CST.c0 * GEO.Height / (2. * GEO.Vs * self.RDB.f_0 * self.RDB.Np_burst * self.RDB.PRI_SAR)

                ind=np.argmin(abs(self.alphapower_LUT_Wght[:, 0] - SWH))
                alpha_power=self.alphapower_LUT_Wght[:, 1][ind]

            elif self.CONF.step == 2 :

                #ind = np.argmin(abs(self.alphap_LUT_Wght[:, 0] - SWH))
                #alpha_p = self.alphap_LUT_Wght[:, 1][ind]
                #Lx = self.CST.c0 * GEO.Height / (2. * GEO.Vs * self.RDB.f_0 * self.RDB.Np_burst * self.RDB.PRI_SAR)

                ind = np.argmin(abs(self.alphap_LUT_NoWght[:, 0] - SWH))
                alpha_p = self.alphap_LUT_NoWght[:, 1][ind]
                Lx = self.CST.c0 * GEO.Height / (2. * GEO.Vs * self.RDB.f_0 * self.RDB.Np_burst * self.RDB.PRI_SAR) * self.CONF.wght_factor

                ind=np.argmin(abs(self.alphapower_LUT_Wght[:, 0] - SWH))
                alpha_power=self.alphapower_LUT_Wght[:, 1][ind]

        elif ~self.CONF.wf_weighted:

            ind = np.argmin(abs(self.alphap_LUT_NoWght[:, 0] - SWH))
            alpha_p = self.alphap_LUT_NoWght[:, 1][ind]
            Lx = self.CST.c0 * GEO.Height / (2. * GEO.Vs * self.RDB.f_0 * self.RDB.Np_burst * self.RDB.PRI_SAR)

            ind=np.argmin(abs(self.alphapower_LUT_NoWght[:, 0] - SWH))
            alpha_power=self.alphapower_LUT_NoWght[:, 1][ind]

        else:

            print('  Waveform Weighting Flag given in input ' + self.CONF.wf_weighted + ' not recognized')
            return np.nan * np.ones( (len(Dk),len(l)) )

        Ly = np.sqrt(self.CST.c0 * GEO.Height / (kappa * self.RDB.Bs))
        Lz = self.CST.c0 / (2. * self.RDB.Bs)
        Lg = kappa/ (2.*GEO.Height*alpha_y)

        sigma_s = (SWH/ (4. * Lz))
        sigma_z = (SWH/ 4.)

        yk = 0 * Dk
        yk[np.where(Dk > 0)] = Ly*np.sqrt(Dk[np.where(Dk > 0)])

        xl = Lx * l

        orbit_slope = GEO.track_sign*((self.CST.R_e**2 - self.CST.b_e**2)/ (2. * earth_radius**2))* np.sin(np.deg2rad(2. * GEO.LAT)) - \
                      (-GEO.Hrate/GEO.Vs)

        ls = self.CONF.flag_slope*orbit_slope* GEO.Height/ (kappa* Lx)

        gl = 1./ np.sqrt(alpha_p**2 + 4. * (alpha_p**2) * (Lx/ Ly)**4 * (l - ls)**2 + np.sign(SWH)* (SWH/ (4. * Lz))**2)

        csi = gl[None,:] * Dk[:,None]

        z = 1. / 4. * csi** 2

        xp = +GEO.Height * GEO.Pitch
        yp = -GEO.Height * GEO.Roll

        Gamma_0 = np.exp(-alpha_y*yp**2 - alpha_x* (xl[None,:] - xp)**2 - xl[None,:]**2*GEO.nu/ GEO.Height**2 -
                      (alpha_y + GEO.nu/GEO.Height**2)*yk[:,None]**2)* np.cosh(2.*alpha_y*yp*yk[:,None])

        T_kappa = np.zeros(np.shape(z))
        T_kappa[np.where(Dk > 0), :]  = ( (1 + GEO.nu/ ((GEO.Height**2)*alpha_y)) - yp/(Ly* np.sqrt(Dk[np.where(Dk > 0)]))* np.tanh(2*alpha_y*yp* Ly*np.sqrt(Dk[np.where(Dk > 0)]))[None,:]).T
        T_kappa[np.where(Dk <= 0),:] = (1 + GEO.nu/ ((GEO.Height**2)*alpha_y)) - 2*alpha_y * yp**2

        csi_max_F0 = np.max(self.F0_LUT[:,0])
        csi_min_F0 = np.min(self.F0_LUT[:, 0])
        clip_F0=np.bitwise_and(csi>=csi_min_F0,csi<=csi_max_F0)

        f0 = np.zeros(np.shape(z))
        Index = np.floor((len(self.F0_LUT[:,0]) - 1) * ((csi[clip_F0] - csi_min_F0) / (csi_max_F0 - csi_min_F0))).astype(np.int)
        f0[clip_F0]=(csi[clip_F0] - self.F0_LUT[Index,0])*((self.F0_LUT[Index + 1,1] - self.F0_LUT[Index,1]) /(self.F0_LUT[Index + 1,0] - self.F0_LUT[Index,0])) + self.F0_LUT[Index,1]
        f0[np.where(csi>csi_max_F0)]=1./2.*np.sqrt(np.pi)/(z[np.where(csi>csi_max_F0)])**(1./4)*(1.+3./(32.*z[np.where(csi>csi_max_F0)])+105./(2048.*(z[(csi>csi_max_F0)])**2) + 10395./(196608.*(z[np.where(csi>csi_max_F0)])**3))
        f0[np.where(csi == 0)] = (1./2.)*(np.pi*2**(3./4.))/(2.*self.CST.gamma_3_4)
        f0[np.where(csi < csi_min_F0)]=0

        csi_max_F1 = np.max(self.F1_LUT[:,0])
        csi_min_F1 = np.min(self.F1_LUT[:, 0])
        clip_F1 = np.bitwise_and(csi >= csi_min_F1, csi <= csi_max_F1)

        f1 = np.zeros(np.shape(z))
        Index = np.floor((len(self.F1_LUT[:,0]) - 1) * ((csi[clip_F1] - csi_min_F1) / (csi_max_F1 - csi_min_F1))).astype(np.int)
        f1[clip_F1]=(csi[clip_F1] - self.F1_LUT[Index,0])*((self.F1_LUT[Index + 1,1] - self.F1_LUT[Index,1]) /(self.F1_LUT[Index + 1,0] - self.F1_LUT[Index,0])) + self.F1_LUT[Index,1]
        f1[ np.where(csi > csi_max_F1)] = (1./2.)*1. / 4. * np.sqrt(np.pi)/ (z[ np.where(csi > csi_max_F1) ])** (3. / 4.)
        f1[np.where(csi == 0)] = -(1./2.)*(2. ** (3. / 4. )) * self.CST.gamma_3_4 / 2.
        f1[np.where(csi < csi_min_F1)] = 0

        f = (f0 + sigma_z / Lg * T_kappa * gl * sigma_s * f1)

        const=np.sqrt(2.*np.pi*alpha_power**4)

        return const*np.sqrt(gl)*Gamma_0*f

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------

    def __Compute_Residuals(self,guess_triplet, tau, wf_norm, LookAngles, MaskRanges, GEO):

        """
         Private Method -> __Compute_Residuals

             Input :

                    - self : class self
                    - guess_triplet : triplet of guess epoch (in ns), SWH (or nu for second step of SAMOSA+), and Pu
                    - tau : time array  (giving the time of each range gate of the waveform, tau=0 is given at the reference gate)
                    - wf_norm : input normalized waveform
                    - LookAngles : input Look Angle Array of each Doppler Beam
                    - MaskRanges : input Mask Range Array
                    - GEO : structure with fields ... (see method Retrack_Samosa)

             Output :

                    - residuals : residuals between model waveform and data waveform
        """

        wf_zp=len(tau)/self.RDB.Npulse
        dr=self.CST.c0/(2*self.RDB.Bs*wf_zp)

        earth_radius = np.sqrt(self.CST.R_e ** 2.0 * (np.cos(np.deg2rad(GEO.LAT))) ** 2 + self.CST.b_e ** 2.0 * (np.sin(np.deg2rad(GEO.LAT))) ** 2)
        kappa = (1. + GEO.Height / earth_radius)

        if LookAngles is None:

            dtheta = GEO.Vs* self.RDB.BRI/ ( GEO.Height * kappa )
            Theta1 = np.pi / 2 + dtheta * self.CONF.N_Look_min
            Theta2 = np.pi / 2 + dtheta * self.CONF.N_Look_max
            LookAngles = np.rad2deg(np.arange(Theta1,Theta2,dtheta))

        DopFreqs = (2*GEO.Vs / self.RDB.lambda_0) * np.cos( np.deg2rad(LookAngles) )
        BeamIndex=np.around(self.CONF.beamsamp_factor*DopFreqs / self.RDB.dfa)/self.CONF.beamsamp_factor
        span=np.where(np.diff(BeamIndex,axis=0)==0)
        BeamIndex=np.delete(BeamIndex,span)

        if self.CONF.rtk_type=='samosa' or self.CONF.step == 1 :

            epoch_ns = guess_triplet[0]
            SWH = guess_triplet[1]
            Pu = guess_triplet[2]

            DDM=self.__Generate_SamosaDDM(epoch_ns,SWH,tau, BeamIndex, GEO)

        elif self.CONF.rtk_type=='samosa+' and self.CONF.step == 2:

            epoch_ns = guess_triplet[0]
            GEO.nu = guess_triplet[1]
            Pu = guess_triplet[2]

            DDM = self.__Generate_SamosaDDM(epoch_ns, 0, tau, BeamIndex, GEO)

        else:

            print('  SAMOSA Retracker Generation given in input ' + self.CONF.rtk_type + ' not recognized')
            return np.nan * np.ones(np.shape(wf_norm))

        if  MaskRanges is None :

            Lx = self.CST.c0 * GEO.Height / (2. * GEO.Vs * self.RDB.f_0 * self.RDB.Np_burst * self.RDB.PRI_SAR)
            MaskRanges_demin = GEO.Height * ( np.sqrt(1 + (kappa * ( (Lx * BeamIndex)  / GEO.Height)**2)) - 1 )

        else:

            MaskRanges = np.delete(MaskRanges, span)
            MaskRanges_demin = MaskRanges - min(MaskRanges)

        R  = np.tile( MaskRanges_demin, (len(wf_norm), 1) )
        Dr = np.tile( dr * np.arange(len(wf_norm)-1,-1,-1),(len(BeamIndex),1) ).T

        DDM[np.where(R >= Dr)] = 0

        Pr = np.sum(DDM, 1) / len(BeamIndex)

        self.max_model = max(Pr)

        Pr = Pu * (Pr/ max(Pr)) + GEO.ThN_norm

        residuals=Pr-np.squeeze(wf_norm)

        return residuals

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------

    def Retrack_Samosa(self,tau,wf,LookAngles,MaskRanges,GEO,CONF):

        """
         Public Method -> Retrack_Samosa -> method to retrack a SAR (Unfocused) Altimetry waveform by SAMOSA or SAMOSA+ retracker

             Input :
                    - self : class self
                    - tau : time array  (giving the time of each range gate of the waveform, tau=0 is given at the reference gate)
                    - wf : input waveform
                    - LookAngles : input Look Angle Array of each Doppler Beams in degrees, set it to None if you dont have this input ( generated looks will be
                                   counted between N_Look_min and N_Look_max, both given in CONF)
                    - MaskRanges : input Mask Range Array in meter, set it to None if you dont have this input (in this case they will be
                                    autonomously computed by the library)
                    - GEO  : structure with fields: LAT (latitude in deg), LON (longitude in deg), Height (Orbit Height in m),
                             Vs (Satellite Velocity in m/sec), Hrate (Orbit Height Rate in m/sec), Pitch (Altimeter Pitch in radiant),
                             Roll (Altimeter Roll in radiant), nu (inverse of mean square slope), ThN (Thermal Noise)
                             track_sign (if Track Ascending => -1, if Track Descending => +1, set it to zero if flag_slope=0 in CONF )
                    - CONF : structure with fields: flag_slope (flag to include in the model the slope of orbit and surface), wf_weighted (set it to True if waveform is weighted)
                             beamsamp_factor (1 means only one beam per resolution cell is generated in the DDM), N_Look_min (number of the first Look to generate in the DDM),
                             N_Look_max (number of the last Look to generate in the DDM), guess_epoch (first-guess epoch in second), guess_swh (first-guess swh in m),
                             guess_pu (first-guess Pu), guess_nu (first-guess nu), lb_epoch (lower bound on epoch in sec), lb_swh (lower bound in swh in m),
                             lb_pu (lower bound on Pu), lb_nu (lower bound on nu), ub_epoch (upper bound on epoch in sec), ub_swh (upper bound in swh in m),
                             ub_pu (upper bound on Pu), ub_nu (upper bound on nu), rtk_type (it can be 'samosa' to retrack the waveform with SAMOSA retracker
                             or it can be 'samosa+' to retrack the waveform with SAMOSA+ retracker)

             Output :

                    - epoch in seconds
                    - SWH in meter
                    - Amplitude Pu
                    - misfit
                    - ocean-like flag (1 means openocean, 0 means non-openocean)

        """
        if not hasattr(GEO,'LAT'):

            print('  Fatal Error: GEO Attribute Latitude not given in LAT -> output padded to nan')
            out = type('', (), {})();out.x = np.full([5], np.nan);return out.x[0],out.x[1],out.x[2],out.x[3],out.x[4]

        if not hasattr(GEO, 'LON'):
            print('  Fatal Error: GEO Attribute Longitude not given in LON -> output padded to nan')
            out = type('', (), {})();out.x = np.full([5], np.nan);return out.x[0],out.x[1],out.x[2],out.x[3],out.x[4]

        if not hasattr(GEO, 'Height'):
            print('  Fatal Error: GEO Attribute Orbit Height not given in Height -> output padded to nan')
            out = type('', (), {})();out.x= np.full([5], np.nan);return out.x[0],out.x[1],out.x[2],out.x[3],out.x[4]

        if not hasattr(GEO, 'Vs'):
            print('  Fatal Error: GEO Attribute Satellite Velocity not given in Vs -> output padded to nan')
            out = type('', (), {})();out.x = np.full([5], np.nan);return out.x[0],out.x[1],out.x[2],out.x[3],out.x[4]

        if not hasattr(GEO, 'Hrate'):
            print('  Fatal Error: GEO Attribute Orbit Height Rate not given in Hrate -> output padded to nan')
            out = type('', (), {})();out.x = np.full([5], np.nan);return out.x[0],out.x[1],out.x[2],out.x[3],out.x[4]

        if not hasattr(GEO, 'Pitch'):
            print('  Fatal Error: GEO Attribute Pitch not given in Pitch -> output padded to nan')
            out = type('', (), {})();out.x = np.full([5], np.nan);return out.x[0],out.x[1],out.x[2],out.x[3],out.x[4]

        if not hasattr(GEO, 'Roll'):
            print('  Fatal Error: GEO Attribute Roll not given in Roll -> output padded to nan')
            out = type('', (), {})();out.x = np.full([5], np.nan);return out.x[0],out.x[1],out.x[2],out.x[3],out.x[4]

        if not hasattr(GEO, 'nu'):
            print('  Fatal Error: GEO Attribute nu not given in nu -> output padded to nan')
            out = type('', (), {})();out.x = np.full([5], np.nan);return out.x[0],out.x[1],out.x[2],out.x[3],out.x[4]

        if not hasattr(GEO, 'track_sign'):
            print('  Fatal Error: GEO Attribute track sign not given in track_sign -> output padded to nan')
            out = type('', (), {})();out.x = np.full([5], np.nan);return out.x[0],out.x[1],out.x[2],out.x[3],out.x[4]

        if not hasattr(GEO, 'ThN'):
            print('  Fatal Error: GEO Attribute Thermal Noise not given in ThN -> output padded to nan')
            out = type('', (), {})();out.x = np.full([5], np.nan);return out.x[0],out.x[1],out.x[2],out.x[3],out.x[4]

        if not self.sucess :
            print('  Fatal Error: SAMOSA Class not initialized with Success: please initialize first the class -> output padded to nan')
            out = type('', (), {})();out.x = np.full([5], np.nan);return out.x[0],out.x[1],out.x[2],out.x[3],out.x[4]

        if not np.ndim(wf)==1 :
            print('  Fatal Error: waveform in wf must have only one dimension -> output padded to nan')
            out = type('', (), {})();out.x = np.full([5], np.nan);return out.x[0],out.x[1],out.x[2],out.x[3],out.x[4]

        if not np.shape(tau)==np.shape(wf) :
            print('  Fatal Error: time in tau and waveform in wf must have the same size -> output padded to nan')
            out = type('', (), {})();out.x = np.full([5], np.nan);return out.x[0],out.x[1],out.x[2],out.x[3],out.x[4]

        if CONF.lb_epoch is None:
            CONF.lb_epoch=tau[0]*1e9

        if CONF.ub_epoch is None:
            CONF.ub_epoch=tau[-1]*1e9

        if CONF.flag_slope:
            CONF.flag_slope=1
        else:
            CONF.flag_slope=0

        CONF.lb = [CONF.lb_epoch, CONF.lb_swh, CONF.lb_pu]
        CONF.ub = [CONF.ub_epoch, CONF.ub_swh, CONF.ub_pu]
        guess_triplet=[CONF.guess_epoch*1e9,CONF.guess_swh,CONF.guess_pu]
        self.CONF = CONF
        self.CONF.step = 1
        wf_norm = wf / max(wf)
        with np.errstate(invalid='ignore')  :
            with np.errstate(divide='ignore'):
                E = - np.nansum(wf_norm ** 2 * np.log2(wf_norm ** 2), axis=0)
        PP = 1. / np.nansum(wf_norm, axis=0)
        wf_zp = len(wf_norm) / self.RDB.Npulse
        GEO.ThN_norm = GEO.ThN / max(wf)

        try:

            out=scipy.optimize.least_squares(self.__Compute_Residuals,guess_triplet, bounds=(self.CONF.lb,self.CONF.ub),loss=self.OPT.loss,
                                             method=self.OPT.method, ftol=self.OPT.ftol, xtol=self.OPT.xtol, gtol=self.OPT.gtol,
                                             max_nfev=self.OPT.max_nfev,args=(tau,wf_norm, LookAngles, MaskRanges, GEO))

            swh = out.x[1]
            misfit = np.sqrt(1. / (len(tau)) * np.sum(out.fun ** 2)) * 100

            if self.CONF.rtk_type == 'samosa+':

                if E * PP < 0.68 or E * PP > 0.78 or (100 * PP) * wf_zp > 8 or (E / misfit) / wf_zp < 4:

                    self.CONF.lb = [CONF.lb_epoch, CONF.lb_nu, CONF.lb_pu]
                    self.CONF.ub = [CONF.ub_epoch, CONF.ub_nu, CONF.ub_pu]
                    self.CONF.step = 2

                    guess_triplet = [CONF.guess_epoch * 1e9, CONF.guess_nu, CONF.guess_pu]

                    out = scipy.optimize.least_squares(self.__Compute_Residuals, guess_triplet, bounds=(self.CONF.lb, self.CONF.ub),loss=self.OPT.loss,
                                                       method=self.OPT.method, ftol=self.OPT.ftol, xtol=self.OPT.xtol, gtol=self.OPT.gtol,
                                                       max_nfev=self.OPT.max_nfev, args=(tau, wf_norm, LookAngles, MaskRanges, GEO))

        except Exception as inst:

            print('  Fatal Error: Catched Exception in retracking: <<' + inst.__str__() + '>> ->output padded to nan')

            out = type('', (), {})();out.x=np.full([5], np.nan);return out.x[0],out.x[1],out.x[2],out.x[3],out.x[4]

        Pu=(out.x[2]*max(wf)/self.max_model).item()
        out.model=out.fun+wf_norm
        oceanlike_flag=~(E * PP < 0.68 or E * PP > 0.78 or (100 * PP) * wf_zp > 8 or (E / misfit) / wf_zp < 4)
        return out.x[0]*1e-9,swh,Pu,misfit,oceanlike_flag

# ==========================================================================================
# ===================   END OF THE C L A S S E S   ===================================
# =========================================================================================