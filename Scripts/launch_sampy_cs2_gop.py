#!/home/salvatore/anaconda3/bin/python3.7
# -*- coding: utf-8 -*-

#### Loading libraries ...
# general python imports
import numpy as np
import xarray as xr
import argparse
import os
import time
# specific SAMPy imports
from sampy import SAMOSA as initialize_SAMOSAlib
from sampy import initialize_epoch, compute_ThNEcho


def get_netcdf_data(fname):
    '''
    Retrieve and reads input file fname
    Input Data are assumed in the ../Data folder with directory structure
        GOP/YYYY/MM/*R1B*.nc
    Return an xarray dataset
    '''
    # Get year and month from filename
    fsplit = fname.split('GOPR1B_')
    yyyy = fsplit[-1][:4]
    mm = fsplit[-1][4:6]
    # Define input folder
    folder=os.path.join('..','Data','GOP',yyyy,mm)
    # Read file
    ds = xr.open_dataset(os.path.join(folder,fname), decode_timedelta=False)
    return ds


def main(fname):
    ##################################################################################
    # Read input file
    ds = get_netcdf_data(fname)
    ##################################################################################
    # Read parameters needed by SAMOSA+
    look_angle_start_20_hr_ku=ds['look_angle_start_20_hr_ku'][:].data.astype(np.float64)
    look_angle_stop_20_hr_ku=ds['look_angle_stop_20_hr_ku'][:].data.astype(np.float64)
    stack_number_after_weighting_20_hr_ku=ds['stack_number_after_weighting_20_hr_ku'][:].data.astype(np.float64)
    lon_20_hr_ku=ds['lon_20_hr_ku'][:].data.astype(np.float64)
    lat_20_hr_ku=ds['lat_20_hr_ku'][:].data.astype(np.float64)
    off_nadir_pitch_angle_str_20_hr_ku=np.pi/180.*ds['off_nadir_pitch_angle_str_20_hr_ku'][:].data.astype(np.float64)
    off_nadir_roll_angle_str_20_hr_ku=np.pi/180.*ds['off_nadir_roll_angle_str_20_hr_ku'][:].data.astype(np.float64)
    orb_alt_rate_20_hr_ku=ds['orb_alt_rate_20_hr_ku'][:].data.astype(np.float64)
    sat_vel_vec_20_hr_ku=ds['sat_vel_vec_20_hr_ku'][:].data.astype(np.float64)
    sat_vel_vec_20_hr_ku=np.sqrt(sat_vel_vec_20_hr_ku[:,0]**2 + sat_vel_vec_20_hr_ku[:,1]**2+sat_vel_vec_20_hr_ku[:,2]**2)
    alt_20_hr_ku=ds['alt_20_hr_ku'][:].data
    pwr_waveform_20_hr_ku=ds['pwr_waveform_20_hr_ku'][:].data.astype(np.float64).T
    window_del_20_hr_ku=ds['window_del_20_hr_ku'][:].data.astype(np.float64)
    uso_cor_20_hr_ku=ds['uso_cor_20_hr_ku'][:].data.astype(np.float64)
    dop_cor_20_hr_ku=ds['dop_cor_20_hr_ku'][:].data.astype(np.float64)
    ##################################################################################

    ### CST is a structure collecting universal constants

    CST = type('', (), {})()

    CST.c0=299792458.                   ## speed of light in m/sec
    CST.R_e=6378137.                    ## Reference Ellipsoid Earh Radius in m
    CST.f_e=1/298.257223563             ## Reference Ellipsoid Earth Flatness
    CST.gamma_3_4=1.2254167024651779    ## Gamma Function Value at 3/4

    ### OPT is a structure collecting parameters relative to the minimization scheme settings

    OPT = type('', (), {})()

    OPT.method='trf'               ## acronym of the minimization solver, see scipy.optimize.least_squares for details
    OPT.ftol=1e-2                  ## exit tolerance on f
    OPT.gtol=1e-2                  ## exit tolerance on gradient norm of f
    OPT.xtol=2*1e-3                ## exit tolerance on x
    OPT.diff_step=None             ## relative step size for the finite difference approximation of the Jacobian
    OPT.max_nfev=None              ## maximum number of function evaluations
    OPT.loss='linear'              ## loss function , see scipy.optimize.least_squares for details

    ### RDB is a structure collecting parameters relative to the sensor radar database

    RDB = type('', (), {})()

    RDB.Np_burst=64                        # number of pulses per burst
    RDB.Npulse=128                         # number of the range gates per pulse (without zero-padding)
    RDB.PRF_SAR=18181.8181818181           # Pulse Repetition Frequency in SAR mode , given in Hz
    RDB.BRI=0.0117929625                   # Burst Repetition Interval, given in sec
    RDB.f_0=13.575e9                       # Carrier Frequency in Hz
    RDB.Bs=320e6                           # Sampled Bandwidth in Hz
    RDB.theta_3x=np.deg2rad(1.10)         # (rad) Antenna 3 dB beamwidth (along-track)
    RDB.theta_3y=np.deg2rad(1.22)         # (rad) Antenna 3 dB beamwidth (cross-track)

    ### LUT is a structure collecting filenames of the all SAMOSA LUT
    ### All the LUT files must be in a folder named auxi and located in the same folder as sampy.py

    LUT = type('', (), {})()

    LUT.F0='LUT_F0.txt'                                                                           ## filename of the F0 LUT
    LUT.F1='LUT_F1.txt'                                                                           ## filename of the F1 LUT
    LUT.alphap_noweight='alphap_table_DX3000_ZP20_SWH20_10_Sept_2019(CS2_NOHAMMING).txt'          ## filename of the alphap LUT ( case no weighting)
    LUT.alphap_weight='alphap_table_DX3000_ZP20_SWH20_10_Sept_2019(CS2_HAMMING).txt'              ## filename of the alphap LUT ( case weighting)
    LUT.alphapower_noweight='alphaPower_table_CONSTANT_SWH20_10_Feb_2020(CS2_NOHAMMING).txt'      ## filename of the alpha power LUT ( case no weighting)
    LUT.alphapower_weight='alphaPower_table_CONSTANT_SWH20_10_Feb_2020(CS2_NOHAMMING).txt'        ## filename of the alpha power LUT ( case weighting)

    ### time array tau : it gives the relative time of each range gate of the radar waveform with respect a time zero
    ### time zero corresponds at the time of the reference gate

    wf_zp=np.shape(pwr_waveform_20_hr_ku)[0]/RDB.Npulse           #### zero-padding factor of the waveform
    Nstart = RDB.Npulse * wf_zp
    Nend  = RDB.Npulse * wf_zp
    dt = 1. / (RDB.Bs * wf_zp)                                    #### time sampling step for the array tau, it includes the zero-padding factor
    tau=np.arange(-(Nstart/2)*dt,((Nend-1)/2)*dt,dt)

    NstartNoise = 2    ## noise range gate counting from 1, no oversampling
    NendNoise   = 6    ## noise range gate counting from 1, no oversampling

    window_del_20_hr_ku_deuso=window_del_20_hr_ku*(uso_cor_20_hr_ku+1)
    Raw_Elevation=alt_20_hr_ku-CST.c0/2*window_del_20_hr_ku_deuso

    ThNEcho=compute_ThNEcho(pwr_waveform_20_hr_ku,NstartNoise*wf_zp,NendNoise*wf_zp)        ### computing Thermal Noise from the waveform matric
    epoch0=initialize_epoch(pwr_waveform_20_hr_ku,tau,Raw_Elevation,CST,size_half_block=10) ### initializing the epoch (first-guess epoch) from the waveform matrix

    samlib=initialize_SAMOSAlib(CST,RDB,OPT,LUT)        #### initializing the SAMOSA library sampy, it's a mandatory step

    epoch_sec=np.full(np.shape(pwr_waveform_20_hr_ku)[1], np.nan,dtype = 'float64')           ### allocation of the output array
    SWH=np.full(np.shape(pwr_waveform_20_hr_ku)[1], np.nan,dtype = 'float64')                 ### allocation of the output array
    Pu=np.full(np.shape(pwr_waveform_20_hr_ku)[1], np.nan,dtype = 'float64')                  ### allocation of the output array
    misfit=np.full(np.shape(pwr_waveform_20_hr_ku)[1], np.nan,dtype = 'float64')              ### allocation of the output array
    oceanlike_flag=np.full(np.shape(pwr_waveform_20_hr_ku)[1], np.nan,dtype = 'float64')      ### allocation of the output array

    ### loop cycle for each record in the product
    T0 = time.perf_counter()
    for k in np.arange( np.shape(pwr_waveform_20_hr_ku)[1] ):

        ### GEO is a structure collecting geophysical input which are varying with the waveform under iteration

        GEO = type('', (), {})()

        GEO.LAT=np.squeeze(lat_20_hr_ku)[k]                              ### latitude in degree for the waveform under iteration
        GEO.LON=np.squeeze(lon_20_hr_ku) [k]                             ### longitude in degree between -180, 180 for the waveform under iteration
        GEO.Height=np.squeeze(alt_20_hr_ku)[k]                           ### Orbit Height in meter for the waveform under iteration
        GEO.Vs=np.squeeze(sat_vel_vec_20_hr_ku)[k]                       ### Satellite Velocity in m/s for the waveform under iteration
        GEO.Hrate=np.squeeze(orb_alt_rate_20_hr_ku)[k]                   ### Orbit Height rate in m/s for the waveform under iteration
        GEO.Pitch=np.squeeze(off_nadir_pitch_angle_str_20_hr_ku)[k]      ### Altimeter Reference Frame Pitch in radiant
        GEO.Roll=np.squeeze(off_nadir_roll_angle_str_20_hr_ku)[k]        ### Altimeter Reference Frame Roll in radiant
        GEO.nu=0                                                         ### Inverse of the mean square slope
        GEO.track_sign=0                                                 ### if Track Ascending => -1, if Track Descending => +1, set it to zero if flag_slope=False in CONF
        GEO.ThN=np.squeeze(ThNEcho)[k]                                   ### Thermal Noise

        ### CONF is a structure collecting input which are relative the SAMOSA retracking configuration
        ### These parameters can be set differently according to the waveform under iteration

        CONF = type('', (), {})()

        CONF.flag_slope = False                    ### flag True commands to include in the model the slope of orbit and surface (this effect usually is included in LookAngles Array)
        CONF.beamsamp_factor = 1                   ### 1 means only one beam per resolution cell is generated in the DDM, the other ones are decimated
        CONF.wf_weighted = False                   ### flag True if the waveform under iteration is weighted
        CONF.N_Look_min = -90                      ### number of the first Look to generate in the DDM (only used if LookAngles array is not passed in input: i.e. set to  None)
        CONF.N_Look_max = 90                       ### number of the last Look to generate in the DDM (only used if LookAngles array is not passed in input: i.e. set to  None)
        CONF.guess_epoch = epoch0[k]               ### first-guess epoch in second
        CONF.guess_swh = 2                         ### first-guess SWH in meter
        CONF.guess_pu = 1                          ### first-guess Pu
        CONF.guess_nu = 2                          ### first-guess nu (only used in second step of SAMOSA+)
        CONF.lb_epoch = None                       ### lower bound on epoch in sec. If set to None, lower bound will be set to the first time in input array tau
        CONF.lb_swh = -0.5                         ### lower bound on SWH in m
        CONF.lb_pu = 0.2                           ### lower bound on Pu
        CONF.lb_nu = 0                             ### lower bound on nu (only used in second step of SAMOSA+)
        CONF.ub_epoch = None                       ### upper bound on epoch in sec. If set to None, upper bound will be set to the last time in input array tau
        CONF.ub_swh = 30                           ### upper bound on SWH in m
        CONF.ub_pu = 1.5                           ### upper bound on Pu
        CONF.ub_nu = 1e9                           ### upper bound on nu (only used in second step of SAMOSA+)
        CONF.rtk_type = 'samosa+'                  ### choose between 'samosa' or 'samosa+'
        CONF.wght_factor= 1.4705                   ### widening factor of PTR main lobe after Weighting Window Application

        wf=np.squeeze(pwr_waveform_20_hr_ku)[:,k]   ### SAR waveform

        MaskRanges=None  ## if you put MaskRangs to None, SAMOSA library will try to autimatically compute it
        LookAngles=90-np.linspace( np.rad2deg(look_angle_start_20_hr_ku[k]),np.rad2deg(look_angle_stop_20_hr_ku[k]),
                             num=int(stack_number_after_weighting_20_hr_ku[k]), endpoint=True )


        #### invoke, from the initialized class, the SAMOSA retrack method
        #### input: time tau, waveform wf, LookAngles array, MaskRanges array, structure GEO, structure CONF
        #### In case LookAngles and MaskRanges are not available, you can set one of them or both of them to None: an approximation of them will be computed by the method
        epoch_sec[k],SWH[k],Pu[k],misfit[k],oceanlike_flag[k]=samlib.Retrack_Samosa(tau,wf,LookAngles,MaskRanges,GEO,CONF)

        if k%100==0 :
            print(f'  Record# {k:d} of {np.shape(pwr_waveform_20_hr_ku)[1]:d}')

    SSHunc=Raw_Elevation-epoch_sec*CST.c0/2                 #### this should be the sea surface height without geo-corrections
    print('  Total processing time: ' +str(time.perf_counter() - T0) + ' seconds')
    # define output dataset
    outds = xr.Dataset({'SSHunc': (['time_20_hr_ku'],SSHunc),
			'Pu': (['time_20_hr_ku'],Pu),
			'misfit' : (['time_20_hr_ku'],misfit),
			'oceanlike_flag' : (['time_20_hr_ku'],oceanlike_flag),
                        'SWH' : (['time_20_hr_ku'],SWH)},
			coords = {'time_20_hr_ku': ds.time_20_hr_ku.values,
                                  'lon_20_hr_ku' : (['time_20_hr_ku'],lon_20_hr_ku),
                                  'lat_20_hr_ku' : (['time_20_hr_ku'],lat_20_hr_ku)},
			attrs= {'description':"Parameters from SAMOSA+ retracker"})
    # Define same folder structure as input data
    outdir = os.path.join('..','Processed','GOP')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    # Get year and month from filename
    fsplit = fname.split('GOPR1B_')
    yyyy = fsplit[-1][:4]
    mm = fsplit[-1][4:6]
    outdir = os.path.join(outdir,yyyy)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    outdir = os.path.join(outdir,mm)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    # Define outpout filename
    fnameout = f'SAMPy_{fname}'
    # Save output
    outds.to_netcdf(os.path.join(outdir,fname))


if __name__=='__main__':
    '''
    Script to launch SAMPy retracker on the waveforms contained in the L1b granule specified in input
    Input Data are assumed in the ../Data folder with directory structure
        GOP/YYYY/MM/*R1B*.nc
    Output netcdf with retracekd variables will be saved in ../Processed

    The example has been tested on GOP L1b files
    >> python launch_sampy_cs2_gop.py --file CS_OFFL_SIR_GOPR1B_20200815T110723_20200815T111407_C001.nc
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",type=str,help="L1b input CryoSat file to be retracked",required=True)
    args = parser.parse_args()
    fname = args.file
    main(fname)
