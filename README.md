# SAMPy

Python implementation of the SAMOSA+ retracker developed within ESA Cryo-TEMPO project

### Folder structure
The repository contains 3 folders:
- `Data`    
    Folder containing the L1b CS granules to process    
    The subfolder structure within the folder should be (i.e. for the GOP product):    
    `GOP\YYYY\MM\*.nc`    
- `Processed`    
    Folder containing the output netcdf with the SAMOSA+ retracked variables    
    (the subfolder structure is analogous to that in `Data`)    
- `Scripts`    
    Folder containing the python implementation of the SAMOSA+ retracker and the script to launch it

### How to use it
To launch the SAMOSA+ retracker on a given L1b granule (i.e. `CS_OFFL_SIR_GOPR1B_20200815T110723_20200815T111407_C001.nc`) simply run   
```python
python launch_sampy_cs2_gop.py --file CS_OFFL_SIR_GOPR1B_20200815T110723_20200815T111407_C001.nc
```

The code has been tested so far on GOP granules.
It is possible that some of the names of the variables used by SAMPy might be modified for the ICE products.   

### General info
- SAMPy makes use of **[xarray](http://xarray.pydata.org/en/stable/)** to read and save netcdf files.
- For the development of the repository I propose to follow the **Gitflow workflow** described **[here](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)**    
   (comments/altrnatives to this are more than welcomed)

<img src="https://wac-cdn.atlassian.com/dam/jcr:61ccc620-5249-4338-be66-94d563f2843c/05%20(2).svg?cdnVersion=1637" width="75%" hover="Gitflow Workflow schematics"/>
