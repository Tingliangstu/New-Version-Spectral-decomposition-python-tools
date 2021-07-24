 #  Spectral decomposition of thermal conductance tools
 ## This is an improved project

**Due to my previous [stupidity](https://github.com/Tingliangstu/Spectral-decomposition-python-tools), I limited the scope of the project. The previous project needed to ensure that the number of atoms on the left and right ends of the illusion interface was equal. But in principle it is not required.**

## Need to be modified files

- [x] relax_thermal.in (input file for lammps, and use to get the atoms velocity)
- [x] forces.in (input file for force_calculate.py, in order to call the python interface of lammps)
- [x] SHC_generate.py (Main program, useful for calculating spectral decomposition thermal conductance)

## Usage

#### 1. According to one specific needs, modify the potential format in the files (relax_thermal.in and forces.in)

- [x] **Modify in relax_thermal.in and forces.in file** 

```python
# ************ Modify in relax_thermal.in and forces.in file ************
# *********************  Potential  function setting  *****************

pair_style                 tersoff
pair_coeff            *      *           BNC.tersoff        B        C       N
```

```python
# ************ Modify in relax_thermal.in and forces.in file ************
#  *************************  For Y direction  **************************************

variable         y_max        equal           ly              # Depends on the direction of heat transport
variable         P            equal     ${y_max}/2-100
variable         P1           equal     ${y_max}/2+100

variable         tmp1         equal     ${P1}-${P}
variable         tmp          equal     ${tmp1}/40

variable         L1           equal     ${P}+3*${tmp}
variable         R1           equal     ${P1}-3*${tmp}
```

- [x] **Modify in relax_thermal.in file** 

```python
variable        dmid         equal            6         ## Set to 6 (A) here, one can modify it
variable        middle       equal        ${y_max}/2 
variable        mid_left     equal     ${middle}-${dmid}
variable        mid_right    equal     ${middle}+${dmid}
```

- [x] **Modify in relax_thermal.in file (For `dn` parameter)**

```python
# **************************  Collect Velocities for the calculation of force constants *************************

variable      dn     equal    15            

dump          vels   interface   custom    ${dn}   vels.dat   id   type   vx   vy   vz
dump_modify   vels   format      line      "%d   %d   %0.8g   %0.8g   %0.8g"
dump_modify   vels   sort        id
```



#### 2. Read the calling part at the end of the SHC_generate.py file and modify it according to needs.

```python
    postprocessor = SHCPostProc(compact_velocities_file,
                                fileprefix,            # forces.in
                                dt_md = md_timestep,   # timestep
                                dn = dn,               # Dump interval
                                scaleFactor = unit_scaling_factor,
                                LAMMPSDumpFile = atomic_velocities_file, # velocity file
                                widthWin = frequency_window_width,
                                LAMMPSInFile = in_file,
                                in_plane = False,
                                out_of_plane = False,
                                reCalcVels = True,
                                reCalcFC = True)
```



#### 3. Compile the compactify_vels.cpp file, and add the generated compactify_vels to the command path.

```c++
g++  compactify_vels.cpp  -o  compactify_vels
```

#### 4. **run**

- **lmp_mpi < relax_thermal.in**

- **python SHC_generate.py**

- ------

  ***Maybe one need to modify the input file according to the error report***
  
  *If you are lucky, maybe you don’t need to modify anything*
  
  

## Output

#### 1. Spectral thermal conductance
<div align=center><img src="https://github.com/Tingliangstu/New-Version-Spectral-decomposition-python-tools/blob/master/SHC_calculate/Fij.dat_SHC.png" style="zoom: 50%;" />



​                                                        **This picture is very similar to the [reference](https://doi.org/10.1016/j.ijheatmasstransfer.2019.118608) (see below)**

![](https://github.com/Tingliangstu/New-Version-Spectral-decomposition-python-tools/blob/master/Ref_paper/ref_fig.jpg)
 #### 2. Accumulated thermal conductance
<div align=center><img src="https://github.com/Tingliangstu/New-Version-Spectral-decomposition-python-tools/blob/master/SHC_calculate/Fij.dat_accumulated_ITC.png" style="zoom:50%;" />
