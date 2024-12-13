#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Kimmo Sääskilahti
ksaaskil@gmail.com

Made some modifications by Liang Ting
liangting.zj@gmail.com
2021/7/23 14:22:15
"""
import time
import numpy as np
import sys, os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from lammps import lammps

__all__ = ["fcCalc"]

def initialize_lammps(in_file):

    lmp = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    with open(in_file, 'r') as file:
        for line in file:
            if "quit" not in line:
                lmp.command(line)
    return lmp

def compute_force_constants(hstep, inds_left, inds_right, i1, in_file):
    """
    Prepare the LAMMPS object for computing force constants.
    Uses the `Python library interface <https://lammps.sandia.gov/doc/Python_library.html>`_
    of LAMMPS so you need to have (1) `lammps` in your `PYTHONPATH` and
    """
    lmp = initialize_lammps(in_file)
    ind1 = inds_left[i1]
    results = np.zeros((3, len(inds_right) * 3))
    inds_right_1d = np.concatenate((3 * inds_right, 3 * inds_right + 1, 3 * inds_right + 2))
    inds_right_1d = np.sort(inds_right_1d)

    for direction in [0, 1, 2]:
        index = 3 * ind1 + direction
        yc = lmp.gather_atoms("x", 1, 3)
        yc[index] += hstep
        lmp.scatter_atoms("x", 1, 3, yc)
        lmp.command("run 0 post no")
        fc1 = lmp.gather_atoms("f", 1, 3)
        fc1 = np.array(fc1, dtype=np.dtype('f8'))

        yc[index] -= 2 * hstep
        lmp.scatter_atoms("x", 1, 3, yc)
        lmp.command("run 0 post no")
        fc2 = lmp.gather_atoms("f", 1, 3)
        fc2 = np.array(fc2, dtype=np.dtype('f8'))

        results[direction, :] = (fc2[inds_right_1d] - fc1[inds_right_1d]) / (2.0 * hstep)
        yc[index] += hstep
        lmp.scatter_atoms("x", 1, 3, yc)

    lmp.close()
    print("\t\t\t Finished moving atom %i / %i" % (i1 + 1, len(inds_left)))
    
    return i1, results

class fcCalc(object):
    """
    Class for computing force constants between atoms.
    Uses the `Python library interface <https://lammps.sandia.gov/doc/Python_library.html>`_
    of LAMMPS so you need to have (1) `lammps` in your `PYTHONPATH` and
    (2) `liblammps.so` available for the Python package.
    :param fileprefix: File prefix (TODO What is this)
    :type fileprefix: str
    :param restartfile: LAMMPS restart file (The restart file is also required, but here not as an attribute)  
    :type restartfile: str
    """
    def __init__(self, fileprefix):

        self.fileprefix = fileprefix
        self.Kij = None
        self.inds_left = None                  ## Used to find the index of the atoms on the left
        self.inds_right = None                 ## Used to find the index of the atoms on the right
        self.inds_interface = None             ## Used to compare with the atomic index in the dump_velovity file
        self.ids_L = None                      ## Used to index the speed ordinal (left)
        self.ids_R = None                      ## Used to index the speed ordinal (right)
        self.lmp = None                        ## Python-Lammps interface

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
        
    def get_ids(self, ids_file):

        if not os.path.exists(ids_file):
           print('\nERROR: file {} not found\n'.format(ids_file))
           exit()
           
        with open(ids_file, 'r') as fin:
           for j in range(3):  # Always skip the first three lines
              fin.readline()
           N_atoms = int(fin.readline().strip().split()[0])
           ids = []
           for j in range(5):  # Always skip the first six lines
              fin.readline()
           for k in range(N_atoms):
              ids.append(int(fin.readline().strip().split()[0]))
           fin.close()
        
        return np.array(ids)
        
    def preparelammps(self, in_lammps = None, group_name_L = "NL", 
                      group_name_R = "NR", group_name_interface = "N", show_log = False):
        """
        Prepare the LAMMPS object for determine the atoms object.
        :param fileprefix: File prefix (lammps in_file)
        :type fileprefix: str
        :param show_log: Whether to write the log file to the generate_force.log file, if true, no log file will be generated. (default False)
        :type show_log: boolean, optional
        :return: None
        """        
        self.in_lammps = in_lammps
        
        cmd_list = ['-log', 'generate_force.log', '-screen', 'none']
        
        if not show_log:
            cmd_list += ['-echo', 'none']
           
        self.lmp = lammps(cmdargs = cmd_list)
        
        if self.in_lammps is not None:
            pass
        else:
            sys.exit('\n\tFile error: The in file that lammps needs to read does not exist!\n')

        #lines = open(self.in_lammps,'r').readlines()
        #for line in lines: self.lmp.command(line) 
        """
        Do the same as lmp.command(filename) but don't allow the script to be
        continued after quit.
        """
        lines = open(self.in_lammps,'r').readlines()
        for line in lines:
           #print(line)
           if "quit" in line and line[0] != "#":
              sys.exit('\n\tQuit error: It is not appropriate '
                       'to exit the execution of the in.file at this time\n')    ## Quit lammps if find the keyword 'quit'
           else:
              self.lmp.command(line)
              
        '''
        (1) minimize            0    1.0e-3   1000   1000     # (after relaxing the structures to the potential minimum)
            run                 0
            After the structure is completely relaxed, the process of energy minimization must be added
        
        (2) The atomic grouping at the interface is already specified in the lammp in (thermal.in) file, so there is no need to deal with it here
        
        (3) The program will automatically identify if the left and right atom ids are the same as the atom id of the dump_velcity_file.
        '''  

        # Note that these indices differ from atom IDs by a factor of one (1), this is required due to python index from 0
        
        self.inds_left = self.get_ids('dump.left')-1  	 ## the files name must the same as the one in forces.in
        self.inds_right = self.get_ids('dump.right')-1    ## the files name must the same as the one in forces.in
        
        # All atom indices sorted by atom ID, duplicates removed
        
        self.inds_interface_compare = self.get_ids('dump.interface')          ## the files name must the same as the one in forces.in
        self.inds_interface = np.unique(np.concatenate((self.inds_left, self.inds_right))) 
        
        if ((self.inds_interface_compare == self.inds_interface).all()):

            print('\n\tLAMMPS SETUP Warning: The atomic id of dump_interface is not equal' 
        					'to the id added by left and right, which may cause errors! ')
        
        # Where are the atoms of the left atom set

        self.ids_L = np.in1d(self.inds_interface, self.inds_left)     # To find the subset
        self.ids_L = np.where(self.ids_L)[0]                          # or np.where(np.in1d(aArray,bArray))[0]
        
        # Atoms of the right set

        self.ids_R = np.in1d(self.inds_interface, self.inds_right)
        self.ids_R = np.where(self.ids_R)[0]
        
        # Get information from the in file 
        
        N = self.lmp.extract_variable(group_name_interface, "all", 0)
        NL = self.lmp.extract_variable(group_name_L, "all", 0)
        NR = self.lmp.extract_variable(group_name_R, "all", 0)
        
        # Determine whether the number of left and right atoms is equal

        print('\nThe numbers atoms in the left = {}.'.format(int(NL)))
        print('The numbers atoms in the right = {}.'.format(int(NR)))
        
        if (len(self.ids_L) != len(self.ids_R) or len(self.ids_L) != NR or len(self.ids_R) != NL):
            print('\nLAMMPS SETUP Warning: Number of atoms on left and right side don\'t match!')

    def fcCalc(self, hstep, n_cores=None):
        """
         Compute force constants and store to `self.Kij`.
        :param hstep: Step to use in finite differences (the finite displacement method)
        :type hstep: float
        :return: None
        
        This function reads in the [dFi/duj] force constanst from 
        LAMMPS compute commands. The structure of the infile should be blocks
        of data that are N atoms long. Here, N means the number of atoms in
        the interface region. Each atom of the N atoms in the block 
        is the 'i' index in d^2Fi/duj. The Python_LAMMPS code loops over all atoms j
        on the left side of the interface; j in the dFi/duj expression. There
        are NL = NR atoms on the left side of the inteface, i.e. the number on 
        each side should (and MUST) be the same. Each atom on the left side, j, is 
        looped over; in each loop iteration the atom is moved in the +x, then -x, 
        then back to equilibrium. Then the atom is moved in the +y, then -y, then
        back to equilibrium. Finally, the atom is moved in the +z, then -z, then 
        back to equilibrium.
    
        Thus, each line in each block in the file corresponds to an atom i. 
        Each element in each line is the force felt by the atom in the x, y, and 
        z direction (the force vector elements). For each block, the atom j is 
        moved iteratively in the +x, then-x, then +y ... etc.
    
        The function returns kij, the matrix elements of the force constants 
        between atoms on either side of the interface.
        kij has the shape [3*nr,3*nl] -> [dFi-x,y,z,duj-x,y,z]. Each individual 
        force constant is defined as:
        
             dFi_a
           ----------
             drj_b
        
        Which is interpreted as the change in force on atom i in the direction
        a = (x,y,z) due to the movement of atom j in b=(x,y,z).
        Each element of first dimension of the kij matrix corresponds to the 
        force on atom i due to the movement of atom j in each direction 
        
         dF1_x            dF1_x            dF1_x           dF1_x        
        ----------       ----------       ----------     ----------  .....
         dr1_x            dr1_y            dr1_z           dr2_x
        
        
          dF1_y           dF1_y             dF1_y          dF1_y        
        ----------       ----------       ----------     ----------  .....
          dr1_x           dr1_y             dr1_z          dr2_x
        
        
          dF1_z           dF1_z             dF1_z          dF1_z        
        ----------       ----------       ----------     ----------  .....
          dr1_x           dr1_y             dr1_z          dr2_x
          
          
          dF2_x           dF2_x             dF2_x          dF2_x        
        ----------       ----------       ----------     ----------  .....
          dr1_x           dr1_y             dr1_z          dr2_x
        
            :               :                  :               :
            :               :                  :               :    
        """
        inds_left = self.inds_left
        inds_right = self.inds_right

        Kij = np.zeros((len(inds_left) * 3, len(inds_right) * 3))
            
        if n_cores is None:
            n_cores = multiprocessing.cpu_count()

        print(f'\nUsing {n_cores} cores for parallel computation.')				
        print('\n  Now start the loop and move the atoms to calculate the force constant !!!')
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = [executor.submit(compute_force_constants, hstep, inds_left, inds_right, i1, self.in_lammps) for i1 in range(len(inds_left))]
            completed = 0
            total_tasks = len(inds_left)            
            for future in futures:
                i1, result = future.result()
                Kij[3 * i1:3 * i1 + 3, :] = result
                completed += 1
                #print(f'Completed {completed} out of {total_tasks} tasks.')                

        self.Kij = Kij
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Total time taken for calculating the force constant: {:.2f} seconds".format(elapsed_time))

    def writeToFile(self):
        """
        Write `self.Kij` to files starting with `self.fileprefix`.
        :return: None
        """
        np.save(self.fileprefix + '.Kij.npy', self.Kij)
        np.save(self.fileprefix + '.ids_L.npy', self.ids_L)
        np.save(self.fileprefix + '.ids_R.npy', self.ids_R)
        np.save(self.fileprefix + '.ids_Interface.npy', self.inds_interface)
        #np.savetxt(self.fileprefix, self.Kij, delimiter=' ')  # just for checking
        
        ##  Delete the dump file, cause the users don't need to use them
        os.remove('dump.left')
        os.remove('dump.right')
        os.remove('dump.interface')
        
        print('\nKij have been written to the ' + self.fileprefix + ' file\n')


if __name__ == "__main__":
    
    group_name_L = "NL"          # group name in forces.in
    group_name_R = "NR"          # group name in forces.in
    group_name_interface = "N"   # group name in forces.in
    
    fileprefix = 'Fij.dat'
    hstep = 0.1                           ## Maybe hstep = 0.01 is a good choice
    in_file = 'forces.in'                 ## The in file for lammps
    
    with fcCalc(fileprefix) as fc:
        fc.preparelammps(in_lammps=in_file,
                         group_name_L=group_name_L,
                         group_name_R=group_name_R,
                         group_name_interface=group_name_interface)
                         
        fc.fcCalc(hstep, n_cores=20)
        fc.writeToFile()
        print('Forces Calculate All Done !\n')
