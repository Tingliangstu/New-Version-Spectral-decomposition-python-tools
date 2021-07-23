// -*- coding: utf-8 -*-
// Kimmo Sääskilahti, 2015

// Usage: compactify_vels lammps_dump_file [output_file]

/* 
Program writes the velocity data file produced by LAMMPS 
in a more compact, ravelled form without the additional data 
appearing for each time step in the dump file. The assumed commands used for dumping the velocities from LAMMPS is as follows:

dump vels interface custom dt_dump filename.vels.dat id type vx vy vz
dump_modify vels format "%d %d %.8g %.8g %.8g"
dump_modify vels sort id

"interface" is the group of atoms at the interface where SHC is calculated
"dt_dump" is the sampling time step (integer) specifying the maximum frequency
*/

#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
using namespace std;

void skip_lines(istream&, size_t);



int main(int argc,char** argv){
	
	
	
	if(argc==1){
		cout<<"Usage: Give the filenames on the command line."<<endl;
		return(0);
	}		
	
	string filename(argv[1]);
	string filename2;
	
	if(argc==2){
		filename2=filename+".compact";
	}
	else if(argc==3)
		filename2.assign(argv[2]);
	else{
		cout<<"Usage: Give only the filenames on the command line."<<endl;
		return(0);
	}
	
	//cin>>filename;
	//cout<<"Reading from file "<<filename<<"."<<endl;
	//cout<<"Give the filename where to write:"<<endl;
	//cin>>filename2;
	
	//cout<<"Writing to file "<<filename2<<"."<<endl;
	
	
	ifstream file1;
	
	double a,b;
	
	//file1.open(filename.c_str());
	file1.open(argv[1]);
	
	string str;
	int t0,t1;
	int N;
	int *ids;
	if(file1.is_open()){ // Initializations, set number of particles etc.
		getline(file1,str);
		//cout<<"Read: "<<str<<endl;
		file1>>t0;
		cout<<endl;
		cout<<"The following is the imformation in dump files"<<"."<<endl;	
		cout<<"       "<<"The initial time step is "<<t0<<"."<<endl;
		getline(file1,str); // The line break
		getline(file1,str);
		//cout<<"Read: "<<str<<endl;
		file1 >> N;
		cout<<"       "<<"The number of atoms is "<<N<<"."<<endl;
		ids=new int[N];
		getline(file1,str); // Line break
		skip_lines(file1,5);
		
		for (int i=0;i<N;i++){
			file1>>ids[i];
			getline(file1,str); // Skip the rest of the line
			//cout<<ids[i]<<endl;
		}	
		skip_lines(file1,1);
		file1>>t1;
		cout<<"       "<<"The difference of time steps is "<<t1-t0<<"."<<endl;
	}
	else{
		cout<<"Could not open file "<<filename<<", exiting."<<endl;
		return 0;
	}
	
	// Rewind back to the beginning
	file1.seekg(0,ios::beg);
	
	int id,type;
	double vx,vy,vz;
	
	size_t iter=0;
	
	ofstream file2;
	file2.open(filename2.c_str());
	
	file2<<"Atoms "<<N<<endl;
	file2<<"d_timestep "<<t1-t0<<endl;
	file2<<"Atom ids:"<<endl;
	for(int i=0;i<N;i++){
		file2<<ids[i]<<endl;
	}
	file2<<"----------"<<endl;
       
  cout<<"The file is being processed, please wait a moment."<<endl;

	//while(!file1.eof()){
	while(file1.peek() != char_traits<wchar_t>::eof()){ // The next character is not EOF
		if ((iter)%100000==0)
			//cout<<"iter="<<iter<<endl;
		iter++;
		//getline(file1,str);
		//cout<<"Read: "<<str<<endl;
		skip_lines(file1,9);
		//getline(file1,str);
		//cout<<"Read: "<<str<<endl;
		// Start reading the velocities
		int i;
		for(i=0;i<N;i++){
			file1>>id>>type>>vx>>vy>>vz;
			//file2<<id<<'\n';
			file2<<vx<<'\n';
			file2<<vy<<'\n';
			file2<<vz<<'\n';
			//cout<<id<<vx<<vy<<vz<<endl;
		}
		//cout<<id<<endl;
		getline(file1,str); // The line break
		
		
		//return 0;
	}
	
	
	file1.close();
	file2.close();
	return 0;
}

void skip_lines(istream& pStream, size_t pLines)
{
    string s;
    for (; pLines; --pLines)
        getline(pStream, s);
}
