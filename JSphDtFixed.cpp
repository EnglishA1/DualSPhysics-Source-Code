/*
<DUALSPHYSICS>  Copyright (C) 2013 by Jose M. Dominguez, Dr Alejandro Crespo, Prof. M. Gomez Gesteira, Anxo Barreiro, Ricardo Canelas
                                      Dr Benedict Rogers, Dr Stephen Longshaw, Dr Renato Vacondio

EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

This file is part of DualSPHysics. 

DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at your option) any later version. 

DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. 

You should have received a copy of the GNU General Public License, along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>. 
*/

/// \file JSphDtFixed.cpp \brief Implements the class \ref JSphDtFixed.

#include "JSphDtFixed.h"
#include "Functions.h"
#include <cstring>
#include <sstream>
#include <iostream>
#include <fstream>
#include <float.h>

using namespace std;

//==============================================================================
/// Constructor.
//==============================================================================
JSphDtFixed::JSphDtFixed(){
  ClassName="JSphDtFixed";
  Times=NULL;
  Values=NULL;
  Reset();
}

//==============================================================================
/// Destructor.
//==============================================================================
JSphDtFixed::~JSphDtFixed(){
  Reset();
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JSphDtFixed::Reset(){
  delete[] Times;  Times=NULL;
  delete[] Values; Values=NULL;
  File="";
  Size=Count=Position=0;
  GetDtError(true);
}

//==============================================================================
/// Resizes memory space for values.
//==============================================================================
void JSphDtFixed::Resize(unsigned size){
  if(size>SIZEMAX)size=SIZEMAX;
  if(size==Size)RunException("Resize","It has reached the maximum size allowed.");
  Times=fun::ResizeAlloc(Times,Count,size);
  Values=fun::ResizeAlloc(Values,Count,size);
  Size=size;
}

//==============================================================================
/// Returns the allocated memory.
//==============================================================================
unsigned JSphDtFixed::GetAllocMemory()const{
  unsigned s=0;
  if(Times)s+=sizeof(float)*Size;
  if(Values)s+=sizeof(float)*Size;
  return(s);
}

//==============================================================================
/// Loads values of dt (MILISECONDS) for different instants (in SECONDS).
//==============================================================================
void JSphDtFixed::LoadFile(std::string file){
  const char met[]="LoadFile";
  Reset();
  //printf("---> LoadFile> [%s]\n",file.c_str());
  ifstream pf;
  pf.open(file.c_str());
  if(pf){
    pf.seekg(0,ios::end);
    unsigned len=(unsigned)pf.tellg();   //printf("FileSize: %u\n",len);
    pf.seekg(0,ios::beg);
    Resize(SIZEINITIAL);
    Count=0;
    while(!pf.eof()){
      float time,value;
      pf >> time;
      pf >> value;
      if(!pf.fail()){
        if(Count>=Size){
          unsigned newsize=unsigned(float(len)/float(pf.tellg())*1.05f*(Count+1))+100;
          //printf("---> Size: %u -> %u   tellg: %u / %u\n",Size,newsize,unsigned(pf.tellg()),len);
          Resize(newsize);
        } 
        Times[Count]=time; Values[Count]=value;
        //printf("[%u]>  t:%f  v:%f\n",Count,time,value);
        Count++;
      }
    }
    //if(pf.fail())RunException(met,"Error while reading data from the file.",fname);
    pf.close();
  }
  else RunException(met,"Cannot open the file.",file);
  if(Count<2)RunException(met,"Cannot be less than two values.",file);
  File=file;
}

//==============================================================================
/// Returns the value of dt (in SECONDS) for a given instant.
//==============================================================================
float JSphDtFixed::GetDt(float timestep,float dtvar){
  float ret=0;
  //-Looks for the interval of a given instant.
  float tini=Times[Position];
  float tnext=(Position+1<Count? Times[Position+1]: tini);
  for(;tnext<timestep&&Position+2<Count;Position++){
    tini=tnext;
    tnext=Times[Position+2];
  }
  //-Computes dt in a given instant.
  if(timestep<=tini)ret=Values[Position]/1000;
  else if(timestep>=tnext)ret=Values[Position+1]/1000;
  else{
    const double tfactor=double(timestep-tini)/double(tnext-tini);
    float vini=Values[Position];
    float vnext=Values[Position+1];
    ret=float((tfactor*(vnext-vini)+vini)/1000);
  }
  float dterror=ret-dtvar;
  if(DtError<dterror)DtError=dterror;
  //printf("---> DtFixed:%f  DtVar:%f  DtError:%f\n",ret,dtvar,DtError);
  return(ret);
}

//==============================================================================
/// Returns the maximum error regarding the dtvariable. max(DtFixed-DtVariable).
//==============================================================================
float JSphDtFixed::GetDtError(bool reset){
  float ret=DtError;
  if(reset)DtError=-FLT_MAX;
  return(ret);
}



