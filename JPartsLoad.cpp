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

/// \file JPartsLoad.cpp \brief Implements the class \ref JPartsLoad.

#include "JPartsLoad.h"
#include "Functions.h"
#include "JPartData.h"
#include <float.h>
#include <cstdio>

using namespace std;

//==============================================================================
/// Constructor.
//==============================================================================
JPartsLoad::JPartsLoad(){
  ClassName="JPartsLoad";
  Idp=NULL; Pos=NULL; Vel=NULL; Rhop=NULL;
  Reset();
}

//==============================================================================
/// Destructor.
//==============================================================================
JPartsLoad::~JPartsLoad(){
  AllocMemory(0);
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JPartsLoad::Reset(){
  LoadOut=LOUT_None;
  Simulate2D=false;
  PartBi2=false;
  MapPosMin=MapPosMax=TFloat3(0);
  PartBegin=0;
  PartBeginTimeStep=0;
  AllocMemory(0);
}

//==============================================================================
/// Resizes memory space for particle data.
//==============================================================================
void JPartsLoad::AllocMemory(unsigned size){
  Count=CountOut=0; Size=size;
  delete[] Idp; Idp=NULL; 
  delete[] Pos; Pos=NULL; 
  delete[] Vel; Vel=NULL; 
  delete[] Rhop; Rhop=NULL; 
  if(Size){
    try{
      Idp=new unsigned[Size];
      Pos=new tfloat3[Size];
      Vel=new tfloat3[Size];
      Rhop=new float[Size];
    }
    catch(const std::bad_alloc){
      RunException("AllocMemory","The requested memory could not be allocated.");
    }
  } 
}

//==============================================================================
/// Returns the allocated memory in CPU.
//==============================================================================
long long JPartsLoad::GetAllocMemory()const{  
  long long s=0;
  //-Allocated in AllocMemory().
  if(Idp)s+=sizeof(unsigned)*Size;
  if(Pos)s+=sizeof(tfloat3)*Size;
  if(Vel)s+=sizeof(tfloat3)*Size;
  if(Rhop)s+=sizeof(float)*Size;
  //Allocated in other objects.
  return(s);
}

//==============================================================================
/// Computes limits of the positions of the loaded particles.
//==============================================================================
void JPartsLoad::CalcPosLimits(unsigned n,const tfloat3 *pos,tfloat3 &posmin,tfloat3 &posmax){
  tfloat3 pmin=TFloat3(FLT_MAX),pmax=TFloat3(-FLT_MAX);
  //-Computes minimum and maximum position. 
  for(unsigned p=0;p<n;p++){
    const tfloat3 *ps=(pos+p);
    if(pmin.x>ps->x)pmin.x=ps->x;
    if(pmin.y>ps->y)pmin.y=ps->y;
    if(pmin.z>ps->z)pmin.z=ps->z;
    if(pmax.x<ps->x)pmax.x=ps->x;
    if(pmax.y<ps->y)pmax.y=ps->y;
    if(pmax.z<ps->z)pmax.z=ps->z;
  }
  posmin=pmin; posmax=pmax;
}

//==============================================================================
/// Returns limits of the domain according to particles or data of PartBegin.
//==============================================================================
void JPartsLoad::GetLimits(tfloat3 bordermin,tfloat3 bordermax,tfloat3 &posmin,tfloat3 &posmax)const{
  if(PartBegin&&!PartBi2){
    posmin=MapPosMin;
    posmax=MapPosMax;
  }
  else{
    //printf("++> JPartsLoad  MapPos=(%f,%f,%f)-(%f,%f,%f)\n",MapPosMin.x,MapPosMin.y,MapPosMin.z,MapPosMax.x,MapPosMax.y,MapPosMax.z);
    posmin=MapPosMin; posmax=MapPosMax;
    if(posmin.x>posmax.x)posmin=posmax=bordermin;
    posmin=posmin-bordermin;
    tfloat3 posmax2=posmax+bordermax;
    for(unsigned c=0;c<100&&(posmax2.x<=posmax.x||posmax2.y<=posmax.y||posmax2.z<=posmax.z);c++){
      if(posmax2.x<=posmax.x)bordermax.x*=2;
      if(posmax2.y<=posmax.y)bordermax.y*=2;
      if(posmax2.z<=posmax.z)bordermax.z*=2;
      printf("*** Recalculation of the BorderMax=(%f,%f,%f)\n",bordermax.x,bordermax.y,bordermax.z);
      posmax2=posmax+bordermax;
    }
    posmax=posmax+bordermax;
  }
}

//==============================================================================
/// Load of particle data.
//==============================================================================
void JPartsLoad::LoadParticles(TpLoadOut loadout,const std::string &casedir,const std::string &casename,unsigned partbegin,const std::string &casedirbegin,unsigned casenp,unsigned casenbound,unsigned casenfixed,unsigned casenmoving,unsigned casenfloat){
  const char* met="LoadParticles";
  Reset();
  LoadOut=loadout;
  PartBegin=partbegin;
  string filepart;
  string filepart1=casedir+casename+".bi2";
  string filepart2=casedir+casename+".bin";
  if(PartBegin){
    JPartData pd;
    string filepart1x=pd.GetFileName(pd.FmtBi2,PartBegin,casedirbegin);
    filepart2="";
    //print("%s\n",fun::VarStr("PartBegin",PartBegin).c_str());
    //print("%s\n",fun::VarStr("PartBeginDir",PartBeginDir).c_str());
    //print("%s\n",fun::VarStr("PartBeginFirst",PartBeginFirst).c_str());
    if(!fun::FileExists(filepart1x))RunException(met,"File of the particles was not found.",filepart1x);
  }
  else if(!fun::FileExists(filepart1)&&!fun::FileExists(filepart2))RunException(met,"File of the particles was not found.",filepart2);
  
  {//-Loads data of BI2 file BI2.
    PartBi2=true;
    JPartData pdini;
    if(PartBegin){
      pdini.LoadFile(JPartData::FmtBi2Out,0,casedirbegin);
      pdini.LoadFile(JPartData::FmtBi2,0,casedirbegin);
      //-Gets limits of the domain in Part 0.
      {
        unsigned np=pdini.GetNp();
        tfloat3 *pos=new tfloat3[np];
        pdini.GetDataSort(np,NULL,pos,NULL,NULL,true);
        CalcPosLimits(np,pos,MapPosMin,MapPosMax);
        delete[] pos;
      }
      pdini.LoadFile(JPartData::FmtBi2,PartBegin,casedirbegin);
      PartBeginTimeStep=pdini.GetPartTime();
      filepart=pdini.GetFileName(pdini.FmtBi2,PartBegin,casedirbegin);
      //printf("++> filepart:[%s]\n",filepart.c_str());
    }
    else{
      if(fun::FileExists(filepart1)){ pdini.LoadFileBi2(0,filepart1); filepart=filepart1; }
      else{ pdini.LoadFileBin(0,filepart2); filepart=filepart2; }
    }
    JPartData::StConfig cf=pdini.GetConfigInfo();     ///<Stores information of configuration of the particles.
    if(cf.h==0)RunException(met,"File data invalid",filepart);
    if(pdini.GetNp()!=casenp||pdini.GetNbound()!=casenbound||pdini.GetNfixed()!=casenfixed||pdini.GetNmoving()!=casenmoving||pdini.GetNfloat()!=casenfloat)RunException(met,"Data file does not match the configuration of the case.",filepart);
    AllocMemory(casenp);
    pdini.GetDataSort(casenp,Idp,Pos,Vel,Rhop,true);
    if(!PartBegin)CalcPosLimits(casenp,Pos,MapPosMin,MapPosMax);
    CountOut=pdini.GetNfluidOut();
    Count=casenp;
    Simulate2D=pdini.GetData2D();
    pdini.Reset();
  }
}





