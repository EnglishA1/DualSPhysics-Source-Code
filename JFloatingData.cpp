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

/// \file JFloatingData.cpp \brief Implements the class \ref JFloatingData.

#include "JFloatingData.h"
#include "JVarsAscii.h"

#include <fstream>
#include <cmath>
#include <cstring>
//#include <iostream>
//#include <sstream>
//#include <cstdlib>
//#include <ctime>
//using namespace std;
#include "Functions.h"

using namespace std;

//##############################################################################
//# JFloatingDataPart
//##############################################################################
//==============================================================================
/// Constructor.
//==============================================================================
JFloatingDataPart::JFloatingDataPart(unsigned part,float timestep,unsigned ftsize):Part(part),TimeStep(timestep),FtSize(ftsize){
  ClassName="JFloatingDataPart";
  FtData=NULL;
  if(FtSize)FtData=new StFloatingData[FtSize];
  FtCount=0;
}

//==============================================================================
/// Destructor.
//==============================================================================
JFloatingDataPart::~JFloatingDataPart(){
  delete[] FtData;
}

//==============================================================================
/// Adds data of floating body.
//==============================================================================
void JFloatingDataPart::AddFloating(unsigned begin,unsigned count,float mass,float massp,const tfloat3 &inertia,const tfloat3 &center,const tfloat3 &fvel,const tfloat3 &fomega){
  if(FtCount>=FtSize)RunException("AddFloating","You can not more floating bodies.");
  StFloatingData *fo=FtData+FtCount;
  fo->begin=begin;
  fo->count=count;
  fo->mass=mass;
  fo->massp=massp;
  fo->inertia=inertia;
  fo->center=center;
  fo->fvel=fvel;
  fo->fomega=fomega;
  //printf("AddFloating================> fvel.z:%f\n",fvel.z);
  FtCount++;
  GetFtData(FtCount-1);
}

//==============================================================================
/// Returns data of a given objecto.
//==============================================================================
JFloatingDataPart::StFloatingData JFloatingDataPart::GetFtData(unsigned num)const{
  if(num>=FtCount)RunException("GetFtData","The requested floating body does not exist.");
  StFloatingData dat=FtData[num];
  //printf("GetFtData[%u]================> fvel.z:%f\n",num,dat.fvel.z);
  return(dat);
}

//##############################################################################
//# JFloatingData
//##############################################################################
//==============================================================================
/// Constructor.
//==============================================================================
JFloatingData::JFloatingData(){
  ClassName="JFloatingData";
  Dist=NULL;
  Reset();
}

//==============================================================================
/// Destructor.
//==============================================================================
JFloatingData::~JFloatingData(){
  Reset();
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JFloatingData::Reset(){
  HeadEmpty=true;
  memset(&Head,0,sizeof(StHeadData));
  ResetData();
}

//==============================================================================
/// Clears data of floating bodies.
//==============================================================================
void JFloatingData::ResetData(){
  delete[] Dist;   Dist=NULL;
  for(int c=0;c<int(Parts.size());c++)delete Parts[c];
  Parts.clear();
}

//==============================================================================
/// Configures object with data of headers.
//==============================================================================
void JFloatingData::Config(float dp,float h,float b,float rhop0,float gamma,bool data2d,unsigned np,unsigned nfixed,unsigned nmoving,unsigned nfloat,unsigned ftcount){
  Reset();
  HeadEmpty=false;
  Head.dp=dp; Head.h=h; Head.b=b; Head.rhop0=rhop0; Head.gamma=gamma; Head.data2d=(data2d? 1: 0);
  Head.np=np; Head.nfixed=nfixed; Head.nmoving=nmoving; Head.nfloat=nfloat; Head.ftcount=ftcount;
}

//==============================================================================
/// Adds disntace of floating particles to the center of the object.
//==============================================================================
void JFloatingData::AddDist(unsigned n,const tfloat3* dist){
  const char met[]="AddDist";
  if(HeadEmpty)RunException(met,"The object is not configured.");
  if(Head.nfloat!=n)RunException(met,"The number of floating particles is invalid.");
  delete[] Dist;   Dist=NULL;
  Dist=new tfloat3[Head.nfloat];
  memcpy(Dist,dist,sizeof(tfloat3)*n);
}

//==============================================================================
/// Returns pointer pointing at Dist checking its validity.
//==============================================================================
const tfloat3* JFloatingData::GetDist(unsigned n)const{
  const char met[]="GetDist";
  if(Head.nfloat!=n)RunException(met,"The number of values is invalid.");
  return(Dist);
}

//==============================================================================
/// Adds info to a new Part.
//==============================================================================
JFloatingDataPart* JFloatingData::AddPart(unsigned part,float timestep){
  const char met[]="AddPart";
  if(HeadEmpty)RunException(met,"The object is not configured.");
  JFloatingDataPart *fpart=new JFloatingDataPart(part,timestep,Head.ftcount);
  Parts.push_back(fpart);
  return(fpart);
}

//==============================================================================
/// Removes previous Part.
//==============================================================================
void JFloatingData::RemoveParts(unsigned partini){
  for(int p=int(GetPartCount())-1;p>=0;p--)if(Parts[p]->Part>=partini){
    JFloatingDataPart* pa=Parts[p];
    Parts.erase(Parts.begin()+p);
    delete pa;
  }
}

//==============================================================================
/// Returns pointer pointing at data of a required Part.
//==============================================================================
const JFloatingDataPart* JFloatingData::GetPart(unsigned num){
  if(num>=GetPartCount())RunException("GetPart","The number of part does not exist.");
  return(Parts[num]);
}

//==============================================================================
/// Returns pointer pointing at data of a required Part.  
/// Returns NULL if it does not exist.
//==============================================================================
const JFloatingDataPart* JFloatingData::GetByPart(unsigned part){
  JFloatingDataPart *ret=NULL;
  const unsigned count=GetPartCount();
  for(unsigned c=0;c<count&&!ret;c++)if(Parts[c]->Part==part)ret=Parts[c];
  return(ret);
}

//==============================================================================
/// Stores data in file.
//==============================================================================
void JFloatingData::SaveFile(std::string file,bool append){
  const char met[]="SaveFile";
  if(append&&!fun::FileExists(file))RunException(met,"Not found the requested file.");
  ofstream pf;
  if(append)pf.open(file.c_str(),ios::binary|ios::out|ios::in);
  else pf.open(file.c_str(),ios::binary|ios::out);
  if(pf){
    if(!append){
      //-Stores header of format.
      StHeadFmt hfmt;
      strcpy(hfmt.titu,"#File FT-Data  ");
      hfmt.ver=1;
      hfmt.bitorder=byte(fun::GetByteOrder());
      hfmt.void1=0;
      hfmt.void2=0;
      pf.write((char*)&hfmt,sizeof(StHeadFmt));
      //-Stores header of configuration.
      pf.write((char*)&Head,sizeof(StHeadData));
      //-Stores distance to the center.
      unsigned nfloat=(!Dist? 0: Head.nfloat);
      pf.write((char*)&nfloat,sizeof(unsigned));
      if(Dist)pf.write((char*)Dist,sizeof(tfloat3)*nfloat);
    }
    else pf.seekp(-int(sizeof(int)),ios::end);
    //-Stores data of Parts.
    int ncount=int(Parts.size());
    pf.write((char*)&ncount,sizeof(int));
    for(int p=0;p<int(Parts.size());p++){
      //printf("------> part:%u\n",p);
      unsigned part=Parts[p]->Part;
      float timestep=Parts[p]->TimeStep;
      pf.write((char*)&part,sizeof(unsigned));
      pf.write((char*)&timestep,sizeof(float));
      unsigned ftcount=Parts[p]->GetFtCount();
      pf.write((char*)&ftcount,sizeof(unsigned));
      for(unsigned c=0;c<ftcount;c++){
        JFloatingDataPart::StFloatingData dat=Parts[p]->GetFtData(c);
        pf.write((char*)&dat,sizeof(JFloatingDataPart::StFloatingData));
      }
    }
    ncount=0;
    pf.write((char*)&ncount,sizeof(int));
    if(pf.fail())RunException(met,"File writing failure.",file);
    pf.close();
  }
  else RunException(met,"Cannot open the file.",file);
  //--DBG--
  //{
  //  JFloatingData fd;
  //  fd.Config(Head.dp,Head.h,Head.b,Head.rhop0,Head.gamma,Head.data2d!=0,Head.np,Head.nfixed,Head.nmoving,Head.nfloat,Head.ftcount);
  //  fd.LoadFile(file);
  //  unsigned nfile=GetPart(GetPartCount()-1)->Part;
  //  string file2=fun::GetWithoutExtension(file)+".csv";
  //  string filecsv=fun::FileNameSec(file2,nfile);
  //  //printf("--> file: [%s] %u\n",filecsv.c_str(),nfile);
  //  while(fun::FileExists(filecsv)){
  //    nfile++; filecsv=fun::FileNameSec(file2,nfile);
  //  }
  //  fd.SaveFileCsv(filecsv);
  //}
}

//==============================================================================
/// Stores data in CSV file.
//==============================================================================
void JFloatingData::SaveFileCsv(std::string file,bool savedist){
  const char met[]="SaveFileCsv";
  ofstream pf;
  pf.open(file.c_str());
  if(pf){
    char cad[1024];
    pf << "dp;h;b;rhop0;gamma;data2d" << endl;
    sprintf(cad,"%f;%f;%f;%f;%f;%s",Head.dp,Head.h,Head.b,Head.rhop0,Head.gamma,(Head.data2d? "True": "False")); pf << cad << endl;
    pf << "np;nfixed;nmoving;nfloat;ftcount" << endl;
    sprintf(cad,"%u;%u;%u;%u;%u",Head.np,Head.nfixed,Head.nmoving,Head.nfloat,Head.ftcount); pf << cad << endl;
    if(savedist&&Dist){
      pf << "Dist:";
      for(unsigned p=0;p<Head.nfloat;p++)pf << ";" << fun::Float3gStr(Dist[p]);
      pf << endl;
    }
    pf << "part;timestep;";
    for(unsigned c=0;c<Head.ftcount;c++)pf << "begin;count;mass;massp;inertia;;;center;;;fvel;;;fomega;;;";
    pf << endl;
    //-Stores data of Parts.
    unsigned n=GetPartCount();
    for(unsigned p=0;p<n;p++){
      const JFloatingDataPart *pa=Parts[p];
      sprintf(cad,"%u;%f",pa->Part,pa->TimeStep); pf << cad;
      unsigned ftcount=pa->GetFtCount();
      for(unsigned c=0;c<ftcount;c++){
        JFloatingDataPart::StFloatingData dat=pa->GetFtData(c);
        sprintf(cad,";%u;%u;%G;%G;%G;%G;%G",dat.begin,dat.count,dat.mass,dat.massp,dat.inertia.x,dat.inertia.y,dat.inertia.z); pf << cad;
        sprintf(cad,";%G;%G;%G;%G;%G;%G;%G;%G;%G",dat.center.x,dat.center.y,dat.center.z,dat.fvel.x,dat.fvel.y,dat.fvel.z,dat.fomega.x,dat.fomega.y,dat.fomega.z); pf << cad;
      }
      pf << endl;
    }
    if(pf.fail())RunException(met,"File writing failure.",file);
    pf.close();
  }
  else RunException(met,"Cannot open the file.",file);
}

//==============================================================================
/// Loads data of a file.
//==============================================================================
void JFloatingData::LoadFile(std::string file){
  const char met[]="LoadFile";
  ResetData();
  ifstream pf;
  pf.open(file.c_str(),ios::binary|ios::in);
  if(pf){
    //-Loads and checks header of format.
    StHeadFmt hfmt;
    pf.read((char*)&hfmt,sizeof(StHeadFmt));
    if(strcmp(hfmt.titu,"#File FT-Data  "))RunException(met,"The format file is invalid.");
    if(hfmt.ver>1)RunException(met,"The version format file is invalid.");
    if(hfmt.bitorder!=byte(fun::GetByteOrder()))RunException(met,"The bit order of data in file is invalid.");
    //-Stores header of configuration.
    StHeadData fhead;
    pf.read((char*)&fhead,sizeof(StHeadData));
    //-Checks validity of configuration.
    if(!HeadEmpty){
      if(fhead.data2d!=Head.data2d||fhead.np!=Head.np||fhead.nfixed!=Head.nfixed||fhead.nmoving!=Head.nmoving||fhead.nfloat!=Head.nfloat||fhead.ftcount!=Head.ftcount)RunException(met,"The configuration of file is invalid for this simulation.");
    }
    else{ Head=fhead; HeadEmpty=false; }
    //-Loads distance to the center.
    unsigned nfloat=0;
    pf.read((char*)&nfloat,sizeof(unsigned));
    if(nfloat){
      if(nfloat!=Head.nfloat)RunException(met,"The value nfloat is invalid.");
      Dist=new tfloat3[Head.nfloat];
      pf.read((char*)Dist,sizeof(tfloat3)*nfloat);
    }
    //-Loads data of Parts.
    int ncount=1;
    while(ncount){
      pf.read((char*)&ncount,sizeof(int));
      for(int p=0;p<ncount;p++){
        unsigned part=0,ftcount=0;
        float timestep=0;
        pf.read((char*)&part,sizeof(unsigned));
        pf.read((char*)&timestep,sizeof(float));
        pf.read((char*)&ftcount,sizeof(unsigned));
        if(ftcount!=Head.ftcount)RunException(met,"The value ftcount is invalid.");
        JFloatingDataPart* fpart=AddPart(part,timestep);
        for(unsigned c=0;c<ftcount;c++){
          JFloatingDataPart::StFloatingData fo;
          pf.read((char*)&fo,sizeof(JFloatingDataPart::StFloatingData));
          fpart->AddFloating(fo.begin,fo.count,fo.mass,fo.massp,fo.inertia,fo.center,fo.fvel,fo.fomega);
        }
      }
    }
    pf.close();
  }
  else RunException(met,"Cannot open the file.",file);
}





