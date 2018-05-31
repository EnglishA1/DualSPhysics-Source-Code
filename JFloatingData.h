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

/// \file JFloatingData.h \brief Declares the class \ref JFloatingData.

#ifndef _JFloatingData_
#define _JFloatingData_

#include "JObject.h"
#include "TypesDef.h"
#include <string>
#include <vector>


//##############################################################################
//# JFloatingDataPart
//##############################################################################
/// \brief Stores information of floating bodies in an instant.

class JFloatingDataPart : protected JObject
{
public:
  /// Structure with the information of the floating object.
  typedef struct{
    unsigned begin;
    unsigned count;
    float mass;
    float massp;
    tfloat3 inertia;
    tfloat3 center;
    tfloat3 fvel;
    tfloat3 fomega;
  }StFloatingData;
private:
  StFloatingData *FtData;
  unsigned FtCount;

public:
  const unsigned FtSize;
  const unsigned Part;
  const float TimeStep;

  JFloatingDataPart(unsigned part,float timestep,unsigned ftsize);
  ~JFloatingDataPart();
  void AddFloating(unsigned begin,unsigned count,float mass,float massp,const tfloat3 &inertia,const tfloat3 &center,const tfloat3 &fvel,const tfloat3 &fomega);
  unsigned GetFtCount()const{ return(FtCount); }
  StFloatingData GetFtData(unsigned num)const;
};


//##############################################################################
//# JFloatingData
//##############################################################################
/// \brief Allows reading/writing files with data of floating bodies.

class JFloatingData : protected JObject
{
public:
  ///Type of format files.
  typedef enum{ VerNull=0,Ver01=1 }TpVerFile; 

private:
  ///Structure that describes the header of binary format files.
  typedef struct{
    char titu[16];           ///<Title of the file "#File FT-Data".
    byte ver;                ///<File version.
    byte bitorder;           ///<1:BigEndian 0:LittleEndian.
    byte void1;              ///<Field empty.
    byte void2;              ///<Field empty.
  }StHeadFmt;//-sizeof(20)  

  //-Structures to be used with the format version 01:
  typedef struct{//-They must be all of 4 bytes due to conversion ByteOrder ...
    float dp,h,b,rhop0,gamma;
    unsigned data2d;         ///<1:Data for a 2D case, 0:3D Case.
    unsigned np,nfixed,nmoving;
    unsigned nfloat;         ///<Number of floating particles.  
    unsigned ftcount;        ///<Number of floating objects.  
  }StHeadData;

  bool HeadEmpty;
  StHeadData Head;

  tfloat3 *Dist;                          ///<Distance to the center.
  std::vector<JFloatingDataPart*> Parts;  ///<Floating body information for each part.

public:
  JFloatingData();
  ~JFloatingData();
  void Reset();
  void ResetData();
  void Config(float dp,float h,float b,float rhop0,float gamma,bool data2d,unsigned np,unsigned nfixed,unsigned nmoving,unsigned nfloat,unsigned ftcount);
  void AddDist(unsigned n,const tfloat3* dist);
  const tfloat3* GetDist(unsigned n)const;
  JFloatingDataPart* AddPart(unsigned part,float timestep);
  void RemoveParts(unsigned partini);

  unsigned GetPartCount()const{ return(unsigned(Parts.size())); }
  const JFloatingDataPart* GetPart(unsigned num);
  const JFloatingDataPart* GetByPart(unsigned part);

  void LoadFile(std::string file);
  void SaveFile(std::string file,bool append);
  void SaveFileCsv(std::string file,bool savedist=false);
};


#endif





