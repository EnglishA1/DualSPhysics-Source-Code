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

/// \file JSphDtFixed.h \brief Declares the class \ref JSphDtFixed.

#ifndef _JSphDtFixed_
#define _JSphDtFixed_

#pragma warning(disable : 4996) //Cancels sprintf() deprecated.

#include "JObject.h"
#include "Types.h"
#include "JTimer.h"
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cstdlib>

//##############################################################################
//# JSphDtFixed
//##############################################################################
/// \brief Manages the use of prefixed values of DT loaded from an input file.

class JSphDtFixed : protected JObject
{
protected:
  static const unsigned SIZEMAX=100000;
  static const unsigned SIZEINITIAL=500;

  std::string File;
  unsigned Size;
  unsigned Count;
  unsigned Position;
  float *Times;
  float *Values;
  float DtError; //- max(DtFixed-DtVariable).

  void Resize(unsigned size);

public:
  JSphDtFixed();
  ~JSphDtFixed();
  void Reset();
  unsigned GetAllocMemory()const;
  void LoadFile(std::string file);
  float GetDt(float timestep,float dtvar);
  float GetDtError(bool reset);
  std::string GetFile()const{ return(File); };
};

#endif


