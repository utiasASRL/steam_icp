#pragma once

#include <fstream>
#include <iostream>
#include <string>

using namespace std;

enum openMode { fileOpenMode_OUT = 0, fileOpenMode_IN = 1 };

class File {
 public:
  File(string path, openMode flag);
  ~File();

 protected:
  string _path;
  const openMode _mode;
  fstream _file;
};