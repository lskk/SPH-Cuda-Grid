#ifndef SHADER_CUH
#define SHADER_CUH
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
using namespace std;

#include <stdlib.h>
#include <string.h>
#include <GL\glew.h>

class Shader{
public:
	GLuint programID;
	void LoadShaderProg(const char * vertex_file_path, const char * fragment_file_path);
private:
	GLuint CreateVertexShader(const char * vertex_file_path);
	GLuint CreateFragmentShader(const char * fragment_file_path);
};

#endif