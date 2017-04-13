#include <stdio.h>
#include <stdlib.h>
#include <GL\glew.h>
//GLFW
#include <GLFW\glfw3.h>
#include <glm\vec3.hpp>
#include <glm\vec4.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include "float3.cuh"
#include <iostream>
#include "Shader.cuh"
#include <windows.h>
#include <time.h>

//SSFRInclude
#include "SsfrParam.cuh"

//SPH
#include "SPHCompute.cuh"
#include "Scene.cuh"
#include "SPHParam.cuh"

//USING CAMERA
#include "Camera.cuh"

//using CSVManager
#include "CSVManager.h"

//Kernel
#include "KernelOption.cuh"

#include "SphClass.cuh"

SsfrParam ssfr;
SPHParam sphparam;
Wall wall;

//AKTIVATOR
bool space_button = false; // untuk emitter


//timer
clock_t deltaTime = 0;
unsigned int frames = 0;
double  frameRate = 30;
double  averageFrameTimeMilliseconds = 33.333;
bool record_data = false;

extern "C" {
	_declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
}

double clockToMilliseconds(clock_t ticks);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void initParam();
void initLight();

int main(){

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	GLFWwindow* window = glfwCreateWindow(800, 600, "LearnOpenGL", nullptr, nullptr);
	if (window == nullptr)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, key_callback);
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK)
	{
		std::cout << "Failed to initialize GLEW" << std::endl;
		return -1;
	}
	glViewport(0, 0, widht, height);

	initLight();
	string file = "E:/Dropbox/My Thesis/Laporan/Data/data_8192_withoutprecomp.csv";
	string limiter = ",";
	CSVManager* csv; 
	if (record_data){
		csv = &CSVManager(file, limiter);
		csv->openCSV();
	}
	//INISIASI SPH =====================================================================
	//initParam();

	SphClass sphsim;
	sphsim.initParam();
	sphsim.initSPH();
	//sphsim.initCustomSPH()

	/*
	DamBreakScene dam(sphparam);
	dam.createScene();
	sphparam = dam.getParam();

	//dam.debugPosition();
	KernelOption kernel = KernelOption(sphparam.kernelSize);
	SPHCompute sph(sphparam, kernel, &dam.getAllParticles()[0], wall);
	sph.initCudaMemoryAlloc();
	*/
	//==================================================================================


	//PERINTAH WAJIB==========================
	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);
	//========================================
	GLuint vertexbuffer;
	// Generate 1 buffer, put the resulting identifier in vertexbuffer
	glGenBuffers(1, &vertexbuffer);
	// The following commands will talk about our 'vertexbuffer' buffer
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	// Give our vertices to OpenGL.

	//EDITED 12-16-2016 Choose 1 only
	//glBufferData(GL_ARRAY_BUFFER, sizeof(Particle) * sphparam.particleCounter, NULL, GL_STREAM_DRAW);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Particle) * sphsim.getSPHParam().particleCounter, NULL, GL_STREAM_DRAW);
	//=================

	//glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glEnable(GL_MODELVIEW_MATRIX);

	Shader depth;
	//depth.LoadShaderProg("depth.vert", "depth.frag");
	depth.LoadShaderProg("pointSplattingVS.vert", "pointSplattingFS.frag");

	// Get a handle for our "MVP" uniform
	GLuint MatrixIDProj = glGetUniformLocation(depth.programID, "Projection");
	GLuint MatrixIDView = glGetUniformLocation(depth.programID, "View");
	GLuint MatrixIDMod = glGetUniformLocation(depth.programID, "Model");
	GLuint RadiusSPH = glGetUniformLocation(depth.programID, "radius");
	GLuint NearCamID = glGetUniformLocation(depth.programID, "zNear");
	GLuint FarCamID = glGetUniformLocation(depth.programID, "zFar");
	GLuint LightPositionID = glGetUniformLocation(depth.programID, "light.position");
	GLuint LightDiffuseID = glGetUniformLocation(depth.programID, "light.diffuse");
	GLuint LightSpecularID = glGetUniformLocation(depth.programID, "light.specular");
	GLuint LightAmbientID = glGetUniformLocation(depth.programID, "light.ambient");

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);
	//glDepthFunc();

	//RENDER INFO
	std::cout << "RENDERING MENGGUNAKAN : " << glGetString(GL_RENDERER) << std::endl;
	std::cout << "VENDOR : " << glGetString(GL_VENDOR) << std::endl;


	int iter = 0;
	while (!glfwWindowShouldClose(window))
	{
		//CalcFPS();
		//printf("frame %d, fps = %f \n", frames, fps);
		iter++;


		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glUseProgram(depth.programID);

		//PASSING VARIABEL
		glUniformMatrix4fv(MatrixIDProj, 1, GL_FALSE, &Projection[0][0]);
		glUniformMatrix4fv(MatrixIDView, 1, GL_FALSE, &View[0][0]);
		glUniformMatrix4fv(MatrixIDMod, 1, GL_FALSE, &Model[0][0]);

		//EDITED 12-16-2016
		//glUniform1f(RadiusSPH, ssfr.getParticleSize());
		glUniform1f(RadiusSPH, sphsim.getSsfrParam().getParticleSize());
		//=================
		glUniform1f(NearCamID, zNear);
		glUniform1f(FarCamID, zFar);
		glUniform4f(LightAmbientID, gLight.ambient.x, gLight.ambient.y, gLight.ambient.z, gLight.ambient.w);
		glUniform4f(LightDiffuseID, gLight.diffuse.x, gLight.diffuse.y, gLight.diffuse.z, gLight.diffuse.w);
		glUniform4f(LightSpecularID, gLight.specular.x, gLight.specular.y, gLight.specular.z, gLight.specular.w);
		glUniform4f(LightPositionID, gLight.position.x, gLight.position.y, gLight.position.z, gLight.position.w);

		//EDITED 12-16-2016
		/*
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glBufferData(GL_ARRAY_BUFFER, (sphparam.particleCounter) * sizeof(Particle), NULL, GL_STREAM_DRAW); // Buffer orphaning, a common way to improve streaming perf. See above link for details.
		glBufferSubData(GL_ARRAY_BUFFER, 0, (sphparam.particleCounter) * sizeof(Particle), &sph.getAllLiquidParticles()[0]);
		*/
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glBufferData(GL_ARRAY_BUFFER, (sphsim.getSPHParam().particleCounter) * sizeof(Particle), NULL, GL_STREAM_DRAW); // Buffer orphaning, a common way to improve streaming perf. See above link for details.
		glBufferSubData(GL_ARRAY_BUFFER, 0, (sphsim.getSPHParam().particleCounter) * sizeof(Particle), sphsim.getAllLiquidParticles());
		//=================

		for (int attrib = 0; attrib <= 1; attrib++){
			glEnableVertexAttribArray(attrib);
		}

		//glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (const void*)0);
		//glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), (const void*)24);

		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

		//EDITED 12-16-2016
		//glDrawArrays(GL_POINTS, 0, sphparam.particleCounter);//*paramSPH.particleSizeReserver());
		glDrawArrays(GL_POINTS, 0, sphsim.getSPHParam().particleCounter);//*paramSPH.particleSizeReserver());
		//=================

		for (int attrib = 0; attrib <= 1; attrib++){
			glDisableVertexAttribArray(attrib);
		}


		glfwPollEvents();
		glfwSwapBuffers(window);
		//EDITED 12-16-2016
		//sph.updateParticles();
		if (space_button) {
			clock_t beginFrame = clock();
			sphsim.updateParticles();
			clock_t endFrame = clock();
			//deltaTime += endFrame - beginFrame;
			frames++;
			deltaTime = endFrame - beginFrame;
			if (record_data) csv->writeFile(frames, clockToMilliseconds(deltaTime));
			//if you really want FPS
			/*
			if (clockToMilliseconds(deltaTime)>1000.0){ //every second
			frameRate = (double)frames*0.5 + frameRate*0.5; //more stable
			frames = 0;
			deltaTime -= CLOCKS_PER_SEC;
			averageFrameTimeMilliseconds = 1000.0 / (frameRate == 0 ? 0.001 : frameRate);
			std::cout << "FrameTime was:" << averageFrameTimeMilliseconds << std::endl;
			}
			*/
		}
		/*
		if (frames == 5000){
			break;
		}
		*/
		//=================

		//sph.debugPosition();
		//printf("iteration (%d) posisi partikel 0 = %f, %f, %f \n", iter, sph.getAllLiquidParticles()[2355].mPosition.x, sph.getAllLiquidParticles()[0].mPosition.y, sph.getAllLiquidParticles()[0].mPosition.z);
		//printf("iteration (%d) particle num = %d \n", iter, sphparam.particleCounter);
	}
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteVertexArrays(1, &VertexArrayID);

	//EDITED 12-16-2016
	//sph.freeParticles();
	sphsim.freeCuda();
	sphsim.freeCPU();
	//=================
	if (record_data) csv->closeCSV();
	glfwTerminate();
	return 0;
}

void initParam(){
	ssfr.setParticleSize(250);
	sphparam.surfaceColor = (vec4(1.00f, 0.95f, 0.00f, 1.0f));
	sphparam.insideColor = (vec4(0.94f, 0.22f, 0.15f, 1.0f));
	sphparam.restDensity = 2.2f;// 2.2 normal
	sphparam.gasConstant = 200.0f;
	sphparam.viscosityCoeff = 3.5f;// (){ return 6.5f; } 1.2 normal
	sphparam.timeStep = (float)1.3f / 60;// (){ return 10 / 80; }
	sphparam.surfaceTreshold = 0.1f;// (){ return 6.0f; }
	sphparam.surfaceTensionCoeff = 1.0f;// (){ return 0.0728; }
	sphparam.gravity = vec3(0.0f, -9.8f, 0.0f);// (){ return vec3(0.0f, 9.8f, 0.0f); }
	sphparam.mParticleRadius = 0.1f;// (){ return 0.5; }
	sphparam.wallDamping = 0.1f;// (){ return 0.2; }
	sphparam.particleSizeReserver = 2200;// (){ return 100; }
	sphparam.particleCounter = 6144;//6144;//5120;//8192;//10240;
	sphparam.kernelSize = 1.5f;
	sphparam.maxParticleCount = 12000;
	sphparam.LOrandomizer = 0.0f;
	sphparam.HIrandomizer = 0.11f;
	sphparam.cudaThreadsPerblock = 32;
	wall.min = vec3(-10.0f, -2.0f, -2.0f); 
	wall.max = vec3(30.0f, 20.0f, 20.0f);
}

void initLight(){
	gLight.ambient = vec4(0.0, 0.0, 0.0, 1.0);
	gLight.diffuse = vec4(1.0, 1.0, 1.0, 1.0);
	gLight.specular = vec4(1.0, 1.0, 1.0, 1.0);
	gLight.position = vec4(0.0, -5.0, 0.0, 1.0);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	// When a user presses the escape key, we set the WindowShouldClose property to true,
	// closing the application
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
	if (key == GLFW_KEY_UP && action == GLFW_REPEAT){
		glm::vec3 direction = vec3(0.0f, 0.0f, -0.1f);
		translateCamera(direction);
	}
	if (key == GLFW_KEY_DOWN && action == GLFW_REPEAT){
		glm::vec3 direction = vec3(0.0f, 0.0f, +0.1f);
		translateCamera(direction);
	}
	if (key == GLFW_KEY_RIGHT && action == GLFW_REPEAT){
		glm::vec3 direction = vec3(+0.1f, 0.0f, 0.0f);
		translateCamera(direction);
	}
	if (key == GLFW_KEY_LEFT && action == GLFW_REPEAT){
		glm::vec3 direction = vec3(-0.1f, 0.0f, 0.0f);
		translateCamera(direction);
	}
	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS){
		space_button = !space_button;
		printf("Space pressed, value = %d \n", space_button);
	}
}

double clockToMilliseconds(clock_t ticks){
	// units/(units/time) => time (seconds) * 1000 = milliseconds
	return (ticks / (double)CLOCKS_PER_SEC)*1000.0;
}