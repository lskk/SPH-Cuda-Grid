#ifndef CAMERA_CUH
#define CAMERA_CUH
#include <GLFW\glfw3.h>
#include <glm\vec3.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <iostream>
#include <windows.h>

using namespace glm;

int widht = 800;
int height = 600;

float zNear = 0.1f;
float zFar = 50.0f;

float cam_speed = 1.5f;
vec3 cam_position = glm::vec3(20.0f, 15.0f, 20.0f); // Camera is at (4,3,-3), in World Space
vec3 cam_target = glm::vec3(0.0f, 0.0f, 0.0f); // and looks at the origin
vec3 cam_up = glm::vec3(0.0f, 1.0f, 0.0f);  // Head is up (set to 0,-1,0 to look upside-down)

// Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
glm::mat4 Projection = perspective(45.0f, (float)widht / (float)height, zNear, zFar);
// Camera matrix
glm::mat4 View = glm::lookAt(
	cam_position,
	cam_target,
	cam_up
	);
// Model matrix : an identity matrix (model will be at the origin)
glm::mat4 Model = glm::mat4(1.0f);
// Our ModelViewProjection : multiplication of our 3 matrices
// glm::mat4 MVP = Projection * View * Model; // Remember, matrix multiplication is the other way around

//VARIABEL LIGHTING
struct Light{
	vec4 position;
	vec4 diffuse;
	vec4 specular;
	vec4 ambient;
};

Light gLight;

void translateCamera(vec3 direction){
	cam_position += direction*cam_speed;
	cam_target += direction*cam_speed;
	View = glm::lookAt(cam_position, cam_target, cam_up);
}

#endif