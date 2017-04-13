#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

// Values that stay constant for the whole mesh.
uniform mat4 Projection;
uniform mat4 View;
uniform mat4 Model;
uniform float radius;

out vec3 pos;
out vec4 col;

void main(){	
	vec4 eye2point =  View * Model * vec4(position,1);
	vec3 eye2point3 = (eye2point/eye2point.w).xyz;
	float dist = length(eye2point3);
	gl_PointSize = radius / dist;
    gl_Position = Projection * eye2point;
	pos = eye2point.xyz;
	col = color;
}

