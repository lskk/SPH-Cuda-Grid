#version 330 core

in vec3 pos;
in vec4 col;

uniform struct Light{
	vec4 position;
	vec4 diffuse;
	vec4 specular;
	vec4 ambient;
} light;

uniform mat4 Model;
uniform mat4 Projection;
uniform float radius;

// Ouput data
out vec4 color;

void main(){
	//compute point coord
	vec2 pointCoord = 2 * (gl_PointCoord - vec2(0.5,0.5)) * vec2(1,-1);
   
	//compute normal (in camera space)
	float mag = dot(pointCoord.xy, pointCoord.xy);
	if(mag > 1) discard;
	vec3 normal = vec3(pointCoord, sqrt(1-mag));
   
	//visualize normal
	color = vec4(normal,1);

	vec3 lightDir = normalize(vec3(light.position.xyz) - pos);
	float diffuse = max(0.0, dot(normal, lightDir));
	color = diffuse*col;
}