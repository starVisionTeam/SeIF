#version 330

layout (location = 0) in vec3 Position;
layout (location = 1) in vec3 Normal;

out vec3 CamNormal;

uniform mat3 RotMat;
uniform mat4 ModelMat;
uniform mat4 PerspMat;
uniform mat4 NormMat;
void main()
{
	gl_Position = PerspMat * ModelMat * NormMat* vec4(RotMat*Position, 1.0);
	//CamNormal = (ModelMat * NormMat* vec4(Normal, 0.0)).xyz;
	CamNormal = (ModelMat * NormMat* vec4(RotMat*Normal, 0.0)).xyz;
}
