#version 330 core

layout (location = 0) in vec3 a_Position;
layout (location = 1) in vec3 a_Color;

out vec3 CamNormal;
out vec3 CamPos;
out vec3 Color;
uniform mat3 RotMat;
uniform mat4 ModelMat;
uniform mat4 PerspMat;
uniform mat4 NormMat;
void main()
{
	gl_Position = PerspMat * ModelMat * NormMat*vec4(RotMat *a_Position, 1.0);
    Color = a_Color;
}