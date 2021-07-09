#version 330 core

layout( location = 0 ) in vec4 in_position;
layout( location = 1 ) in vec4 in_color;

out vec4 vertex_color;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_pointsize;

void main()
{
	if (in_position.w < 0)
	{
		gl_Position = u_projection * u_view * u_model * vec4(10,-10,10,1);
		vertex_color = in_color;
		vertex_color.w = -1;
		gl_PointSize = u_pointsize;
	}
	else
	{
		gl_Position = u_projection * u_view * u_model * in_position;
		vertex_color = in_color;
		gl_PointSize = u_pointsize;
	}
}
