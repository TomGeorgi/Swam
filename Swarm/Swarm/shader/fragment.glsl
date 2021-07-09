#version 330 core

in vec4 vertex_color;
out vec4 frag_color;

void main()
{
	vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
	if ( dot( circCoord, circCoord ) > 1.0 || vertex_color.w < 0)
	{
		discard;
	}

	frag_color = vertex_color;
}
