#version 150 core

in vec3 pos;

uniform mat4 mView;
uniform mat4 projection;
uniform float pointRadius;

out vec4 fragColor;

void main() {
	//calculate normal
	vec3 normal;
	normal.xy = gl_PointCoord * 2.0 - 1.0;
	float r2 = dot(normal.xy, normal.xy);
	
	if (r2 > 1.0) {
		discard;
	}
	
	fragColor = vec4(1);
}