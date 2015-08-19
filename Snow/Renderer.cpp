#include "Renderer.h"

using namespace std;

static const float radius = 0.008f;

Renderer::Renderer(int width, int height, solverParams* sp) :
width(width),
height(height),
plane(Shader("plane.vert", "plane.frag")),
snow(Shader("snow.vert", "snow.frag"))
{
	this->sp = sp;
	aspectRatio = float(width) / float(height);

	GLfloat floorVertices[] = {
		sp->boxCorner2.x, sp->boxCorner1.y, sp->boxCorner2.z,
		sp->boxCorner2.x, sp->boxCorner1.y, sp->boxCorner1.z,
		sp->boxCorner1.x, sp->boxCorner1.y, sp->boxCorner1.z,
		sp->boxCorner1.x, sp->boxCorner1.y, sp->boxCorner2.z
	};

	GLfloat wallVertices[] = {
		sp->boxCorner1.x, sp->boxCorner1.y, sp->boxCorner2.z,
		sp->boxCorner1.x, sp->boxCorner1.y, sp->boxCorner1.z,
		sp->boxCorner1.x, sp->boxCorner2.y, sp->boxCorner1.z,
		sp->boxCorner1.x, sp->boxCorner2.y, sp->boxCorner2.z
	};

	GLuint indices[] = {
		0, 1, 3,
		1, 2, 3
	};

	//Wall
	glGenVertexArrays(1, &wallBuffers.vao);

	glGenBuffers(1, &wallBuffers.vbo);
	glBindBuffer(GL_ARRAY_BUFFER, wallBuffers.vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(wallVertices), wallVertices, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &wallBuffers.ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, wallBuffers.ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	//Floor
	glGenVertexArrays(1, &floorBuffers.vao);

	glGenBuffers(1, &floorBuffers.vbo);
	glBindBuffer(GL_ARRAY_BUFFER, floorBuffers.vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(floorVertices), floorVertices, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &floorBuffers.ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, floorBuffers.ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

}

Renderer::~Renderer() {
	if (snowBuffers.vao != 0) {
		glDeleteVertexArrays(1, &snowBuffers.vao);
		glDeleteBuffers(1, &snowBuffers.positions);
	}

	if (wallBuffers.vao != 0) {
		glDeleteVertexArrays(1, &wallBuffers.vao);
		glDeleteBuffers(1, &wallBuffers.vbo);
		glDeleteBuffers(1, &wallBuffers.ebo);
	}

	if (floorBuffers.vao != 0) {
		glDeleteVertexArrays(1, &floorBuffers.vao);
		glDeleteBuffers(1, &floorBuffers.vbo);
		glDeleteBuffers(1, &floorBuffers.ebo);
	}
}

void Renderer::setProjection(glm::mat4 projection) {
	this->projection = projection;
}

void Renderer::initSnowBuffers(int numParticles) {
	glGenVertexArrays(1, &snowBuffers.vao);

	glGenBuffers(1, &snowBuffers.positions);
	glBindBuffer(GL_ARRAY_BUFFER, snowBuffers.positions);
	glBufferData(GL_ARRAY_BUFFER, numParticles * 3 * sizeof(float), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	cudaGraphicsGLRegisterBuffer(&resource, snowBuffers.positions, cudaGraphicsRegisterFlagsWriteDiscard);

	snowBuffers.numParticles = numParticles;
}

void Renderer::render(Camera& cam) {
	//Set model view matrix
	mView = cam.getMView();

	//Clear buffer
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//Plane
	renderPlane(wallBuffers);
	renderPlane(floorBuffers);

	//Snow
	renderSnow(cam);
}

void Renderer::renderPlane(planeBuffers &buf) {
	glUseProgram(plane.program);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	plane.setUniformmat4("mView", mView);
	plane.setUniformmat4("projection", projection);

	glBindVertexArray(buf.vao);
	glBindBuffer(GL_ARRAY_BUFFER, buf.vbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buf.ebo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void Renderer::renderSnow(Camera& cam) {
	glUseProgram(snow.program);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	snow.setUniformmat4("mView", mView);
	snow.setUniformmat4("projection", projection);
	snow.setUniformf("pointRadius", radius);
	snow.setUniformf("pointScale", width / aspectRatio * (1.0f / tanf(cam.zoom * 0.5f)));

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable(GL_CULL_FACE);

	//Draw snow
	glBindVertexArray(snowBuffers.vao);
	glBindBuffer(GL_ARRAY_BUFFER, snowBuffers.positions);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	glDrawArrays(GL_POINTS, 0, GLsizei(snowBuffers.numParticles));
}