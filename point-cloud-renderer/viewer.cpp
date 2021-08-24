#include "viewer.h"
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;

// Took the shaders from the ZED code :) 
GLchar* OBJ_VERTEX_SHADER =
"#version 130\n"
"in vec3 in_Vertex;\n"
"in vec3 in_Color;\n"
"uniform mat4 u_mvpMatrix;\n"
"out vec3 b_color;\n"
"void main() {\n"
"   b_color = in_Color;\n"
"	gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);\n"
"}"; 

GLchar* OBJ_FRAGMENT_SHADER =
"#version 130\n"
"in vec3 b_color;\n"
"out vec4 out_Color;\n"
"void main() {\n"
"   out_Color = vec4(b_color, 1);\n"
"}";

GLchar* PC_VERTEX_SHADER =
"#version 130\n"
"in vec3 in_Vertex;\n"
"in uint in_Color;\n"
"uniform mat4 u_mvpMatrix;\n"
"out vec4 b_color;\n"
"void main() {\n"
"   uint q = in_Color;\n"
//"   uint q = uint(0xFF0000FF);\n"
"   float r = float(q & uint(0x000000FF))/255.0f;\n"
"   float g = float( (q & uint(0x0000FF00)) >> 8 )/255.0f;\n"
"   float b = float( (q & uint(0x00FF0000)) >> 16)/255.0f;\n"
"   b_color = vec4(r, g, b, 1.f);\n"
//"   b_color = vec4(0.0f, 1.0f, 0.0f, 1.f);\n"
"	gl_Position = u_mvpMatrix * vec4(in_Vertex.xyz, 1);\n"
"}";

GLchar* PC_FRAGMENT_SHADER =
"#version 130\n"
"in vec4 b_color;\n"
"out vec4 out_Color;\n"
"void main() {\n"
"   out_Color = b_color;\n"
"}";

/*
 * Shader 
 */

Shader::Shader(GLchar* vs, GLchar* fs) {
    if (!compile(verterxId_, GL_VERTEX_SHADER, vs)) {
        cout << "ERROR: while compiling vertex shader" << endl;
    }
    if (!compile(fragmentId_, GL_FRAGMENT_SHADER, fs)) {
        cout << "ERROR: while compiling vertex shader" << endl;
    }

    programId_ = glCreateProgram();

    glAttachShader(programId_, verterxId_);
    glAttachShader(programId_, fragmentId_);

    glBindAttribLocation(programId_, ATTRIB_VERTICES_POS, "in_Vertex");
    glBindAttribLocation(programId_, ATTRIB_COLOR_POS, "in_Color");

    glLinkProgram(programId_);

    GLint errorlk(0);
    glGetProgramiv(programId_, GL_LINK_STATUS, &errorlk);
    if (errorlk != GL_TRUE) {
        cout << "ERROR: while linking shader" << endl;
        GLint errorSize(0);
        glGetProgramiv(programId_, GL_INFO_LOG_LENGTH, &errorSize);

        char *error = new char[errorSize + 1];
        glGetShaderInfoLog(programId_, errorSize, &errorSize, error);
        error[errorSize] = '\0';
        std::cout << error << std::endl;

        delete[] error;
        glDeleteProgram(programId_);
    }
}

Shader::~Shader() {
    if (verterxId_ != 0)
        glDeleteShader(verterxId_);
    if (fragmentId_ != 0)
        glDeleteShader(fragmentId_);
    if (programId_ != 0)
        glDeleteShader(programId_);
}

GLuint Shader::getProgramId() {
    return programId_;
}

bool Shader::compile(GLuint &shaderId, GLenum type, GLchar* src) {
    shaderId = glCreateShader(type);
    if (shaderId == 0) {
        return false;
    }
    glShaderSource(shaderId, 1, (const char**) &src, 0);
    glCompileShader(shaderId);

    GLint errorCp(0);
    glGetShaderiv(shaderId, GL_COMPILE_STATUS, &errorCp);
    if (errorCp != GL_TRUE) {
        cout << "ERROR: while compiling shader" << endl;
        GLint errorSize(0);
        glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &errorSize);

        char *error = new char[errorSize + 1];
        glGetShaderInfoLog(shaderId, errorSize, &errorSize, error);
        error[errorSize] = '\0';
        std::cout << error << std::endl;

        delete[] error;
        glDeleteShader(shaderId);
        return false;
    }
    return true;
}


/* 
 * 3D Object
 */

Object3D::Object3D() {
    glGenVertexArrays(1, &vaoID);
    glGenBuffers(1, &pointsGPU);
    glGenBuffers(1, &colorsGPU);
    glGenBuffers(1, &indiciesGPU);
}

Object3D::Object3D(std::vector<vec3> &pts, std::vector<vec3> &colors, std::vector<int> &idcs) {
    glGenVertexArrays(1, &vaoID);
    glGenBuffers(1, &pointsGPU);
    glGenBuffers(1, &colorsGPU);
    glGenBuffers(1, &indiciesGPU);
    update(pts, colors, idcs);
}


void Object3D::draw() {
    glBindVertexArray(vaoID);
    //std::cout << indicies.size() << std::endl;
    glDrawElements(GL_TRIANGLES, (GLsizei) indicies.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Object3D::update(std::vector<vec3> &pts, std::vector<vec3> &colors, std::vector<int> &idcs) {
    // Update internal CPU representations 
    // We might not actually even need this 
    points = pts;
    colors = colors;
    indicies = idcs;
    
    //Provide default initialization with: ??
    //std::iota(indicies.begin(), indicies.end(), 0);

    // Update GPU data for rendering 
    glBindVertexArray(vaoID);
    // Points
    glBindBuffer(GL_ARRAY_BUFFER, pointsGPU);
    glBufferData(GL_ARRAY_BUFFER, points.size()*sizeof(vec3), &points[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    // Colors
    glBindBuffer(GL_ARRAY_BUFFER, colorsGPU);
    glBufferData(GL_ARRAY_BUFFER, colors.size()*sizeof(vec3), &colors[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
    // Indicies
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indiciesGPU);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indicies.size()*sizeof(unsigned int), &indicies[0], GL_DYNAMIC_DRAW);
    // Unbind 
    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

Object3D::~Object3D() {
    glDeleteVertexArrays(1, &vaoID);
    glDeleteBuffers(1, &pointsGPU);
    glDeleteBuffers(1, &colorsGPU);
    glDeleteBuffers(1, &indiciesGPU);
}

/*
 * Point Cloud
 */

PointCloud::PointCloud() {
    glGenVertexArrays(1, &vaoID);
    glGenBuffers(1, &pointsGPU);
}

PointCloud::~PointCloud() {
    glDeleteVertexArrays(1, &vaoID);
    glDeleteBuffers(1, &pointsGPU);
}

void PointCloud::update(std::vector<vec4> &pts) {
    update(&pts[0], pts.size());
}

void PointCloud::update(vec4* pts, int size) {
    this->size = size;

    // Update GPU data for rendering 
    glBindVertexArray(vaoID);
    // Points
    glBindBuffer(GL_ARRAY_BUFFER, pointsGPU);
    glBufferData(GL_ARRAY_BUFFER, size*sizeof(vec4), pts, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vec4), 0);
    glEnableVertexAttribArray(0);
    // Color
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(vec4), (void*)offsetof(vec4, w));
    glEnableVertexAttribArray(1);
    // Unbind 
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void PointCloud::draw() {
    glBindVertexArray(vaoID);
    glDrawArrays(GL_POINTS, 0, size);
    glBindVertexArray(0);
}

/* 
 * Camera
 */
void Camera::updateView() {
    view = glm::lookAt(eye, target, glm::vec3(0.0f, -1.0f, 0.0f));
}

/* 
 * Viewer
 */

Viewer* curInstance; 

Viewer::Viewer(int argc, char **argv) {
    // Window stuff
    glutInit(&argc, argv);
    int wnd_w = 1920*0.7;//glutGet(GLUT_SCREEN_WIDTH);
    int wnd_h = 1080*0.7;//glutGet(GLUT_SCREEN_HEIGHT);
    glutInitWindowSize(1920*0.7, 1080*0.7);
    glutInitWindowPosition(wnd_w*0.05, wnd_h*0.05);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutCreateWindow("Display");

    // Glew (loads opengl api backend)
    GLenum err = glewInit();
    if (GLEW_OK != err)
        cout << "Error w/ viewer";

    // Options
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
    glEnable(GL_DEPTH_TEST | GL_PROGRAM_POINT_SIZE);

    // Callbacks
    cout << "creating callbacks" << endl;
    glutMouseFunc(Viewer::mouseButtonCallback);
    glutMotionFunc(Viewer::mouseMotionCallback);
    //glutKeyboardFunc(keyboard);
    cout << "callbacks created" << endl;
    curInstance = this;

    // Shader
    objectShader = Shader(OBJ_VERTEX_SHADER, OBJ_FRAGMENT_SHADER);
    pcShader = Shader(PC_VERTEX_SHADER, PC_FRAGMENT_SHADER);

    // Camera
    camera.projection = glm::perspective(glm::radians(45.0f), 1920.0f/1080.0f, 0.1f, 100000.0f);
    camera.updateView();
}

// Viewer tick
void Viewer::update() {
    // Basic drawing setup 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0, 0, 0.0, 1.f);
    //glLineWidth(2.f);
    glPointSize(6.f); // 1, 3
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // Draw 3D Objects
    glUseProgram(objectShader.getProgramId());
    for(auto &object : objects) {
        glm::mat4 mvp_mat = camera.projection * camera.view;
        glUniformMatrix4fv(glGetUniformLocation(objectShader.getProgramId(), "u_mvpMatrix"), 1, GL_FALSE, glm::value_ptr(mvp_mat));
        object.draw();
    }
    viewer_mutex.lock();
    for(auto &object : ephemeralObjects) {
        glm::mat4 mvp_mat = camera.projection * camera.view;
        glUniformMatrix4fv(glGetUniformLocation(objectShader.getProgramId(), "u_mvpMatrix"), 1, GL_FALSE, glm::value_ptr(mvp_mat));
        object.draw();
    }
    viewer_mutex.unlock();

    glUseProgram(pcShader.getProgramId());
    pc_mutex.lock();
    for (auto &pc : pointClouds) {
        glm::mat4 mvp_mat = camera.projection * camera.view;
        glUniformMatrix4fv(glGetUniformLocation(objectShader.getProgramId(), "u_mvpMatrix"), 1, GL_FALSE, glm::value_ptr(mvp_mat));
        pc.draw();
    }
    pc_mutex.unlock();

    // Update display
    glutMainLoopEvent();
    glutSwapBuffers();
    glutPostRedisplay();
}

void Viewer::addObject(Object3D &obj, bool ephemeral) {
    viewer_mutex.lock();
    if(ephemeral) ephemeralObjects.push_back(obj);
    else objects.push_back(obj);
    viewer_mutex.unlock();
}

void Viewer::clearEphemerals() {
    viewer_mutex.lock();
    ephemeralObjects.clear();
    viewer_mutex.unlock();
}

void Viewer::updatePointCloud(int idx, vec4* pts, int size) {
    pc_mutex.lock();
    pointClouds[idx].update(pts, size);
    pc_mutex.unlock();
}

void Viewer::addPointCloud() {
    pc_mutex.lock();
    pointClouds.push_back(PointCloud());
    pc_mutex.unlock();
}

void Viewer::mouseButtonCallback(int button, int state, int x, int y) {
    /*
    if (button < 5) {
        if (button < 3) {
            currentInstance_->mouseButton_[button] = state == GLUT_DOWN;
        } else {
            currentInstance_->mouseWheelPosition_ += button == MOUSE_BUTTON::WHEEL_UP ? 1 : -1;
        }
        currentInstance_->mouseCurrentPosition_[0] = x;
        currentInstance_->mouseCurrentPosition_[1] = y;
        currentInstance_->previousMouseMotion_[0] = x;
        currentInstance_->previousMouseMotion_[1] = y;
    }*/
    cout << "button" << endl;

    // wheel up
    if(button == 3) {
        curInstance->camera.eye += 20.0f*glm::normalize(curInstance->camera.target - curInstance->camera.eye); 
    } 
    // wheel down
    else if(button == 4) {
        curInstance->camera.eye -= 20.0f*glm::normalize(curInstance->camera.target - curInstance->camera.eye); 

    }

    curInstance->camera.updateView();
    glutPostRedisplay();

}

void Viewer::mouseMotionCallback(int x, int y) {
    //cout << "mouse" << endl;
    
    //curInstance->camera.view = glm::lookAt(glm::vec3(0.0f, 0.0f, -3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
    //curInstance->prevMouseX = 0;

}

void Viewer::keyPressedCallback(unsigned char c, int x, int y) {
    cout << "key press" << endl;
}


/* 
 * Main
 */

int main(int argc, char **argv) {
    cout << "Hello" << endl;
    Viewer viewer(argc, argv);

    vector<vec3> points = {vec3(-0.5f, 0.5f, -0.5f), vec3(0.5f, 0.5f, -0.5f), vec3(0.5f, -0.5f, -0.5f), vec3(-0.5f, -0.5f, -0.5f)};
    vector<vec3> colors = {vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, 0.0f, 1.0f)};
    vector<int> indicies = {0, 1, 2/*, 3, 2, 0*/};
    Object3D obj(points, colors, indicies);
    //viewer.addObject(obj, false);

    
    //vec4 pc[4] = {vec4(-0.5f, 0.5f, -0.5f, 9.18340948595e-41), vec4(0.5f, 0.5f, -0.5f, 3.57331108403e-43), vec4(0.5f, -0.5f, -0.5f, 9.14767637511e-41), vec4(-0.5f, -0.5f, -0.5f, 2.34184088514e-38)};
    PCDReader pcd_reader;
    vector<vec4> pc = pcd_reader.readCloud("data/pcl28.pcd");
    //cout << "pc size " <<  pc.size() << endl;
    viewer.addPointCloud();
    viewer.updatePointCloud(0, &pc[0], pc.size());
    //viewer.updatePointCloud(0, pc, 4);

    while(true) {
        //cout << "updating" << endl;
        /*
        points[0].x -= 0.01f;
        Object3D ob2(points, colors, indicies);
        viewer.addObject(ob2, true); */

        viewer.update();
        viewer.clearEphemerals();
    }
}