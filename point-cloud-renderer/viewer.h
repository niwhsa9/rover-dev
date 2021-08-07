#include <GL/glew.h>
#include <GL/freeglut.h>
#include <vector>
#include <numeric>
#include <glm/mat4x4.hpp> 

// change this to cuda floatX types
#include <glm/vec3.hpp> // glm::vec3
#include <glm/vec4.hpp> // glm::vec4
typedef glm::vec3 vec3;
typedef glm::vec4 vec4;

// Point Cloud Graphics Object
class PointCloud {
    public:
        PointCloud() {
            
        }

};

// 3D Object
class Object3D {
    public:
        Object3D();
        ~Object3D();

        // Change the underlying model 
        void update(std::vector<vec3> pts, std::vector<vec3> colors, std::vector<int> idcs = std::vector<int>());
        void draw();

        // Allow rotation and translation of the underlying model
        void setTranslation(float x, float y, float z);
        void setRotation(float pitch, float roll, float yaw);
    
    private:
        // Model
        std::vector<vec3> points;
        std::vector<vec3> colors;
        std::vector<int> indicies;

        // State variables
        glm::mat4 translation;
        glm::mat4 rotation;

        // GPU representation 
        // https://gamedev.stackexchange.com/questions/8042/whats-the-purpose-of-opengls-vertex-array-objects
        GLuint vaoID;
        GLuint pointsGPU;
        GLuint colorsGPU;
        GLuint indiciesGPU;
};

// Defines the view 
struct Camera {
    vec3 position;
    glm::mat4 view;
    glm::mat4 projection;
};

// Vertex and Frag shader, stole this straight from ZED example 
// but this is basically the same in every OpenGL program so 
// it will work even without ZED 
class Shader {
public:

    Shader() {}
    Shader(GLchar* vs, GLchar* fs);
    ~Shader();
    GLuint getProgramId();

    static const GLint ATTRIB_VERTICES_POS = 0;
    static const GLint ATTRIB_COLOR_POS = 1;
private:
    bool compile(GLuint &shaderId, GLenum type, GLchar* src);
    GLuint verterxId_;
    GLuint fragmentId_;
    GLuint programId_;
};

class Viewer {
    public:
        // Creates a window
        Viewer(int argc, char **argv);

        // Updates the window and draws graphics (graphics thread)
        void update();

        void addObject(Object3D &obj);

    private:
        // Internals
        Camera camera;
        std::vector<Object3D> objects;
        Shader objectShader;
        std::vector<PointCloud> pointClouds;
        
};

