// g++ simulation.cpp -o simulation -lglfw -lGLEW -lGL -ldl -lm

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>
#include <iostream>
#include <cmath>

struct Body {
    glm::dvec3 position;
    glm::dvec3 velocity;
    double mass;
    float radius;
    glm::vec3 color;
};

const double G = 6.67430e-11;
const double SCALE = 1e9;
const double TIMESTEP = 60*60*24; // 1 day in seconds per step

std::vector<Body> bodies;

void computeGravity() {
    std::vector<glm::dvec3> acceleration(bodies.size(), glm::dvec3(0));

    for (size_t i = 0; i < bodies.size(); ++i) {
        for (size_t j = i + 1; j < bodies.size(); ++j) {
            glm::dvec3 offset = bodies[j].position - bodies[i].position;
            double distSqr = glm::dot(offset, offset);
            double dist = std::sqrt(distSqr);
            double force = G * bodies[i].mass * bodies[j].mass / distSqr;
            glm::dvec3 forceDir = offset / dist;

            acceleration[i] += forceDir * force / bodies[i].mass;
            acceleration[j] -= forceDir * force / bodies[j].mass;
        }
    }

    for (size_t i = 0; i < bodies.size(); ++i) {
        bodies[i].velocity += acceleration[i] * TIMESTEP;
    }
}

void updatePositions() {
    for (auto& body : bodies) {
        body.position += body.velocity * TIMESTEP;
    }
}

static void error_callback(int error, const char* description) {
    std::cerr << "Error: " << description << std::endl;
}

GLFWwindow* initWindow(int width, int height) {
    glfwSetErrorCallback(error_callback);
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return nullptr;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(width, height, "3D Celestial Simulation", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);
    glewInit();

    glEnable(GL_DEPTH_TEST);
    return window;
}


struct SphereMesh {
    std::vector<float> vertices;
    std::vector<unsigned int> indices;

    SphereMesh(unsigned int sectorCount = 36, unsigned int stackCount = 18) {
        float x, y, z, xy;                             
        float sectorStep = 2 * M_PI / sectorCount;
        float stackStep = M_PI / stackCount;
        float sectorAngle, stackAngle;

        for(unsigned int i = 0; i <= stackCount; ++i) {
            stackAngle = M_PI / 2 - i * stackStep;
            xy = cosf(stackAngle);
            z = sinf(stackAngle);

            for(unsigned int j = 0; j <= sectorCount; ++j) {
                sectorAngle = j * sectorStep;

                x = xy * cosf(sectorAngle);
                y = xy * sinf(sectorAngle);
                vertices.push_back(x);
                vertices.push_back(y);
                vertices.push_back(z);
            }
        }

        unsigned int k1, k2;
        for(unsigned int i = 0; i < stackCount; ++i) {
            k1 = i * (sectorCount + 1);
            k2 = k1 + sectorCount + 1;

            for(unsigned int j = 0; j < sectorCount; ++j, ++k1, ++k2) {
                if(i != 0) {
                    indices.push_back(k1);
                    indices.push_back(k2);
                    indices.push_back(k1 + 1);
                }
                if(i != (stackCount-1)) {
                    indices.push_back(k1 + 1);
                    indices.push_back(k2);
                    indices.push_back(k2 + 1);
                }
            }
        }
    }
};


GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info[512];
        glGetShaderInfoLog(shader, 512, nullptr, info);
        std::cerr << "Shader compile error: " << info << std::endl;
    }
    return shader;
}

GLuint createShaderProgram(const char* vertexSrc, const char* fragmentSrc) {
    GLuint vertShader = compileShader(GL_VERTEX_SHADER, vertexSrc);
    GLuint fragShader = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertShader);
    glAttachShader(program, fragShader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char info[512];
        glGetProgramInfoLog(program, 512, nullptr, info);
        std::cerr << "Program link error: " << info << std::endl;
    }

    glDeleteShader(vertShader);
    glDeleteShader(fragShader);
    return program;
}


float camYaw = 0.0f, camPitch = 0.0f;
double lastX = -1, lastY = -1;
bool firstMouse = true;

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }
    double xoffset = xpos - lastX;
    double yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    camYaw += (float)xoffset * sensitivity;
    camPitch += (float)yoffset * sensitivity;

    // Clamp pitch rotation
    if(camPitch > 89.0f) camPitch = 89.0f;
    if(camPitch < -89.0f) camPitch = -89.0f;
}

int main() {
    GLFWwindow* window = initWindow(1280, 720);
    if (!window) return -1;

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(window, mouse_callback);

    SphereMesh sphere;

    GLuint vao, vbo, ebo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sphere.vertices.size() * sizeof(float), sphere.vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphere.indices.size() * sizeof(unsigned int), sphere.indices.data(), GL_STATIC_DRAW);


    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    const char* vertexShaderSource = R"glsl(
        #version 330 core
        layout(location = 0) in vec3 aPos;
        uniform mat4 MVP;
        void main() {
            gl_Position = MVP * vec4(aPos, 1.0);
        }
    )glsl";

    const char* fragmentShaderSource = R"glsl(
        #version 330 core
        out vec4 FragColor;
        uniform vec3 uColor;
        void main() {
            FragColor = vec4(uColor, 1.0);
        }
    )glsl";

    GLuint shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);


    bodies.push_back({glm::dvec3(0,0,0), glm::dvec3(0,0,0), 1.989e30, 20.f, glm::vec3(1.0f, 1.0f, 0.f)}); // Sun

    bodies.push_back({glm::dvec3(1.496e11,0,0), glm::dvec3(0,29783,0), 5.972e24, 7.f, glm::vec3(0.2f, 0.5f, 1.f)});

    bodies.push_back({glm::dvec3(2.279e11,0,0), glm::dvec3(0,24130,0), 6.4169e23, 5.f, glm::vec3(1.f, 0.2f, 0.2f)});

    glEnable(GL_DEPTH_TEST);

    while (!glfwWindowShouldClose(window)) {
        computeGravity();
        updatePositions();

        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1280.f/720.f, 0.1f, 1000.f);
        glm::vec3 camPos = glm::vec3(
            300 * cos(glm::radians(camYaw)) * cos(glm::radians(camPitch)), 
            300 * sin(glm::radians(camPitch)),
            300 * sin(glm::radians(camYaw)) * cos(glm::radians(camPitch))
        );
        glm::mat4 view = glm::lookAt(camPos, glm::vec3(0,0,0), glm::vec3(0,1,0));
        glm::mat4 VP = projection * view;

        glUseProgram(shaderProgram);
        glBindVertexArray(vao);

        for (auto& body : bodies) {
            glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(body.position) / (float)SCALE);
            model = glm::scale(model, glm::vec3(body.radius / 10.f));  // scale spheres visually
            glm::mat4 MVP = VP * model;
            GLuint mvpLoc = glGetUniformLocation(shaderProgram, "MVP");
            glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, &MVP[0][0]);
            GLuint colorLoc = glGetUniformLocation(shaderProgram, "uColor");
            glUniform3fv(colorLoc, 1, &body.color[0]);
            glDrawElements(GL_TRIANGLES, sphere.indices.size(), GL_UNSIGNED_INT, 0);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
