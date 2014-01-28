struct Camera {
    vec3 pos;

    vec3 up;    // up direction
    vec3 dir;   // camera pointing direction
    vec3 right;  // right direction

    float f;        // foco length
    float w, h;     // canvas size
} caminfo;

const int POINT_LIGHT = 0;
const int DIRECTIONAL_LIGHT = 1;
const int SPOT_LIGHT = 2;

struct Light {
    int type;	// POINT = 0, DIRECTIONAL = 1, SPOT = 2

    float intensity;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    vec3 pos;
    vec3 dir;

    float spotExponent;
    float spotCutOff;
    float spotCosCutOff;

    vec3 attenuation;   // K0, K1, K2
};

// shape types
const int SPHERE = 0;
const int PLANE = 1;
const int ELLIPSOID = 2;
const int CYLINDER = 3;
const int CONE = 4;
const int HYPERBOLOID = 5;

struct Material {
    // material property
    vec3 emission;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    vec3 kcool;
    vec3 kwarm;

    float shininess;
};

struct Shape {
    int type;           // sphere = 0, plane = 1, ellipsoid = 2
                        // cylinder = 3, cone = 4, hyperboloid = 5

    vec3 p;             // center for sphere and ellipsoid, plane
                        // vertex for cylinder and cone, hyperboloid
    vec3 axis0, axis1, axis2;       // axes for ellipsoid, main axis for cylinder, cone and hyperboloid
                        // normal and u, v for plane
    vec3 radius;    // radius for sphere, ellipsoid, cylinder, width and height for plane
    float angle;        // open angle for cone
    float height;       // for cylinder and cone

    mat3 m;             // for ellipsoid


    // material property
    vec3 emission;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    vec3 kcool;
    vec3 kwarm;

    float shininess;
    float alpha, beta;

    bool hasTexture;
    int tex;

    bool hasNormalMap;
    int nTex;
};

struct Ray {
    vec3 origin;
    vec3 dir;
};

struct Hit {
    float t;
    vec3 color;
};
