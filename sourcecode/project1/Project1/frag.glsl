uniform sampler2D qt_Texture0;
varying vec4 qt_TexCoord0;
uniform vec2 windowSize;

struct Camera {
    vec3 pos;

    vec3 up;    // up direction
    vec3 dir;   // camera pointing direction
    vec3 left;  // left direction

    float f;        // foco length
    float w, h;     // canvas size
} caminfo;

struct Light {
  vec3 eyePosOrDir;
  bool isDirectional;
  vec3 intensity;
  float attenuation;
} lights[16];

struct Sphere {
    vec3 center;
    float radius;
    vec3 color;
} spheres[128];

struct Ellipsoid {
    vec3 center;
    vec3 axis;
    vec3 color;
} ellipsoids[128];

struct Cylinder {
    vec3 center;
    vec3 axis;
    vec3 color;
    vec3 length;
} cylinders[128];

struct Cone {
    vec3 vertex;
    vec3 axis;
    float angle;
    vec3 length;
    vec3 color;
} cones[128];

struct Hyperboloid {
    vec3 vertex;

} hyperboloids[128];

struct Plane {
    vec3 point;
    vec3 normal;
    vec3 color;
} planes[128];

struct Ray {
    vec3 origin;
    vec3 dir;
};

struct Hit {
    float t;
    vec3 color;
};

void initializeCamera() {
    caminfo.pos = vec3(0.0, 0.0, -5.0);
    caminfo.up = vec3(0.0, 1.0, 0.0);
    caminfo.dir = vec3(0.0, 0.0, 1.0);
    caminfo.left = vec3(-1.0, 0.0, 0.0);
    caminfo.f = 1.0;
    caminfo.w = 1.0;
    caminfo.h = windowSize.y / windowSize.x;
}

void initializeSpheres() {
    spheres[0].center = vec3(0.0, 0.0, 1.0);
    spheres[0].radius = 1.25;
    spheres[0].color = vec3(0.25, 0.5, 1.0);
}


// initial rays
Ray constructRay(vec2 pos) {
    Ray r;
    r.origin = caminfo.pos;

    float x = pos.x / windowSize.x - 0.5;
    float y = 0.5 - pos.y / windowSize.y;

    // find the intersection point on the canvas
    vec3 pcanvas;

    vec3 canvasCenter = caminfo.f * caminfo.dir + caminfo.pos;
    pcanvas = canvasCenter + x * caminfo.w * caminfo.left + y * caminfo.h * caminfo.up;

    r.dir = normalize(pcanvas - caminfo.pos);
    return r;
}

struct Roots {
    int length;
    float x0, x1;
};

Hit rayIntersectsSphere(Ray r, Sphere s) {
    Hit h;

    vec3 pq = r.origin - s.center;
    float a = 1.0;
    float b = 2.0 * dot(pq, r.dir);
    float c = length(pq)*length(pq) - s.radius * s.radius;

    // solve the quadratic equation
    Roots roots;
    float delta = b*b - 4.0*a*c;
    if( delta < 0.0 )
    {
        roots.length = 0;
    }
    else
    {
        roots.length = 2;
        roots.x0 = (-b+sqrt(delta))/(2.0*a);
        roots.x1 = (-b-sqrt(delta))/(2.0*a);
    }

    if( roots.length == 0 )
    {
        h.t = -1.0;
        h.color = vec3(0.0, 0.0, 0.0);
        return h;
    }
    else
    {
        float THRES = 1e-8;

        float r1 = roots.x0, r2 = roots.x1;
        if( r1 > r2 ){
            r1 = roots.x1;
            r2 = roots.x0;
        }

        if( r2 < THRES ) {
            h.t = -1.0;
            h.color = vec3(0.0, 0.0, 0.0);
            return h;
        }
        else
        {
            if( r1 < THRES ) h.t = r2;
            else h.t = r1;
            h.color = s.color;
            return h;
        }
    }
}

vec3 phongShading() {
    vec3 c;
    // iterate through all lights
    return c;
}

vec3 lambertShading() {
    vec3 c;
    // iterate through all lights
    return c;
}

vec3 goochShading() {
    vec3 c;
    // iterate through all lights
    return c;
}

void main(void)
{
    initializeCamera();
    initializeSpheres();

    vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
    for(int i=0;i<16;i++) {
        float x = floor(float(i) * 0.25);
        float y = mod(float(i), 4.0);
        vec4 offsets = vec4(x*0.25, y*0.25, 0, 0);
        vec4 pos = gl_FragCoord + offsets;

        Ray r = constructRay(pos.xy);

        // test if the ray hits the sphere
        Hit hit = rayIntersectsSphere(r, spheres[0]);

        color = color + vec4(hit.color, 1.0);
    }

    gl_FragColor = color * 0.0625;
}
