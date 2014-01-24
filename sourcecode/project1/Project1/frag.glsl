uniform sampler2D qt_Texture0;
varying vec4 qt_TexCoord0;

uniform vec2 windowSize;
uniform int lightCount;
uniform int shadingMode;    // 1 = lambert, 2 = phong, 3 = gooch, 4 = cook-torrance

// for gooch shading
uniform vec3 kcool, kwarm;
uniform vec3 kdiff, kspec;
uniform vec3 kpos;
uniform float alpha, beta;

struct Camera {
    vec3 pos;

    vec3 up;    // up direction
    vec3 dir;   // camera pointing direction
    vec3 right;  // right direction

    float f;        // foco length
    float w, h;     // canvas size
} caminfo;

struct Light {
    float intensity;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    vec3 pos;

    bool isDirectional;
    bool isSpot;
    vec3 dir;
    float spotExponent;
    float spotCutoff;
    float spotCosCutoff;

    vec3 attenuation;   // K0, K1, K2
} lights[16];

struct Sphere {
    vec3 center;
    float radius;

    // material property
    vec3 emission;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
} spheres[128];

struct Ellipsoid {
    vec3 center;
    vec3 axis[3];
    float radius[3];

    // material property
    vec3 emission;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
} ellipsoids[128];

struct Cylinder {
    vec3 center;
    vec3 axis;
    float length;
    float radius;

    // material property
    vec3 emission;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
} cylinders[128];

struct Cone {
    vec3 vertex;
    vec3 axis;
    float angle;
    float length;

    // material property
    vec3 emission;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
} cones[128];

struct Hyperboloid {
    vec3 vertex;

    // material property
    vec3 emission;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
} hyperboloids[128];

struct Plane {
    vec3 point;
    vec3 normal, u, v;
    float w, h;

    // material property
    vec3 emission;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
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
    caminfo.right = vec3(1.0, 0.0, 0.0);
    caminfo.f = 1.0;
    caminfo.w = 1.0;
    caminfo.h = windowSize.y / windowSize.x;
}

void initializeLights() {
    lights[0].intensity = 1.0;
    lights[0].ambient = vec3(1.0, 1.0, 1.0);
    lights[0].diffuse = vec3(0.75, 0.95, 0.75);
    lights[0].specular = vec3(1.0, 1.0, 0.0);
    lights[0].pos = vec3(-2.0, 2.0, -2.0);

    lights[1].intensity = 0.5;
    lights[1].ambient = vec3(1.0, 1.0, 1.0);
    lights[1].diffuse = vec3(0.75, 0.95, 0.75);
    lights[1].specular = vec3(0.0, 1.0, 1.0);
    lights[1].pos = vec3(4.0, 4.0, -4.0);
}

void initializeSpheres() {
    spheres[0].center = vec3(0.0, 0.0, 1.0);
    spheres[0].radius = 1.0;

    spheres[0].diffuse = vec3(0.25, 0.5, 1.0);
    spheres[0].specular = vec3(1.0, 1.0, 1.0);
    spheres[0].ambient = vec3(0.05, 0.10, 0.15);
    spheres[0].shininess = 50.0;
}


// initial rays
Ray constructRay(vec2 pos) {
    Ray r;
    r.origin = caminfo.pos;

    float x = pos.x / windowSize.x - 0.5;
    float y = pos.y / windowSize.y - 0.5;

    // find the intersection point on the canvas
    vec3 pcanvas;

    vec3 canvasCenter = caminfo.f * caminfo.dir + caminfo.pos;
    pcanvas = canvasCenter + x * caminfo.w * caminfo.right + y * caminfo.h * caminfo.up;

    r.dir = normalize(pcanvas - caminfo.pos);
    return r;
}

struct Roots {
    int length;
    float x0, x1;
};


vec3 phongShading(vec3 v, vec3 N, vec3 eyePos, vec3 diffuse, vec3 ambient, vec3 specular, float shininess) {
    vec3 c = vec3(0, 0, 0);

    // iterate through all lights
    for(int i=0;i<lightCount;i++) {

        vec3 L = normalize(lights[i].pos - v);
        vec3 E = normalize(eyePos-v);
        vec3 R = normalize(-reflect(L,N));

        //calculate Ambient Term:
        vec3 Iamb = ambient * lights[i].ambient;

        //calculate Diffuse Term:
        vec3 Idiff = diffuse * lights[i].diffuse * max(dot(N,L), 0.0);
        Idiff = clamp(Idiff, 0.0, 1.0);

        // calculate Specular Term:
        vec3 Ispec = specular * lights[i].specular
                     * pow(max(dot(R,E),0.0),0.3*shininess);
        Ispec = clamp(Ispec, 0.0, 1.0);

        c = c + (Idiff + Ispec + Iamb) * lights[i].intensity;
    }

    return c;
}

vec3 lambertShading(vec3 v, vec3 N, vec3 eyePos, vec3 diffuse) {
    vec3 c = vec3(0, 0, 0);

    // iterate through all lights
    for(int i=0;i<lightCount;i++) {
        vec3 L = normalize(lights[i].pos - v);

        vec3 Idiff = clamp(diffuse * lights[i].diffuse * max(dot(N, L), 0.0), 0.0, 1.0);

        c = c + Idiff * lights[i].intensity;
    }

    return c;
}

vec3 goochShading(vec3 v, vec3 N, vec3 eyePos, vec3 diffuse, vec3 specular, float shininess) {
    vec3 L = normalize(kpos - v);
    vec3 E = normalize(eyePos-v);
    vec3 R = normalize(-reflect(L,N));

    float NdotL = dot(N, L);
    vec3 Idiff = diffuse * kdiff * NdotL;

    vec3 kcdiff = min(kcool + alpha * Idiff, 1.0);
    vec3 kwdiff = min(kwarm + beta * Idiff, 1.0);

    vec3 kfinal = mix(kcdiff, kwdiff, (NdotL+1.0)*0.5);

    // calculate Specular Term:
    vec3 Ispec = specular * kspec
                 * pow(max(dot(R,E),0.0),0.3*shininess);
    Ispec = clamp(Ispec, 0.0, 1.0);

    return min(kfinal + Ispec, 1.0);
}


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

            // hit point
            vec3 p = h.t * r.dir + r.origin;
            // normal at hit point
            vec3 n = normalize(p - s.center);

            if( shadingMode == 1 )
                h.color = lambertShading(p, n, r.origin, s.diffuse);
            else if( shadingMode == 2 )
                h.color = phongShading(p, n, r.origin, s.diffuse, s.ambient, s.specular, s.shininess);
            else if( shadingMode == 3 )
                h.color = goochShading(p, n, r.origin, s.diffuse, s.specular, s.shininess);

            return h;
        }
    }
}

void main(void)
{
    initializeCamera();
    initializeLights();
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
