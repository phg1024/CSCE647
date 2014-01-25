uniform sampler2D qt_Texture0;
varying vec4 qt_TexCoord0;

uniform vec2 windowSize;
uniform int lightCount;
uniform int shapeCount;
uniform int shadingMode;    // 1 = lambert, 2 = phong, 3 = gooch, 4 = cook-torrance
uniform int AAsamples;

// camera info
uniform vec3 camPos, camUp, camDir;
uniform float camF;


// for gooch shading
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
} lights[4];

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
    vec3 axis[3];       // axes for ellipsoid, main axis for cylinder, cone and hyperboloid
                        // normal and u, v for plane
    float radius[3];    // radius for sphere, ellipsoid, cylinder, width and height for plane
    float angle;        // open angle for cone

    float height;       // for cylinder and cone

    // material property
    vec3 emission;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    vec3 kcool;
    vec3 kwarm;

    float shininess;
} shapes[8];

struct Ray {
    vec3 origin;
    vec3 dir;
};

struct Hit {
    float t;
    vec3 color;
};

Hit background;

void initializeCamera() {
    caminfo.pos = camPos;
    caminfo.up = camUp;
    caminfo.dir = camDir;

    caminfo.right = cross(caminfo.dir, caminfo.up);
    caminfo.f = camF;
    caminfo.w = 1.0;
    caminfo.h = windowSize.y / windowSize.x;
}

void initializeLights() {
    int idx = 0;
    lights[idx].intensity = 1.0;
    lights[idx].ambient = vec3(1.0, 1.0, 1.0);
    lights[idx].diffuse = vec3(0.75, 0.75, 0.75);
    lights[idx].specular = vec3(1.0, 1.0, 0.0);
    lights[idx].pos = vec3(-2.0, 2.0, -10.0);

    idx++;
    lights[idx].intensity = 0.25;
    lights[idx].ambient = vec3(1.0, 1.0, 1.0);
    lights[idx].diffuse = vec3(0.75, 0.75, 0.75);
    lights[idx].specular = vec3(0.0, 1.0, 1.0);
    lights[idx].pos = vec3(4.0, 4.0, -10.0);

    idx++;
    lights[idx].intensity = 0.25;
    lights[idx].ambient = vec3(1.0, 1.0, 1.0);
    lights[idx].diffuse = vec3(0.75, 0.75, 0.75);
    lights[idx].specular = vec3(0.0, 1.0, 1.0);
    lights[idx].pos = vec3(0.0, 1.0, -10.0);
}

void initializeShapes() {
    int idx = 0;
    shapes[idx].type = SPHERE;
    shapes[idx].p = vec3(0.0, 0.0, 1.0);
    shapes[idx].radius[0] = 1.0;

    shapes[idx].diffuse = vec3(0.25, 0.5, 1.0);
    shapes[idx].specular = vec3(1.0, 1.0, 1.0);
    shapes[idx].ambient = vec3(0.05, 0.10, 0.15);
    shapes[idx].shininess = 50.0;
    shapes[idx].kcool = vec3(0, 0, 0.4);
    shapes[idx].kwarm = vec3(0.4, 0.4, 0);


    idx++;
    shapes[idx].type = PLANE;
    shapes[idx].p = vec3(0.0, -1.0, 0.0);
    shapes[idx].axis[0] = vec3(0.0, 1.0, 0.0);    // normal
    shapes[idx].axis[1] = vec3(1.0, 0.0, 0.0);    // u
    shapes[idx].axis[2] = vec3(0.0, 0.0, 1.0);    // v
    shapes[idx].radius[0] = 3.0;  // width
    shapes[idx].radius[1] = 6.0;  // height

    shapes[idx].diffuse = vec3(0.75, 0.75, 0.75);
    shapes[idx].specular = vec3(1.0, 1.0, 1.0);
    shapes[idx].ambient = vec3(0.05, 0.05, 0.05);
    shapes[idx].shininess = 50.0;

    idx++;
    shapes[idx].type = SPHERE;
    shapes[idx].p = vec3(-0.5, 0.5, -1.0);
    shapes[idx].radius[0] = 0.25;

    shapes[idx].diffuse = vec3(0.25, 0.75, 0.25);
    shapes[idx].specular = vec3(1.0, 1.0, 1.0);
    shapes[idx].ambient = vec3(0.05, 0.05, 0.05);
    shapes[idx].shininess = 5.0;

    shapes[idx].kcool = vec3(0, 0.4, 0.0);
    shapes[idx].kwarm = vec3(0.4, 0.0, 0.4);

    idx++;
    shapes[idx].type = SPHERE;
    shapes[idx].p = vec3(0.75, -0.5, -0.5);
    shapes[idx].radius[0] = 0.5;

    shapes[idx].diffuse = vec3(0.75, 0.75, 0.25);
    shapes[idx].specular = vec3(1.0, 1.0, 1.0);
    shapes[idx].ambient = vec3(0.15, 0.05, 0.05);
    shapes[idx].shininess = 100.0;

    shapes[idx].kcool = vec3(0.9, 0.1, 0.6);
    shapes[idx].kwarm = vec3(0.05, 0.45, 0.05);
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

// light ray intersection tests
float lightRayIntersectsSphere(Ray r, Shape s) {
    vec3 pq = r.origin - s.p;
    float a = 1.0;
    float b = 2.0 * dot(pq, r.dir);
    float c = length(pq)*length(pq) - s.radius[0] * s.radius[0];

    // solve the quadratic equation
    float delta = b*b - 4.0*a*c;
    if( delta < 0.0 )
    {
        return -1.0;
    }
    else
    {
        float x0 = (-b+sqrt(delta))/(2.0*a);
        float x1 = (-b-sqrt(delta))/(2.0*a);

        float THRES = 1e-4;

        float r1 = x0, r2 = x1;
        if( r1 > r2 ){
            r1 = x1;
            r2 = x0;
        }

        if( r2 < THRES ) {
            return -1.0;
        }
        else
        {
            if( r1 < THRES ) return r2;
            else return r1;
        }
    }
}

float lightRayIntersectsPlane(Ray r, Shape s) {
    float t = 1e10;
    float THRES = 1e-4;
    vec3 pq = s.p - r.origin;
    float ldotn = dot(s.axis[0], r.dir);
    if( abs(ldotn) < THRES ) return -1.0;
    else {
        t = dot(s.axis[0], pq) / ldotn;

        if( t >= THRES ) {
            vec3 p = r.origin + t * r.dir;

            // compute u, v coordinates
            vec3 pp0 = p - s.p;
            float u = dot(pp0, s.axis[1]);
            float v = dot(pp0, s.axis[2]);
            if( abs(u) > s.radius[0] || abs(v) > s.radius[1] ) return -1.0;
            else {
                return t;
            }
        }
        else return -1.0;
    }
}

float lightRayIntersectsShape(Ray r, Shape s) {
    if(s.type == SPHERE) return lightRayIntersectsSphere(r, s);
    else if( s.type == PLANE ) return lightRayIntersectsPlane(r, s);
    else return -1.0;
}

float lightRayIntersectsShapes(Ray r) {
    // go through a list of shapes and find closest hit
    float t = 1e10;

    for(int i=0;i<shapeCount;i++) {
        float hitT = lightRayIntersectsShape(r, shapes[i]);
        if( (hitT >= 0.0) && (hitT < t) ) {
            t = hitT;
        }
    }

    return t;
}

bool checkLightVisibility(vec3 p, Light lt) {
    float dist = length(p - lt.pos);
    Ray r;
    r.origin = p;
    r.dir = normalize(lt.pos - p);
    float t = lightRayIntersectsShapes(r);

    float THRES = 1e-6;
    return t < THRES || t > dist;
}


vec3 phongShading(vec3 v, vec3 N, vec3 eyePos, vec3 diffuse, vec3 ambient, vec3 specular, float shininess) {
    vec3 c = vec3(0, 0, 0);

    // iterate through all lights
    for(int i=0;i<lightCount;i++) {

        // determine if this light is visible
        bool isVisible = checkLightVisibility(v, lights[i]);

        //calculate Ambient Term:
        vec3 Iamb = ambient * lights[i].ambient;

        if( isVisible ) {
            vec3 L = normalize(lights[i].pos - v);
            vec3 E = normalize(eyePos-v);
            vec3 R = normalize(-reflect(L,N));

            //calculate Diffuse Term:
            vec3 Idiff = diffuse * lights[i].diffuse * max(dot(N,L), 0.0);
            Idiff = clamp(Idiff, 0.0, 1.0);

            // calculate Specular Term:
            vec3 Ispec = specular * lights[i].specular
                         * pow(max(dot(R,E),0.0),0.3*shininess);
            Ispec = clamp(Ispec, 0.0, 1.0);

            c = c + (Idiff + Ispec + Iamb) * lights[i].intensity;
        }
        else {
            c = c + Iamb * lights[i].intensity;
        }
    }

    return c;
}

vec3 lambertShading(vec3 v, vec3 N, vec3 eyePos, vec3 diffuse) {
    vec3 c = vec3(0, 0, 0);

    // iterate through all lights
    for(int i=0;i<lightCount;i++) {
        // determine if this light is visible
        bool isVisible = checkLightVisibility(v, lights[i]);

        if( isVisible ) {
            vec3 L = normalize(lights[i].pos - v);

            vec3 Idiff = clamp(diffuse * lights[i].diffuse * max(dot(N, L), 0.0), 0.0, 1.0);

            c = c + Idiff * lights[i].intensity;
        }
    }

    return c;
}

vec3 goochShading(vec3 v, vec3 N, vec3 eyePos, vec3 kcool, vec3 kwarm, float shininess) {

    vec3 c = vec3(0, 0, 0);

    for(int i=0;i<lightCount;i++) {
        // determine if this light is visible
        bool isVisible = checkLightVisibility(v, lights[i]);

        if( isVisible ) {
            vec3 L = normalize(lights[i].pos - v);
            vec3 E = normalize(eyePos - v);
            vec3 R = normalize(-reflect(L,N));

            float NdotL = dot(N, L);
            vec3 Idiff = kdiff * NdotL;

            vec3 kcdiff = min(kcool + alpha * Idiff, 1.0);
            vec3 kwdiff = min(kwarm + beta * Idiff, 1.0);

            vec3 kfinal = mix(kcdiff, kwdiff, (NdotL+1.0)*0.5);

            // calculate Specular Term:
            vec3 Ispec = kspec
                         * pow(max(dot(R,E),0.0),0.3*shininess);
            //Ispec = clamp(Ispec, 0.0, 1.0);
            Ispec = step(vec3(0.5, 0.5, 0.5), Ispec);

            float EdotN = dot(E, N);
            if( EdotN >= 0.2 ) c = c + min(kfinal + Ispec, 1.0) * lights[i].intensity;
        }
    }

    return c;
}

vec3 computeShading(vec3 p, vec3 n, Ray r, Shape s) {
    if( shadingMode == 1 )
        return lambertShading(p, n, r.origin, s.diffuse);
    else if( shadingMode == 2 )
        return phongShading(p, n, r.origin, s.diffuse, s.ambient, s.specular, s.shininess);
    else if( shadingMode == 3 )
        if( s.type == PLANE ) return lambertShading(p, n, r.origin, s.diffuse);
        else return goochShading(p, n, r.origin, s.kcool, s.kwarm, s.shininess);
}

// ray intersection test with shading computation
Hit rayIntersectsSphere(Ray r, Shape s) {
    Hit h;

    vec3 pq = r.origin - s.p;
    float a = 1.0;
    float b = 2.0 * dot(pq, r.dir);
    float c = length(pq)*length(pq) - s.radius[0] * s.radius[0];

    // solve the quadratic equation
    float delta = b*b - 4.0*a*c;
    if( delta < 0.0 )
    {
        return background;
    }
    else
    {
        float x0 = (-b+sqrt(delta))/(2.0*a);
        float x1 = (-b-sqrt(delta))/(2.0*a);
        float THRES = 1e-8;

        float r1 = x0, r2 = x1;
        if( r1 > r2 ){
            r1 = x1;
            r2 = x0;
        }

        if( r2 < THRES ) {
            return background;
        }
        else
        {
            if( r1 < THRES ) h.t = r2;
            else h.t = r1;

            // hit point
            vec3 p = h.t * r.dir + r.origin;
            // normal at hit point
            vec3 n = normalize(p - s.p);

            h.color = computeShading(p, n, r, s);
            return h;
        }
    }
}

Hit rayIntersectsPlane(Ray r, Shape s) {
    Hit h;
    h.t = 1e10;
    h.color = vec3(0, 0, 0);
    vec3 pq = s.p - r.origin;
    float ldotn = dot(s.axis[0], r.dir);
    if( abs(ldotn) < 1e-3 ) return background;
    else {
        h.t = dot(s.axis[0], pq) / ldotn;

        if( h.t > 0.0 ) {
            vec3 p = r.origin + h.t * r.dir;

            // compute u, v coordinates
            vec3 pp0 = p - s.p;
            float u = dot(pp0, s.axis[1]);
            float v = dot(pp0, s.axis[2]);
            if( abs(u) > s.radius[0] || abs(v) > s.radius[1] ) return background;
            else {
                h.color = computeShading(p, s.axis[0], r, s);
                return h;
            }
        }
        else return background;
    }
}

Hit rayIntersectsShape(Ray r, Shape s) {
    if(s.type == SPHERE) return rayIntersectsSphere(r, s);
    else if( s.type == PLANE ) return rayIntersectsPlane(r, s);
    else return background;
}

Hit rayIntersectsShapes(Ray r) {
    // go through a list of shapes and find closest hit
    Hit h;
    h.t = 1e10;
    h.color = background.color;

    for(int i=0;i<shapeCount;i++) {
        Hit hit = rayIntersectsShape(r, shapes[i]);
        if( (hit.t > 0.0) && (hit.t < h.t) ) {
            h.t = hit.t;
            h.color = hit.color;
        }
    }

    return h;
}

void init() {
    background.t = -1.0;
    background.color = vec3(0.85, 0.85, 0.85);
}

void main(void)
{
    init();
    initializeCamera();
    initializeLights();
    initializeShapes();

	float edgeSamples = sqrt(AAsamples);
	float step = 1.0 / edgeSamples;

    vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
    for(int i=0;i<AAsamples;i++) {
        float x = floor(float(i) * step);
        float y = mod(float(i), edgeSamples);
        vec4 offsets = vec4(x*step, y*step, 0, 0);
        vec4 pos = gl_FragCoord + offsets;

        Ray r = constructRay(pos.xy);

        // test if the ray hits the sphere
        Hit hit = rayIntersectsShapes(r);

        color = color + vec4(hit.color, 1.0);
    }

    gl_FragColor = color / AAsamples;
}
