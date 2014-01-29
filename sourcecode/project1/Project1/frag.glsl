#version 120
uniform vec2 windowSize;
uniform int lightCount;
uniform int shapeCount;
uniform int shadingMode;    // 1 = lambert, 2 = phong, 3 = gooch, 4 = cook-torrance
uniform int AAsamples;

// camera info
uniform vec3 camPos, camUp, camDir;
uniform float camF;

// for gooch shading
uniform float alpha, beta;

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
    float spotCutoff;
    float spotCosCutoff;

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
    vec3 axis[3];       // axes for ellipsoid, main axis for cylinder, cone and hyperboloid
                        // normal and u, v for plane
    float radius[3];    // radius for sphere, ellipsoid, cylinder(height also), cone(height and angle also), width and height for plane
	mat3 m;				// for ellipsoid

    // material property
    vec3 emission;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    vec3 kcool;
    vec3 kwarm;

    float shininess;

	bool hasTexture;
	sampler2D tex;

	bool hasNormalMap;
	sampler2D nTex;
};

struct Ray {
    vec3 origin;
    vec3 dir;
};

struct Hit {
    float t;
    vec3 color;
};

uniform Light lights[4];
uniform Shape shapes[16];
uniform Hit background;


vec2 spheremap(vec3 p) {
	const float PI = 3.1415926536;
	return vec2((atan(p.z, p.x) / PI + 1.0) * 0.5,
                -((asin(p.y) / PI + 0.5)));
}


vec3 sphere_tangent(vec3 p) {
	const float PI = 3.1415926536;
	float phi = asin(p.y);
	
	vec2 bn = normalize(vec2(p.x, p.z)) * sin(phi);
	return vec3(bn.x, -cos(phi), bn.y);
}

void initializeCamera() {
    caminfo.pos = camPos;
    caminfo.up = camUp;
    caminfo.dir = camDir;

    caminfo.right = cross(caminfo.dir, caminfo.up);
    caminfo.f = camF;
    caminfo.w = 1.0;
    caminfo.h = windowSize.y / windowSize.x;
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
    pcanvas = canvasCenter - x * caminfo.w * caminfo.right + y * caminfo.h * caminfo.up;

    r.dir = normalize(pcanvas - caminfo.pos);
    return r;
}

// light ray intersection tests
float lightRayIntersectsSphere(Ray r, Shape s) {
    vec3 pq = r.origin - s.p;
    float a = 1.0;
    float b = dot(pq, r.dir);
    float c = dot(pq, pq) - s.radius[0] * s.radius[0];

    // solve the quadratic equation
    float delta = b*b - a*c;
    if( delta < 0.0 )
    {
        return -1.0;
    }
    else
    {
        delta = sqrt(delta);
        float x0 = -b+delta; // a = 1, no need to do the division
        float x1 = -b-delta;
        float THRES = 1e-3;

        if( x0 < THRES ) {
            return -1.0;
        }
        else
        {
            if( x1 < THRES ) return x0;
            else return x1;
        }
    }
}

float lightRayIntersectsPlane(Ray r, Shape s) {
    float t = 1e10;
    float THRES = 1e-3;
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

float lightRayIntersectsEllipsoid(Ray r, Shape s) {
	vec3 pq = s.p - r.origin;
	float a = dot(r.dir, s.m*r.dir);
	float b = -dot(pq, s.m*r.dir);
	float c = dot(pq, s.m*pq) - 1;

    float delta = b*b - a*c;
    if (delta < 0.0)
    {
		return -1.0;
    }
    else
    {
		delta = sqrt(delta);        
		float inv = 1.0 / a;

		float x0 = (-b-delta)*inv;
		float x1 = (-b+delta)*inv;
        
		const float THRES = 1e-3;
		if( x1 < THRES ) {
			return -1.0;
        }
        else
        {			
			if( x0 < THRES ) return x1;
			else return x0;
		}
    }
}

float lightRayIntersectsCylinder( Ray r, Shape s ) {
	vec3 m = s.p - r.origin;
	float dDa = dot(r.dir, s.axis[0]);
	float mDa = dot(m, s.axis[0]);
	float a = dot(r.dir, r.dir) - dDa * dDa;
	float b = mDa * dDa - dot(m, r.dir);
	float c = dot(m, m) - mDa * mDa - s.radius[0] * s.radius[0];

	// degenerate cases
	if( abs(a) < 1e-6 ) {
		if( abs(b) < 1e-6 ) {
			return -1.0;
		}
		else {
			float t = -c/b*0.5;
			vec3 p = t * r.dir + r.origin;
			vec3 pq = p - s.p;
			float hval = dot(pq, s.axis[0]);

			if( hval < 0 || hval > s.radius[1] ) return t;
			else return -1.0;
		}
	}

    float delta = b*b - a*c;
    if (delta < 0.0)
    {
		return -1.0;
    }
    else
    {
		delta = sqrt(delta);        
		float inv = 1.0 / a;

		float t0 = (-b-delta)*inv;
		float t1 = (-b+delta)*inv;

		float x0 = min(t0, t1);
		float x1 = max(t0, t1);
        
		const float THRES = 1e-3;
		if( x1 < THRES ) {
			return -1.0;
        }
        else
        {
			float t;
			if( x0 < THRES ) t = x1;
			else t = x0;

			// hit point
			vec3 p = t * r.dir + r.origin;
			vec3 pq = p - s.p;
			float hval = dot(pq, s.axis[0]);

			if( hval < 0 || hval > s.radius[1] ) {
				if( t < x1 ) {
					// try x1
					t = x1;
					vec3 p = t * r.dir + r.origin;

					vec3 pq = p - s.p;
					float hval1 = dot(pq, s.axis[0]);
					if( hval1 < 0 || hval1 > s.radius[1] ) return -1.0;
					else return t;
				}
				else return -1.0;
			}
			else return t;	
		}
    }
}

float lightRayIntersectsCone(Ray r, Shape s) {
	vec3 m = s.p - r.origin;
	float cosTheta = cos(s.radius[2]);
	float dDa = dot(r.dir, s.axis[0]);
	float mDa = dot(m, s.axis[0]);
	float a = dot(r.dir, r.dir) * cosTheta - dDa*dDa;
	float b = dDa * mDa - dot(m, r.dir) * cosTheta;
	float c = dot(m, m) * cosTheta - mDa * mDa;

	// degenerate case
	if( abs(a) < 1e-6 ) {
		if( abs(b) < 1e-6 ) {
			// impossible
			return -1.0;
		}
		else {
			float t = -c/b*0.5;
			vec3 p = t * r.dir + r.origin;
			vec3 pq = p - s.p;
			float hval = dot(pq, s.axis[0]);

			if( hval < 0 || hval > s.radius[1] ) return t;
			else return -1.0;
		}		
	}

    float delta = b*b - a*c;
    if (delta < 0.0)
    {
		return -1.0;
    }
    else
    {
		delta = sqrt(delta);
	        
		float inv = 1.0 / a;

		float t0 = (-b-delta)*inv;
		float t1 = (-b+delta)*inv;

		float x0 = min(t0, t1);
		float x1 = max(t0, t1);
        
		const float THRES = 1e-3;
		if( x1 < THRES ) {
			return -1.0;
        }
        else
        {
			float t;
			if( x0 < THRES ) t = x1;
			else t = x0;

			// hit point
			vec3 p = t * r.dir + r.origin;
			vec3 pq = p - s.p;
			float hval = dot(pq, s.axis[0]);

			if( hval < 0 || hval > s.radius[1] ) {
				if( t < x1 ) {
					// try x1
					t = x1;
					vec3 p = t * r.dir + r.origin;

					vec3 pq = p - s.p;
					float hval = dot(pq, s.axis[0]);
					if( hval < 0 || hval > s.radius[1] ) return -1.0;
					else return t;
				}
				else return -1.0;
			}
			else return t;
		}
    }
}

float lightRayIntersectsHyperboloid(Ray r, Shape s) {
	return -1.0;
}

float lightRayIntersectsShape(Ray r, Shape s) {
    if(s.type == SPHERE) return lightRayIntersectsSphere(r, s);
    else if( s.type == PLANE ) return lightRayIntersectsPlane(r, s);
	else if( s.type == ELLIPSOID ) return lightRayIntersectsEllipsoid(r, s);
	else if( s.type == CONE ) return lightRayIntersectsCone(r, s);
	else if( s.type == CYLINDER ) return lightRayIntersectsCylinder(r, s);	
	else if( s.type == HYPERBOLOID ) return lightRayIntersectsHyperboloid(r, s);
    else return -1.0;
}

float lightRayIntersectsShapes(Ray r) {
	float T_INIT = 1e10;
    // go through a list of shapes and find closest hit
    float t = T_INIT;

	float THRES = 1e-3;

    for(int i=0;i<shapeCount;i++) {
        float hitT = lightRayIntersectsShape(r, shapes[i]);
        if( (hitT > -THRES) && (hitT < t) ) {
            t = hitT;
        }
    }

	if( t < T_INIT )
		return t;
	else return -1.0;
}

bool checkLightVisibility(vec3 p, vec3 N, Light lt) {
	if( lt.type == POINT_LIGHT ) {
		float dist = length(p - lt.pos);
		Ray r;
		r.origin = p;
		r.dir = normalize(lt.pos - p);
		float t = lightRayIntersectsShapes(r);

		float THRES = 1e-3;
		return t < THRES || t > dist;
	}
	else if( lt.type == SPOT_LIGHT ) {
	}
	else if( lt.type == DIRECTIONAL_LIGHT ) {
		Ray r;
		r.origin = p;
		r.dir = -lt.dir;
		float t = lightRayIntersectsShapes(r);

		float THRES = 1e-3;
		return (t < THRES && (dot(N, lt.dir)<0));
	}
}


vec3 phongShading(vec3 v, vec3 N, vec2 t, Ray r, Shape s) {
    vec3 c = vec3(0, 0, 0);

    // iterate through all lights
    for(int i=0;i<lightCount;i++) {

        // determine if this light is visible
        bool isVisible = checkLightVisibility(v, N, lights[i]);

        //calculate Ambient Term:
        vec3 Iamb = s.ambient * lights[i].ambient;

        if( isVisible ) {
            vec3 L;
			if( lights[i].type != DIRECTIONAL_LIGHT)
				L = normalize(lights[i].pos - v);
			else
				L = -lights[i].dir;

            vec3 E = normalize(caminfo.pos-v);
            vec3 R = normalize(-reflect(L,N));

			float NdotL, RdotE;
			if( s.hasNormalMap ) {
				// normal defined in tangent space
				vec3 n_normalmap = normalize(texture2D(s.nTex, t).rgb * 2.0 - 1.0);

				vec3 tangent = normalize(sphere_tangent(N));
				vec3 bitangent = cross(N, tangent);

				// find the mapping from tangent space to camera space
				mat3 m_t = transpose(mat3(tangent, bitangent, N));

				NdotL = dot(n_normalmap, normalize(m_t*L));
				RdotE = dot(m_t*R, normalize(m_t*E));
			}
			else {
				NdotL = dot(N, L);
				RdotE = dot(R, E);
			}

            //calculate Diffuse Term:
            vec3 Idiff = s.diffuse * lights[i].diffuse * max(NdotL, 0.0);
            Idiff = clamp(Idiff, 0.0, 1.0);

            // calculate Specular Term:
            vec3 Ispec = s.specular * lights[i].specular
                         * pow(max(RdotE,0.0),0.3*s.shininess);
            Ispec = clamp(Ispec, 0.0, 1.0);

			if( s.hasTexture ) {
				vec3 Itexture = texture2D (s.tex, t).rgb;
				c = c + Itexture * ((Idiff + Ispec) + Iamb) * lights[i].intensity;
			}
			else
	            c = c + ((Idiff + Ispec) + Iamb) * lights[i].intensity;
        }
        else {
			if( s.hasTexture ) {
				vec3 Itexture = texture2D (s.tex, t).rgb;
				c = c + Itexture * Iamb * lights[i].intensity;
			}
			else
	            c = c + Iamb * lights[i].intensity;
        }
    }

    return c;
}

vec3 lambertShading(vec3 v, vec3 N, vec2 t, Ray r, Shape s) {
    vec3 c = vec3(0, 0, 0);

    // iterate through all lights
    for(int i=0;i<lightCount;i++) {
        // determine if this light is visible
        bool isVisible = checkLightVisibility(v, N, lights[i]);

        if( isVisible ) {
			vec3 L;
			if( lights[i].type != DIRECTIONAL_LIGHT)
				L = normalize(lights[i].pos - v);
			else
				L = -lights[i].dir;

			vec3 E = normalize(caminfo.pos - v);

			float NdotL;

			// change the normal with normal map
			if( s.hasNormalMap ) {
				// normal defined in tangent space
				vec3 n_normalmap = normalize(texture2D(s.nTex, t).rgb * 2.0 - 1.0);

				vec3 tangent = normalize(sphere_tangent(N));
				vec3 bitangent = cross(N, tangent);

				// find the mapping from tangent space to camera space
				mat3 m_t = transpose(mat3(tangent, bitangent, N));

				// convert the normal to to camera space
				NdotL = dot(n_normalmap, normalize(m_t*L));
			}
			else {
				NdotL = dot(N, L);
			}

			vec3 Itexture;
			if( s.hasTexture ) {
				Itexture = texture2D (s.tex, t).rgb;
			}
			else Itexture = vec3(1, 1, 1);

            vec3 Idiff = clamp(s.diffuse * lights[i].diffuse * max(NdotL, 0.0), 0.0, 1.0);

            c = c + Itexture * Idiff * lights[i].intensity;
        }
    }

    return c;
}

vec3 goochShading(vec3 v, vec3 N, vec2 t, Ray r, Shape s) {

    vec3 c = vec3(0, 0, 0);

    for(int i=0;i<lightCount;i++) {
		vec3 L;
		if( lights[i].type != DIRECTIONAL_LIGHT)
			L = normalize(lights[i].pos - v);
		else
			L = -lights[i].dir;

        vec3 E = normalize(caminfo.pos - v);
        vec3 R = normalize(-reflect(L,N));
        float NdotL = dot(N, L);

		vec3 diffuse;
		if( s.hasTexture ) {
			diffuse = texture2D (s.tex, t).rgb;
		}
		else diffuse = s.diffuse;

        vec3 Idiff = diffuse * NdotL;
        vec3 kcdiff = min(s.kcool + alpha * Idiff, 1.0);
        vec3 kwdiff = min(s.kwarm + beta * Idiff, 1.0);
        vec3 kfinal = mix(kcdiff, kwdiff, (NdotL+1.0)*0.5);
        // calculate Specular Term:
            vec3 Ispec = s.specular
                         * pow(max(dot(R,E),0.0),0.3*s.shininess);
        Ispec = step(vec3(0.5, 0.5, 0.5), Ispec);
        // edge effect
            float EdotN = dot(E, N);
        if( EdotN >= 0.2 ) c = c + min(kfinal + Ispec, 1.0) * lights[i].intensity;
    }

    return c;
}

vec3 computeShading(vec3 p, vec3 n, vec2 t, Ray r, Shape s) {
    if( shadingMode == 1 )
        return lambertShading(p, n, t, r, s);
    else if( shadingMode == 2 )
        return phongShading(p, n, t, r, s);
    else if( shadingMode == 3 )
        if( s.type == PLANE ) return lambertShading(p, n, t, r, s);
        else return goochShading(p, n, t, r, s);
}

// ray intersection test with shading computation
Hit rayIntersectsSphere(Ray r, Shape s) {
	float ti = lightRayIntersectsSphere(r, s);

	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
        // hit point
        vec3 p = h.t * r.dir + r.origin;
        // normal at hit point
        vec3 n = normalize(p - s.p);
        // hack, move the point a little bit outer
		p = s.p + (s.radius[0] + 1e-6) * n;
        vec2 t = spheremap(n);
        h.color = computeShading(p, n, t, r, s);
        return h;
    }
	else return background;
}

Hit rayIntersectsPlane(Ray r, Shape s) {
	float ti = lightRayIntersectsPlane(r, s);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
        vec3 p = r.origin + h.t * r.dir;
        // compute u, v coordinates
            vec3 pp0 = p - s.p;
        float u = dot(pp0, s.axis[1]);
        float v = dot(pp0, s.axis[2]);
        if( abs(u) > s.radius[0] || abs(v) > s.radius[1] ) return background;
        else {
            vec2 t = clamp((vec2(u / s.radius[0], v / s.radius[1]) + vec2(1.0, 1.0))*0.5, 0.0, 1.0);
            h.color = computeShading(p, s.axis[0], t, r, s);
            return h;
        }
    }
	else return background;
}

Hit rayIntersectsEllipsoid(Ray r, Shape s) {
	float ti = lightRayIntersectsEllipsoid(r, s);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
		// hit point
		vec3 p = h.t * r.dir + r.origin;
        // normal at hit point
		vec3 n = normalize(2.0 * s.m * (p - s.p));
        vec2 t = spheremap(n);
        h.color = computeShading(p, n, t, r, s);
        return h;
    }

	else return background;
}

Hit rayIntersectsCone( Ray r, Shape s ) {
	float ti = lightRayIntersectsCone(r, s);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
		vec3 p = h.t * r.dir + r.origin;
        vec3 pq = p - s.p;
        float hval = dot(pq, s.axis[0]);
        // normal at hit point
		vec3 n = normalize(cross(cross(s.axis[0], pq), s.axis[0]));
        vec2 t = vec2(0, 0);
        // dummy
		h.color = computeShading(p, n, t, r, s);
        return h;
    }
	else return background;
}

Hit rayIntersectsCylinder(Ray r, Shape s) {
	float ti = lightRayIntersectsCylinder(r, s);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
        vec3 p = h.t * r.dir + r.origin;
        vec3 pq = p - s.p;
        float hval = dot(pq, s.axis[0]);
        // normal at hit point
		vec3 n = normalize(pq - hval*s.axis[0]);
        vec2 t = vec2(0, 0);
        // dummy
		h.color = computeShading(p, n, t, r, s);
        return h;
    }
	else return background;
}

Hit rayIntersectsHyperboloid(Ray r, Shape s) {
	Hit h;
	return h;
}

Hit rayIntersectsShape(Ray r, Shape s) {
    if(s.type == SPHERE) return rayIntersectsSphere(r, s);
    else if( s.type == PLANE ) return rayIntersectsPlane(r, s);
	else if( s.type == ELLIPSOID ) return rayIntersectsEllipsoid(r, s);
	else if( s.type == CONE ) return rayIntersectsCone(r, s);
	else if( s.type == CYLINDER ) return rayIntersectsCylinder(r, s);	
	else if( s.type == HYPERBOLOID ) return rayIntersectsHyperboloid(r, s);
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

float rand(vec2 co){
  return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main(void)
{
    initializeCamera();
    //initializeLights();
    //initializeShapes();

    float edgeSamples = sqrt(float(AAsamples));
    float step = 1.0 / edgeSamples;

    vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
    for(int i=0;i<AAsamples;i++) {
        // uniform jitter
        float x = floor(float(i) * step);
        float y = mod(float(i), edgeSamples);

        float xoffset = rand(gl_FragCoord.xy + vec2(x*step, y*step)) - 0.5;
        float yoffset = rand(gl_FragCoord.xy + vec2(y*step, x*step)) - 0.5;
        vec4 pos = gl_FragCoord + vec4((x + xoffset) * step, (y + yoffset) * step, 0, 0);

        Ray r = constructRay(pos.xy);

        // test if the ray hits the sphere
        Hit hit = rayIntersectsShapes(r);

        color = color + vec4(hit.color, 1.0);
    }

    gl_FragColor = color / AAsamples;
}
