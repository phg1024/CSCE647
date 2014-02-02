#include <thrust/random.h>
#include <helper_math.h>    // includes cuda.h and cuda_runtime_api.h

#include "definitions.h"
#include "utils.h"

__device__ int shadingMode;

// light ray intersection tests
__device__ float lightRayIntersectsSphere( Ray r, Shape* shapes, int sid ) {
    vec3 pq = r.origin - shapes[sid].p;
    float a = 1.0;
    float b = dot(pq, r.dir);
    float c = dot(pq, pq) - shapes[sid].radius[0] * shapes[sid].radius[0];

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


__device__ float lightRayIntersectsPlane( Ray r, Shape* shapes, int sid ) {
    float t = 1e10;
    float THRES = 1e-3;
    vec3 pq = shapes[sid].p - r.origin;
    float ldotn = dot(shapes[sid].axis[0], r.dir);
    if( abs(ldotn) < THRES ) return -1.0;
    else {
        t = dot(shapes[sid].axis[0], pq) / ldotn;

        if( t >= THRES ) {
            vec3 p = r.origin + t * r.dir;

            // compute u, v coordinates
            vec3 pp0 = p - shapes[sid].p;
            float u = dot(pp0, shapes[sid].axis[1]);
            float v = dot(pp0, shapes[sid].axis[2]);
            if( abs(u) > shapes[sid].radius[0] || abs(v) > shapes[sid].radius[1] ) return -1.0;
            else {
                return t;
            }
        }
        else return -1.0;
    }
}

__device__ float lightRayIntersectsEllipsoid( Ray r, Shape* shapes, int sid ) {
	vec3 pq = shapes[sid].p - r.origin;
	float a = dot(r.dir, shapes[sid].m*r.dir);
	float b = -dot(pq, shapes[sid].m*r.dir);
	float c = dot(pq, shapes[sid].m*pq) - 1;

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

__device__ float lightRayIntersectsCylinder( Ray r, Shape* shapes, int sid ) {
	vec3 m = shapes[sid].p - r.origin;
	float dDa = dot(r.dir, shapes[sid].axis[0]);
	float mDa = dot(m, shapes[sid].axis[0]);
	float a = dot(r.dir, r.dir) - dDa * dDa;
	float b = mDa * dDa - dot(m, r.dir);
	float c = dot(m, m) - mDa * mDa - shapes[sid].radius[0] * shapes[sid].radius[0];

	// degenerate cases
	if( abs(a) < 1e-6 ) {
		if( abs(b) < 1e-6 ) {
			return -1.0;
		}
		else {
			float t = -c/b*0.5;
			vec3 p = t * r.dir + r.origin;
			vec3 pq = p - shapes[sid].p;
			float hval = dot(pq, shapes[sid].axis[0]);

			if( hval < 0 || hval > shapes[sid].radius[1] ) return t;
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
			vec3 pq = p - shapes[sid].p;
			float hval = dot(pq, shapes[sid].axis[0]);

			if( hval < 0 || hval > shapes[sid].radius[1] ) {
				if( t < x1 ) {
					// try x1
					t = x1;
					vec3 p = t * r.dir + r.origin;

					vec3 pq = p - shapes[sid].p;
					float hval1 = dot(pq, shapes[sid].axis[0]);
					if( hval1 < 0 || hval1 > shapes[sid].radius[1] ) return -1.0;
					else return t;
				}
				else return -1.0;
			}
			else return t;	
		}
    }
}

__device__ float lightRayIntersectsCone(Ray r, Shape* shapes, int sid) {
	vec3 m = shapes[sid].p - r.origin;
	float cosTheta = cos(shapes[sid].radius[2]);
	float dDa = dot(r.dir, shapes[sid].axis[0]);
	float mDa = dot(m, shapes[sid].axis[0]);
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
			vec3 pq = p - shapes[sid].p;
			float hval = dot(pq, shapes[sid].axis[0]);

			if( hval < 0 || hval > shapes[sid].radius[1] ) return t;
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
			vec3 pq = p - shapes[sid].p;
			float hval = dot(pq, shapes[sid].axis[0]);

			if( hval < 0 || hval > shapes[sid].radius[1] ) {
				if( t < x1 ) {
					// try x1
					t = x1;
					vec3 p = t * r.dir + r.origin;

					vec3 pq = p - shapes[sid].p;
					float hval = dot(pq, shapes[sid].axis[0]);
					if( hval < 0 || hval > shapes[sid].radius[1] ) return -1.0;
					else return t;
				}
				else return -1.0;
			}
			else return t;
		}
    }
}

__device__ float lightRayIntersectsHyperboloid(Ray r, Shape* shapes, int sid) {
	return -1.0;
}


__device__ float lightRayIntersectsShape(Ray r, Shape* shapes, int sid) {
	int stype = shapes[sid].t;
    if( stype == Shape::SPHERE ) return lightRayIntersectsSphere(r, shapes, sid);
    else if( stype == Shape::PLANE ) return lightRayIntersectsPlane(r, shapes, sid);
	else if( stype == Shape::ELLIPSOID ) return lightRayIntersectsEllipsoid(r, shapes, sid);
	else if( stype == Shape::CONE ) return lightRayIntersectsCone(r, shapes, sid);
	else if( stype == Shape::CYLINDER ) return lightRayIntersectsCylinder(r, shapes, sid);	
	else if( stype == Shape::HYPERBOLOID ) return lightRayIntersectsHyperboloid(r, shapes, sid);
    else return -1.0;
}

__device__ float lightRayIntersectsShapes(Ray r, Shape* shapes, int shapeCount) {
	float T_INIT = 1e10;
    // go through a list of shapes and find closest hit
    float t = T_INIT;

	float THRES = 1e-3;

    for(int i=0;i<shapeCount;i++) {
        float hitT = lightRayIntersectsShape(r, shapes, i);
        if( (hitT > -THRES) && (hitT < t) ) {
            t = hitT;
        }
    }

	if( t < T_INIT )
		return t;
	else return -1.0;
}

__device__ bool checkLightVisibility(vec3 p, vec3 N, Shape* shapes, int nShapes, Light* lights, int lid) {
	if( lights[lid].t == Light::POINT ) {
		float dist = (p - lights[lid].pos).norm();
		Ray r;
		r.origin = p;
		r.dir = normalize(lights[lid].pos - p);
		float t = lightRayIntersectsShapes(r, shapes, nShapes);

		float THRES = 1e-3;
		return t < THRES || t > dist;
	}
	else if( lights[lid].t == Light::SPOT ) {
	}
	else if( lights[lid].t == Light::DIRECTIONAL ) {
		Ray r;
		r.origin = p;
		r.dir = -lights[lid].dir;
		float t = lightRayIntersectsShapes(r, shapes, nShapes);

		float THRES = 1e-3;
		return (t < THRES && (dot(N, lights[lid].dir)<0));
	}
}

__device__ vec3 phongShading(vec3 v, vec3 N, vec2 t, Ray r, Shape* shapes, int nShapes, Light* lights, int lightCount, int sid) {
    vec3 c = vec3(0, 0, 0);

    // iterate through all lights
    for(int i=0;i<lightCount;i++) {

        // determine if this light is visible
        bool isVisible = checkLightVisibility(v, N, shapes, nShapes, lights, i);

        //calculate Ambient Term:
		vec3 Iamb = shapes[sid].material.ambient * lights[i].ambient;

        if( isVisible ) {
            vec3 L;
			if( lights[i].t != Light::DIRECTIONAL)
				L = normalize(lights[i].pos - v);
			else
				L = -lights[i].dir;

			vec3 E = normalize(r.origin-v);
            vec3 R = normalize(-reflect(L,N));

			float NdotL, RdotE;
			if( shapes[sid].hasNormalMap ) {
				/*
				// normal defined in tangent space
				vec3 n_normalmap = normalize(texture2D(shapes[sid].nTex, t).rgb * 2.0 - 1.0);

				vec3 tangent = normalize(sphere_tangent(N));
				vec3 bitangent = cross(N, tangent);

				// find the mapping from tangent space to camera space
				mat3 m_t = transpose(mat3(tangent, bitangent, N));

				NdotL = dot(n_normalmap, normalize(m_t*L));
				RdotE = dot(m_t*R, normalize(m_t*E));
				*/
			}
			else {
				NdotL = dot(N, L);
				RdotE = dot(R, E);
			}

            //calculate Diffuse Term:
            vec3 Idiff = shapes[sid].material.diffuse * lights[i].diffuse * max(NdotL, 0.0);
            Idiff = clamp(Idiff, 0.0, 1.0);

            // calculate Specular Term:
            vec3 Ispec = shapes[sid].material.specular * lights[i].specular
                         * pow(max(RdotE,0.0),0.3 * shapes[sid].material.shininess);
            Ispec = clamp(Ispec, 0.0, 1.0);

			if( shapes[sid].hasTexture ) {
				/*
				vec3 Itexture = texture2D (shapes[sid].tex, t).rgb;
				c = c + Itexture * ((Idiff + Ispec) + Iamb) * lights[i].intensity;
				*/
			}
			else
	            c = c + ((Idiff + Ispec) + Iamb) * lights[i].intensity;
        }
        else {
			if( shapes[sid].hasTexture ) {
				/*
				vec3 Itexture = texture2D (shapes[sid].tex, t).rgb;
				c = c + Itexture * Iamb * lights[i].intensity;
				*/
			}
			else
	            c = c + Iamb * lights[i].intensity;
        }
    }

    return c;
}

__device__ vec3 lambertShading(vec3 v, vec3 N, vec2 t, Ray r, Shape* shapes, int nShapes, Light* lights, int lightCount, int sid) {
    vec3 c = vec3(0, 0, 0);

    // iterate through all lights
    for(int i=0;i<lightCount;i++) {
        // determine if this light is visible
        bool isVisible = checkLightVisibility(v, N, shapes, nShapes, lights, i);

        if( isVisible ) {
			vec3 L;
			if( lights[i].t != Light::DIRECTIONAL)
				L = normalize(lights[i].pos - v);
			else
				L = -lights[i].dir;

			vec3 E = normalize(r.origin - v);

			float NdotL;

			// change the normal with normal map
			if( shapes[sid].hasNormalMap ) {
				/*
				// normal defined in tangent space
				vec3 n_normalmap = normalize(texture2D(shapes[sid].nTex, t).rgb * 2.0 - 1.0);

				vec3 tangent = normalize(sphere_tangent(N));
				vec3 bitangent = cross(N, tangent);

				// find the mapping from tangent space to camera space
				mat3 m_t = transpose(mat3(tangent, bitangent, N));

				// convert the normal to to camera space
				NdotL = dot(n_normalmap, normalize(m_t*L));
				*/
			}
			else {
				NdotL = dot(N, L);
			}

			vec3 Itexture;
			if( shapes[sid].hasTexture ) {
				/*
				Itexture = texture2D (shapes[sid].tex, t).rgb;
				*/
			}
			else Itexture = vec3(1, 1, 1);

            vec3 Idiff = clamp(shapes[sid].material.diffuse * lights[i].diffuse * max(NdotL, 0.0), 0.0, 1.0);

            c = c + Itexture * Idiff * lights[i].intensity;
        }
    }

    return c;
}

__device__ vec3 goochShading(vec3 v, vec3 N, vec2 t, Ray r, Shape* shapes, int nShapes, Light* lights, int lightCount, int sid) {

    vec3 c = vec3(0, 0, 0);

    for(int i=0;i<lightCount;i++) {
		vec3 L;
		if( lights[i].t != Light::DIRECTIONAL)
			L = normalize(lights[i].pos - v);
		else
			L = -lights[i].dir;

        vec3 E = normalize(r.origin - v);
        vec3 R = normalize(-reflect(L,N));
        float NdotL = dot(N, L);

		vec3 diffuse;
		if( shapes[sid].hasTexture ) {
			/*
			diffuse = texture2D (shapes[sid].tex, t).rgb;
			*/
		}
		else diffuse = shapes[sid].material.diffuse;

        vec3 Idiff = diffuse * NdotL;
        vec3 kcdiff = min(shapes[sid].material.kcool + shapes[sid].material.alpha * Idiff, 1.0);
        vec3 kwdiff = min(shapes[sid].material.kwarm + shapes[sid].material.beta * Idiff, 1.0);
        vec3 kfinal = mix(kcdiff, kwdiff, (NdotL+1.0)*0.5);
        // calculate Specular Term:
            vec3 Ispec = shapes[sid].material.specular
                         * pow(max(dot(R,E),0.0),0.3*shapes[sid].material.shininess);
        Ispec = step(vec3(0.5, 0.5, 0.5), Ispec);
        // edge effect
            float EdotN = dot(E, N);
        if( EdotN >= 0.2 ) c = c + min(kfinal + Ispec, 1.0) * lights[i].intensity;
    }

    return c;
}

__device__ vec3 computeShading(vec3 p, vec3 n, vec2 t, Ray r, Shape* shapes, int nShapes, Light* lights, int nLights, int sid) {
    if( shadingMode == 1 )
        return lambertShading(p, n, t, r, shapes, nShapes, lights, nLights, sid);
    else if( shadingMode == 2 )
        return phongShading(p, n, t, r, shapes, nShapes, lights, nLights, sid);
    else if( shadingMode == 3 )
        if( shapes[sid].t == Shape::PLANE ) return lambertShading(p, n, t, r, shapes, nShapes, lights, nLights, sid);
        else return goochShading(p, n, t, r, shapes, nShapes, lights, nLights, sid);
}

__device__ __inline__ Hit background() {
	Hit h;
	h.t = -1.0;
	h.color = vec3(0.75, 0.75, 0.75);
	return h;
}

__device__ Ray generateRay(Camera* cam, float u, float v) {
	Ray r;
	r.origin = cam->pos;

    // find the intersection point on the canvas
    vec3 pcanvas;

    vec3 canvasCenter = cam->f * cam->dir + cam->pos;
    pcanvas = canvasCenter + u * cam->w * cam->right + v * cam->h * cam->up;

    r.dir = normalize(pcanvas - cam->pos);

	return r;
}

__device__ float rand(float x, float y){
  float val = sin(x * 12.9898 + y * 78.233) * 43758.5453;
  return val - floorf(val);
}

// ray intersection test with shading computation
__device__ Hit rayIntersectsSphere(Ray r, Shape* shapes, int nShapes, Light* lights, int nLights, int sid) {
	float ti = lightRayIntersectsSphere(r, shapes, sid);

	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
        // hit point
        vec3 p = h.t * r.dir + r.origin;
        // normal at hit point
		vec3 n = normalize(p - shapes[sid].p);
        // hack, move the point a little bit outer
		p = shapes[sid].p + (shapes[sid].radius[0] + 1e-6) * n;
        vec2 t = spheremap(n);
		h.color = computeShading(p, n, t, r, shapes, nShapes, lights, nLights, sid);
        return h;
    }
	else return background();
}


__device__ Hit rayIntersectsPlane(Ray r, Shape* shapes, int nShapes, Light* lights, int nLights, int sid) {
	float ti = lightRayIntersectsPlane(r, shapes, sid);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
        vec3 p = r.origin + h.t * r.dir;
        // compute u, v coordinates
        vec3 pp0 = p - shapes[sid].p;
        float u = dot(pp0, shapes[sid].axis[1]);
        float v = dot(pp0, shapes[sid].axis[2]);
        if( abs(u) > shapes[sid].radius[0] || abs(v) > shapes[sid].radius[1] ) return background();
        else {
            vec2 t = clamp((vec2(u / shapes[sid].radius[0], v / shapes[sid].radius[1]) + vec2(1.0, 1.0))*0.5, 0.0, 1.0);
            h.color = computeShading(p, shapes[sid].axis[0], t, r, shapes, nShapes, lights, nLights, sid);
            return h;
        }
    }
	else return background();
}

__device__ Hit rayIntersectsEllipsoid(Ray r, Shape* shapes, int nShapes, Light* lights, int nLights, int sid) {
	float ti = lightRayIntersectsEllipsoid(r, shapes, sid);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
		// hit point
		vec3 p = h.t * r.dir + r.origin;
        // normal at hit point
		vec3 n = normalize(2.0 * shapes[sid].m * (p - shapes[sid].p));
        vec2 t = spheremap(n);
        h.color = computeShading(p, n, t, r, shapes, nShapes, lights, nLights, sid);
        return h;
    }

	else return background();
}

__device__ Hit rayIntersectsCone(Ray r, Shape* shapes, int nShapes, Light* lights, int nLights, int sid) {
	float ti = lightRayIntersectsCone(r, shapes, sid);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
		vec3 p = h.t * r.dir + r.origin;
        vec3 pq = p - shapes[sid].p;
        float hval = dot(pq, shapes[sid].axis[0]);
        // normal at hit point
		vec3 n = normalize(cross(cross(shapes[sid].axis[0], pq), shapes[sid].axis[0]));
        vec2 t = vec2(0, 0);
        // dummy
		h.color = computeShading(p, n, t, r, shapes, nShapes, lights, nLights, sid);
        return h;
    }
	else return background();
}

__device__ Hit rayIntersectsCylinder(Ray r, Shape* shapes, int nShapes, Light* lights, int nLights, int sid) {
	float ti = lightRayIntersectsCylinder(r, shapes, sid);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
        vec3 p = h.t * r.dir + r.origin;
        vec3 pq = p - shapes[sid].p;
        float hval = dot(pq, shapes[sid].axis[0]);
        // normal at hit point
		vec3 n = normalize(pq - hval*shapes[sid].axis[0]);
        vec2 t = vec2(0, 0);
        // dummy
		h.color = computeShading(p, n, t, r, shapes, nShapes, lights, nLights, sid);
        return h;
    }
	else return background();
}

__device__ Hit rayIntersectsHyperboloid(Ray r, Shape* shapes, int nShapes, Light* lights, int nLights, int sid) {
	Hit h;
	return h;
}


__device__ Hit rayIntersectsShape(Ray r, Shape* shapes, int nShapes, Light* lights, int nLights, int sid) {
	int stype = shapes[sid].t;
    if( stype == Shape::SPHERE ) return rayIntersectsSphere(r, shapes, nShapes, lights, nLights, sid);
    else if( stype == Shape::PLANE ) return rayIntersectsPlane(r, shapes, nShapes, lights, nLights, sid);
	else if( stype == Shape::ELLIPSOID ) return rayIntersectsEllipsoid(r, shapes, nShapes, lights, nLights, sid);
	else if( stype == Shape::CONE ) return rayIntersectsCone(r, shapes, nShapes, lights, nLights, sid);
	else if( stype == Shape::CYLINDER ) return rayIntersectsCylinder(r, shapes, nShapes, lights, nLights, sid);	
	else if( stype == Shape::HYPERBOLOID ) return rayIntersectsHyperboloid(r, shapes, nShapes, lights, nLights, sid);
    else return background();
}

__device__ Hit rayIntersectsShapes(Ray r, int nShapes, Shape* shapes, int nLights, Light* lights) {
	Hit h;
	h.t = 1e10;
	h.color = background().color;

    for(int i=0;i<nShapes;i++) {
        Hit hit = rayIntersectsShape(r, shapes, nShapes, lights, nLights, i);
        if( (hit.t > 0.0) && (hit.t < h.t) ) {
            h.t = hit.t;
            h.color = hit.color;
        }
    }

	return h;
}

__global__ void init_kernel(int sMode) {
	shadingMode = sMode;
}

///////////////////////////////////////////////////////////////////////////////
//! main entry of the ray tracing program
///////////////////////////////////////////////////////////////////////////////
__global__ void raytrace(float3 *pos, Camera* cam, 
						 int nLights, Light* lights, 
						 int nShapes, Shape* shapes, 
						 unsigned int width, unsigned int height,
						 int AASamples)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x > width - 1 || y > height - 1 ) return;
	
	Color c(0.0f, 0.0f, 0.0f, 0.0f);
	int edgeSamples = sqrtf(AASamples);
	float step = 1.0 / edgeSamples;

	for(int i=0;i<AASamples;i++) {
		float px = floor(i*step);
		float py = i % edgeSamples;

		float xoffset = rand(x + x*step, y + y * step) - 0.5;
        float yoffset = rand(x + y*step, y + x * step) - 0.5;

		float u = x + (px + xoffset) * step;
		float v = y + (py + yoffset) * step;

		u = u / (float) width - 0.5;
		v = v / (float) height - 0.5;

		Ray r = generateRay(cam, u, v);

		Hit h = rayIntersectsShapes(r, nShapes, shapes, nLights, lights);

		c.c = c.c + vec4(clamp(h.color, 0.0, 1.0), 1.0);
	}

	c.c = c.c / (float)AASamples;



	// write output vertex
	pos[y*width+x] = make_float3(x, y, c.toFloat());
}