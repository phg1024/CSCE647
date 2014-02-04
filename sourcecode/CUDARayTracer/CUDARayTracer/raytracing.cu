#include <thrust/random.h>
#include <helper_math.h>    // includes cuda.h and cuda_runtime_api.h

#include "definitions.h"
#include "utils.h"

__device__ int shadingMode;
__device__ bool circularSpecular = false;
__device__ bool rectangularSpecular = false;
__device__ bool rampedSpecular = true;
__device__ bool hasReflection = false;
__device__ bool hasRefraction = false;

// light ray intersection tests
__device__ float lightRayIntersectsSphere( Ray r, d_Shape* shapes, int sid ) {
    float3 pq = r.origin - shapes[sid].p;
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


__device__ float lightRayIntersectsPlane( Ray r, d_Shape* shapes, int sid ) {
    float t = 1e10;
    float THRES = 1e-3;
    float3 pq = shapes[sid].p - r.origin;
    float ldotn = dot(shapes[sid].axis[0], r.dir);
    if( abs(ldotn) < THRES ) return -1.0;
    else {
        t = dot(shapes[sid].axis[0], pq) / ldotn;

        if( t >= THRES ) {
            float3 p = r.origin + t * r.dir;

            // compute u, v coordinates
            float3 pp0 = p - shapes[sid].p;
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

__device__ float lightRayIntersectsEllipsoid( Ray r, d_Shape* shapes, int sid ) {
	float3 pq = shapes[sid].p - r.origin;
	float a = dot(r.dir, mul(shapes[sid].m, r.dir));
	float b = -dot(pq, mul(shapes[sid].m, r.dir));
	float c = dot(pq, mul(shapes[sid].m, pq)) - 1;

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

__device__ float lightRayIntersectsCylinder( Ray r, d_Shape* shapes, int sid ) {
	float3 m = shapes[sid].p - r.origin;
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
			float3 p = t * r.dir + r.origin;
			float3 pq = p - shapes[sid].p;
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
        {
			// hit point
			float3 p = x0 * r.dir + r.origin;
			float3 pq = p - shapes[sid].p;
			float hval = dot(pq, shapes[sid].axis[0]);

			if( hval < 0 || hval > shapes[sid].radius[1] ) x0 = -1.0;
			
			p = x1 * r.dir + r.origin;
			pq = p - shapes[sid].p;
			float hval1 = dot(pq, shapes[sid].axis[0]);
			if( hval1 < 0 || hval1 > shapes[sid].radius[1] ) x1 = -1.0;

			// x0 and x1 are on cylinder surface
			// find out the interscetion on caps

			// top cap
			// defined by p+radius[1]*axis[0] and axis[0]
			float x2;
			float3 top = shapes[sid].p + shapes[sid].radius[1] * shapes[sid].axis[0];
		    pq = top - r.origin;
			float ldotn = dot(shapes[sid].axis[0], r.dir);
			if( abs(ldotn) < THRES ) x2 = -1.0;
			else{
				x2 = dot(shapes[sid].axis[0], pq) / ldotn;
				p = x2 * r.dir + r.origin;
				float hval = length(p - top);
				if( hval > shapes[sid].radius[0] ) x2 = -1.0;
			}

			// bottom cap
			// defined by p and -axis[0]
			float x3;
		    pq = shapes[sid].p - r.origin;
			float ldotn2 = dot(-shapes[sid].axis[0], r.dir);
			if( abs(ldotn2) < THRES ) x3 = -1.0;
			else{
				x3 = dot(-shapes[sid].axis[0], pq) / ldotn2;
				p = x3 * r.dir + r.origin;
				float hval = length(p - shapes[sid].p);
				if( hval > shapes[sid].radius[0] ) x3 = -1.0;
			}

			float t = 1e10;
			if( x0 > THRES ) t = x0;
			if( x1 > THRES ) t = min(x1, t);
			if( x2 > THRES ) t = min(x2, t);
			if( x3 > THRES ) t = min(x3, t);
			if( t >= 9e9 ) return -1.0;
			else return t;
		}
    }
}

__device__ float lightRayIntersectsCone(Ray r, d_Shape* shapes, int sid) {
	float3 m = shapes[sid].p - r.origin;
	float cosTheta = cos(shapes[sid].radius[2]);
	float dDa = dot(r.dir, shapes[sid].axis[0]);
	float mDa = dot(m, shapes[sid].axis[0]);
	float a = dot(r.dir, r.dir) * cosTheta * cosTheta - dDa*dDa;
	float b = dDa * mDa - dot(m, r.dir) * cosTheta * cosTheta;
	float c = dot(m, m) * cosTheta * cosTheta - mDa * mDa;

	// degenerate case

	if( abs(a) < 1e-6 ) {
		if( abs(b) < 1e-6 ) {
			// impossible
			return -1.0;
		}
		else {
			float t = -c/b*0.5;
			float3 p = t * r.dir + r.origin;
			float3 pq = p - shapes[sid].p;
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

		float x0 = (-b-delta)*inv;
		float x1 = (-b+delta)*inv;

		const float THRES = 1e-3;
		// hit point
		float3 p = x0 * r.dir + r.origin;
		float3 pq = p - shapes[sid].p;
		float hval = dot(pq, shapes[sid].axis[0]);
		if( hval < 0 || hval > shapes[sid].radius[1] ) x0 = -1.0;

		p = x1 * r.dir + r.origin;
		pq = p - shapes[sid].p;
		hval = dot(pq, shapes[sid].axis[0]);
		if( hval < 0 || hval > shapes[sid].radius[1] ) x1 = -1.0;

		// cap
		float x2;
		float3 top = shapes[sid].p + shapes[sid].radius[1] * shapes[sid].axis[0];
		pq = top - r.origin;
		float ldotn = dot(shapes[sid].axis[0], r.dir);
		if( abs(ldotn) < 1e-6 ) x2 = -1.0;
		else{
			x2 = dot(shapes[sid].axis[0], pq) / ldotn;
			p = x2 * r.dir + r.origin;
			float hval = length(p - top);
			if( hval > shapes[sid].radius[1] * tanf(shapes[sid].radius[2]) ) x2 = -1.0;
		}

		float t = 1e10;
		if( x0 > THRES ) t = min(x0, t);
		if( x1 > THRES ) t = min(x1, t);
		if( x2 > THRES ) t = min(x2, t);
		if( t > 9e9 ) t = -1.0;
		return t;
    }
}

__device__ float lightRayIntersectsHyperboloid(Ray r, d_Shape* shapes, int sid) {
	float3 pq = shapes[sid].p - r.origin;
	float a = dot(r.dir, mul(shapes[sid].m, r.dir));
	float b = -dot(pq, mul(shapes[sid].m, r.dir));
	float c = dot(pq, mul(shapes[sid].m, pq)) - 1;

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

__device__ float lightRayIntersectsHyperboloid2(Ray r, d_Shape* shapes, int sid) {
	float3 pq = shapes[sid].p - r.origin;
	float a = dot(r.dir, mul(shapes[sid].m, r.dir));
	float b = -dot(pq, mul(shapes[sid].m, r.dir));
	float c = dot(pq, mul(shapes[sid].m, pq)) - 1;

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
		float t = 1e10;
		if( x0 > THRES ) t = min(x0, t);
		if( x1 > THRES ) t = min(x1, t);
		if( t > 9e9 ) t = -1.0;
		return t;

		//return fminf(x0, x1);
    }
}

__device__ float lightRayIntersectsShape(Ray r, d_Shape* shapes, int sid) {
	int stype = shapes[sid].t;
    if( stype == Shape::SPHERE ) return lightRayIntersectsSphere(r, shapes, sid);
    else if( stype == Shape::PLANE ) return lightRayIntersectsPlane(r, shapes, sid);
	else if( stype == Shape::ELLIPSOID ) return lightRayIntersectsEllipsoid(r, shapes, sid);
	else if( stype == Shape::CONE ) return lightRayIntersectsCone(r, shapes, sid);
	else if( stype == Shape::CYLINDER ) return lightRayIntersectsCylinder(r, shapes, sid);	
	else if( stype == Shape::HYPERBOLOID ) return lightRayIntersectsHyperboloid(r, shapes, sid);
	else if( stype == Shape::HYPERBOLOID2 ) return lightRayIntersectsHyperboloid2(r, shapes, sid);
    else return -1.0;
}

__device__ float lightRayIntersectsShapes(Ray r, d_Shape* shapes, int shapeCount) {
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

__device__ bool checkLightVisibility(float3 p, float3 N, d_Shape* shapes, int nShapes, d_Light* lights, int lid) {
	if( lights[lid].t == Light::POINT ) {
		float dist = length(p - lights[lid].pos);
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

__device__ float3 phongShading(float3 v, float3 N, float2 t, Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int lightCount, int sid) {
    float3 c = make_float3(0, 0, 0);

    // iterate through all lights
    for(int i=0;i<lightCount;i++) {

        // determine if this light is visible
        bool isVisible = checkLightVisibility(v, N, shapes, nShapes, lights, i);

        //calculate Ambient Term:
		float3 Iamb = shapes[sid].material.ambient * lights[i].ambient;

        if( isVisible ) {
            float3 L;
			if( lights[i].t != Light::DIRECTIONAL)
				L = normalize(lights[i].pos - v);
			else
				L = -lights[i].dir;

			float3 E = normalize(r.origin-v);
            float3 R = normalize(-reflect(L,N));

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
            float3 Idiff = shapes[sid].material.diffuse * lights[i].diffuse * max(NdotL, 0.0);
            Idiff = clamp(Idiff, 0.0, 1.0);

            // calculate Specular Term:
			float specFactor = pow(max(RdotE,0.0),0.3 * shapes[sid].material.shininess);
			if( circularSpecular ) specFactor = step(0.8, specFactor);
			if( rampedSpecular ) specFactor = toonify(specFactor, 4);

            float3 Ispec = shapes[sid].material.specular * lights[i].specular
                         * specFactor;
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

__device__ float3 lambertShading(float3 v, float3 N, float2 t, Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int lightCount, int sid) {
    float3 c = make_float3(0, 0, 0);

    // iterate through all lights
    for(int i=0;i<lightCount;i++) {
        // determine if this light is visible
        bool isVisible = checkLightVisibility(v, N, shapes, nShapes, lights, i);

        if( isVisible ) {
			float3 L;
			if( lights[i].t != Light::DIRECTIONAL)
				L = normalize(lights[i].pos - v);
			else
				L = -lights[i].dir;

			float3 E = normalize(r.origin - v);

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

			float3 Itexture;
			if( shapes[sid].hasTexture ) {
				/*
				Itexture = texture2D (shapes[sid].tex, t).rgb;
				*/
			}
			else Itexture = make_float3(1, 1, 1);

            float3 Idiff = clamp(shapes[sid].material.diffuse * lights[i].diffuse * max(NdotL, 0.0), 0.0, 1.0);

            c = c + Itexture * Idiff * lights[i].intensity;
        }
    }

    return c;
}

__device__ float3 goochShading(float3 v, float3 N, float2 t, Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int lightCount, int sid) {

    float3 c = make_float3(0, 0, 0);

    for(int i=0;i<lightCount;i++) {
		float3 L;
		if( lights[i].t != Light::DIRECTIONAL)
			L = normalize(lights[i].pos - v);
		else
			L = -lights[i].dir;

        float3 E = normalize(r.origin - v);
        float3 R = normalize(-reflect(L,N));
        float NdotL = dot(N, L);

		float3 diffuse;
		if( shapes[sid].hasTexture ) {
			/*
			diffuse = texture2D (shapes[sid].tex, t).rgb;
			*/
		}
		else diffuse = shapes[sid].material.diffuse;

        float3 Idiff = diffuse * NdotL;
        float3 kcdiff = fminf(shapes[sid].material.kcool + shapes[sid].material.alpha * Idiff, 1.0f);
        float3 kwdiff = fminf(shapes[sid].material.kwarm + shapes[sid].material.beta * Idiff, 1.0f);
        float3 kfinal = mix(kcdiff, kwdiff, (NdotL+1.0)*0.5);
        // calculate Specular Term:
        float3 Ispec = shapes[sid].material.specular
                         * pow(max(dot(R,E),0.0),0.3*shapes[sid].material.shininess);
        Ispec = step(make_float3(0.5, 0.5, 0.5), Ispec);
        // edge effect
            float EdotN = dot(E, N);
        if( fabs(EdotN) >= 0.2 ) c = c + fminf(kfinal + Ispec, 1.0) * lights[i].intensity;
    }
    return c;
}

__device__ float3 computeShading(float3 p, float3 n, float2 t, Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int nLights, int sid) {
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
	h.color = make_float3(0.05, 0.05, 0.05);
	return h;
}

__device__ Ray generateRay(Camera* cam, float u, float v) {
	Ray r;
	r.level = 0;
	r.origin = cam->pos.data;

    // find the intersection point on the canvas
    float3 canvasCenter = cam->f * cam->dir.data + cam->pos.data;
    float3 pcanvas = canvasCenter + u * cam->w * cam->right.data + v * cam->h * cam->up.data;

    r.dir = normalize(pcanvas - cam->pos.data);

	return r;
}

__device__ float rand(float x, float y){
  float val = sin(x * 12.9898 + y * 78.233) * 43758.5453;
  return val - floorf(val);
}

__device__ Hit rayIntersectsShapes(Ray r, int nShapes, d_Shape* shapes, int nLights, d_Light* lights);

__device__ Hit computeReflectedHit(Ray r, float3 p, float3 n, int nShapes, d_Shape* shapes, int nLights, d_Light* lights) {
	Ray rr;
	rr.dir = reflect(r.dir, n);
	rr.origin = p;
	rr.level = r.level + 1;
	return rayIntersectsShapes(rr, nShapes, shapes, nLights, lights);
}

__device__ Hit computeRefractedHit(Ray r, float3 p, float3 n, int nShapes, d_Shape* shapes, int nLights, d_Light* lights) {
	Hit h;
	h.color = make_float3(0, 0, 0);
	h.t = -1.0;
	return h;
}

// ray intersection test with shading computation
__device__ Hit rayIntersectsSphere(Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int nLights, int sid) {
	float ti = lightRayIntersectsSphere(r, shapes, sid);

	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
        // hit point
        float3 p = h.t * r.dir + r.origin;
        // normal at hit point
		float3 n = normalize(p - shapes[sid].p);
        // hack, move the point a little bit outer
		p = shapes[sid].p + (shapes[sid].radius[0] + 1e-6) * n;
        float2 t = spheremap(n);
		
		// shade color
		h.color = computeShading(p, n, t, r, shapes, nShapes, lights, nLights, sid);

		Hit rh, fh;
		if( hasReflection )
		// reflected ray
			rh = computeReflectedHit(r, p, n, nShapes, shapes, nLights, lights);

		if( hasRefraction )
		// refracted ray
			fh = computeRefractedHit(r, p, n, nShapes, shapes, nLights, lights);

		if( hasReflection || hasRefraction )
			h.color = mix(h.color, rh.color, fh.color, shapes[sid].material.Ks, shapes[sid].material.Kr, shapes[sid].material.Kf);
			
        return h;
    }
	else return background();
}

__device__ Hit rayIntersectsPlane(Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int nLights, int sid) {
	float ti = lightRayIntersectsPlane(r, shapes, sid);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
        float3 p = r.origin + h.t * r.dir;
        // compute u, v coordinates
        float3 pp0 = p - shapes[sid].p;
        float u = dot(pp0, shapes[sid].axis[1]);
        float v = dot(pp0, shapes[sid].axis[2]);
        if( abs(u) > shapes[sid].radius[0] || abs(v) > shapes[sid].radius[1] ) return background();
        else {
            float2 t = clamp((make_float2(u / shapes[sid].radius[0], v / shapes[sid].radius[1]) + make_float2(1.0, 1.0))*0.5, 0.0, 1.0);
            h.color = computeShading(p, shapes[sid].axis[0], t, r, shapes, nShapes, lights, nLights, sid);

			float3 n = shapes[sid].axis[0];

			Hit rh, fh;
			if( hasReflection )
			// reflected ray
				rh = computeReflectedHit(r, p, n, nShapes, shapes, nLights, lights);

			if( hasRefraction )
			// refracted ray
				fh = computeRefractedHit(r, p, n, nShapes, shapes, nLights, lights);

			if( hasReflection || hasRefraction )
				h.color = mix(h.color, rh.color, fh.color, shapes[sid].material.Ks, shapes[sid].material.Kr, shapes[sid].material.Kf);

            return h;
        }
    }
	else return background();
}

__device__ Hit rayIntersectsEllipsoid(Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int nLights, int sid) {
	float ti = lightRayIntersectsEllipsoid(r, shapes, sid);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
		// hit point
		float3 p = h.t * r.dir + r.origin;
        // normal at hit point
		float3 n = normalize(2.0 * mul(shapes[sid].m, (p - shapes[sid].p)));
        float2 t = spheremap(n);
        h.color = computeShading(p, n, t, r, shapes, nShapes, lights, nLights, sid);

		Hit rh, fh;
		if( hasReflection )
			// reflected ray
				rh = computeReflectedHit(r, p, n, nShapes, shapes, nLights, lights);

		if( hasRefraction )
			// refracted ray
				fh = computeRefractedHit(r, p, n, nShapes, shapes, nLights, lights);

		if( hasReflection || hasRefraction )
			h.color = mix(h.color, rh.color, fh.color, shapes[sid].material.Ks, shapes[sid].material.Kr, shapes[sid].material.Kf);

        return h;
    }

	else return background();
}

__device__ Hit rayIntersectsCone(Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int nLights, int sid) {
	float ti = lightRayIntersectsCone(r, shapes, sid);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
		float3 p = h.t * r.dir + r.origin;
        float3 pq = p - shapes[sid].p;
        float hval = dot(pq, shapes[sid].axis[0]);
        // normal at hit point
		float3 n;
		if(fabsf(hval - shapes[sid].radius[1])<1e-4) 
			n = shapes[sid].axis[0];
		else 
			n = normalize(cross(cross(shapes[sid].axis[0], pq), shapes[sid].axis[0]));

        float2 t = make_float2(0, 0);

		h.color = computeShading(p, n, t, r, shapes, nShapes, lights, nLights, sid);

		Hit rh, fh;
		if( hasReflection )
			// reflected ray
				rh = computeReflectedHit(r, p, n, nShapes, shapes, nLights, lights);

		if( hasRefraction )
			// refracted ray
				fh = computeRefractedHit(r, p, n, nShapes, shapes, nLights, lights);

		if( hasReflection || hasRefraction )
			h.color = mix(h.color, rh.color, fh.color, shapes[sid].material.Ks, shapes[sid].material.Kr, shapes[sid].material.Kf);
		

        return h;
    }
	else return background();
}

__device__ Hit rayIntersectsCylinder(Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int nLights, int sid) {
	float ti = lightRayIntersectsCylinder(r, shapes, sid);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
        float3 p = h.t * r.dir + r.origin;
        float3 pq = p - shapes[sid].p;
        float hval = dot(pq, shapes[sid].axis[0]);
        // normal at hit point
		float3 n = normalize(pq - hval*shapes[sid].axis[0]);
		if( fabsf(hval) < 1e-3 ) n = -shapes[sid].axis[0];
		else if( fabsf(hval - shapes[sid].radius[1]) < 1e-3 ) n = shapes[sid].axis[0];
        float2 t = make_float2(0, 0);

		h.color = computeShading(p, n, t, r, shapes, nShapes, lights, nLights, sid);

		Hit rh, fh;
		if( hasReflection )
			// reflected ray
				rh = computeReflectedHit(r, p, n, nShapes, shapes, nLights, lights);

		if( hasRefraction )
			// refracted ray
				fh = computeRefractedHit(r, p, n, nShapes, shapes, nLights, lights);

		if( hasReflection || hasRefraction )
			h.color = mix(h.color, rh.color, fh.color, shapes[sid].material.Ks, shapes[sid].material.Kr, shapes[sid].material.Kf);

        return h;
    }
	else return background();
}

__device__ Hit rayIntersectsHyperboloid(Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int nLights, int sid) {
	float ti = lightRayIntersectsHyperboloid(r, shapes, sid);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
		// hit point
		float3 p = h.t * r.dir + r.origin;
        // normal at hit point
		float3 n = normalize(2.0 * mul(shapes[sid].m, (p - shapes[sid].p)));
        float2 t = spheremap(n);
        h.color = computeShading(p, n, t, r, shapes, nShapes, lights, nLights, sid);

		Hit rh, fh;
		if( hasReflection )
			// reflected ray
				rh = computeReflectedHit(r, p, n, nShapes, shapes, nLights, lights);

		if( hasRefraction )
			// refracted ray
				fh = computeRefractedHit(r, p, n, nShapes, shapes, nLights, lights);

		if( hasReflection || hasRefraction )
			h.color = mix(h.color, rh.color, fh.color, shapes[sid].material.Ks, shapes[sid].material.Kr, shapes[sid].material.Kf);

        return h;
    }

	else return background();
}

__device__ Hit rayIntersectsHyperboloid2(Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int nLights, int sid) {
	float ti = lightRayIntersectsHyperboloid2(r, shapes, sid);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
		// hit point
		float3 p = h.t * r.dir + r.origin;
        // normal at hit point
		float3 n = normalize(2.0 * mul(shapes[sid].m, (shapes[sid].p - p)));

        float2 t = spheremap(n);
        h.color = computeShading(p, n, t, r, shapes, nShapes, lights, nLights, sid);

		Hit rh, fh;
		if( hasReflection )
			// reflected ray
				rh = computeReflectedHit(r, p, n, nShapes, shapes, nLights, lights);

		if( hasRefraction )
			// refracted ray
				fh = computeRefractedHit(r, p, n, nShapes, shapes, nLights, lights);

		if( hasReflection || hasRefraction )
			h.color = mix(h.color, rh.color, fh.color, shapes[sid].material.Ks, shapes[sid].material.Kr, shapes[sid].material.Kf);

        return h;
    }

	else return background();
}

__device__ Hit rayIntersectsShape(Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int nLights, int sid) {
	int stype = shapes[sid].t;
    if( stype == Shape::SPHERE ) return rayIntersectsSphere(r, shapes, nShapes, lights, nLights, sid);
    else if( stype == Shape::PLANE ) return rayIntersectsPlane(r, shapes, nShapes, lights, nLights, sid);
	else if( stype == Shape::ELLIPSOID ) return rayIntersectsEllipsoid(r, shapes, nShapes, lights, nLights, sid);
	else if( stype == Shape::CONE ) return rayIntersectsCone(r, shapes, nShapes, lights, nLights, sid);
	else if( stype == Shape::CYLINDER ) return rayIntersectsCylinder(r, shapes, nShapes, lights, nLights, sid);	
	else if( stype == Shape::HYPERBOLOID ) return rayIntersectsHyperboloid(r, shapes, nShapes, lights, nLights, sid);
	else if( stype == Shape::HYPERBOLOID2 ) return rayIntersectsHyperboloid2(r, shapes, nShapes, lights, nLights, sid);
    else return background();
}

__device__ Hit rayIntersectsShapes(Ray r, int nShapes, d_Shape* shapes, int nLights, d_Light* lights) {
	if( r.level > 4 ) return background();

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

///////////////////////////////////////////////////////////////////////////////
//! main entry of the ray tracing program
///////////////////////////////////////////////////////////////////////////////
__global__ void raytrace(float3 *pos, Camera* cam, 
						 int nLights, Light* lights, 
						 int nShapes, Shape* shapes, 
						 unsigned int width, unsigned int height,
						 int sMode, int AASamples)
{
	// load scene information into block
	shadingMode = sMode;
	__shared__ int inLightsCount;
	__shared__ int inShapesCount;
	__shared__ d_Shape inShapes[16];
	__shared__ d_Light inLights[8];
	
	inLightsCount = nLights;
	inShapesCount = nShapes;
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int tid = tidy * blockDim.x + tidx;
	if( tid < nLights )	inLights[tid].init(lights[tid]);
	if( tid < nShapes )	inShapes[tid].init(shapes[tid]);

	//__threadfence_block();
	__syncthreads();

	unsigned int x = blockIdx.x*blockDim.x + tidx;
	unsigned int y = blockIdx.y*blockDim.y + tidy;

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

		Hit h = rayIntersectsShapes(r, inShapesCount, inShapes, inLightsCount, inLights);

		c.c = c.c + vec4(clamp(h.color, 0.0, 1.0), 1.0);
	}

	c.c = c.c / (float)AASamples;

	// write output vertex
	pos[y*width+x] = make_float3(x, y, c.toFloat());
}

///////////////////////////////////////////////////////////////////////////////
//! main entry of the ray tracing program, with load balancing
///////////////////////////////////////////////////////////////////////////////
__global__ void raytrace2(float3 *pos, Camera* cam, 
						  int nLights, Light* lights, 
						  int nShapes, Shape* shapes, 
						  unsigned int width, unsigned int height,
						  int sMode, int AASamples, int gx, int gy, int gmx, int gmy)
{
	// load scene information into block
	shadingMode = sMode;
	__shared__ int inLightsCount;
	__shared__ int inShapesCount;
	__shared__ d_Shape inShapes[16];
	__shared__ d_Light inLights[8];

	inLightsCount = nLights;
	inShapesCount = nShapes;
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int tid = tidy * blockDim.x + tidx;
	if( tid < nLights )	inLights[tid].init(lights[tid]);
	if( tid < nShapes )	inShapes[tid].init(shapes[tid]);
	__syncthreads();

	for(int gi=0;gi<gmy;gi++) {
		for(int gj=0;gj<gmx;gj++) {

			unsigned int xoffset = gj * gx * blockDim.x;
			unsigned int yoffset = gi * gy * blockDim.y;

			unsigned int x = blockIdx.x*blockDim.x + tidx + xoffset;
			unsigned int y = blockIdx.y*blockDim.y + tidy + yoffset;

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

				//Hit h = rayIntersectsShapes(r, nShapes, shapes, nLights, lights);
				Hit h = rayIntersectsShapes(r, inShapesCount, inShapes, inLightsCount, inLights);

				c.c = c.c + vec4(clamp(h.color, 0.0, 1.0), 1.0);
			}

			c.c = c.c / (float)AASamples;

#define SHOW_AFFINITY 0
#if SHOW_AFFINITY
			
			c.c.r = blockIdx.x / (float)gx;
			c.c.g = blockIdx.y / (float)gy;
			c.c.b = 0.0;
			
			/*
			c.c.r = threadIdx.x / (float)blockDim.x;
			c.c.g = threadIdx.y / (float)blockDim.y;
			c.c.b = 0.0;
			*/
#endif

			// write output vertex
			pos[y*width+x] = make_float3(x, y, c.toFloat());
		}
	}
}


__device__ int curb;

__global__ void initCurrentBlock(int v) {
	curb = v;
}

///////////////////////////////////////////////////////////////////////////////
//! main entry of the ray tracing program, with load balancing
///////////////////////////////////////////////////////////////////////////////
__global__ void raytrace3(float3 *pos, Camera* cam, 
						  int nLights, Light* lights, 
						  int nShapes, Shape* shapes, 
						  unsigned int width, unsigned int height,
						  int sMode, int AASamples, int bmx, int bmy, int ttlb)
{
	// load scene information into block
	shadingMode = sMode;
	__shared__ int inLightsCount;
	__shared__ int inShapesCount;
	__shared__ d_Shape inShapes[16];
	__shared__ d_Light inLights[4];

	inLightsCount = nLights;
	inShapesCount = nShapes;
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int tid = tidy * blockDim.x + tidx;
	if( tid < nLights )	inLights[tid].init(lights[tid]);
	if( tid < nShapes )	inShapes[tid].init(shapes[tid]);

	__syncthreads();

	__shared__ int currentBlock;
	if( tid == 0 ){
		currentBlock = curb;
	}
	__threadfence_system();

	// total number of blocks, current block
	do {
		int bx = currentBlock % bmx;
		int by = currentBlock / bmx;

		unsigned int xoffset = bx * blockDim.x;
		unsigned int yoffset = by * blockDim.y;

		unsigned int x = tidx + xoffset;
		unsigned int y = tidy + yoffset;

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

			//Hit h = rayIntersectsShapes(r, nShapes, shapes, nLights, lights);
			Hit h = rayIntersectsShapes(r, inShapesCount, inShapes, inLightsCount, inLights);

			c.c = c.c + vec4(clamp(h.color, 0.0, 1.0), 1.0);
		}

		c.c = c.c / (float)AASamples;

#define SHOW_AFFINITY 0
#if SHOW_AFFINITY
		c.c.r = blockIdx.x / (float)gridDim.x;
		c.c.g = blockIdx.y / (float)gridDim.y;
		c.c.b = 0.0;

		/*
		c.c.r = threadIdx.x / (float)blockDim.x;
		c.c.g = threadIdx.y / (float)blockDim.y;
		c.c.b = 0.0;
		*/
#endif

		// write output vertex
		pos[y*width+x] = make_float3(x, y, c.toFloat());		
		__syncthreads();

		if( tid == 0 ){
			currentBlock = atomicAdd(&curb, 1);
		}
		__threadfence_block();


	}while(currentBlock < ttlb);
}