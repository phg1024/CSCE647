#include <thrust/random.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <helper_math.h>    // includes cuda.h and cuda_runtime_api.h

#include "definitions.h"
#include "utils.h"
#include "randvec.h"

__device__ int shadingMode;
__device__ bool circularSpecular = false;
__device__ bool rectangularSpecular = true;
__device__ bool rampedSpecular = false;
__device__ bool cartoonShading = false;
__device__ bool hasReflection = false;
__device__ bool hasRefraction = false;
__device__ bool isRayTracing = false;
__device__ bool isDirectionalLight = false;
__device__ bool isSpotLight = false;
__device__ bool fakeSoftShadow = false;
__device__ bool rainbowLight = false;
__device__ bool jittered = true;
__device__ uchar4* textures[32];
__device__ int2 textureSize[32];
__device__ int envMapIdx;

__global__ void setParams(int specType, int tracingType, int envmap) {
	envMapIdx = envmap;

	circularSpecular = false;
	rectangularSpecular = false;
	rampedSpecular = false;
	cartoonShading = false;
	hasReflection = false;
	hasRefraction = false;

	switch( specType ) {
	case 1:
		circularSpecular = true;
		break;
	case 2:
		rectangularSpecular = true;
		break;
	case 3:
		rampedSpecular = true;
		break;
	case 4:
		cartoonShading = true;
		break;
	default:
		break;
	}

	switch( tracingType ) {
	case 0:
		isRayTracing = true;
		jittered = false;
		break;
	case 1:
		isRayTracing = true;
		jittered = true;
		break;
	case 2:
		isRayTracing = false;
		jittered = true;
		break;
	default:
		break;
	}
}

__global__ void bindTexture(TextureObject* texs, int texCount) {
	for(int i=0;i<texCount;i++){
		textures[i] = (uchar4*)texs[i].addr;
		textureSize[i] = texs[i].size;
	}
}

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
    if( abs(ldotn) < 1e-6 ) return -1.0;
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

__device__ float lightRayIntersectsQuadraticSurface(Ray r, d_Shape* shapes, int sid) {
	float3 pq = shapes[sid].p - r.origin;
	float a = dot(r.dir, mul(shapes[sid].m, r.dir));
	float b = -dot(pq, mul(shapes[sid].m, r.dir));
	float c = dot(pq, mul(shapes[sid].m, pq)) - shapes[sid].constant2;

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
    }
}

__device__  __forceinline__ float lightRayIntersectsShape(Ray r, d_Shape* shapes, int sid) {
	switch( shapes[sid].t ) {
	case Shape::PLANE:
		return lightRayIntersectsPlane(r, shapes, sid);
	case Shape::ELLIPSOID:
	case Shape::HYPERBOLOID:
	case Shape::HYPERBOLOID2:
	case Shape::CYLINDER:
	case Shape::CONE:
		return lightRayIntersectsQuadraticSurface(r, shapes, sid);
	//case Shape::CONE:
	//	return lightRayIntersectsCone(r, shapes, sid);
	//case Shape::CYLINDER:
	//	return lightRayIntersectsCylinder(r, shapes, sid);	
	default:
		return -1.0;
	}
}

__device__ float lightRayIntersectsShapes(Ray r, d_Shape* shapes, int shapeCount) {
	float T_INIT = 1e10;
    // go through a list of shapes and find closest hit
    float t = T_INIT;

	float THRES = 1e-3;

    for(int i=0;i<shapeCount;i++) {
		// hack
		if( hasRefraction && shapes[i].material.Kf != 0.0 ) continue;
        float hitT = lightRayIntersectsShape(r, shapes, i);
		if( (hitT > -THRES) && (hitT < t) ) {
            t = hitT;
        }
    }

	if( t < T_INIT )
		return t;
	else return -1.0;
}

__device__ float lightRayIntersectsShapes2(Ray r, d_Shape* shapes, int shapeCount, int sid) {
	float T_INIT = 1e10;
    // go through a list of shapes and find closest hit
    float t = T_INIT;

	float THRES = 1e-3;

    for(int i=0;i<shapeCount;i++) {
		// hack
		if( hasRefraction && shapes[i].material.Kf != 0.0 ) continue;
		if( i == sid ) continue;
        float hitT = lightRayIntersectsShape(r, shapes, i);
		if( (hitT > -THRES) && (hitT < t) ) {
            t = hitT;
        }
    }

	if( t < T_INIT )
		return t;
	else return -1.0;
}

__device__ float travelPathLengthQuadraticSurface(Ray r, d_Shape* shapes, int sid, float threshold) {
	float3 pq = shapes[sid].p - r.origin;
	float a = dot(r.dir, mul(shapes[sid].m, r.dir));
	float b = -dot(pq, mul(shapes[sid].m, r.dir));
	float c = dot(pq, mul(shapes[sid].m, pq)) - shapes[sid].constant2;

    float delta = b*b - a*c;
    if (delta < 0.0)
    {
		return 0.0;
    }
    else
    {
		delta = sqrt(delta);        
		float inv = 1.0 / a;

		float x0 = (-b-delta)*inv;
		float x1 = (-b+delta)*inv;
        
		const float THRES = 1e-6;
		float t = 1e10;
		if( x0 >= THRES && x1 >= THRES && x0 <= threshold-1e-6 && x1 <= threshold-1e-6 ) {
			return fabs(x0 - x1);
		}
		else if( x0 <= THRES && x1 <= THRES ){
			return 0.0;
		}
		else if( x0 >= threshold-1e-6 && x1 >= threshold-1e-6 )
			return 0.0;
		else if(fmaxf(x0, x1) <= threshold-1e-6) {
			return fmaxf(x0, x1);
		}
		else return 0.0;
    }
}

__device__ float travelPathLength(Ray r, d_Shape* shapes, int shapeCount, int sid, int lid, float dist) {
	float L = 0.0;

    for(int i=0;i<shapeCount;i++) {
		if( i == sid || i == lid ) continue;
		if( shapes[i].t != Shape::PLANE )
			L += travelPathLengthQuadraticSurface(r, shapes, i, dist);
    }
	return L;
}


__device__ bool checkLightVisibility(float3 p, float3 N, d_Shape* shapes, int nShapes, d_Light* lights) {
	switch( lights->t ) {
	case Light::POINT:
	case Light::SPHERE: {
		float dist = length(p - lights->pos);
		Ray r;
		r.origin = p;
		r.dir = normalize(lights->pos - p);
		float t = lightRayIntersectsShapes(r, shapes, nShapes);

		float THRES = 1e-3;
		return t < THRES || t > dist;
	}
	case Light::SPOT: {
	}
	case Light::DIRECTIONAL: {
		Ray r;
		r.origin = p;
		r.dir = -lights->dir;
		float t = lightRayIntersectsShapes(r, shapes, nShapes);

		float THRES = 1e-3;
		return (t < THRES && (dot(N, lights->dir)<0));
	}
	default:
		return false;
	}
}

__device__ bool checkLightVisibility2(float3 l, float3 p, float3 N, d_Shape* shapes, int nShapes, int sid) {
	float dist = length(p - l);

	Ray r;
	r.origin = p;
	if( isDirectionalLight ) r.dir = normalize(l);
	else r.dir = normalize(l - p);
	float t = lightRayIntersectsShapes2(r, shapes, nShapes, sid);

	float THRES = 1e-3;
	return t < THRES || t > dist-THRES;
}

__device__ float3 phongShading(int2 res, float time, int x, int y, float3 v, float3 N, float2 t, Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int lightCount, int sid) {
    //float3 c = make_float3(0, 0, 0);
	float3 c = shapes[sid].material.emission;

    // iterate through all lights
    for(int i=0;i<lightCount;i++) {

		if( lights[i].t == Light::SPHERE ) {
			const int lightSamples = 1;
			float3 contribution = make_float3(0, 0, 0);
			d_Light tmpLt;
			tmpLt.init(lights[i]);
			float3 opos = tmpLt.pos;

			float3 cc = make_float3(0.0);
			for(int j=0;j<lightSamples;j++) {
				float3 offset = generateRandomOffsetFromThread(res, time, x, y);
				tmpLt.pos = opos + offset * lights[i].radius;
				// determine if this light is visible
				bool isVisible = checkLightVisibility(v, N, shapes, nShapes, &tmpLt);

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
					if( rectangularSpecular ) {
						float3 pq = lights[i].pos - shapes[sid].p;
						float3 uvec = normalize(cross(pq, N));
						float3 vvec = normalize(cross(uvec, N));

						float ufac = fabsf(dot(R-E, uvec));
						float vfac = fabsf(dot(R-E, vvec));

						specFactor = filter(ufac, 0.0, 0.25) * filter(vfac, 0.0, 0.25) * specFactor;
					}

					float3 Ispec = shapes[sid].material.specular * lights[i].specular
						* specFactor;
					Ispec = clamp(Ispec, 0.0, 1.0);

					if( shapes[sid].hasTexture ) {
						float3 Itexture = texel_supersample(textures[shapes[sid].texId], textureSize[shapes[sid].texId], t);
						cc = cc + Itexture * ((Idiff + Ispec) + Iamb) * lights[i].intensity;
					}
					else{
						if( cartoonShading )
							cc = cc + toonify((Idiff + Ispec) + Iamb, 8) * lights[i].intensity;
						else 
							cc = cc + ((Idiff + Ispec) + Iamb) * lights[i].intensity;
					}
				}
				else {
					if( shapes[sid].hasTexture ) {
						float3 Itexture = texel_supersample(textures[shapes[sid].texId], textureSize[shapes[sid].texId], t);
						cc = cc + Itexture * Iamb * lights[i].intensity;
					}
					else{
						cc = cc + Iamb * lights[i].intensity;
					}
				}
			}
			c += (cc/lightSamples);
		}
		else {
			// determine if this light is visible
			bool isVisible = checkLightVisibility(v, N, shapes, nShapes, lights+i);

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
				if( rectangularSpecular ) {
					float3 pq = lights[i].pos - shapes[sid].p;
					float3 uvec = normalize(cross(pq, N));
					float3 vvec = normalize(cross(uvec, N));

					float ufac = fabsf(dot(R-E, uvec));
					float vfac = fabsf(dot(R-E, vvec));

					specFactor = filter(ufac, 0.0, 0.25) * filter(vfac, 0.0, 0.25) * specFactor;
				}

				float3 Ispec = shapes[sid].material.specular * lights[i].specular
					* specFactor;
				Ispec = clamp(Ispec, 0.0, 1.0);

				if( shapes[sid].hasTexture ) {
					float3 Itexture = texel_supersample(textures[shapes[sid].texId], textureSize[shapes[sid].texId], t);
					c = c + Itexture * ((Idiff + Ispec) + Iamb) * lights[i].intensity;
				}
				else{
					if( cartoonShading )
						c = c + toonify((Idiff + Ispec) + Iamb, 8) * lights[i].intensity;
					else 
						c = c + ((Idiff + Ispec) + Iamb) * lights[i].intensity;
				}
			}
			else {
				if( shapes[sid].hasTexture ) {
					float3 Itexture = texel_supersample(textures[shapes[sid].texId], textureSize[shapes[sid].texId], t);
					c = c + Itexture * Iamb * lights[i].intensity;
				}
				else{
					c = c + Iamb * lights[i].intensity;
				}
			}
		}
    }

    return c;
}

__device__ float3 lambertShading(int2 res, float time, int x, int y, float3 v, float3 N, float2 t, Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int lightCount, int sid) {
    float3 c = make_float3(0, 0, 0);

    // iterate through all lights
    for(int i=0;i<lightCount;i++) {
		if( lights[i].t == Light::SPHERE ) {
			const int lightSamples = 1;
			float3 contribution = make_float3(0, 0, 0);
			d_Light tmpLt;
			tmpLt.init(lights[i]);
			float3 opos = tmpLt.pos;
			for(int j=0;j<lightSamples;j++) {
				float3 offset = generateRandomOffsetFromThread(res, time, x, y);
				tmpLt.pos = opos + offset * lights[i].radius;
				// determine if this light is visible
				bool isVisible = checkLightVisibility(v, N, shapes, nShapes, &tmpLt);
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

					float diffuseFactor = max(NdotL, 0.0);
					if( cartoonShading ) diffuseFactor = toonify(diffuseFactor, 8);

					float3 Idiff = clamp(shapes[sid].material.diffuse * tmpLt.diffuse * diffuseFactor, 0.0, 1.0);

					contribution = contribution + Idiff * tmpLt.intensity;
				}
			}
			float3 Itexture;
			if( shapes[sid].hasTexture ) {
				Itexture = texel_supersample(textures[shapes[sid].texId], textureSize[shapes[sid].texId], t);
			}
			else Itexture = make_float3(1.0);
			c = c + Itexture * contribution / (float)lightSamples;

			// restore position
			lights[i].pos = opos;
		}
		else {
			// determine if this light is visible
			bool isVisible = checkLightVisibility(v, N, shapes, nShapes, lights+i);

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
					Itexture = texel_supersample(textures[shapes[sid].texId], textureSize[shapes[sid].texId], t);
				}
				else Itexture = make_float3(1, 1, 1);

				float diffuseFactor = max(NdotL, 0.0);
				if( cartoonShading ) diffuseFactor = toonify(diffuseFactor, 8);

				float3 Idiff = clamp(shapes[sid].material.diffuse * lights[i].diffuse * diffuseFactor, 0.0, 1.0);

				c = c + Itexture * Idiff * lights[i].intensity;
			}
		}
    }

    return c;
}

__device__ float3 goochShading(int2 res, float time, int x, int y, float3 v, float3 N, float2 t, Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int lightCount, int sid) {

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

__device__ float3 phongShading2(int2 res, float time, int x, int y, float3 v, float3 N, float2 t, Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int lightCount, int sid) {
    float3 c = make_float3(0, 0, 0);

    // iterate through all lights
    for(int i=0;i<nShapes;i++) {
		if( i == sid ) continue;
		if( shapes[i].material.emission.x != 0 || shapes[i].material.emission.y != 0 || shapes[i].material.emission.z != 0 )
		{
			// this is a light, create a shadow ray
			float3 lpos;
			if( isRayTracing )
				lpos = shapes[i].p;
			else
				lpos = shapes[i].randomPointOnSurface(res, time, x, y);

			// determine if this light is visible
			bool isVisible = checkLightVisibility2(lpos, v, N, shapes, nShapes, i);

			//calculate Ambient Term:
			float3 Iamb = shapes[sid].material.ambient * shapes[i].material.emission;


			if( isVisible ) {
				float3 L;
				if( isDirectionalLight )
					L = normalize(lpos);
				else
					L = normalize(lpos - v);

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

				float spotLightFactor = 1.0;
				if( isSpotLight ) {
					float3 ldir = normalize(lpos);
					const float spotlightCutoff = 0.75;
					float cosLL = clamp(dot(ldir, L), spotlightCutoff, 1.0);
					
					// hard spot light
					//spotLightFactor = step(spotlightCutoff, cosLL);

					// soft spot light
					spotLightFactor = (cosLL - spotlightCutoff)/(1.0-spotlightCutoff);
					NdotL *= spotLightFactor;
				}

				//calculate Diffuse Term:
				float3 Idiff = shapes[sid].material.diffuse * shapes[i].material.emission * max(NdotL, 0.0);
				Idiff = clamp(Idiff, 0.0, 1.0);

				// calculate Specular Term:
				float specFactor = pow(max(RdotE,0.0),0.3 * shapes[sid].material.shininess);
				if( circularSpecular ) specFactor = step(0.8, specFactor);
				if( rampedSpecular ) specFactor = toonify(specFactor, 4);
				if( rectangularSpecular ) {
					float3 pq = r.origin - shapes[sid].p;
					float3 uvec = normalize(cross(pq, N));
					float3 vvec = normalize(cross(uvec, N));

					float ufac = dot(R-E, uvec);
					float vfac = dot(R-E, vvec);

					specFactor = filter(ufac, -0.25, 0.25) * filter(vfac, 0.1, 0.5);
				}
				if( isSpotLight ) {
					specFactor *= spotLightFactor;
				}

				float3 Ispec = shapes[sid].material.specular * shapes[i].material.emission
					* specFactor;
				Ispec = clamp(Ispec, 0.0, 1.0);

				if( shapes[sid].hasTexture ) {
					float3 Itexture = texel_supersample(textures[shapes[sid].texId], textureSize[shapes[sid].texId], t);
					c = c + Itexture * ((Idiff + Ispec) + Iamb);
				}
				else{
					if( cartoonShading )
						c = c + toonify((Idiff + Ispec) + Iamb, 8);
					else 
						c = c + ((Idiff + Ispec) + Iamb);
				}
			}
			else {
				if( shapes[sid].hasTexture ) {
					float3 Itexture = texel_supersample(textures[shapes[sid].texId], textureSize[shapes[sid].texId], t);
					c = c + Itexture * Iamb;
				}
				else{
					c = c + Iamb;
				}
			}
		}
    }

    return c;
}

__device__ float3 lambertShading2(int2 res, float time, int x, int y, float3 v, float3 N, float2 t, Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int lightCount, int sid) {
    float3 c = make_float3(0, 0, 0);

    // iterate through all lights
    for(int i=0;i<nShapes;i++) {
		if( i == sid ) continue;
		if( shapes[i].material.t == Material::Emissive )
		{
			// this is a light, create a shadow ray
			float3 lpos;
			if( isRayTracing )
				lpos = shapes[i].p;
			else
				lpos = shapes[i].randomPointOnSurface(res, time, x, y);

			// determine if this light is visible
			bool isVisible = checkLightVisibility2(lpos, v, N, shapes, nShapes, i);

			//calculate Ambient Term:
			float3 Iamb = shapes[sid].material.ambient * shapes[i].material.emission;

			if( isVisible || fakeSoftShadow ) {
				float3 L;
				if( isDirectionalLight )
					L = normalize(lpos);
				else
					L = normalize(lpos - v);
				float3 E = normalize(r.origin - v);

				float NdotL;

				// change the normal with normal map
				if( shapes[sid].hasNormalMap ) {
					
					// normal defined in tangent space
					float3 n_normalmap = normalize(texel_supersample(textures[shapes[sid].normalTexId], textureSize[shapes[sid].normalTexId], t) * 2.0 - 1.0);
										
					float3 tangent;
					if( shapes[sid].t == Shape::PLANE )
						tangent = shapes[sid].axis[1];
					else
						tangent = normalize(sphere_tangent(N));
					float3 bitangent = cross(N, tangent);

					// find the mapping from tangent space to camera space
					mat3 m_t = mat3(tangent, bitangent, N);

					// convert the normal to to camera space
					NdotL = dot(n_normalmap, normalize(m_t*L));					
				}
				else {
					NdotL = dot(N, L);
				}

				if( isSpotLight ) {
					float3 ldir = normalize(lpos);
					NdotL *= step(0.75, dot(ldir, L));
				}


				float3 Itexture;
				if( shapes[sid].hasTexture ) {
					Itexture = texel_supersample(textures[shapes[sid].texId], textureSize[shapes[sid].texId], t);
				}
				else Itexture = make_float3(1, 1, 1);

				float diffuseFactor = max(NdotL, 0.0);
				if( cartoonShading ) diffuseFactor = toonify(diffuseFactor, 8);

				float3 Idiff;
				if( rainbowLight ) {
					float3 lightColor = mix(make_float3(1, 0.5, 0.5), make_float3(0.5, 1, 0.5), make_float3(0.5, 0.5, 1), diffuseFactor);
					Idiff = clamp(shapes[sid].material.diffuse * lightColor * diffuseFactor, 0.0, 1.0);
				}
				else{
					Idiff = clamp(shapes[sid].material.diffuse * shapes[i].material.emission * diffuseFactor, 0.0, 1.0);
				}
								
				float shadowFactor = 1.0;
				if( fakeSoftShadow ) {
					Ray tr;
					tr.origin = v;
					tr.dir = lpos - v;
					float dist = length(tr.dir);
					tr.dir = normalize(tr.dir);
					const float maxLength = 5.0;
					shadowFactor = clamp((maxLength - travelPathLength(tr, shapes, nShapes, sid, i, dist))/maxLength, 0.0f, 1.0f);
				}

				c += Itexture * (Idiff + Iamb) * shadowFactor;
			}
			else {
				float3 Itexture;
				if( shapes[sid].hasTexture ) {
					Itexture = texel_supersample(textures[shapes[sid].texId], textureSize[shapes[sid].texId], t);
				}
				else Itexture = make_float3(1, 1, 1);

				c += Itexture * Iamb;
			}
		}
    }

    return c;
}

__device__ float3 goochShading2(int2 res, float time, int x, int y, float3 v, float3 N, float2 t, Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int lightCount, int sid) {

    float3 c = make_float3(0, 0, 0);

    // iterate through all lights
    for(int i=0;i<nShapes;i++) {
		if( i == sid ) continue;
		if( shapes[i].material.t == Material::Emissive )
		{
			// this is a light, create a shadow ray
			float3 lpos;
			if( isRayTracing )
				lpos = shapes[i].p;
			else
				lpos = shapes[i].randomPointOnSurface(res, time, x, y);

			// determine if this light is visible
			bool isVisible = checkLightVisibility2(lpos, v, N, shapes, nShapes, i);

			float3 L;
			if( isDirectionalLight )
				L = normalize(lpos);
			else
				L = normalize(lpos - v);

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

			float shadowFactor = 1.0;
			if( fakeSoftShadow ) {
				Ray tr;
				tr.origin = v;
				tr.dir = lpos - v;
				float dist = length(tr.dir);
				tr.dir = normalize(tr.dir);
				const float maxLength = 5.0;
				shadowFactor = clamp((maxLength - travelPathLength(tr, shapes, nShapes, sid, i, dist))/maxLength, 0.0f, 1.0f);
			}
			else {
				shadowFactor = isVisible?1.0:0.0;
			}

			float3 Idiff = diffuse * NdotL * shadowFactor;
			float3 kcdiff = fminf(shapes[sid].material.kcool + shapes[sid].material.alpha * Idiff, 1.0f);
			float3 kwdiff = fminf(shapes[sid].material.kwarm + shapes[sid].material.beta * Idiff, 1.0f);
			float3 kfinal = mix(kcdiff, kwdiff, (NdotL+1.0)*0.5) * shapes[i].material.emission;
			// calculate Specular Term:
			float3 Ispec = shapes[sid].material.specular * shapes[i].material.emission
				* pow(max(dot(R,E),0.0),0.3*shapes[sid].material.shininess) * (isVisible?1.0:0.0);
			Ispec = step(make_float3(0.5, 0.5, 0.5), Ispec);
			// edge effect
			//float EdotN = dot(E, N);
			//if( fabs(EdotN) >= 0.2 ) 
				c = c + clamp(kfinal + Ispec, 0.0, 1.0);
		}
    }
    return clamp(c, 0.0f, 1.0f);
}

__device__ __forceinline__ Hit background() {
	Hit h;
	h.t = -1.0;
	h.objIdx = -1;
	return h;
}

__device__ float3 computeShading(int2 res, float time, int x, int y, float3 p, float3 n, float2 t, Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int nLights, int sid) {
    switch( shadingMode ) {
	case 1:
        return lambertShading(res, time, x, y, p, n, t, r, shapes, nShapes, lights, nLights, sid);
	case 2:
        return phongShading(res, time, x, y, p, n, t, r, shapes, nShapes, lights, nLights, sid);
	case 3:
        return goochShading(res, time, x, y, p, n, t, r, shapes, nShapes, lights, nLights, sid);
	default:
		return make_float3(0, 0, 0);
	}
}

__device__ float3 computeShading2(int2 res, float time, int x, int y, float3 p, float3 n, float2 t, Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int nLights, int sid) {
    switch( shadingMode ) {
	case 1:
        return lambertShading2(res, time, x, y, p, n, t, r, shapes, nShapes, lights, nLights, sid);
	case 2:
        return phongShading2(res, time, x, y, p, n, t, r, shapes, nShapes, lights, nLights, sid);
	case 3:
        return goochShading2(res, time, x, y, p, n, t, r, shapes, nShapes, lights, nLights, sid);
	default:
		return make_float3(0, 0, 0);
	}
}

__device__ __forceinline__ Ray generateRay(Camera* cam, float u, float v) {
	Ray r;
	r.origin = cam->pos.data;

    // find the intersection point on the canvas
    float3 canvasCenter = cam->f * cam->dir.data + cam->pos.data;
    float3 pcanvas = canvasCenter + u * cam->w * cam->right.data + v * cam->h * cam->up.data;

    r.dir = normalize(pcanvas - cam->pos.data);

	return r;
}

__device__ __forceinline__ float rand(float x, float y){
  float val = sin(x * 12.9898 + y * 78.233) * 43758.5453;
  return val - floorf(val);
}

// ray intersection test with shading computation
__device__ Hit rayIntersectsSphere(Ray r, d_Shape* shapes, int nShapes, int sid) {
	float ti = lightRayIntersectsSphere(r, shapes, sid);

	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
        // hit point
        h.p = h.t * r.dir + r.origin;
        // normal at hit point
		h.n = normalize(h.p - shapes[sid].p);
        // hack, move the point a little bit outer
		h.p = shapes[sid].p + (shapes[sid].radius[0] + 1e-6) * h.n;
        h.tex = spheremap(h.n);
		h.objIdx = sid;
        return h;
    }
	else return background();
}

__device__ Hit rayIntersectsPlane(Ray r, d_Shape* shapes, int nShapes, int sid) {
	float ti = lightRayIntersectsPlane(r, shapes, sid);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
        h.p = r.origin + h.t * r.dir;
        // compute u, v coordinates
        float3 pp0 = h.p - shapes[sid].p;
        float u = dot(pp0, shapes[sid].axis[1]);
        float v = dot(pp0, shapes[sid].axis[2]);
        if( abs(u) > shapes[sid].radius[0] || abs(v) > shapes[sid].radius[1] ) return background();
        else {
            float2 t = clamp((make_float2(u / shapes[sid].radius[0], v / shapes[sid].radius[1]) + make_float2(1.0, 1.0))*0.5, 0.0, 1.0);
			float scale = 0.5 * (shapes[sid].radius[0]/5.0 + shapes[sid].radius[1]/5.0);
			h.tex = t * scale;
			h.n = shapes[sid].axis[0];
			h.objIdx = sid;

			return h;
        }
    }
	else return background();
}

__device__ Hit rayIntersectsQuadraticSurface(Ray r, d_Shape* shapes, int nShapes, int sid) {
	float ti = lightRayIntersectsQuadraticSurface(r, shapes, sid);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
		// hit point
		h.p = h.t * r.dir + r.origin;
        // normal at hit point
		h.n = normalize(2.0 * mul(shapes[sid].m, (h.p - shapes[sid].p)) * shapes[sid].constant);
		
		h.tex = spheremap(h.n);
		
		h.objIdx = sid;
        return h;
    }

	else return background();
}

__device__ Hit rayIntersectsCone(Ray r, d_Shape* shapes, int nShapes, int sid) {
	float ti = lightRayIntersectsCone(r, shapes, sid);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
		h.p = h.t * r.dir + r.origin;
        float3 pq = h.p - shapes[sid].p;
        float hval = dot(pq, shapes[sid].axis[0]);
        // normal at hit point
		if(fabsf(hval - shapes[sid].radius[1])<1e-4) 
			h.n = shapes[sid].axis[0];
		else 
			h.n = normalize(cross(cross(shapes[sid].axis[0], pq), shapes[sid].axis[0]));

        h.tex = make_float2(0, 0);
		h.objIdx = sid;

        return h;
    }
	else return background();
}

__device__ Hit rayIntersectsCylinder(Ray r, d_Shape* shapes, int nShapes, int sid) {
	float ti = lightRayIntersectsCylinder(r, shapes, sid);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
        h.p = h.t * r.dir + r.origin;
        float3 pq = h.p - shapes[sid].p;
        float hval = dot(pq, shapes[sid].axis[0]);
        // normal at hit point
		h.n = normalize(pq - hval*shapes[sid].axis[0]);
		if( fabsf(hval) < 1e-3 ) h.n = -shapes[sid].axis[0];
		else if( fabsf(hval - shapes[sid].radius[1]) < 1e-3 ) h.n = shapes[sid].axis[0];
        h.tex = make_float2(0, 0);
		h.objIdx = sid;

        return h;
    }
	else return background();
}

__device__ __forceinline__ Hit rayIntersectsShape(Ray r, d_Shape* shapes, int nShapes, int sid) {
	switch( shapes[sid].t ) {
	case Shape::PLANE:
		return rayIntersectsPlane(r, shapes, nShapes, sid);
	case Shape::ELLIPSOID:
	case Shape::HYPERBOLOID:
	case Shape::HYPERBOLOID2:
	case Shape::CYLINDER:
	case Shape::CONE:
		return rayIntersectsQuadraticSurface(r, shapes, nShapes, sid);
	//case Shape::CONE:
	//	return rayIntersectsCone(r, shapes, nShapes, sid);
	//case Shape::CYLINDER:
	//	return rayIntersectsCylinder(r, shapes, nShapes, sid);	
	default:
		return background();
	}
}

__device__ Hit rayIntersectsShapes(Ray r, int nShapes, d_Shape* shapes) {
	Hit h;
	h.t = 1e10;
	h.objIdx = -1;

    for(int i=0;i<nShapes;i++) {
        Hit hit = rayIntersectsShape(r, shapes, nShapes, i);
        if( (hit.t > 0.0) && (hit.t < h.t) ) {
            h = hit;
        }
    }

	return h;
}

__device__ float3 traceRay_simple(float time, int2 res, int x, int y, Ray r, int nShapes, d_Shape* shapes, int nLights, d_Light* lights) {
	Hit h = rayIntersectsShapes(r, nShapes, shapes);
	if( h.objIdx == -1 ) {
		// no hit, sample environment map
		if( envMapIdx >= 0 ) {
			float2 t = spheremap(r.dir);
			return texel_supersample(textures[envMapIdx], textureSize[envMapIdx], t);
		}
		else
			return make_float3(0, 0, 0);
	}
	else {
		// hit a light
		if( shapes[h.objIdx].material.emission.x != 0 || shapes[h.objIdx].material.emission.y != 0 || shapes[h.objIdx].material.emission.z != 0 )
		{
			return shapes[h.objIdx].material.emission;
		}
		else {
			return computeShading2(res, time, x, y, h.p, h.n, h.tex, r, shapes, nShapes, lights, nLights, h.objIdx);
		}
	}
}

__device__ float3 traceRay_reflection(float time, int2 res, int x, int y, Ray r, int nShapes, d_Shape* shapes, int nLights, d_Light* lights) {
	const int maxBounces = 10;
	float3 accumulatedColor = make_float3(0.0);
	float3 colormask = make_float3(1.0);
	Ray ray = r;
	for(int bounces=0;bounces<maxBounces;bounces++) {

		Hit h = rayIntersectsShapes(ray, nShapes, shapes);
		
		float3 color;

		if( h.objIdx == -1 ) {
			// no hit
			color = make_float3(0, 0, 0);
			break;
		}
		else {
			// hit a light
			if( shapes[h.objIdx].material.emission.x != 0 || shapes[h.objIdx].material.emission.y != 0 || shapes[h.objIdx].material.emission.z != 0 )
			{
				accumulatedColor += shapes[h.objIdx].material.emission;
			}
			else {
				color = clamp(computeShading2(res, time, x, y, h.p, h.n, h.tex, ray, shapes, nShapes, lights, nLights, h.objIdx), 0.0, 1.0);

				accumulatedColor += colormask * color * shapes[h.objIdx].material.Ks;
				colormask *= color;
			}
		}

		// get the reflected ray
		ray.origin = h.p;
		ray.dir = reflect(ray.dir, h.n);
	}

	return accumulatedColor;
}

__device__ float3 traceRay_refraction(float time, int2 res, int x, int y, Ray r, int nShapes, d_Shape* shapes, int nLights, d_Light* lights) {
	const int maxBounces = 5;
	float3 accumulatedColor = make_float3(0.0);
	float3 colormask = make_float3(1.0);

	Ray ray = r;
	for(int bounces=0;bounces<maxBounces;bounces++) {
		Hit h = rayIntersectsShapes(ray, nShapes, shapes);
		
		float3 color;

		if( h.objIdx == -1 ) {
			// no hit
			color = make_float3(0, 0, 0);
			accumulatedColor += color * colormask;
			break;
		}
		else {
			color = clamp(computeShading(res, time, x, y, h.p, h.n, h.tex, ray, shapes, nShapes, lights, nLights, h.objIdx), 0.0, 1.0);

			accumulatedColor += color * colormask * shapes[h.objIdx].material.Ks;
			float Kf = shapes[h.objIdx].material.Kf;
			if( Kf == 0.0 ) break;
			colormask *= Kf;
		}

		// get the reflected ray
		ray.origin = h.p;

		if( dot(ray.dir, h.n) < 0 ) {
			// enter ray
			ray.dir = refract(ray.dir, h.n, 1.0/shapes[h.objIdx].material.eta);
		}
		else 
		{
			// leaving ray
			ray.dir = refract(ray.dir, -h.n, shapes[h.objIdx].material.eta);
		}

		/*
		float2 uv = generateRandomNumberFromThread2(res, time, x, y);
		ray.dir += 0.1 * normalize(calculateRandomDirectionInHemisphere(h.n, uv.x, uv.y));
		*/
	}

	return accumulatedColor;
}

__device__ float3 computeShadow(int2 res, float time, int x, int y, float3 v, float3 N, float2 t, Ray r, d_Shape* shapes, int nShapes, d_Light* lights, int lightCount, int sid) {
    float3 c = make_float3(0, 0, 0);

    // iterate through all lights
    for(int i=0;i<nShapes;i++) {
		if( i == sid ) continue;
		if( shapes[i].material.t == Material::Emissive )
		{
			// this is a light, create a shadow ray
			float3 lpos = shapes[i].randomPointOnSurface(res, time, x, y);

			// determine if this light is visible
			bool isVisible = checkLightVisibility2(lpos, v, N, shapes, nShapes, i);

			if( isVisible ) {
				float3 L = normalize(lpos - v);
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
					Itexture = texel_supersample(textures[shapes[sid].texId], textureSize[shapes[sid].texId], t);
				}
				else Itexture = make_float3(1, 1, 1);

				float diffuseFactor = max(NdotL, 0.0);
				if( cartoonShading ) diffuseFactor = toonify(diffuseFactor, 8);

				float3 Idiff = clamp(shapes[sid].material.diffuse * shapes[i].material.emission * diffuseFactor, 0.0, 1.0);

				c += Itexture * Idiff;
			}
		}
    }

    return c;
}

__device__ float3 traceRay_general(float time, int2 res, int x, int y, Ray r, int nShapes, d_Shape* shapes, int nLights, d_Light* lights) {
	const int maxBounces = 10;
	float3 accumulatedColor = make_float3(0.0);
	float3 colormask = make_float3(1.0);

	Ray ray = r;
	int bounces = 0;
	for(;bounces<maxBounces;bounces++) {
		Hit h = rayIntersectsShapes(ray, nShapes, shapes);		
		float3 color = make_float3(0.0f);

		if( h.objIdx == -1 ) {	
			// no hit, sample environment map
			if( envMapIdx >= 0 ) {
				float2 t = spheremap(ray.dir);
				colormask *= texel_supersample(textures[envMapIdx], textureSize[envMapIdx], t);
			}
			break;
		}

		switch( shapes[h.objIdx].material.t ) {
		case Material::Emissive:
			{				
				//float maxf = fmaxf(shapes[h.objIdx].material.emission.x, shapes[h.objIdx].material.emission.y);
				//maxf = fmaxf(shapes[h.objIdx].material.emission.z, maxf);				

				// hit a light
				accumulatedColor += shapes[h.objIdx].material.emission * colormask;
				colormask *= shapes[h.objIdx].material.emission;

				// send it out again
				ray.origin = h.p + 1e-3 * h.n;
				float2 uv = generateRandomNumberFromThread2(res, time, x, y);
				
				ray.dir = calculateRandomDirectionInHemisphere(h.n, uv.x, uv.y);

				break;
			}
		case Material::Diffuse:
			{
				// direct lighting
				//float3 shading = computeShadow(res, time, x, y, h.p, h.n, h.tex, ray, shapes, nShapes, lights, nLights, h.objIdx);
				//accumulatedColor += shading * colormask;

				accumulatedColor += shapes[h.objIdx].material.emission * colormask;

				if( shapes[h.objIdx].hasTexture ) {
					color = texel_supersample(textures[shapes[h.objIdx].texId], textureSize[shapes[h.objIdx].texId], h.tex);
				}
				else color = shapes[h.objIdx].material.diffuse;
				
				colormask *= color * 0.5;

				ray.origin = h.p;
				float2 uv = generateRandomNumberFromThread2(res, time, x, y);
				
				ray.dir = calculateRandomDirectionInHemisphere(h.n, uv.x, uv.y);
				break;
			}
		case Material::Glossy:
			{
				// direct lighting
				//float3 shading = computeShadow(res, time, x, y, h.p, h.n, h.tex, ray, shapes, nShapes, lights, nLights, h.objIdx);
				//accumulatedColor += shading * colormask;

				float3 color;
				if( shapes[h.objIdx].hasTexture ) {
					color = texel_supersample(textures[shapes[h.objIdx].texId], textureSize[shapes[h.objIdx].texId], h.tex);
				}
				else color = shapes[h.objIdx].material.diffuse;

				colormask *= color;

				ray.origin = h.p + 1e-3 * h.n;
				ray.dir = reflect(ray.dir, h.n);
				float2 uv = generateRandomNumberFromThread2(res, time, x, y);
				
				ray.dir += calculateRandomDirectionInHemisphere(h.n, uv.x, uv.y) * shapes[h.objIdx].material.Kr;
				ray.dir = normalize(ray.dir);
				break;
			}
		case Material::DiffuseScatter:
			{
				// direct lighting
				float3 shading = computeShadow(res, time, x, y, h.p, h.n, h.tex, ray, shapes, nShapes, lights, nLights, h.objIdx);
				accumulatedColor += shading * colormask * (1.0 - shapes[h.objIdx].material.Kr);

				// get a random number
				float Xi = generateRandomNumberFromThread1(res, time, x, y);

				if( Xi > shapes[h.objIdx].material.Kr ) { 
					// emission is zero, no need to add it
					//accumulatedColor += shapes[h.objIdx].material.emission * colormask;
					colormask *= shapes[h.objIdx].material.diffuse;

					ray.origin = h.p + 1e-3 * h.n;
					float2 uv = generateRandomNumberFromThread2(res, time, x, y);
				
					ray.dir = calculateRandomDirectionInHemisphere(h.n, uv.x, uv.y);
				}
				else {
					colormask *= shapes[h.objIdx].material.diffuse;

					// get the reflected ray
					ray.origin = h.p + 1e-3 * h.n;
					ray.dir = reflect(ray.dir, h.n);
				}
				break;
			}
		case Material::Specular:
			{
				// direct lighting
				colormask *= shapes[h.objIdx].material.diffuse;

				// get the reflected ray
				ray.origin = h.p + 1e-3 * h.n;
				ray.dir = reflect(ray.dir, h.n);
				break;
			}
		case Material::Refractive:
			{
				color = shapes[h.objIdx].material.diffuse;
				colormask *= color;

				// get the reflected ray
				Ray rf;
				rf.origin = h.p; rf.dir = normalize(reflect(ray.dir, h.n));
				
				double dDn = dot(h.n, ray.dir);
				float3 n1 = dDn<0?h.n:-h.n;
				bool into = dDn<0;
				double nc=1.0, nt=shapes[h.objIdx].material.eta;
				double nnt=into?nc/nt:nt/nc, ddn = dot(ray.dir, n1);
				double cos2t=1.0-nnt*nnt*(1-ddn*ddn);

				if( cos2t<0.0 ) {
					// total internal reflection
					rf.origin -= 1e-3 * h.n;
					ray = rf;
				}
				else {
					// choose either reflection or refraction
					Ray rr;
					rr.origin = h.p - 1e-3 * n1; 
					rr.dir = normalize(nnt*ray.dir-(into?1.0:-1.0)*(ddn*nnt+sqrtf(cos2t))*h.n);

					float a = nt-nc, b = nt+nc, R0 = a*a/(b*b), c = 1.0 - (into?-ddn:dot(rr.dir, h.n));
					float Re = R0 + (1-R0)*powf(c, 5.0), Tr =1.0 - Re, P=0.25+0.5*Re, RP = Re/P, TP = Tr/(1.0-P);

					float Xi = generateRandomNumberFromThread1(res, time, x, y);

					if( Xi < P ) {
						ray = rf;
						//colormask *= RP;
					}
					else {
						ray = rr;
						//colormask *= TP;
					}
				}
				break;
			}
		}
	}

	// finally, collect it
	accumulatedColor += colormask;

	return accumulatedColor;
}

__device__ d_Light dev_lights[16];
__device__ d_Shape dev_shapes[256];

__global__ void initScene(int nLights, Light* lights, 
						  int nShapes, Shape* shapes) 
{
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int tid = tidy * blockDim.x + tidx;
	if( tid < nLights )	dev_lights[tid].init(lights[tid]);
	if( tid < nShapes )	dev_shapes[tid].init(shapes[tid]);
}

__global__ void clearCumulatedColor(float3* color, int w, int h) {
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x > w-1 || y > h-1 ) return;

	int idx = y*w+x;
	color[idx] = make_float3(0.0);
}

__global__ void copy2pbo(float3 *color, float3 *pos, int iters, int w, int h, float gamma) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x > w-1 || y > h-1 ) return;

	int idx = y*w+x;

	float3 inC = clamp(color[idx]/iters, 0.0, 1.0);
	inC = pow(inC, 1.0/gamma);

	Color c;
	c.c.data = make_float4(inC, 1.0);

	pos[idx] = make_float3(x, y, c.toFloat());
}

///////////////////////////////////////////////////////////////////////////////
//! main entry of the ray tracing program
///////////////////////////////////////////////////////////////////////////////
__global__ void raytrace(float time, float3 *color, Camera* cam, 
						 int nLights, Light* lights, 
						 int nShapes, Shape* shapes, 
						 unsigned int width, unsigned int height,
						 int sMode, int AASamples)
{
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	// load scene information into block
	shadingMode = sMode;
	
	__shared__ int inLightsCount;
	__shared__ int inShapesCount;
	__shared__ d_Shape inShapes[64];
	__shared__ d_Light inLights[4];
	
	inLightsCount = nLights;
	inShapesCount = nShapes;
	int tid = tidy * blockDim.x + tidx;
	if( tid < nLights )	inLights[tid].init(lights[tid]);
	if( tid < nShapes )	inShapes[tid].init(shapes[tid]);
	
	__syncthreads();

	int2 resolution = make_int2(width, height);

	unsigned int x = blockIdx.x*blockDim.x + tidx;
	unsigned int y = blockIdx.y*blockDim.y + tidy;

	if( x > width - 1 || y > height - 1 ) return;
	
	float3 c = make_float3(0.0);
	int edgeSamples = sqrtf(AASamples);
	float step = 1.0 / edgeSamples;

	for(int i=0;i<AASamples;i++) {
		float px = floor(i*step);
		float py = i % edgeSamples;

		float2 offset = make_float2(0, 0);
		if( jittered )
			offset = generateRandomOffsetFromThread2(resolution, time, x, y);

		float u = x + (px + offset.x) * step;
		float v = y + (py + offset.y) * step;
		u = u / (float) width - 0.5;
		v = v / (float) height - 0.5;

		Ray r = generateRay(cam, u, v);

		//if( isRayTracing ) {
			c += traceRay_simple(time, resolution, x, y, r, inShapesCount, inShapes, inLightsCount, inLights);
		//}
		//else c += traceRay_general(time, resolution, x, y, r, inShapesCount, inShapes, inLightsCount, inLights);
	}

	c /= (float)AASamples;

	// write output vertex
	color[y*width+x] += c;
}

///////////////////////////////////////////////////////////////////////////////
//! main entry of the ray tracing program, with load balancing
///////////////////////////////////////////////////////////////////////////////
__global__ void raytrace2(float time, float3 *color, Camera* cam, 
						  int nLights, Light* lights, 
						  int nShapes, Shape* shapes, 
						  unsigned int width, unsigned int height,
						  int sMode, int AASamples, int gx, int gy, int gmx, int gmy)
{
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	// load scene information into block
	shadingMode = sMode;
	
	__shared__ int inLightsCount;
	__shared__ int inShapesCount;
	__shared__ d_Shape inShapes[64];
	__shared__ d_Light inLights[4];

	inLightsCount = nLights;
	inShapesCount = nShapes;
	int tid = tidy * blockDim.x + tidx;
	if( tid < nLights )	inLights[tid].init(lights[tid]);
	if( tid < nShapes )	inShapes[tid].init(shapes[tid]);
	__syncthreads();
	
		
	int2 resolution = make_int2(width, height);
	for(int gi=0;gi<gmy;gi++) {
		for(int gj=0;gj<gmx;gj++) {

			unsigned int xoffset = gj * gx * blockDim.x;
			unsigned int yoffset = gi * gy * blockDim.y;

			unsigned int x = blockIdx.x*blockDim.x + tidx + xoffset;
			unsigned int y = blockIdx.y*blockDim.y + tidy + yoffset;

			if( x > width - 1 || y > height - 1 ) return;

			float3 c = make_float3(0, 0, 0);
			int edgeSamples = sqrtf(AASamples);
			float step = 1.0 / edgeSamples;

			for(int i=0;i<AASamples;i++) {
				float px = floor(i*step);
				float py = i % edgeSamples;

				float2 offset = generateRandomOffsetFromThread2(resolution, time, x, y);

				float u = x + (px + offset.x) * step;
				float v = y + (py + offset.y) * step;

				u = u / (float) width - 0.5;
				v = v / (float) height - 0.5;

				Ray r = generateRay(cam, u, v);

				if( isRayTracing ) {
					c += traceRay_simple(time, resolution, x, y, r, inShapesCount, inShapes, inLightsCount, inLights);
				}
				else c += traceRay_general(time, resolution, x, y, r, inShapesCount, inShapes, inLightsCount, inLights);
			}

			c /= (float)AASamples;

#define SHOW_AFFINITY 0
#if SHOW_AFFINITY
			
			c.x = blockIdx.x / (float)gx;
			c.y = blockIdx.y / (float)gy;
			c.z = 0.0;
			
			/*
			c.c.r = threadIdx.x / (float)blockDim.x;
			c.c.g = threadIdx.y / (float)blockDim.y;
			c.c.b = 0.0;
			*/			
#endif

			// write output vertex
			color[y*width+x] += c;
			__syncthreads();
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
__global__ void raytrace3(float time, float3 *color, Camera* cam, 
						  int nLights, Light* lights, 
						  int nShapes, Shape* shapes,
						  unsigned int width, unsigned int height,
						  int sMode, int AASamples, int bmx, int bmy, int ttlb)
{
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int tid = tidy * blockDim.x + tidx;

	// load scene information into block
	shadingMode = sMode;	

	__shared__ int inLightsCount;
	__shared__ int inShapesCount;
	__shared__ d_Shape inShapes[64];
	__shared__ d_Light inLights[4];

	inLightsCount = nLights;
	inShapesCount = nShapes;

	//if( tid < nLights )	inLights[tid].init(lights[tid]);
	if( tid < nShapes )	inShapes[tid].init(shapes[tid]);
	__syncthreads();

	__shared__ int currentBlock;
	if( tid == 0 ){
		currentBlock = curb;
	}
	//__threadfence_system();
	__threadfence();
	//__syncthreads();

	int2 resolution = make_int2(width, height);

	// total number of blocks, current block
	do {
		int bx = currentBlock % bmx;
		int by = currentBlock / bmx;

		unsigned int xoffset = bx * blockDim.x;
		unsigned int yoffset = by * blockDim.y;

		unsigned int x = tidx + xoffset;
		unsigned int y = tidy + yoffset;

		if( x > width - 1 || y > height - 1 ) return;

		float3 c;
		float2 offset = generateRandomOffsetFromThread2(resolution, time, x, y);

		float u = x + offset.x;
		float v = y + offset.y;

		u = u / (float) width - 0.5;
		v = v / (float) height - 0.5;

		Ray r = generateRay(cam, u, v);

		if( isRayTracing ) {
			c = traceRay_simple(time, resolution, x, y, r, inShapesCount, inShapes, inLightsCount, inLights);
		}
		else c = traceRay_general(time, resolution, x, y, r, inShapesCount, inShapes, inLightsCount, inLights);

#define SHOW_AFFINITY 0
#if SHOW_AFFINITY
		c.x = blockIdx.x / (float)gridDim.x;
		c.y = blockIdx.y / (float)gridDim.y;
		c.z = 1.0 - c.x;

		/*
		c.c.r = threadIdx.x / (float)blockDim.x;
		c.c.g = threadIdx.y / (float)blockDim.y;
		c.c.b = 0.0;
		*/
#endif
		//__syncthreads();
		// write output vertex
		color[y*width+x] += c;	
		__syncthreads();
		//__threadfence_block();

		if( tid == 0 ){
			currentBlock = atomicAdd(&curb, 1);
		}
		__threadfence();


	}while(currentBlock < ttlb);
}