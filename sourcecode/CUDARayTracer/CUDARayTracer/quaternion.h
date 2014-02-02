#pragma once

#include "helper_math.h"
#include "element.h"

class quaternion {
public:
	union {
		struct {
			float s; //!< the real component
			vec3 v; //!< the imaginary components
		};
		struct { float elem[4]; }; //! the raw elements of the quaternion
	};

	//! constructors
	__host__ __device__ quaternion() {}
	__host__ __device__ quaternion(float real, float x, float y, float z): s(real), v(x,y,z) {}
	__host__ __device__ quaternion(float real, const vec3 &i): s(real), v(i) {}

	//! from 3 euler angles
	__host__ __device__ quaternion(float theta_z, float theta_y, float theta_x)
	{
		float cos_z_2 = cosf(0.5*theta_z);
		float cos_y_2 = cosf(0.5*theta_y);
		float cos_x_2 = cosf(0.5*theta_x);

		float sin_z_2 = sinf(0.5*theta_z);
		float sin_y_2 = sinf(0.5*theta_y);
		float sin_x_2 = sinf(0.5*theta_x);

		// and now compute quaternion
		s   = cos_z_2*cos_y_2*cos_x_2 + sin_z_2*sin_y_2*sin_x_2;
		v.x = cos_z_2*cos_y_2*sin_x_2 - sin_z_2*sin_y_2*cos_x_2;
		v.y = cos_z_2*sin_y_2*cos_x_2 + sin_z_2*cos_y_2*sin_x_2;
		v.z = sin_z_2*cos_y_2*cos_x_2 - cos_z_2*sin_y_2*sin_x_2;

	}
	
	//! from 3 euler angles 
	__host__ __device__ quaternion(const vec3 &angles)
	{	
		float cos_z_2 = cosf(0.5*angles.z);
		float cos_y_2 = cosf(0.5*angles.y);
		float cos_x_2 = cosf(0.5*angles.x);

		float sin_z_2 = sinf(0.5*angles.z);
		float sin_y_2 = sinf(0.5*angles.y);
		float sin_x_2 = sinf(0.5*angles.x);

		// and now compute quaternion
		s   = cos_z_2*cos_y_2*cos_x_2 + sin_z_2*sin_y_2*sin_x_2;
		v.x = cos_z_2*cos_y_2*sin_x_2 - sin_z_2*sin_y_2*cos_x_2;
		v.y = cos_z_2*sin_y_2*cos_x_2 + sin_z_2*cos_y_2*sin_x_2;
		v.z = sin_z_2*cos_y_2*cos_x_2 - cos_z_2*sin_y_2*sin_x_2;		
	} 
		
	//! basic operations
	__host__ __device__ quaternion &operator =(const quaternion &q)		
	{ s= q.s; v= q.v; return *this; }

	__host__ __device__ const quaternion operator +(const quaternion &q) const	
	{ return quaternion(s+q.s, v+q.v); }

	__host__ __device__ const quaternion operator -(const quaternion &q) const	
	{ return quaternion(s-q.s, v-q.v); }

	__host__ __device__ const quaternion operator *(const quaternion &q) const	
	{	return quaternion(s*q.s - v.dot(q.v),
				  v.y*q.v.z - v.z*q.v.y + s*q.v.x + v.x*q.s,
				  v.z*q.v.x - v.x*q.v.z + s*q.v.y + v.y*q.s,
				  v.x*q.v.y - v.y*q.v.x + s*q.v.z + v.z*q.s);
	}

	__host__ __device__ const quaternion operator /(const quaternion &q) const	
	{
		quaternion p(q); 
		p.invert(); 
		return *this * p;
	}

	__host__ __device__ const quaternion operator *(float scale) const
	{ return quaternion(s*scale,v*scale); }

	__host__ __device__ const quaternion operator /(float scale) const
	{ return quaternion(s/scale,v/scale); }

	__host__ __device__ const quaternion operator -() const
	{ return quaternion(-s, -v); }
	
	__host__ __device__ const quaternion &operator +=(const quaternion &q)		
	{ v+=q.v; s+=q.s; return *this; }

	__host__ __device__ const quaternion &operator -=(const quaternion &q)		
	{ v-=q.v; s-=q.s; return *this; }

	__host__ __device__ const quaternion &operator *=(const quaternion &q)		
	{			
		float x= v.x, y= v.y, z= v.z, sn= s*q.s - v.dot(q.v);
		v.x= y*q.v.z - z*q.v.y + s*q.v.x + x*q.s;
		v.y= z*q.v.x - x*q.v.z + s*q.v.y + y*q.s;
		v.z= x*q.v.y - y*q.v.x + s*q.v.z + z*q.s;
		s= sn;
		return *this;
	}
	
	__host__ __device__ const quaternion &operator *= (float scale)	{ v*=scale; s*=scale; return *this; }
	__host__ __device__ const quaternion &operator /= (float scale)	{ v/=scale; s/=scale; return *this; }	

	//! gets the length of this quaternion
	__host__ __device__ float length() const { return (float)sqrt(s*s + v.dot(v)); }

	//! gets the squared length of this quaternion
	__host__ __device__ float length_squared() const { return (float)(s*s + v.dot(v)); }

	//! normalizes this quaternion
	__host__ __device__ void normalize() { *this/=length(); }

	//! returns the normalized version of this quaternion
	__host__ __device__ quaternion normalized() const { return  *this/length(); }

	//! computes the conjugate of this quaternion
	__host__ __device__ void conjugate() { v=-v; }

	//! inverts this quaternion
	__host__ __device__ void invert() { conjugate(); *this/=length_squared(); }
	
	//! returns the logarithm of a quaternion = v*a where q = [cos(a),v*sin(a)]
	__host__ __device__ quaternion log() const
	{
		float a = (float)acos(s);
		float sina = (float)sin(a);
		quaternion ret;

		ret.s = 0;
		if (sina > 0)
		{
			ret.v.x = a*v.x/sina;
			ret.v.y = a*v.y/sina;
			ret.v.z = a*v.z/sina;
		} else {
			ret.v.x= ret.v.y= ret.v.z= 0;
		}
		return ret;
	}

	//! returns e^quaternion = exp(v*a) = [cos(a),vsin(a)]
	__host__ __device__ quaternion exp() const
	{
		float a = (float)v.norm();
		float sina = (float)sin(a);
		float cosa = (float)cos(a);
		quaternion ret;

		ret.s = cosa;
		if (a > 0)
		{
			ret.v.x = sina * v.x / a;
			ret.v.y = sina * v.y / a;
			ret.v.z = sina * v.z / a;
		} else {
			ret.v.x = ret.v.y = ret.v.z = 0;
		}
		return ret;
	}

	//! casting to a 4x4 isomorphic matrix for right multiplication with vector
	__host__ __device__ operator mat4() const
	{			
		return mat4(
			s,  -v.x, -v.y,-v.z,
			v.x,  s,  -v.z, v.y,
			v.y, v.z,    s,-v.x,
			v.z,-v.y,  v.x,   s
			);
	}
	
	//! casting to 3x3 rotation matrix
	__host__ __device__ operator mat3() const
	{
		//assert(length() > 0.9999 && length() < 1.0001);
		return mat3(1-2*(v.y*v.y+v.z*v.z), 2*(v.x*v.y-s*v.z),   2*(v.x*v.z+s*v.y),   
				2*(v.x*v.y+s*v.z),   1-2*(v.x*v.x+v.z*v.z), 2*(v.y*v.z-s*v.x),   
				2*(v.x*v.z-s*v.y),   2*(v.y*v.z+s*v.x),   1-2*(v.x*v.x+v.y*v.y));
	}

	//! computes the dot product of 2 quaternions
	__host__ __device__ static inline float dot(const quaternion &q1, const quaternion &q2) 
	{ return q1.v.dot(q2.v) + q1.s*q2.s; }

	//! linear quaternion interpolation
	__host__ __device__ static quaternion lerp(const quaternion &q1, const quaternion &q2, float t) 
	{ return (q1*(1-t) + q2*t).normalized(); }

	//! spherical linear interpolation
	__host__ __device__ static quaternion slerp(const quaternion &q1, const quaternion &q2, float t) 
	{
		quaternion q3;
		float dot = quaternion::dot(q1, q2);

		/*	dot = cos(theta)
			if (dot < 0), q1 and q2 are more than 90 degrees apart,
			so we can invert one to reduce spinning	*/
		if (dot < 0)
		{
			dot = -dot;
			q3 = -q2;
		} else q3 = q2;
		
		if (dot < 0.95f)
		{
			float angle = acosf(dot);
			return (q1*sinf(angle*(1-t)) + q3*sinf(angle*t))/sinf(angle);
		} else // if the angle is small, use linear interpolation								
			return lerp(q1,q3,t);		
	}

	//! This version of slerp, used by squad, does not check for theta > 90.
	__host__ __device__ static quaternion slerpNoInvert(const quaternion &q1, const quaternion &q2, float t) 
	{
		float dot = quaternion::dot(q1, q2);

		if (dot > -0.95f && dot < 0.95f)
		{
			float angle = acosf(dot);			
			return (q1*sinf(angle*(1-t)) + q2*sinf(angle*t))/sinf(angle);
		} else  // if the angle is small, use linear interpolation								
			return lerp(q1,q2,t);			
	}

	//! spherical cubic interpolation
	__host__ __device__ static quaternion squad(const quaternion &q1,const quaternion &q2,const quaternion &a,const quaternion &b,float t)
	{
		quaternion c= slerpNoInvert(q1,q2,t),
			       d= slerpNoInvert(a,b,t);		
		return slerpNoInvert(c,d,2*t*(1-t));
	}

	//! Shoemake-Bezier interpolation using De Castlejau algorithm
	__host__ __device__ static quaternion bezier(const quaternion &q1,const quaternion &q2,const quaternion &a,const quaternion &b,float t)
	{
		// level 1
		quaternion q11= slerpNoInvert(q1,a,t),
				q12= slerpNoInvert(a,b,t),
				q13= slerpNoInvert(b,q2,t);		
		// level 2 and 3
		return slerpNoInvert(slerpNoInvert(q11,q12,t), slerpNoInvert(q12,q13,t), t);
	}

	//! Given 3 quaternions, qn-1,qn and qn+1, calculate a control point to be used in spline interpolation
	__host__ __device__ static quaternion spline(const quaternion &qnm1,const quaternion &qn,const quaternion &qnp1)
	{
		quaternion qni(qn.s, -qn.v);	
		return qn * (( (qni*qnm1).log()+(qni*qnp1).log() )/-4).exp();
	}

	//! converts from a normalized axis - angle pair rotation to a quaternion
	__host__ __device__ static inline quaternion from_axis_angle(const vec3 &axis, float angle)
	{ return quaternion(cosf(angle/2), axis*sinf(angle/2)); }

	//! returns the axis and angle of this unit quaternion
	__host__ __device__ void to_axis_angle(vec3 &axis, float &angle) const
	{
		angle = acosf(s);

		// pre-compute to save time
		float sinf_theta_inv = 1.0/sinf(angle);

		// now the vector
		axis.x = v.x*sinf_theta_inv;
		axis.y = v.y*sinf_theta_inv;
		axis.z = v.z*sinf_theta_inv;

		// multiply by 2
		angle*=2;
	}

	//! rotates v by this quaternion (quaternion must be unit)
	__host__ __device__ vec3 rotate(const vec3 &v)
	{   
		quaternion V(0, v);
		quaternion conjugate(*this);
		conjugate.conjugate();
		quaternion tmp = (*this) * V * conjugate;
		return tmp.v;
	}

	//! returns the euler angles from a rotation quaternion
	__host__ __device__ vec3 euler_angles(bool homogenous=true) const
	{
		float sqw = s*s;    
		float sqx = v.x*v.x;    
		float sqy = v.y*v.y;    
		float sqz = v.z*v.z;    

		vec3 euler;
		if (homogenous) {
			euler.x = atan2f(2.f * (v.x*v.y + v.z*s), sqx - sqy - sqz + sqw);    		
			euler.y = asinf(-2.f * (v.x*v.z - v.y*s));
			euler.z = atan2f(2.f * (v.y*v.z + v.x*s), -sqx - sqy + sqz + sqw);    
		} else {
			euler.x = atan2f(2.f * (v.z*v.y + v.x*s), 1 - 2*(sqx + sqy));
			euler.y = asinf(-2.f * (v.x*v.z - v.y*s));
			euler.z = atan2f(2.f * (v.x*v.y + v.z*s), 1 - 2*(sqy + sqz));
		}
		return euler;
	}
};