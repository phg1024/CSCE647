#pragma once

#include "element.h"
#include "quaternion.h"

class TrackBall
{
public:
	TrackBall(void){}
	TrackBall(const TrackBall& ball):
		R(ball.R), incR(ball.incR),	T(ball.T),
		m_x(ball.m_x), m_y(ball.m_y),
		m_width(ball.m_width), m_height(ball.m_height),
		m_r(ball.m_r),
		m_scale(ball.m_scale)
	{
		for(int i=0;i<16;i++)
			M[i] = ball.M[i];
	}
	void init();

	void setSceneScale(float s) {m_sceneScale = s;}

	void reshape(int w, int h);
	void mouse_rotate(int u, int v);
	void mouse_translate(int u, int v);
	void motion_rotate(int u, int v);
	void motion_translate(int u, int v);
	void joystick(unsigned int mask, int x, int y, int z);
	void wheel(int delta);

	/*
	void applyRotation();
	void applyTransform();
	void applyInverseTransform();
	void applyInverseRotation();
	*/

	float getX() const {return m_x;};
	float getY() const {return m_y;};
	float getWidth() const {return m_width;};
	float getHeight() const {return m_height;};
	float getR() const {return m_r;};
	float getScale() const {return m_scale;}
	vec3 getNormDir();
	quaternion getRotation() const ;
	vec3 getTranslation() const ;
	quaternion getIncRotation() const ;

	void reset();
	void loadStatus(const char *filename);
	void saveStatus(const char *filename);

	void setRotation(const quaternion &Q);
	void rotate(const quaternion &Q);

	float* getMatrix();
	float* getInverseMatrix();

	float M[16];


protected:
	quaternion R, incR;
	vec3 T;

	void m_Quaternion2Matrix();

private:
	int m_x, m_y;
	int m_width, m_height;
	float m_r; 		    //!< min(m_width, m_height)
	float m_scale;
	float m_sceneScale;
};

inline void TrackBall::init()
{
	R = quaternion(1.0f, 0.0f, 0.0f, 0.0f); 	// no rotation
	T = vec3(0.0, 0.0, 0.0);           // no translation

	m_x = m_y = 0;
	m_r = 0;
	m_width = m_height = 0;
	m_scale = 1.0;
	m_sceneScale = 1.0;

	m_Quaternion2Matrix();
}

inline void TrackBall::reset()
{
	R = quaternion(1.0, 0.0, 0.0, 0.0);
	m_scale = 1.0;
	m_Quaternion2Matrix();
}

inline vec3 TrackBall::getNormDir()
{
	return R.v;
}

inline quaternion TrackBall::getRotation() const
{
	return R;
}

inline quaternion TrackBall::getIncRotation() const
{
	return incR;
}

inline vec3 TrackBall::getTranslation() const
{
	return T;
}

inline void TrackBall::reshape(int w, int h)
{
	m_width = w;
	m_height = h;

	m_r = min(m_width, m_height) / 2.f;
}

inline void TrackBall::mouse_rotate(int u, int v)
{
	v = m_height - v;

	m_x = u;
	m_y = v;
}

inline void TrackBall::mouse_translate(int u, int v)
{
	v = m_height - v;

	m_x = u;
	m_y = v;
}

inline void TrackBall::motion_rotate(int u, int v)
{
	vec3 N;
	vec3 v0(0, 0, 0), v1(0, 0, 0);
	float theta;

	v = m_height - v;

	if (u<0 || u>m_width || v<0 || v>m_height) {
		//cout << "out 1" << endl;
		return;
	}

	v0.x = (m_x - m_width/2.f);
	v0.y = (m_y - m_height/2.f);
	v0.z = (0.f);

	if (v0.length() < 2){
		//cout << "out 2" << endl;
		return;
	}
	v0.z = (m_r*m_r/2.f / v0.length());

	v1.x = (u - m_width/2.f);
	v1.y = (v - m_height/2.f);
	v1.z = (0.f);

	if (v1.length() < 2){
		//cout << "out 3" << endl;
		return;
	}
	v1.z = (m_r*m_r/2.f / v1.length());

	v0.normalize();
	v1.normalize();
	N = v0.cross(v1);
	N.normalize();

	float dotProduct = v0.dot(v1);

	if(dotProduct > 1.0)
		dotProduct = 1.0;

	if(dotProduct < -1.0)
		dotProduct = -1.0;

	theta = acos(dotProduct);

	incR = quaternion(cos(theta/2.f), N*sin(theta/2.f));

	R = incR*R;

	m_Quaternion2Matrix();

	m_x = u, m_y = v;
}

inline void TrackBall::motion_translate(int u, int v)
{
	v = m_height - v;

	// a straight forward scheme
	vec3 screenMovementVector = vec3((float)(u - m_x)/(float)m_width, (float)(v - m_y)/(float)m_height, 0.0);

	m_Quaternion2Matrix();

	float t;
	mat4 modelMatrix = mat4(M);
	vec3 movementVector = modelMatrix * screenMovementVector;

	T += movementVector / m_scale * m_sceneScale;

	m_x = u, m_y = v;
}

inline void TrackBall::rotate(const quaternion &Q)
{
	R = Q*R;

	m_Quaternion2Matrix();
}

inline void TrackBall::setRotation(const quaternion &Q)
{
	R = Q;

	m_Quaternion2Matrix();
}

inline void TrackBall::wheel(int delta)
{
	if (delta > 0)
		m_scale *= 1.025;
	else
		m_scale *= 0.975;
}

/*
void TrackBall::applyRotation()
{
	m_Quaternion2Matrix();
#if TRACKBALL_USE_DOUBLE
	glMultMatrixd(M);
#else
	glMultMatrixf(M);
#endif
}

void TrackBall::applyTransform()
{
	m_Quaternion2Matrix();
#if TRACKBALL_USE_DOUBLE
	glMultMatrixd(M);
#else
	glMultMatrixf(M);
#endif
	//    glScalef(m_scale, m_scale, m_scale);
	glTranslatef(T.x(), T.y(), T.z());
}

void TrackBall::applyInverseTransform()
{
	quaternion S = R;
	R.setScalar(-R.scalar());
	m_Quaternion2Matrix();
	R = S;

#if TRACKBALL_USE_DOUBLE
	glMultMatrixd(M);
#else
	glMultMatrixf(M);
#endif
	glScalef(1.f/m_scale, 1.f/m_scale, 1.f/m_scale);
}
*/

inline float* TrackBall::getMatrix()
{
	m_Quaternion2Matrix();
	return M;
}

inline float* TrackBall::getInverseMatrix()
{
	quaternion S = R;
	R.s = -R.s;
	m_Quaternion2Matrix();
	R = S;
	return M;
}

/*
void TrackBall::applyInverseRotation()
{
	quaternion S = R;
	R.setScalar(-R.scalar());
	m_Quaternion2Matrix();
	R = S;

#if TRACKBALL_USE_DOUBLE
	glMultMatrixd(M);
#else
	glMultMatrixf(M);
#endif
}
*/

inline void TrackBall::m_Quaternion2Matrix()
{
	float q0 = R.s,	q1 = R.v.x,	q2 = R.v.y,	q3 = R.v.z;

	M[0] = q0*q0 + q1*q1 - q2*q2 - q3*q3;
	M[1] = 2*(q1*q2 + q0*q3);
	M[2] = 2*(q1*q3 - q0*q2);
	M[3] = 0;

	M[4] = 2*(q1*q2 - q0*q3);
	M[5] = q0*q0 + q2*q2 - q1*q1 - q3*q3;
	M[6] = 2*(q2*q3 + q0*q1);
	M[7] = 0;

	M[8] = 2*(q1*q3 + q0*q2);
	M[9] = 2*(q2*q3 - q0*q1);
	M[10]= q0*q0 + q3*q3 - q1*q1 - q2*q2;
	M[11]= 0;

	M[12]= 0;
	M[13]= 0;
	M[14]= 0;
	M[15]= 1;
}

inline void TrackBall::loadStatus(const char *filename)
{
	ifstream ifs;
	float x, y, z, w, s;

	ifs.open(filename);

	if (!ifs) {
		cerr << "[TrackBall::loadStatus] cannot open trackball file. " << endl;
		return;
	}

	ifs >> x >> y >> z >> w >> s;

	R = quaternion(w, x, y, z);
	m_scale = s;

	ifs.close();

	m_Quaternion2Matrix();
}

inline void TrackBall::saveStatus(const char *filename)
{
	ofstream ofs;

	ofs.open(filename);

	if (!ofs) {
		cerr << "[CGLTraclball::saveStatus] cannot save trackball file. " << endl;
		return;
	}

	ofs << R.v.x << '\t' << R.v.y << '\t' << R.v.z << '\t' << R.s << '\t' << m_scale;
	ofs.close();
}