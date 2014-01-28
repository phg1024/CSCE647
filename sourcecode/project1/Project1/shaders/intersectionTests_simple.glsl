// light ray intersection tests
float lightRayIntersectsSphere(Ray r, Shape s) {
    vec3 pq = r.origin - s.p;
    float a = 1.0;
    float b = dot(pq, r.dir);
    float c = dot(pq, pq) - s.radius.x * s.radius.x;

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

float lightRayIntersectsPlane(Ray r, Shape s) {
    float t = 1e10;
    float THRES = 1e-3;
    vec3 pq = s.p - r.origin;
    float ldotn = dot(s.axis0, r.dir);
    if( abs(ldotn) < THRES ) return -1.0;
    else {
        t = dot(s.axis0, pq) / ldotn;

        if( t >= THRES ) {
            vec3 p = r.origin + t * r.dir;

            // compute u, v coordinates
            vec3 pp0 = p - s.p;
            float u = dot(pp0, s.axis1);
            float v = dot(pp0, s.axis2);
            if( abs(u) > s.radius.x || abs(v) > s.radius.y ) return -1.0;
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

float lightRayIntersectsShape(Ray r, Shape s) {
    if(s.type == SPHERE) return lightRayIntersectsSphere(r, s);
    else if( s.type == PLANE ) return lightRayIntersectsPlane(r, s);
    else if( s.type == ELLIPSOID ) return lightRayIntersectsEllipsoid(r, s);
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
    float dist = length(p - lt.pos);
    Ray r;
    r.origin = p;
    r.dir = normalize(lt.pos - p);
    float t = lightRayIntersectsShapes(r);

    float THRES = 1e-3;
    return t < THRES || t > dist;
}
