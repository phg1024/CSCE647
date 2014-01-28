// ray intersection test with shading computation
Hit rayIntersectsSphere(Ray r, Shape s) {
    Hit h;

    vec3 pq = r.origin - s.p;
    float a = 1.0;
    float b = dot(pq, r.dir);
    float c = dot(pq, pq) - s.radius.x * s.radius.x;

    // solve the quadratic equation
    float delta = b*b - a*c;
    if( delta < 0.0 )
    {
        return background;
    }
    else
    {
        delta = sqrt(delta);
        float x0 = -b+delta; // a = 1, no need to do the division
        float x1 = -b-delta;
        float THRES = 1e-3;

        if( x0 < THRES ) {
            return background;
        }
        else
        {
            if( x1 < THRES ) h.t = x0;
            else h.t = x1;

            // hit point
            vec3 p = h.t * r.dir + r.origin;
            // normal at hit point
            vec3 n = normalize(p - s.p);

            // hack, move the point a little bit outer
            p = s.p + (s.radius.x + 1e-6) * n;

            vec2 t = spheremap(n);

            h.color = computeShading(p, n, t, r, s);
            return h;
        }
    }
}

Hit rayIntersectsPlane(Ray r, Shape s) {
    vec3 pq = s.p - r.origin;
    float ldotn = dot(s.axis0, r.dir);
    if( abs(ldotn) < 1e-3 ) return background;
    else {
        Hit h;
        h.t = dot(s.axis0, pq) / ldotn;

        if( h.t > 0.0 ) {
            vec3 p = r.origin + h.t * r.dir;

            // compute u, v coordinates
            vec3 pp0 = p - s.p;
            float u = dot(pp0, s.axis1);
            float v = dot(pp0, s.axis2);
            if( abs(u) > s.radius.x || abs(v) > s.radius.x ) return background;
            else {
                vec2 t = clamp((vec2(u / s.radius.x, v / s.radius.y) + vec2(1.0, 1.0))*0.5, 0.0, 1.0);
                h.color = computeShading(p, s.axis0, t, r, s);
                return h;
            }
        }
        else return background;
    }
}

Hit rayIntersectsEllipsoid(Ray r, Shape s) {
    vec3 pq = s.p - r.origin;
    float a = dot(r.dir, s.m*r.dir);
    float b = -dot(pq, s.m*r.dir);
    float c = dot(pq, s.m*pq) - 1;

    float delta = b*b - a*c;
    if (delta < 0.0)
    {
        return background;
    }
    else
    {
        delta = sqrt(delta);
        float inv = 1.0 / a;

        float x0 = (-b-delta)*inv;
        float x1 = (-b+delta)*inv;

        const float THRES = 1e-3;
        if( x1 < THRES ) {
            return background;
        }
        else
        {
            Hit h;
            if( x0 < THRES ) h.t = x1;
            else h.t = x0;

            // hit point
            vec3 p = h.t * r.dir + r.origin;
            // normal at hit point
            vec3 n = normalize(2.0 * s.m * (p - s.p));

            vec2 t = spheremap(n);
            h.color = computeShading(p, n, t, r, s);
            return h;
        }
    }
}

Hit rayIntersectsShape(Ray r, Shape s) {
    if(s.type == SPHERE) return rayIntersectsSphere(r, s);
    else if( s.type == PLANE ) return rayIntersectsPlane(r, s);
    else if( s.type == ELLIPSOID ) return rayIntersectsEllipsoid(r, s);
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
