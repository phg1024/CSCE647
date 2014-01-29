// ray intersection test with shading computation
Hit rayIntersectsSphere(Ray r, int i) {
    float ti = lightRayIntersectsSphere(r, i);
    if( ti > 0.0 ) {
        Hit h;
        h.t = ti;
        // hit point
        vec3 p = h.t * r.dir + r.origin;
        // normal at hit point
        vec3 n = normalize(p - shapes[i].p);

        // hack, move the point a little bit outer
        p = shapes[i].p + (shapes[i].radius.x + 1e-6) * n;

        vec2 t = spheremap(n);

        h.color = computeShading(p, n, t, r, i);
        return h;
    }
    else return background;
}

Hit debug() {
    Hit hh;
    hh.color = vec3(1, 0, 0);
    hh.t = 1.0;
    return hh;
}

Hit rayIntersectsPlane(Ray r, int i) {
    float ti = lightRayIntersectsPlane(r, i);
    if( ti > 0.0 ) {
        Hit h;
        h.t = ti;

        if( h.t > 0.0 ) {
            vec3 p = r.origin + h.t * r.dir;

            // compute u, v coordinates
            vec3 pp0 = p - shapes[i].p;
            float u = dot(pp0, shapes[i].axis1);
            float v = dot(pp0, shapes[i].axis2);
            if( abs(u) > shapes[i].radius.x || abs(v) > shapes[i].radius.x ) return background;
            else {
                vec2 t = clamp((vec2(u / shapes[i].radius.x, v / shapes[i].radius.y) + vec2(1.0, 1.0))*0.5, 0.0, 1.0);
                h.color = computeShading(p, shapes[i].axis0, t, r, i);
                return h;
            }
        }
        else return background;
    }
    else return background;
}

Hit rayIntersectsEllipsoid(Ray r, int i) {
    float ti = lightRayIntersectsEllipsoid(r, i);
    if( ti > 0.0 )
    {
        //
    }
    else {
        return background;
    }

    /*
    vec3 pq = shapes[i].p - r.origin;
    float a = dot(r.dir, shapes[i].m*r.dir);
    float b = -dot(pq, shapes[i].m*r.dir);
    float c = dot(pq, shapes[i].m*pq) - 1;

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
            vec3 n = normalize(2.0 * shapes[i].m * (p - shapes[i].p));

            vec2 t = spheremap(n);
            h.color = computeShading(p, n, t, r, i);
            return h;
        }
    }
    */
}

Hit rayIntersectsCylinder(Ray r, int i) {
    Hit h;
    return h;
}

Hit rayIntersectsShape(Ray r, int i) {
    if(shapes[i].type == SPHERE) return rayIntersectsSphere(r, i);
    else if( shapes[i].type == PLANE ) return rayIntersectsPlane(r, i);
    else if( shapes[i].type == ELLIPSOID ) return rayIntersectsEllipsoid(r, i);
    else if( shapes[i].type == CYLINDER ) return rayIntersectsCylinder(r, i);
    else if( shapes[i].type == CONE ) return rayIntersectsCylinder(r, i);
    else if( shapes[i].type == CYLINDER ) return rayIntersectsCylinder(r, i);
    else return background;
}

Hit rayIntersectsShapes(Ray r) {
    // go through a list of shapes and find closest hit
    Hit h;
    h.t = 1e10;
    h.color = background.color;

    for(int i=0;i<shapeCount;i++) {
        Hit hit = rayIntersectsShape(r, i);

        if( (hit.t > 0.0) && (hit.t < h.t) ) {
            h.t = hit.t;
            h.color = hit.color;
        }
    }

    return h;
}
