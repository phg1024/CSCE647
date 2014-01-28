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

float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}