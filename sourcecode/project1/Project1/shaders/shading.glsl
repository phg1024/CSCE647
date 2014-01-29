vec3 phongShading(vec3 v, vec3 N, vec2 t, Ray r, int sid) {
    vec3 c = vec3(0, 0, 0);

    // iterate through all lights
    for(int i=0;i<lightCount;i++) {

        // determine if this light is visible
        bool isVisible = checkLightVisibility(v, N, lights[i]);

        //calculate Ambient Term:
        vec3 Iamb = shapes[sid].ambient * lights[i].ambient;

        if( isVisible ) {
            vec3 L;
            if( lights[i].type != DIRECTIONAL_LIGHT)
                L = normalize(lights[i].pos - v);
            else
                L = -lights[i].dir;

            vec3 E = normalize(r.origin-v);
            vec3 R = normalize(-reflect(L,N));

            float NdotL, RdotE;
            if( shapes[sid].hasNormalMap ) {
                // normal defined in tangent space
                vec3 n_normalmap = normalize(texture(textures[shapes[sid].nTex], t).rgb * 2.0 - 1.0);

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
            vec3 Idiff = shapes[sid].diffuse * lights[i].diffuse * max(NdotL, 0.0);
            Idiff = clamp(Idiff, 0.0, 1.0);

            // calculate Specular Term:
            vec3 Ispec = shapes[sid].specular * lights[i].specular
                    * pow(max(RdotE,0.0),0.3*shapes[sid].shininess);
            Ispec = clamp(Ispec, 0.0, 1.0);

            if( shapes[sid].hasTexture ) {
                vec3 Itexture = texture (textures[shapes[sid].tex], t).rgb;
                c = c + Itexture * (Idiff + Ispec + Iamb) * lights[i].intensity;
            }
            else
                c = c + (Idiff + Ispec + Iamb) * lights[i].intensity;
        }
        else {
            if( shapes[sid].hasTexture ) {
                vec3 Itexture = texture (textures[shapes[sid].tex], t).rgb;
                c = c + Itexture * Iamb * lights[i].intensity;
            }
            else
                c = c + Iamb * lights[i].intensity;
        }
    }

    return c;
}

vec3 lambertShading(vec3 v, vec3 N, vec2 t, Ray r, int sid) {
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

            float NdotL;

            // change the normal with normal map
            if( shapes[sid].hasNormalMap ) {
                // normal defined in tangent space
                vec3 n_normalmap = normalize(texture(textures[shapes[sid].nTex], t).rgb * 2.0 - 1.0);

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
            if( shapes[sid].hasTexture ) {
                Itexture = texture (textures[shapes[sid].tex], t).rgb;
            }
            else Itexture = vec3(1, 1, 1);

            vec3 Idiff = clamp(shapes[sid].diffuse * lights[i].diffuse * max(NdotL, 0.0), 0.0, 1.0);

            c = c + Itexture * Idiff * lights[i].intensity;
        }
    }

    return c;
}

vec3 goochShading(vec3 v, vec3 N, vec2 t, Ray r, int sid) {

    vec3 c = vec3(0, 0, 0);

    for(int i=0;i<lightCount;i++) {
        vec3 L;
        if( lights[i].type != DIRECTIONAL_LIGHT)
            L = normalize(lights[i].pos - v);
        else
            L = -lights[i].dir;

        vec3 E = normalize(r.origin - v);
        vec3 R = normalize(-reflect(L,N));
        float NdotL = dot(N, L);

        vec3 diffuse;
        if( shapes[sid].hasTexture ) {
            diffuse = texture (textures[shapes[sid].tex], t).rgb;
        }
        else diffuse = shapes[sid].diffuse;

        vec3 Idiff = diffuse * NdotL;
        vec3 kcdiff = min(shapes[sid].kcool + shapes[sid].alpha * Idiff, 1.0);
        vec3 kwdiff = min(shapes[sid].kwarm + shapes[sid].beta * Idiff, 1.0);
        vec3 kfinal = mix(kcdiff, kwdiff, (NdotL+1.0)*0.5);
        // calculate Specular Term:
        vec3 Ispec = shapes[sid].specular
                * pow(max(dot(R,E),0.0),0.3*shapes[sid].shininess);
        Ispec = step(vec3(0.5, 0.5, 0.5), Ispec);
        // edge effect
        float EdotN = dot(E, N);
        if( EdotN >= 0.2 ) c = c + min(kfinal + Ispec, 1.0) * lights[i].intensity;
    }

    return c;
}

vec3 computeShading(vec3 p, vec3 n, vec2 t, Ray r, int sid) {
    if( shadingMode == 1 )
        return lambertShading(p, n, t, r, sid);
    else if( shadingMode == 2 )
        return phongShading(p, n, t, r, sid);
    else if( shadingMode == 3 )
        if( shapes[sid].type == PLANE ) return lambertShading(p, n, t, r, sid);
        else return goochShading(p, n, t, r, sid);
}