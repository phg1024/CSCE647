out vec4 fragColor;

void main(void)
{
    initializeCamera();
    //initializeLights();
    //initializeShapes();

    float edgeSamples = sqrt(float(AAsamples));
    float step = 1.0 / edgeSamples;


    vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
    for(int i=0;i<AAsamples;i++) {
        // uniform jitter
        float x = floor(float(i) * step);
        float y = mod(float(i), edgeSamples);

        float xoffset = rand(gl_FragCoord.xy + vec2(x*step, y*step)) - 0.5;
        float yoffset = rand(gl_FragCoord.xy + vec2(y*step, x*step)) - 0.5;
        vec4 pos = gl_FragCoord + vec4((x + xoffset) * step, (y + yoffset) * step, 0, 0);

        Ray r = constructRay(pos.xy);

        // test if the ray hits the sphere
        Hit hit = rayIntersectsShapes(r);

        color = color + vec4(hit.color, 1.0);
    }

    fragColor = color / AAsamples;
}
