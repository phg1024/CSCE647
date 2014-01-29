uniform vec2 windowSize;
uniform int lightCount;
uniform int shapeCount;
uniform int shadingMode;    // 1 = lambert, 2 = phong, 3 = gooch, 4 = cook-torrance
uniform int AAsamples;

// camera info
uniform vec3 camPos, camUp, camDir;
uniform float camF;

uniform Hit background;
uniform sampler2D textures[8];

//uniform Light lights[4];
//uniform Shape shapes[8];
