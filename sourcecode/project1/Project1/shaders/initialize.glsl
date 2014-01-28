void initializeCamera() {
    caminfo.pos = camPos;
    caminfo.up = camUp;
    caminfo.dir = camDir;

    caminfo.right = cross(caminfo.dir, caminfo.up);
    caminfo.f = camF;
    caminfo.w = 1.0;
    caminfo.h = windowSize.y / windowSize.x;
}