#include "scene.h"

string Scene::createInitializationSourceCode() const
{
//    // initialize camera
//    string camStr = "void initializeCamera(){";

//    camStr += "\n}\n";

    // initialize shapes
    string shapeStr = "void initializeShapes(){";
    for(int i=0;i<shapes.size();i++) {
        shapeStr += "shapes[" + PhGUtils::toString(i) + "] = " + "shapes" + PhGUtils::toString(i) + ";\n";
    }
    shapeStr += "\n}\n";

    // initialize lights
    string lightStr = "void initializeLights(){";
    for(int i=0;i<lights.size();i++) {
        lightStr += "lights[" + PhGUtils::toString(i) + "] = " + "lights" + PhGUtils::toString(i) + ";\n";
    }
    lightStr += "\n}\n";

    return shapeStr + lightStr;
}

string Scene::createShapesSourceCode() const
{
    string shapeStr = "";
    for(int i=0;i<shapes.size();i++){
        shapeStr += shapes[i].toString("shapes", i);
    }
    return shapeStr;
}

string Scene::createLightSourceCode() const
{
    string lightStr = "";
    for(int i=0;i<lights.size();i++) {
        lightStr += lights[i].toString("lights", i);
    }
    return lightStr;
}
