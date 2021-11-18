from logica import Raytracer, V3
from objeto import *
from figures import *
import random

width = 1000
height = 500

wood = Material(diffuse = (0.6,0.2,0.2), spec = 64)
stone = Material(diffuse = (0.4,0.4,0.4), spec = 64)

gold = Material(diffuse = (1, 0.8, 0 ),spec = 32, matType = REFLECTIVE)
mirror = Material(spec = 128, matType = REFLECTIVE)

water = Material(spec = 64, ior = 1.33, matType = TRANSPARENT)
glass = Material(spec = 64, ior = 1.5, matType = TRANSPARENT)
diamond = Material(spec = 64, ior = 2.417, matType = TRANSPARENT)


materiales=[mirror,stone,gold,diamond,wood]

rtx = Raytracer(width,height)
rtx.envmap = EnvMap('env1.bmp')

rtx.ambLight = AmbientLight(strength = 0.1)
rtx.dirLight = DirectionalLight(direction = V3(1, -1, -2), intensity = 0.5)
rtx.pointLights.append( PointLight(position = V3(0, 2, 0), intensity = 0.5))
s=1
z=1

for m in range(0,len(materiales)):
    s=random.uniform(0.9,1.2)
    z=random.uniform(0.9,1.2)
    rtx.scene.append( Sphere(V3((m*3)-7,3,-8), random.uniform(0.3,0.7), materiales[m]) )
    rtx.scene.append( AABB(V3((m*3)-7,0,-8), V3(s,s,s), materiales[m]) )

    rtx.scene.append(AABB(V3((m*3)-7,-3,-8), V3(1.3,0.6,0.6),materiales[m]))
    rtx.scene.append(AABB(V3((m*3)-7,-3,-8), V3(0.6,1.3,0.6),materiales[m]))
    rtx.scene.append(AABB(V3((m*3)-7,-3,-8), V3(0.6,0.6,1.3),materiales[m]))
rtx.glRender()
rtx.glFinish('proyecto2.bmp')



