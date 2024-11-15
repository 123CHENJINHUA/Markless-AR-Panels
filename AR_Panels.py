import sys
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import pywavefront

# 初始化Pygame和OpenGL
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

# 读取OBJ文件
scene = pywavefront.Wavefront('SolarPanel.obj', collect_faces=True)

# 渲染函数
def draw_scene():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glBegin(GL_TRIANGLES)
    for mesh in scene.mesh_list:
        for face in mesh.faces:
            for vertex_i in face:
                glVertex3fv(scene.vertices[vertex_i])
    glEnd()
    pygame.display.flip()

# 主循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    draw_scene()
    pygame.time.wait(10)

pygame.quit()
