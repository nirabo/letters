import bpy

import os
import math
import random
import time

from mathutils import Euler, Color
from pathlib import Path

#bpy.context.scene.objects["A"].rotation_euler = Euler((pi/2, 0, 0),"XYZ")

def rand_rot_obj(obj):
    rx = random.random() * 2 * math.pi
    ry = random.random() * 2 * math.pi
    rz = random.random() * 2 * math.pi
    obj.rotation_euler = Euler((rx,ry,rz), "XYZ")
    
def rand_col_obj(mat):
    color = Color()
    hue = random.random()
    color.hsv = (hue, 1, 1)
    rgba = [color.r, color.g, color.b, 1]
    mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = rgba
    
#rand_rot_obj(bpy.context.scene.objects["B"])
#rand_col_obj(bpy.data.materials['Letter Material'])
#bpy.context.scene.render.filepath = "/home/lpetrov/projects/blender/letters/b.png"
#bpy.ops.render.render(write_still=True)

obj_names = ["A", "B", "C"]
obj_count = len(obj_names)
obj_renders_per_split = [('train',500), ('val', 200), ('test',100)]

output_path = "./data/letters"

total_render_count = sum([obj_count * r[1] for r in obj_renders_per_split])

for name in obj_names:
    bpy.context.scene.objects[name].hide_render = True
    
start_idx = 0

start_time = time.time()
for split_name, renders_per_obj in obj_renders_per_split:
    print(f"Starting split: {split_name} | Total renders: {renders_per_obj * obj_count}")
    
    for obj_name in obj_names:
        print(f"Starting obj: {split_name}/{obj_name}")
        
        obj2render = bpy.context.scene.objects[obj_name]
        obj2render.hide_render = False
        
        for i in range(start_idx, start_idx + renders_per_obj):
            rand_rot_obj(obj2render)
            rand_col_obj(obj2render.material_slots[0].material)
            
            print(f"Rendering image {i + 1} of {total_render_count}")
            seconds_per_render = (time.time() - start_time / (i + 1))
            seconds_remaining = seconds_per_render * (total_render_count - i - 1)
        
            bpy.context.scene.render.filepath = os.path.join(output_path, split_name, obj_name, f"{str(i).zfill(6)}.png")
            bpy.ops.render.render(write_still=True)
        # Hide the object
        obj2render.hide_render = True
        start_idx += renders_per_obj
        
for name in obj_names:
    bpy.context.scene.objects[name].hide_render = True