.PHONY: data

BLENDER := /home/lpetrov/opt/blender-2.93.4-linux-x64/blender

data:
	${BLENDER} blender/letters.blend --background --python blender/render_letters.py