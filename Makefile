.PHONY: data

BLENDER := /home/lpetrov/opt/blender-3.4.1-linux-x64/blender

PYVERSION := 3
VENV_NAME?=.venv
PYTHON=${VENV_NAME}/bin/python

# Requirements are in setup.py, so whenever setup.py is changed, re-run installation of dependencies.
venv: $(VENV_NAME)/bin/activate

$(VENV_NAME)/bin/activate: requirements.txt
		test -d venv || virtualenv -p python$(PYVERSION) $(VENV_NAME)
		${PYTHON} -m pip install -U pip
		${PYTHON} -m pip install -r requirements.txt
		touch $(VENV_NAME)/bin/activate

data: venv
	${BLENDER} blender/letters.blend --background --python blender/render_letters.py

train: venv
	@echo "Not implemented yet. Please go through the Letters.ipynb notebook to train the model"

test: venv
	${PYTHON} test.py
