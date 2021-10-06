# Document Segmentation Assemble - DOSA

## Installation and requirements

Tested for Ubuntu 18.04/20.04.

Use of a GPU significantly speeds up generation of detection outputs, but it is possible to run the inference demo code on CPU.

### Python Virtual Environment

1. Set up python = 3.7.x environment:
`pyenv install 3.7.12`
`pyenv virtualenv 3.7.12 dosa-env`

3. Activate the environment
`pyenv shell dosa-env`

4. Update pip & setuptools
`python -m pip install --upgrade pip setuptools`

### Models required

5. Install requirements
`pip install -r requirements.txt`
	- (for GPU-enabled installation: `pip install -r requirements_gpu.txt`)

### Mask R-CNN & DocParser

6. Install Mask R-CNN
`pip install -e ./Mask_RCNN`

7. Install DocParser
- `pip install -e ./DocParser`
- Download model weights follow instruction in `DocParser/docparser/default_models/README.md`

### PaddleOCR

8. Insall PaddlePaddle
`pip install paddlepaddle==2.1.3`
	- (for GPU-enabled installation: `pip install paddlepaddle-gpu==2.1.3`)

Installing paddlepaddle will raise warning error about dependency of gast==0.2.2 in tensorflow==1.15.5 vs. gast==0.4.0 in paddlepaddle==2.1.3.
**Just ignore it!**

9. Insall PaddleOCR
`pip install -e ./PaddleOCR`

### fastAPI server

10. Install poetry following instruction
`https://github.com/python-poetry/poetry#osx--linux--bashonwindows-install-instructions`

11. Install server dependencies
`poetry install`

## Run Server and Demo

Try each model with script in `./demos`, or running API server in `./server` and `./demos/server_api.py`
