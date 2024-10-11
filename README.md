# INSTALLATION:

> git clone https://github.com/StanfordWELab/pyCascade.git

> cd pyCascade

## \[PROBABLY WHAT YOU WANT\] To install as an editable package (package links to local repo, so any updates are automatically included without updating the package)
pip
> pip install -e .

conda
> conda develop .

## To install as a non-editable package (like most pip installs)
> pip install ./dist/pyCascade-0.1.0-py3-none-any.whl

## To build custom version
### \[First time only\] Create a virtual environment:
conda:
> conda env create -n \<env\> -f ./environment.yml

pip (untested):
> python3 -m venv env
> source env/bin/activate
> pip install -r requirements.txt
### Activate environemnt
> conda activate \<env\>

> python setup.py bdist_wheel

> conda deactivate

### To install the new-build as a non-editable package
> pip install ./dist/\<new_wheel_file\>.whl


