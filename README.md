# ISMRM 2015 Tractography Challenge Scoring system

This system contains the scripts and tools that can be used to 
recreate the results of the ISMRM 2015 Tractography Challenge and to
evaluate new datasets.

The release used to produce results for the [website](http://www.tractometer.org/ismrm_2015_challenge/) and
paper is archived on Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.580063.svg)](https://doi.org/10.5281/zenodo.580063).


Configuration and installation
------------------------------

Make sure the "pip" version is recent enough. You can update it with

```bash
pip install -U pip
```

Then, install the needed dependencies using the requirements file.
The team recommends using a [virtual environment](https://pypi.python.org/pypi/virtualenv)
(with the [virtual env wrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)), but
it is not mandatory. Once setup, run

```bash
pip install -r requirements.txt
pip install -r requirements_additional.txt
```

You can then clone or download the scoring system. Once cloned or
downloaded, go inside the ```ismrm_2015_tractography_challenge_scoring```
directory, and run

```bash
python setup.py build_all
```

Once those steps are all done, the system is configured.

Fetching the Ground Truth Dataset
---------------------------------

To be able to run the scoring system, a directory containing the ground
truth dataset is needed. It can be downloaded from
[the Tractometer website](http://www.tractometer.org/downloads/downloads/scoring_data_tractography_challenge.tar.gz).

