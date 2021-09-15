# Optimal histograms with Outliers: a Python Implementation and Experimentation.
A python implementation of the algorithms described in the following paper:

Rachel Behar, Sara Cohen. Optimal Histograms with Outliers. EDBT 2020: 181-192. https://openproceedings.org/2020/conf/edbt/paper_93.pdf

This project was implemented under the frame of the Database Systems course of the DSIT MSc programme @ NKUA.



## Download and set up the project

1. Clone the repository.
  `$:git clone https://github.com/AlexandrosKal/opt_hist.git`

2. Go to opt_hist directory. 
  `$:cd opt_hist`
3. Set up a virtual enviroment and install the requirements.
  - `$:python3 -m venv ./venv`
  - `$:source ./venv/bin/activate`
  - `$:pip3 install wheel && pip3 install -r requirements.txt` 
4. Run jupyter lab:
  `$:jupyter lab`. Make sure you run the command inside the virtualenv you created. 
5. In jupyter lab find the notebooks in the experiments folder and run the exeriments (there are instructions there).
Make sure you select the virtualenv you created as the kernel of the notebook you wish to run (probably named `Python 3 (ipykernel)`
if you started jupyter lab inside the virtualenv).

NOTE: The project was implemented and tested on Python 3.8.5. Some requirements may not work for previous versions.
