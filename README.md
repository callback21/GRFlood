
![flood](GRFlood.png)

<h1>Introduction</h1>

Train several popular statistical models to predict regional flooding,and compare the performance of each model.
We will use the Root Mean Square Error(RMSE) metric to evaluate each model’s performance, and then normalize each of 
these statistics by dividing each of them by the difference of the max and min training values in order to make performance
comparisons across models


[![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Wcxd9iBJChn7gJgV--Vc03gSdljZbp2b?usp=sharing)
[![image](https://img.shields.io/pypi/v/geemap.svg)](https://pypi.org/project/GRFlood/)
[![image](https://img.shields.io/conda/vn/conda-forge/geemap.svg)](https://anaconda.org/conda-forge/geemap)
[![Downloads](https://static.pepy.tech/personalized-badge/grflood?period=month&units=international_system&left_color=yellow&right_color=orange&left_text=Downloads)](https://pepy.tech/project/grflood)
[![image](https://github.com/giswqs/geemap/workflows/build/badge.svg)](https://github.com/callback21/GRFlood)

[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A Python package for flood forecasting**

-   GitHub repo: <https://github.com/callback21/GRFlood>
-   PyPI: <https://pypi.org/project/GRFlood>
-   Conda-forge: <https://anaconda.org/conda-forge/GRFlood>
-   Free software: [MIT license](https://opensource.org/licenses/MIT)


<h1>Installation</h1>



<h2>Install from PyPI</h2>

<br>

`GRFlood` is available on PyPI. To install `GRFlood`, run this command in your terminal:

```sh
pip install GRFlood
```
<h2>Install from conda-forge</h2>


<br>

`GRFlood` is also available on conda-forge. If you have Anaconda or Miniconda installed on your computer, you can create a conda Python environment to install `GRFlood`:

```sh
conda install -c defaults -c conda-forge GRFlood
```
<h2>Install from GitHub</h2> 

<br>
To install the development version from GitHub using Git, run the following command in your terminal:

```sh
pip install git+https://github.com/callback21/GRFlood
```
<h2>Colaboratory</h2> 

<br>
Colaboratory lets you connect to a local runtime using Jupyter. This allows you to execute code on your local hardware and have access to your local file system.

**Step 1: Install Jupyter**<br>
Install [Jupyter](https://jupyter.org/install) on your local machine.

**Step 2: Install and enable the jupyter_http_over_ws jupyter extension (one-off)** <br>
The ```jupyter_http_over_ws``` extension is authored by the Colaboratory team and available on [GitHub](https://github.com/googlecolab/jupyter_http_over_ws).

```sh
pip install jupyter_http_over_ws
jupyter serverextension enable --py jupyter_http_over_ws
```
**Step 3: Start server and authenticate**<br>
New notebook servers are started normally, though you will need to set a flag to explicitly trust WebSocket connections from the Colaboratory frontend.

```sh
jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --NotebookApp.port_retries=0
```   
Once the server has started, it will print a message with the initial backend URL used for authentication. Make a copy of this URL as you'll need to provide this in the next step.

**Step 4: Connect to the local runtime**<br>
In Colaboratory, click the 'Connect' button and select 'Connect to local runtime…'. Enter the URL from the previous step in the dialogue that appears and click the 'Connect' button. After this, you should now be connected to your local runtime.

<h2>Upgrade GRFlood</h2>
<br>

If you have installed `GRFlood` before and want to upgrade to the latest version, you can run the following command in your terminal:

```sh
pip install -U GRFlood
```
If you use conda, you can update `GRFlood` to the latest version by running the following command in your terminal:

```sh
conda update GRFlood
```
