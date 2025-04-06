<h1 align="center">RAPID - Reconstruct Any Pathology In 3 Dimensions</h2>
<p align="center">
</p>

<p align="center">
  <img width="900" height="304" src="./img/flowchart.png">
</p>
    
## What is RAPID?
RAPID is a deep learning-based algorithm to convert a stack of unregistered whole slide images (WSIs) into a full resolution three-dimensional reconstruction. This 3D reconstruction process essentially aims to recover the original shape of the specimen, which is a crucial step for several downstream applications. For example, any qualitative or quantitative comparison with 3D imaging will greatly benefit from a 3D-3D rather than a 2D-3D comparison. Additionally, these 3D reconstructions pave the way for fully automated quantitative 3D specimen analysis, enabling insights in tumor growth patterns, vasculature patterns etc.

## Does RAPID also work on my data?
We have primarily tested RAPID on sparsely sampled prostatectomy specimens where slides were acquired every ~4 mm. The main strength of RAPID is that it was designed to be robust against arbitrary slice gaps, providing a valuable addition to other methods which can only handle serial sections with micron-scale slice gaps. Therefore, if your data has sub 4 mm slice gaps and is ordered along the desired reconstruction dimension, we expect that RAPID will be able to reconstruct it in 3D. Moreover, RAPID expects that each input slide has multiple resolution layers, also known as a pyramidal file. RAPID also heavily relies on a tissue foreground mask. If you have one available, be sure to pass it as input, otherwise RAPID will automatically generate masks for you using simple thresholding heuristics.

## How do I run RAPID?
#### Docker container 
It is highly recommended to run RAPID as a Docker container, since RAPID uses some libraries that need to be built from source. The Docker container comes prepackaged with these libraries and any additional model weights, and should run out-of-the-box. We will soon provide a link to directly pull the pre-built container, but for now you can build the container yourself locally with the provided Dockerfile in /build.

	docker build . --tag dnschouten/rapid:v0.1

#### Data preparation
RAPID wil do the heavy lifting in terms of data preprocessing for you, just make sure that your WSIs are in pyramidal format. To figure out which slides to use for the reconstruction, RAPID requires a .csv/.xlsx with two columns. First, the "imagepath" column entails the absolute path to your WSI. Second, the "case" column should be used to indicate which WSIs belong to the same case, as RAPID will use all of the WSIs for a given case. If you have a tissue masks, you can provide the absolute path to it in the optional column "maskpath". 

#### Usage instructions
After preparing your data and the .csv/.xlsx with the reconstruction instructions, you can run the RAPID container with:

    docker run -it --gpus all -v /home/user:/home/user --network host dnschouten/rapid:v0.1 --joboverview /path/to/joboverview.xlsx --savedir /path/to/results

If you do not specify any input arguments (i.e., omit --joboverview and --savedir), the container should run in debug mode. You can then attach your IDE of choice to the running container to experiment with any tweaks or different versions.  

#### Sample data 
If you don't have any data available, but are still curious to try RAPID, we will soon provide some sample data on Zenodo.

#### Acknowledgements
We gratefully acknowledge the work of [RoMa](https://github.com/Parskatt/RoMa), which we use heavily in our code and inspired us to solve the reconstruction problem in global feature space.

## Licensing
The source code of RAPID is licensed under the [GNU Lesser General Public License (LGPL)](https://www.gnu.org/licenses/lgpl-3.0.nl.html). 