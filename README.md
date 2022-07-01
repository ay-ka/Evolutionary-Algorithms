<div id="top"></div>


[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/ay-ka/Algorithms">
    <img style="float:left" src="images/basic.jpg" alt="evolution" width="700" height="400">
  </a>
  <h5 align="center">Implementation of various Evolutionary & Swarm Intelligence methods including ACO, ACOR, GENETIC, PSO, GWO</h5>
</div>

<br />
<br />

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#Prerequisites-Installation">Prerequisites & Installation</a></li>
        <li><a href="#How-To-Run">How To Run</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<ul>
  <li>
    <b>Genetic Algorithm: </b> A genetic algorithm is a search heuristic that is inspired by Charles Darwin’s theory of natural evolution. This algorithm                                  reflects the process of natural selection where the fittest individuals are selected for reproduction in order to produce                                  offspring of the next generation.
  </li>
  <li>
    <b>Partickle Swarm Optimization: (PSO):</b> Particle swarm optimization (PSO) is one of the bio-inspired algorithms and it is a simple one to search                                                   foran optimal solution in the solution space. It is different from other optimization algorithms in such a                                                 way that only the objective function is needed and it is not dependent on the gradient or any differential                                                 form of the objective. It also has very few hyperparameters
  </li>
   <li>
     <b>Ant Colony Optimization (ACO):</b> The ant colony optimization algorithm (ACO) is a probabilistic technique for solving computational problems                                                which can be reduced to finding good paths through graphs. Artificial ants stand for multi-agent methods                                                    inspired by the   behavior of real ants. The pheromone-based communication of biological ants is often the                                                  predominant paradigm  used Combinations of artificial ants and local search algorithms have become a method of                                              choice for numerous optimization tasks involving some sort of graph, e.g., vehicle routing and internet routing
   </li>
   <li>
     <b>Ant Colony Optimization Continues Spaces (ACOR):</b> The ACOR algorithm is an Ant Colony Optimization (ACO) extended to continuous domains, and has                                                              been used for training neural network.
   </li>
    <li>
     <b>Gray Wolf Optimization (GWO):</b> Grey wolf optimization algorithm (GWO) is a meta-heuristic optimization technology. Its principle is to imitate                                             the behavior of grey wolves in nature to hunt in a cooperative way. GWO is different from others in terms of                                               model structure. It is a large-scale search method centered on three optimal samples, and which is also the                                                 research object of manY scholars.
   </li>
</ul>





<p align="right">(<a href="#top">back to top</a>)</p>



## Getting Started

this section explain instructions which should be followed to setting up the project. it is easier to run scripts from terminal by passing argument with the help of argparse library and this project also follow this pattern

### Prerequisites-Installation

<ul>
  <li>install all library specified whithin requirement.txt on  your virtual env</li>
  <li>if you are running codes on <b>linux</b> or <b>WSL</b>, you should have a <b>graphical interfece</b> installed on your system. for example for WSl          --> <b>GWSL</b> can be used
  </li>
  <li>
    Benchmark used for evaluating algorithms are listed below. for new Benchmarks manipulate Cost Function of algorithms inside codes.
    <ul>
      <li>ACO --> TSP</li>
      <li>ACOR --> Sphere</li>
      <li>Genetic --> Sphere</li>
      <li>PSO --> Sphere</li>
      <li>GWO --> Sphere</li>
    </ul>
  </li>
</ul>

### How To Run


<b>Run: For all algorithms</b> 

```sh
git clone https://github.com/ay-ka/Evolutionary-Algorithms.git
cd Evolutionary-Algorithms
export PYTHONPATH=$PWD
python src/<algorithm-directory-name>/main.py --arguments
```

for example for Genetic:

```sh
git clone https://github.com/ay-ka/Evolutionary-Algorithms.git
cd Evolutionary-Algorithms
export PYTHONPATH=$PWD
python src/Genetic/main.py --pop_size ? --otherarguments ?
```

for example for GWO

```sh
git clone https://github.com/ay-ka/Evolutionary-Algorithms.git
cd Evolutionary-Algorithms
export PYTHONPATH=$PWD
python src/GWO/main.py --pop_size ? --otherarguments ?
```


<h3><b>Take arguments list: </b></h3> All algorithms take various argument which can be passed through terminal. to get list of all argument used in                                             algorithm use --help: <br/><br/>



<b>For all algorithms</b> <br/>
```sh
git clone https://github.com/ay-ka/Evolutionary-Algorithms.git
cd Evolutionary-Algorithms
export PYTHONPATH=$PWD
python src/<algorithm-directory-name>/main.py --help
```

for example for ACO:

```sh
git clone https://github.com/ay-ka/Evolutionary-Algorithms.git
cd Evolutionary-Algorithms
export PYTHONPATH=$PWD
python src/ACO/main.py --help
```





<!-- CONTRIBUTING -->
## Contributing


If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

aidin kazempour - [linkdin](https://www.linkedin.com/in/aidin-kazempour-b647811ba/) - aydinkazempour7@gmail.com

Project Link: [https://github.com/ay-ka/Algorithms](https://github.com/ay-ka/WordCloud)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [pytopia - a platform for learning python](https://www.pytopia.ai/)


<p align="right">(<a href="#top">back to top</a>)</p>


[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/aidin-kazempour-b647811ba/
[product-screenshot]: images/screenshot.png
