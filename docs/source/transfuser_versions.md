# History

There are many different version of TransFuser by now, and we sometimes see the wrong papers getting cited.
Here is a short history over the different transFuser versions and how to cite them correctly.
Some of the methods have the same name. This may be confusing, if you prefer you can also name them TransFuser v[id] which is indicated after the dash.

### TransFuser (CVPR 2021) - TransFuser v1
The first version of TransFuser. The paper [Multi-Modal Fusion Transformer for End-to-End Autonomous Driving](https://www.cvlibs.net/publications/Prakash2021CVPR.pdf) introduced the architecture back in 2021.
The code is still available and can be found [here](https://github.com/autonomousvision/transfuser/tree/cvpr2021).
The model was developed in the early days of the CARLA leaderboard code / community, where dataset quality was quite poor. There is not much of a point to compare to this model anymore, as its performance is quite weak by today's standards.
```BibTeX
@inproceedings{Prakash2021CVPR,
  author = {Prakash, Aditya and
            Chitta, Kashyap and
            Geiger, Andreas},
  title = {Multi-Modal Fusion Transformer for End-to-End Autonomous Driving},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2021}
}
```

### TransFuser+ (Master Thesis University of Tübingen 2021) - TransFuser v2
The thesis [Expert Drivers for Autonomous Driving](https://kait0.github.io/assets/pdf/master_thesis_bernhard_jaeger.pdf) investigated the data quality issue of the original TransFuser work.
It proposes a stronger automatic labeling algorithm.
Together with auxiliary training, this pushed the performance of TransFuser a lot without changing the architecture.
The document is available online but not formally published as it is only relevant to the core CARLA community.
We would still appreciate it if you cite it where relevant.
The code and models are not directly released, but the relevant code was published as part of the PAMI project.
```BibTeX
@mastersthesis{Jaeger2021Thesis,
	author = {Bernhard Jaeger},
	title = {Expert Drivers for Autonomous Driving},
	year = {2021},
	school = {University of Tübingen},
}
```

### TransFuser (T-PAMI 2022) - TransFuser v3
The journal update to the CVPR paper. The paper [TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving](https://www.cvlibs.net/publications/Chitta2022PAMI.pdf) is published in Transaction on Pattern Analysis and Machine Intelligence.
At the core it is still the TransFuser architecture, but it features better data, sensors, backbones, training and a rigorous set of ablations that shows what is important and what is not.
The paper features a version called Latent TransFuser, which is a camera only TransFuser that replaces the LiDAR input by a positional encoding.
The final models are roughly 4x better on the CARLA leaderboard than the CVPR TransFuser.
Code, models and data are available [online](https://github.com/autonomousvision/transfuser/).
```BibTeX
@article{Chitta2022PAMI,
  author = {Chitta, Kashyap and
            Prakash, Aditya and
            Jaeger, Bernhard and
            Yu, Zehao and
            Renz, Katrin and
            Geiger, Andreas},
  title = {TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving},
  journal = {Pattern Analysis and Machine Intelligence (PAMI)},
  year = {2022},
}
```

### TransFuser++ (ICCV 2023) - TransFuser v4
The ICCV 2023 paper [Hidden Biases of End-to-End Driving Models](https://arxiv.org/abs/2306.07957) offers some explanations why TransFuser and related approaches work so well.
It also improved the TransFuser family with a better sensor setup, architecture and training recipe. The model called TransFuser++ (as well as a WP variant that uses waypoints as output representation).
[Code](https://github.com/autonomousvision/carla_garage) is available online. TransFuser v4 is, at the time of writing (Feb.2025), still the best open-source model on many CARLA leaderboard 1.0 benchmarks.
```BibTeX
@article{Jaeger2023ICCV,
  title={Hidden Biases of End-to-End Driving Models},
  author={Bernhard Jaeger and Kashyap Chitta and Andreas Geiger},
  booktitle={Proc. of the IEEE International Conf. on Computer Vision (ICCV)},
  year={2023}
}
```

### TransFuser (NeurIPS 2024) - TransFuser v3.5
The NAVSIM paper [NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking](https://arxiv.org/abs/2406.15349) uses a baseline called TransFuser. This baseline uses the architecture and sensor setup of TransFuser++ ICCV, with the training schedule and output representation of TransFuser T-PAMI. You can cite either paper or both to refer to this baseline. Typically, [TransFuser T-PAMI](#transfuser-t-pami-2022) is cited. We gave it the version 3.5 because it is a mix between version 3 and 4.
```BibTeX
@inproceedings{Dauner2024NeurIPS,
  author       = {Daniel Dauner and
                  Marcel Hallgarten and
                  Tianyu Li and
                  Xinshuo Weng and
                  Zhiyu Huang and
                  Zetong Yang and
                  Hongyang Li and
                  Igor Gilitschenski and
                  Boris Ivanovic and
                  Marco Pavone and
                  Andreas Geiger and
                  Kashyap Chitta},
  title        = {NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking},
  booktitle    = {Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024 (NeurIPS)},
  year         = {2024},
}

```

### TransFuser++ (Tech Report & Master Thesis) - TransFuser v5
Adapted version of TransFuser++ for the needs of the CARLA leaderboard 2.0 and PDM-Lite expert.
The model reached 2nd place in the [CVPR 2024 CARLA challenge](https://opendrivelab.com/challenge2024/#carla).
A [technical report](https://arxiv.org/abs/2412.09602) is available on ArXiv, which describes the changes and has some analysis on changes and the DS metric.
The model is released in this repository.
This model was created as part of a master thesis project which describes the approach in more depth.
```BibTeX
@article{Zimmerlin2024ArXiv,
  author       = {Julian Zimmerlin and
                  Jens Bei{\ss}wenger and
                  Bernhard Jaeger and
                  Andreas Geiger and
                  Kashyap Chitta},
  title        = {Hidden Biases of End-to-End Driving Datasets},
  journal      = {ArXiv.org},
  volume       = {2412.09602},
  year         = {2024}
}

@mastersthesis{Zimmerlin2024thesis,
  title={Tackling CARLA Leaderboard 2.0 with End-to-End Imitation Learning},
  author={Julian Zimmerlin},
  school={University of Tübingen},
  howpublished={\textsc{url:}~\url{https://kashyap7x.github.io/assets/pdf/students/Zimmerlin2024.pdf}},
  year={2024}
}
```

### TransFuser v6 (ArXiv report)
Latest version in the TransFuser series, which minimizes the asymmetry between the expert driver and the TransFuser student.
We have fully transitioned to using the version number in the name.
An earlier (latent) version of the model reached 2nd place in the [CVPR 2025 Waymo Vision-based End-to-End Driving Challenge](https://waymo.com/open/challenges).
A [report](https://arxiv.org/abs/2512.20563v1) is available on ArXiv, describing all changes and the research insights.
This paper is an extension of a master's thesis project. TransFuser v6 is, at the time of writing (Dec. 2025), the best model for all CARLA leaderboard 2.0 benchmarks.

```BibTeX
@article{Nguyen2025ARXIV,
  author       = {Long Nguyen and Micha Fauth and Bernhard Jaeger and Daniel Dauner and Maximilian Igl and Andreas Geiger and Kashyap Chitta},
  title        = {LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving},
  journal      = {ArXiv.org},
  volume       = {2512.20563},
  year         = {2025}
}

@mastersthesis{Nguyen2025thesis,
  title={Addressing the Fundamental Barriers towards End-to-End Driving in Simulation},
  author={Long Nguyen},
  school={University of Tübingen},
  howpublished={\textsc{url:}~\url{https://ln2697.github.io/assets/pdf/Nguyen2025Master.pdf}},
  year={2025}
}
```
