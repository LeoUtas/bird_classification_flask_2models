<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li><a href="#demo">Demo</a></li>
    <li><a href="#technical-tools">Technical Tools</a></li>
    <li><a href="#data-source">Data source</a></li>
    <li><a href="#the-design">The design</a></li>
    <li><a href="#how-to-use-the-source-code">How to use the source code</a></li>
    <li><a href="#the-bottom-line">The Bottom Line</a></li>
    <li><a href="#reference">Reference</a></li>
  </ol>
</details>

### Introduction

This repository hosts the source code for a web application to classify bird species. It integrates two models: a customization of MobileNet architecture (Howard, A.G., et al., 2017) and a model based on the You Only Look Once (YOLO) framework (Redmon, J. et al., 2016). The application features a user-friendly interface for easy image uploading and rapid species identification.

### Demo

<p align="center">
  <a href="GIF" style="display: flex; justify-content: center; align-items: center; margin-bottom:80px">
    <img src="/video/bird-app524.gif" width="440" alt="" style="margin-right: 10px;"/>
    <img src="/video/bird-app-yolov8.gif" width="400" alt="" style="margin-left: 10px;"/>
  </a>
</p>

### Technical tools

-   The orginal paper of MobileNet <a href="https://arxiv.org/pdf/1704.04861.pdf">(Howard, A.G. et al., 2017)</a>.

-   The application documentation of <a href="https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/MobileNet"> MobileNet </a> using TensorFlow v2.14.0.

-   The orginal paper of You Only Look Once YOLO architecture <a href="https://arxiv.org/pdf/1506.02640.pdf">(Redmon, J. et al., 2016)</a>.

-   The application documentation of <a href="https://docs.ultralytics.com/"> YOLOv8 </a> by Ultralytics.

*   Pytorch
*   YOLOv8
*   TensorFlow
*   numpy
*   pandas
*   Flask
*   JavaScript (plain)
*   HTML
*   CSS (Bootstrap)
*   Docker

### Data source

This project utilizes a bird species dataset provided by <a href="https://www.kaggle.com/gpiosenka">Gerry</a>, available on Kaggle. For detailed information, visit <a href="https://www.kaggle.com/datasets/gpiosenka/100-bird-species/data"> birds 525 species- image classification </a>.

### The design

I developed a bird classification web application with three distinct approaches:

-   A 2-stage model using YOLOv8 architecture, <a href="https://github.com/LeoUtas/bird_classification_flask_YOLOv8.git">(source code)</a>;
-   A 1-stage model using MobileNet architectures, <a href="https://github.com/LeoUtas/bird_classification_flask_MobileNet.git">(source code)</a>; and
-   A combination of the YOLOv8 and MobileNet architectures, <a href="https://github.com/LeoUtas/bird_classification_flask_2models.git">(source code)</a>

Only the MobileNet architecture was chosen for the <a href="https://bird-classification524-b310a542793a.herokuapp.com/"> final web application </a> after evaluating different models. However, this repository could offer a nice experience about comparing the performance of different models on real images.

### How to use the source code

##### Using the source code for development

-   Fork this repository (https://github.com/LeoUtas/bird_classification_flask_2models.git)
-   Get the docker container ready

    -   Run docker build (it might take a while for installing all the required dependencies to your local docker image)

    ```cmd
    docker build -t <name of the docker image> .
    ```

    -   Run the Docker Container (once the docker image is built, you will run a docker container, map it to the port 5000)

    ```cmd
    docker run -p 5000:5000 -v "$(PWD):/app" --name <name of the container> <name of the docker image>
    ```

-   Run the app.py on the docker container

    -   For windows users

    ```cmd
    python app.py
    ```

    -   For MacOS and Linux users

    ```bash
    python3 app.py
    ```

    -   Change debug=False to True in app.py for development (it's crucial to asign debug=True for the ease of tracking bugs when customizing the code)

    ```python
    # the last chunk of code in the app.py
    if __name__ == "__main__":
    port = int(
        os.environ.get("PORT", 5000)
    )  # define port so we can map container port to localhost
    app.run(host="0.0.0.0", port=port, debug=False)  # define 0.0.0.0 for Docker
    ```

-   Stop running the container when you're done:

    ```cmd
    docker stop <name of the container>
    ```

### The bottom line

I'm excited to share this repository! Please feel free to explore its functionalities. Thank you for this far. Have a wonderful day ahead!

Best,
Hoang Ng

### Reference

Howard, A.G. et al., 2017. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. CoRR, abs/1704.04861. Available at: http://arxiv.org/abs/1704.04861.

Redmon, J. et al., 2016. You Only Look Once: Unified, Real-Time Object Detection. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 779–788.
