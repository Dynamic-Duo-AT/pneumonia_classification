# pneumonia
The overall goal of this project is to develop an automated model for detecting pneumonia in paediatric chest X-ray images. The model is intended to serve as a diagnostic tool, specifically capable of classifying images of the lungs as either NORMAL or PNEUMONIA. The model will be developed within a standardized framework that defines training and evaluation procedures to create standardized experiment pipelines which will ensure reproducibility and comparability. We will further expand on the framework using different tools for logging, experiment setup, deployment and monitorization, to streamline all aspects of the project life cycle.  

The dataset which initially is going to be used is the “Chest X-Ray Images (Pneumonia)” found on Kaggle, which consists of 5856 Chest X-Ray Images of both healthy and Pneumonia (both bacterial and viral) patients. The images are of one- to five-year-olds and are from Guangzhou Women and Children’s Medical Center, Guangzhou. The labels (diagnosis) were graded by two expert physicians. The images are grayscale, meaning that they consist of a single channel. The images are of various sizes meaning that a preprocessing step will include resizing the images into an appropriate shape. The dataset is pre-distributed into a training set consisting of 5216 images, validation set consisting of 16 images, and a test set consisting of 624 images. We recognize that the validation set is extremely small and will therefore randomly select roughly 500 images from the training set and instead insert them into the validation set resulting in the following train, validation, test split = 4716, 516, 624.  

We aim to create a classic binary classification CNN model as our baseline model. We will start by creating a small CNN with a few convolutional layers into a fully connected layer. We will expand on this classic CNN structure and lastly download a state-of-the-art model architecture such as resnet or potentially a visual transformer encoder structure and compare the performance to the classic CNN. 

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
