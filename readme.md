
# Exo Explorer

## 🌌 A 2025 NASA Space Apps Challenge: **A World Away: Hunting for Exoplanets with AI**

This project was developed for the NASA Space Apps Challenge 2025 under the challenge **A World Away: Hunting for Exoplanets with AI**.

Our goal is to train machine learning models to classify astronomical objects detected by missions such as Kepler, K2, and TESS, distinguishing between confirmed exoplanets, candidates, and false positives. The project also provides an API + interactive frontend to make exploration easy and accessible.

## 🚀 Motivation
Thousands of exoplanets have been discovered via the transit method, but most detections are still manually validated. Using modern machine learning techniques, we can:

- Reduce the time required for classification.

- Improve detection accuracy.

- Suggest which candidates are more likely to be true planets.

- Democratize access to exoplanet ML models through a simple web interface for investigators that don't know how to program or use an AI model.

## 📊​ Datasets

We used publicly available datasets from NASA (see Appendix):

- Kepler (original mission).

- K2 (Kepler’s extended mission).

- TESS / TOI (Transiting Exoplanet Survey Satellite).


## 🧠 Machine Learning Models

We trained and compared several models:

- 🌲 Random Forest

- ⚡ XGBoost

- 🔍 Basic neural net with 2 layers

XGBoost achieved the best performance with 80% accuracy on validation data.

## ⚙️ Project structure

nasa_exoplanets/

│── data/                 # datasets (Kepler, K2, TESS)

│   ├── raw/              # raw datasets

│   └── processed/        # clean and merged data

│── models/               # trained models

│── scripts/              # API, preprocessing, training, evaluation

│── web/                  # UI

│── .gitignore

│── requirements.txt    

└── README.md             

## 🔧 Installation & Execution

### Dependencies
Install the dependencies:

``` bash
pip install -r requirements
```
### Backend
Consist in the steps: 
- **Preprocessing**
``` bash
python ./scripts/preprocess.py
```

- **Training**
``` bash
python ./scripts/train_ML.py
python ./scripts/train_NN.py
```

- **Evaluation**
``` bash
python ./scripts/evaluation_ML.py
python ./scripts/evaluation_NN.py
```
### Frontend
1. Navigate to frontend directory:
``` bash
cd frontend
npm install
npm run dev
```

2. Open http://localhost:3000

## 🚀 Deployment

### Backend
Deployed on [Railway](https://railway.app) using Docker containerization.
- **URL**: https://api.exoexplorer.study
- **CI/CD**: Automatic deployments from GitHub, set "backend" as base directory. 

### Frontend
Deployed on [Vercel](https://vercel.com).
- **URL**: https://exoexplorer.study
- **CI/CD**: Automatic deployments from GitHub, set "frontend" as base directory. 

<img width="2492" height="1660" alt="arch2" src="https://github.com/user-attachments/assets/60dbd4d7-65d8-4389-ae51-e1533d669e13" />


## 🌍 Impact

This project aims to:

- Support astronomers and enthusiasts in exoplanet classification.

- Accelerate validation of candidates

- Promote accessible science through interactive tools.

## 👥 Team

[Javier Trujillo Castro](https://github.com/javitrucas) – Machine Learning & content creator

[Ángel Sanchez Guerrero](https://github.com/Angeloyo) – Frontend Development & UI/UX Design

[Raúl Martínez Alonso](https://github.com/raulmart03) – Data Science & documentation

[Pablo Tamayo López](https://github.com/pablotl0) – Backend Development & content creator







## 🔗​ Links of interest

### Datasets
- [Kepler Objects of Interest (KOI)](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)

- [TESS Objects of Interest (TOI)](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI)

- [K2 Planets and Candidates](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc)


### References

- [Exoplanet Detection Using Machine Learning](https://academic.oup.com/mnras/article/513/4/5505/6472249?login=false)

- [Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification](https://www.mdpi.com/2079-9292/13/19/3950)
## License

[MIT](https://choosealicense.com/licenses/mit/)

