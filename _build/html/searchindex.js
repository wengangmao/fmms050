Search.setIndex({"docnames": ["README", "contents/data-filter/fft", "contents/data-filter/kalmar", "contents/data-filter/kalmar_codes", "contents/data-filter/kalmar_tutorial", "contents/data-filter/moving-average", "contents/data-filter/wavelet", "contents/deep-learning/cnn", "contents/deep-learning/rnn", "contents/deep-learning/spatio-temporal", "contents/deep-learning/spde", "contents/introduction", "contents/machine-learning/ann", "contents/machine-learning/boost", "contents/machine-learning/logistic", "contents/machine-learning/svm", "contents/machine-learning/trees", "contents/ml-engineering/basic-math", "contents/ml-engineering/intro_applications", "contents/ml-engineering/ml-types", "contents/regression/gam-mem", "contents/regression/gam-mem_codes", "contents/regression/gam-mem_tutorial", "contents/regression/glr", "contents/regression/glr-tutorial", "contents/regression/glr_codes", "contents/regression/interperations", "contents/regression/poly-spline", "contents/regression/poly-spline_codes", "contents/regression/poly-spline_tutorial", "contents/regression/reg-gradient", "contents/time-series/acf", "contents/time-series/ar-ma", "contents/time-series/arima", "contents/time-series/gaussian-transform", "contents/time-series/stationary-gaussian"], "filenames": ["README.md", "contents\\data-filter\\fft.md", "contents\\data-filter\\kalmar.md", "contents\\data-filter\\kalmar_codes.ipynb", "contents\\data-filter\\kalmar_tutorial.md", "contents\\data-filter\\moving-average.md", "contents\\data-filter\\wavelet.md", "contents\\deep-learning\\cnn.md", "contents\\deep-learning\\rnn.md", "contents\\deep-learning\\spatio-temporal.md", "contents\\deep-learning\\spde.md", "contents\\introduction.md", "contents\\machine-learning\\ann.md", "contents\\machine-learning\\boost.md", "contents\\machine-learning\\logistic.md", "contents\\machine-learning\\svm.md", "contents\\machine-learning\\trees.md", "contents\\ml-engineering\\basic-math.md", "contents\\ml-engineering\\intro_applications.md", "contents\\ml-engineering\\ml-types.md", "contents\\regression\\gam-mem.md", "contents\\regression\\gam-mem_codes.ipynb", "contents\\regression\\gam-mem_tutorial.md", "contents\\regression\\glr.md", "contents\\regression\\glr-tutorial.md", "contents\\regression\\glr_codes.ipynb", "contents\\regression\\interperations.md", "contents\\regression\\poly-spline.md", "contents\\regression\\poly-spline_codes.ipynb", "contents\\regression\\poly-spline_tutorial.md", "contents\\regression\\reg-gradient.md", "contents\\time-series\\acf.md", "contents\\time-series\\ar-ma.md", "contents\\time-series\\arima.md", "contents\\time-series\\gaussian-transform.md", "contents\\time-series\\stationary-gaussian.md"], "titles": ["FMMS050 - Statistical regression and machine learning", "Fast Fourier Transform", "Kalmar filter", "Example codes to solve the tutorial problems", "Tutorial of Kalmar filter", "Moving average", "<span style = \"color:red; font-weight: 500;\">PyWavelets - Wavelet Transforms in Python</span>", "Convolutional neural network", "Recursive neural networks", "Spatio-temporal model", "Statistical partical differentical equations", "FMMS050 - Statistical regression and machine learning", "Artificial Neural Network", "Boosting method (XGBoost)", "Logistical regression and classification", "Support vector machine", "Decision trees and ensemble algorithm", "Basic mathematics and statistics for machine learning methods", "Basic definitions of machine learning", "Categories of machine learning methods", "Generalized additive model and mixed effect model", "Example codes for GAM and MEM tutorials", "Tutorial for GAM and MEM methods", "Generalized linear regression", "Tutorials of using the GLA methods", "Example codes for the GLR problems", "Regression and its interpretation", "Polynomial and Spline fitting", "Example codes for the tutorial examples for Polynomial and Spline fitting", "Several tutorial examples to use Polynomial and Spline fitting", "Gradient for regression (parameter estimations)", "Autocorrelation and Conditional expectation", "Auto regressive models and Moving average models", "ARIMA models", "Gaussian transformation method", "Basic properties of stationary Gaussian process"], "terms": {"The": [0, 11], "given": [0, 11], "divis": [0, 11], "marin": [0, 11], "technolog": [0, 11], "depart": [0, 11], "maritim": [0, 11], "scienc": [0, 11], "univers": [0, 11], "gothenburg": [0, 11], "sweden": [0, 11], "credit": [0, 11], "7": [0, 11], "5": [0, 11], "grade": [0, 11], "pass": [0, 11], "fail": [0, 11], "educ": [0, 11], "cycl": [0, 11], "doctor": [0, 11], "student": [0, 11], "major": [0, 11], "subject": [0, 11], "ship": [0, 11], "appli": [0, 9, 10, 11], "30": [0, 11], "AND": [0, 11], "phone": [0, 11], "031": [0, 11], "7721483": [0, 11], "email": [0, 11], "se": [0, 11], "senior": [0, 11], "master": [0, 11], "mathemat": [0, 11, 26, 30], "includ": 0, "linear": [0, 11], "algebra": [0, 11], "numer": [0, 11], "analysi": [0, 8, 11, 30], "multi": [0, 9, 10, 11], "variabl": [0, 11], "calculu": [0, 11], "strength": [0, 11], "materi": [0, 11], "applic": [0, 9], "should": [0, 11], "have": [0, 11, 17, 26], "need": [0, 11, 17], "deal": [0, 11], "some": [0, 10, 11, 17, 18], "experiment": [0, 11], "full": [0, 11], "scale": [0, 11], "data": [0, 1, 2, 5, 11, 18], "from": [0, 2, 5, 11, 19], "purpos": [0, 11], "give": [0, 10, 11, 17], "method": [0, 2, 5, 8, 10, 18, 26], "test": [0, 11], "order": [0, 11, 17], "understand": [0, 11, 17], "hidden": [0, 11], "characterist": [0, 11], "within": [0, 11], "In": [0, 11, 17, 18, 26], "thi": [0, 8, 9, 10, 17, 18, 26], "so": [0, 11], "call": [0, 11], "cover": [0, 11], "two": [0, 11], "type": [0, 11], "i": [0, 11], "e": [0, 11], "independ": [0, 11], "observ": [0, 11], "uncertainti": [0, 11], "nois": [0, 2, 11], "strongli": [0, 11], "depend": [0, 11], "time": [0, 8, 9, 11], "seri": [0, 8, 11], "signal": [0, 11], "addit": [0, 11], "direct": [0, 11], "implement": [0, 11], "skill": [0, 11], "practic": [0, 4, 11], "also": [0, 11, 17], "fundament": [0, 11, 17], "theori": [0, 10, 11], "connect": [0, 11, 17, 26], "each": [0, 11], "outcom": [0, 11], "after": [0, 11], "finish": [0, 11], "profession": [0, 11], "knowledg": [0, 11, 17], "interpret": [0, 11], "variou": [0, 11], "random": [0, 11], "natur": [0, 10, 11], "involv": [0, 11], "problem": [0, 4, 11], "model": [0, 10, 11], "explain": [0, 11, 26, 30], "scenario": [0, 11], "contain": [0, 11], "fft": 1, "veri": [1, 17], "us": [1, 2, 4, 17, 18], "clean": [1, 2, 5], "certain": 1, "frequenc": 1, "ha": [2, 8], "been": 2, "develop": [2, 8, 19], "dure": 2, "past": 2, "decad": 2, "we": [2, 8, 9, 10, 17, 18, 26], "present": [2, 9], "basic": [2, 11, 19, 26], "idea": 2, "sever": [2, 4, 8, 9], "evolut": [2, 8], "origin": [2, 5, 19], "kalmer": 2, "clearn": 2, "pre": [2, 11], "process": [2, 11], "But": 3, "you": [3, 4, 17, 26], "ar": [3, 9, 11, 17, 18, 19, 30], "suppos": 3, "first": 3, "try": 3, "find": [3, 18], "solut": 3, "write": 3, "yourself": 3, "exampl": [4, 11], "demonstr": 4, "how": [4, 10, 17, 26, 30], "solv": 4, "an": [5, 11], "import": 5, "tool": 5, "smooth": 5, "unexpect": 5, "unrealist": 5, "spike": 5, "measur": 5, "open": 6, "sourc": 6, "softwar": 6, "It": [6, 8, 9, 19], "combin": [6, 8], "simpl": 6, "high": 6, "level": 6, "interfac": 6, "low": 6, "c": [6, 11], "cython": 6, "perform": 6, "good": [8, 10], "predict": [8, 11], "further": [8, 19], "can": [8, 9, 10, 19, 26], "typic": 8, "machin": [8, 9, 26], "learn": [8, 9, 26], "extract": 8, "static": 8, "condit": [8, 11], "introduc": 8, "rnn": 8, "base": 8, "statist": [9, 19], "construct": 9, "due": 9, "intern": 9, "correl": [9, 10], "between": [9, 18], "input": 9, "featur": 9, "sometim": 9, "dimension": [9, 10], "both": 9, "space": 9, "small": 9, "branch": 9, "concept": [9, 17], "its": [9, 11], "more": [9, 11, 17, 26], "advanc": [9, 11, 17, 26], "usag": 9, "For": 9, "cours": [9, 18], "part": [9, 11, 17, 26], "option": 9, "prove": 10, "statistician": 10, "phenomenon": 10, "short": 10, "summari": 10, "about": [10, 18], "especi": [10, 18], "show": 10, "metocean": 10, "environ": 10, "mechan": 11, "chalmer": [11, 18], "wengang": 11, "mao": 11, "engin": 11, "divid": 11, "four": 11, "forecast": 11, "detail": 11, "below": 11, "clarif": 11, "differ": [11, 17, 18], "terminolog": [11, 18], "field": 11, "ai": [11, 18], "ml": [11, 18], "overview": 11, "categori": 11, "radient": 11, "paramet": 11, "estim": 11, "polynomi": 11, "spline": 11, "fit": [11, 19, 26], "gener": 11, "mix": 11, "effect": 11, "logist": 11, "classif": [11, 19], "neural": 11, "network": 11, "support": 11, "vector": 11, "decis": 11, "tree": 11, "ensembl": 11, "algorithm": [11, 26], "boost": 11, "xgboost": 11, "gaussian": 11, "transform": 11, "properti": 11, "stationari": 11, "autocorrel": 11, "expect": 11, "auto": 11, "move": 11, "averag": 11, "arima": 11, "assign": 11, "seminar": 11, "relat": 11, "form": 11, "18": 11, "last": 11, "2": 11, "4": 11, "hour": 11, "end": 11, "where": 11, "hasti": 11, "t": 11, "ribshirani": 11, "r": 11, "friedman": 11, "j": 11, "2003": 11, "mine": 11, "infer": [11, 19], "springer": 11, "shalizi": 11, "2019": 11, "elementari": 11, "point": 11, "view": 11, "print": 11, "shumwai": 11, "h": 11, "stoffer": 11, "d": 11, "s": 11, "2016": 11, "fourth": 11, "edit": 11, "wei": 11, "w": 11, "2006": 11, "univari": 11, "multivari": 11, "second": 11, "pearson": 11, "addison": 11, "weslei": 11, "consist": 11, "final": 11, "figur": 11, "http": 11, "github": 11, "com": 11, "wengangmao": 11, "blob": 11, "main": 11, "imag": 11, "201": 11, "20": 11, "20cours": 11, "20content": 11, "pdf": 11, "height": 11, "200px": 11, "name": 11, "fund": 11, "agenc": 11, "alt": 11, "larg": 11, "our": 11, "research": 11, "root": 17, "knowledge": 17, "while": 17, "onli": 17, "interest": 17, "packag": 17, "mai": [17, 26], "deep": [17, 19], "those": 17, "mathematica": 17, "importantli": 17, "who": 17, "dig": 17, "provid": 17, "refer": 17, "studi": [17, 26], "bit": 17, "section": 18, "describ": 18, "talk": 18, "big": 18, "scientist": 18, "which": 18, "often": 18, "confus": 18, "societi": 18, "save": 18, "earth": 18, "next": 18, "pleas": 18, "download": 18, "lectur": 18, "through": 18, "1": 18, "content": 18, "A": [], "demo": [], "voyag": [], "optim": [], "come": 19, "differnet": 19, "regressiong": 19, "supervis": 19, "unsupevis": 19, "regress": 19, "mani": 26, "alreadi": 26, "interpr": 26, "wai": 30, "fmms050": 18}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"fmms050": [0, 11], "statist": [0, 10, 11, 17], "regress": [0, 11, 14, 23, 26, 30, 32], "machin": [0, 11, 15, 17, 18, 19], "learn": [0, 11, 17, 18, 19], "A": 0, "chalmer": 0, "phd": 0, "cours": [0, 11], "engin": 0, "mechan": 0, "teach": [0, 11], "languag": [0, 11], "english": [0, 11], "examin": [0, 11], "lectur": [0, 11], "wengang": 0, "mao": 0, "elig": [0, 11], "specif": [0, 11], "prerequisit": [0, 11], "aim": [0, 11], "object": [0, 11], "fast": 1, "fourier": 1, "transform": [1, 6, 34], "kalmar": [2, 3, 4], "filter": [2, 4], "exampl": [3, 21, 25, 28, 29], "code": [3, 21, 25, 28], "solv": 3, "tutori": [3, 4, 21, 22, 24, 28, 29], "problem": [3, 25], "some": 3, "us": [3, 24, 29], "packag": 3, "implement": 3, "fitler": 3, "method": [3, 11, 13, 17, 19, 22, 24, 34], "move": [5, 32], "averag": [5, 32], "span": 6, "style": 6, "color": 6, "red": 6, "font": 6, "weight": 6, "500": 6, "pywavelet": 6, "wavelet": 6, "python": 6, "convolut": 7, "neural": [7, 8, 12], "network": [7, 8, 12], "recurs": 8, "spatio": 9, "tempor": 9, "model": [9, 20, 32, 33], "partic": 10, "different": 10, "equat": 10, "syllabu": 11, "thi": 11, "content": 11, "basi": 11, "applic": 11, "organ": 11, "literatur": 11, "includ": 11, "compulsori": 11, "element": 11, "present": 11, "1": 11, "artifici": 12, "boost": 13, "xgboost": 13, "logist": 14, "classif": 14, "support": 15, "vector": 15, "decis": 16, "tree": 16, "ensembl": 16, "algorithm": 16, "basic": [17, 18, 35], "mathemat": 17, "definit": 18, "categori": 19, "gener": [20, 23], "addit": 20, "mix": 20, "effect": 20, "gam": [21, 22], "mem": [21, 22], "linear": 23, "gla": 24, "glr": 25, "its": 26, "interpret": 26, "polynomi": [27, 28, 29], "spline": [27, 28, 29], "fit": [27, 28, 29], "sever": 29, "gradient": 30, "paramet": 30, "estim": 30, "autocorrel": 31, "condit": 31, "expect": 31, "auto": 32, "arima": 33, "gaussian": [34, 35], "properti": 35, "stationari": 35, "process": 35}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx": 56}})