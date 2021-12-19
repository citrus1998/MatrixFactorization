![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
![Packagist](https://img.shields.io/badge/Pytorch-1.7.1-red.svg)
# Matrix Factorization
## Datasets
- Movielens
- Jester
## Train
Fundamentally, you can use `python3 main.py` or `python3 jester.py`.
However, you can use shellscript better as follow:
~~~
bash scripts/main.sh
~~~
In addition, if you wanna compute these programs on background, you can use `nohup` command to learn as follow:
~~~
bash scripts/{dataset}_nhp.sh
~~~  
