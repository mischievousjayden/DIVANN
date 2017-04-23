
run: diva.py
	python3 example_diva.py 0.8 0.01 8 result1.csv

run_tensorboard:
	tensorboard --logdir=logs/set1

init:
	pip install -r requirements.txt

freeze:
	pip freeze > requirements.txt

clean:
	rm -rf *.pyc __pycache__ logs

