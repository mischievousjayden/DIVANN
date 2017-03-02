
run: diva.py
	python3 example_diva.py

run_tensorboard:
	tensorboard --logdir=logs/set1

init:
	pip install -r requirements.txt

freeze:
	pip freeze > requirements.txt

clean:
	rm -rf *.pyc __pycache__ logs

