
run: diva.py
	python3 example_diva.py

run_tensorboard:
	tensorboard --logdir=logs/set1

clean:
	rm -rf *.pyc __pycache__ logs

