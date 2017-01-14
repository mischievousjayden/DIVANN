
run: diva.py
	python3 diva.py

run_tensorboard:
	tensorboard --logdir=logs/diva

clean:
	rm -rf *.pyc __pycache__ logs

