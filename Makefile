.PHONY: install train demo

install:
	pip install -r requirements.txt

train:
	python victim_model.py
	python detector.py

demo:
	python app.py
