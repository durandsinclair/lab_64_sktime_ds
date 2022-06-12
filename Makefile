install:
	# Install libraries 
	pip install --upgrade pip &&\
	pip install -r requirements.txt
lint:
	pylint -d=R,C addtwo.py
format:
	black *.py
test:
	# pytest -vv --cov=addtwo test_addtwo.py
	print "No tests yet. (Best get on that)."
	
