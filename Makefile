install:
	# Install libraries 
	pip install --upgrade pip &&\
	pip install -r requirements.txt
lint:
	pylint -d=R,C *.py
$(info No file types have been specified for linting.)
format:
	black *.py
test:
# pytest -vv --cov=addtwo test_addtwo.py
	$(info No tests yet. (Best get on that!))
	
