install:
	conda config --set env_prompt '({name}) '
	conda env create --prefix=.env

activate:
	conda activate ./.env

clean:
	git clean -fX