conda-install: are-you-sure
	conda config --set env_prompt '({name}) '
	conda env create --prefix=.env

conda-export: are-you-sure
	conda env export | sed "s/name.*/name: .env/" | sed "s/prefix.*/prefix: .\/.env/" > environment.yml

dataset-download: are-you-sure
	rm -rf ./dataset
	mkdir -p ./dataset
	curl -SL https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1285/SentiCoref_1.0.zip -o tmp.zip
	unzip -d ./dataset tmp.zip
	rm tmp.zip
	rm -rf ./dataset/__MACOSX

git-clean: are-you-sure
	git clean -fX

are-you-sure:
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]