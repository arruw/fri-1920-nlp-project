conda-install:
	conda config --set env_prompt '({name}) '
	conda env create --prefix=.env

dataset-download:
	rm -rf ./dataset
	mkdir -p ./dataset
	curl -SL https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1285/SentiCoref_1.0.zip -o tmp.zip
	unzip -d ./dataset tmp.zip
	rm tmp.zip
	rm -rf ./dataset/__MACOSX

git-clean:
	git clean -fX