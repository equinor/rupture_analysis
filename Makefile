build:
	docker build -t img_rupture .
run:
	docker rm rupture || true
	docker run -v ${HOME_CODE}:/code -i --name rupture -t img_rupture
