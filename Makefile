
all: aj

arxiv:
	tar czf arxiv.tgz unwise.tex \
	plots5/npole-03.pdf plots5/npole-05.pdf \
	plots6/cov-00.pdf plots6/cov-01.pdf plots6/cov-02.pdf plots6/cov-03.pdf \
	plots6/cov-04.pdf \
	plots3/boxes-00.pdf plots3/boxes-01.pdf plots3/boxes-02.pdf \
	plots3/boxes-03.pdf plots3/boxes-04.pdf plots3/boxes-05.pdf \
	plots3/boxes-06.pdf plots3/boxes-07.pdf plots3/boxes-08.pdf \
	plots3/boxes-09.pdf plots3/boxes-10.pdf plots3/boxes-11.pdf \
	wisepsf-00.pdf wisepsf-01.pdf wisepsf-02.pdf wisepsf-03.pdf \
	wisepsf-04.pdf wisepsf-05.pdf wisepsf-06.pdf wisepsf-07.pdf \
	wisepsf-08.pdf wisepsf-09.pdf wisepsf-10.pdf wisepsf-11.pdf \
	wisepsf-12.pdf \
	plots1/sequels-009.pdf plots1/sequels-010.pdf plots1/sequels-011.pdf \
	plots1/sequels-015.pdf plots1/sequels-016.pdf plots1/sequels-017.pdf \
	plots1/sequels-024.pdf plots1/sequels-025.pdf plots1/sequels-026.pdf \
	plots1/sequels-027.pdf plots1/sequels-029.pdf plots1/sequels-030.pdf \
	plots1/sequels-035.pdf plots1/sequels-037.pdf plots1/sequels-038.pdf \
	plots2/co-00.pdf plots2/co-01.pdf plots2/co-02.pdf plots2/co-03.pdf \
	plots2/co-04.pdf plots2/co-05.pdf plots2/co-00.pdf plots2/co-06.pdf \
	plots2/co-07.pdf plots2/co-08.pdf \
	plots2/co-09.pdf \
	plots2/co-12-bw.pdf plots2/co-13-bw.pdf \
	plots2/co-10.pdf plots2/co-11.pdf \
	plots4/medfilt-00.pdf plots4/medfilt-01.pdf plots4/medfilt-02.pdf \
	plots4/medfilt-04.pdf plots4/medfilt-05.pdf plots4/medfilt-06.pdf \
	plots7/medfilt-bad-00.pdf plots7/medfilt-bad-01.pdf \
	plots7/medfilt-bad-02.pdf plots7/medfilt-bad-03.pdf

aj:
	python -c 'import re; txt=open("unwise.tex").read(); txt=re.sub(r"\\bwfig{(.*?)}", r"\1-bw", txt); f=open("unwise-aj.tex","w"); f.write(txt); f.close()'
	./mkapj unwise-aj

#bwpdf:
#	python -c 'import re; txt=open("unwise.tex").read(); txt=re.sub(r"\\bwfig{(.*?)}", r"\1-bw", txt); f=open("unwise-bwpdf.tex","w"); f.write(txt); f.close()'
#	./mkapj-pdf unwise-bwpdf

ajpdf:
	rm -R apj apj-bw
	python -c 'import re; txt=open("unwise.tex").read(); txt=re.sub(r"\\bwfig{(.*?)}", r"\1-bw", txt); f=open("unwise-bwpdf.tex","w"); f.write(txt); f.close()'
	./mkapj-pdf unwise-bwpdf
	mv apj apj-bw
	python -c 'import re; txt=open("unwise.tex").read(); txt=re.sub(r"\\bwfig{(.*?)}", r"\1", txt); f=open("unwise-ajpdf.tex","w"); f.write(txt); f.close()'
	./mkapj-pdf unwise-ajpdf
	(cd apj; md5sum -b * > md5sums)
	(cd apj-bw; md5sum -c ../apj/md5sums) | grep FAILED | sed 's/.pdf: FAILED//g' | xargs -t -n 1 -I X cp apj-bw/X.pdf apj/X-bw.pdf
	rm apj/md5sums


.PHONY: aj

