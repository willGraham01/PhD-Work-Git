# Will's thesis makefile is here: https://bitbucket.org/wrs20/thesis/src/master/Makefile

# directories for files, all relative to Paper_Sept2020/

# directory containing chapter files
CHAPDIR := ./Sections
# names (without extensions) of chapter files
CHAPNAMES := Abstract Introduction
# create absolute path to chapter files
CHAP_ABS := $(addprefix $(CHAPDIR)/, $(CHAPNAMES))
# append .pdf extension to chapter names for targets
CHAP_TEX := $(CHAP_ABS:=.tex)
# location of bibliography files
BIB_FILES := Paper_Scalar2020.bib
# name of report file to make
NAME := Paper_Scalar2020

# directory containing tikz .tex files
TIKZDIR := ../Diagrams
# directory to place rendered tikz pictures as pdfs
TIKZ_PDF_DUMP := $(TIKZDIR)/Diagram_PDFs#folder to contain all the tikz pdfs
# all tikz tex filenames, without extensions
TIKZ_FNAMES := $(shell ls -p $(TIKZDIR)/ | grep -v / | sed -e 's/\.tex$///' | tr '\n' ' ')
# create pdf filenames for tikz files
TIKZ_PDFS := $(addprefix $(TIKZDIR)/, $(TIKZ_FNAMES:=.pdf))

# render article, report, paper, etc
MKLATEX := latexmk -bibtex -pdf -latexoption=""
# render tikz figures in .tex files in $(TIKZDIR)/
MKTIKZ := latexmk -cd -pdf -outdir=$(TIKZ_PDF_DUMP)
# clear auxillary files produced by latexmk
SOFT_CLEAN := latexmk -c
# clear auxillary files and pdfs produced by fdb_latexmk
HARD_CLEAN := latexmk -C
# only run pdflatex
PDFLATEX := pdflatex --shell-escape
# print what is currently being assembled
PRINT = @/bin/echo -e "\e[1;34mBuilding $<\e[0m"

all: $(NAME).pdf

$(NAME).pdf: $(NAME).tex $(TIKZ_PDFS) $(BIB_FILES) $(CHAP_TEX)
	$(PRINT)
	$(MKLATEX) $(NAME).tex
	$(SOFT_CLEAN) $(NAME).tex

# assembly method for tex files containing tikz images
$(TIKZ_PDFS): %.pdf: %.tex
	$(PRINT)
	$(MKTIKZ) $< #use latexmk to build the image and place it into the output directory
	$(SOFT_CLEAN) -outdir=$(TIKZ_PDF_DUMP) $< #cleanup auxillary files created by the build

# make all tikz pictures and place into output folder
tikz: $(TIKZ_PDFS)

# make the file without re-rendering the tikz images (only for use when diagrams haven't been changed)
no_tikz: $(NAME).tex $(BIB_FILES) $(CHAP_TEX)
	$(PRINT)
	$(MKLATEX) $(NAME).tex
	$(SOFT_CLEAN) $(NAME).tex

# redirect
.PHONY: clean

# clear auxillary files and PDF of FullReport
clean:
	$(HARD_CLEAN) $(NAME).tex

# clear axuillary files of FullReport
clear:
	$(SOFT_CLEAN) $(NAME).tex

clearfigs:
	@/bin/echo -e "Clearing tikz PDFs..."
	$(HARD_CLEAN) -outdir=$(TIKZ_PDF_DUMP) $(TIKZDIR)/*.tex
