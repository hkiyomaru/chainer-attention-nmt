wget http://www.statmt.org/wmt10/training-giga-fren.tar
wget http://www.statmt.org/wmt15/dev-v2.tgz

tar -xf training-giga-fren.tar
gunzip giga-fren.release2.fixed.en.gz
gunzip giga-fren.release2.fixed.fr.gz
tar zxf dev-v2.tgz

python wmt_preprocess.py giga-fren.release2.fixed.en giga-fren.preprocess.en --vocab-file vocab.en
python wmt_preprocess.py giga-fren.release2.fixed.fr giga-fren.preprocess.fr --vocab-file vocab.fr
python wmt_preprocess.py dev/newstest2013.en newstest2013.preprocess.en
python wmt_preprocess.py dev/newstest2013.fr newstest2013.preprocess.fr

rm training-giga-fren.tar
rm giga-fren.release2.fixed.en
rm giga-fren.release2.fixed.fr
rm dev-v2.tgz
rm -r dev
