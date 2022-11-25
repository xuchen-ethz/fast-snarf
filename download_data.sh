mkdir data

echo Downloading motion sequences...

wget https://scanimate.is.tue.mpg.de/media/upload/demo_data/aist_demo_seq.zip
unzip aist_demo_seq.zip -d ./data/
rm aist_demo_seq.zip
mv ./data/gLO_sBM_cAll_d14_mLO1_ch05 ./data/aist_demo

echo Done!


mkdir outputs

echo Downloading pretrained models ...
wget https://dataset.ait.ethz.ch/downloads/fast-snarf/cape.zip
unzip cape.zip -d ./outputs/
rm cape.zip

wget https://dataset.ait.ethz.ch/downloads/fast-snarf/dfaust.zip
unzip dfaust.zip -d ./outputs/
rm dfaust.zip