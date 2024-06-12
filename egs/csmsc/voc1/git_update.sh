version=1.4
message="try causal and test feature"

git add /home/xintong/ParallelWaveGAN

git commit -m "$message Version $version"

git tag -a v$version -m "verison $version of the model"

git push origin v$version
git push origin master