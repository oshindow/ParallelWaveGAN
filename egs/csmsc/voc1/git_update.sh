version=1.5
message="train aishell3 and all model"

git add /home/xintong/ParallelWaveGAN

git commit -m "$message Version $version"

git tag -a v$version -m "verison $version of the model"

git push origin v$version
git push origin master